import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.time import Time

from .influx_query import InfluxQueryClient
from .observatory_status import get_tma_limits

try:
    from rubin_scheduler.scheduler.model_observatory import KinemModel, rotator_movement, tma_movement
    from rubin_scheduler.site_models import Almanac, SeeingModel
    from rubin_scheduler.utils import (
        Site,
        SysEngVals,
        angular_separation,
        approx_altaz2pa,
        approx_ra_dec2_alt_az,
        rotation_converter,
    )

    HAS_RUBIN_SCHEDULER = True
except ModuleNotFoundError:
    HAS_RUBIN_SCHEDULER = False

logger = logging.getLogger(__name__)

PLATESCALE = 0.2
GAUSSIAN_FWHM_OVER_SIGMA: float = 2.0 * np.sqrt(2.0 * np.log(2.0))

__all__ = ["add_rubin_scheduler_cols", "add_model_slew_times"]


def add_rubin_scheduler_cols(
    visits: pd.DataFrame,
    instrument: str = "lsstcam",
    cols_from: str = "visit1_quicklook",
) -> pd.DataFrame:
    """Add columns that require rubin_scheduler (including Almanac)
    parallactic angle and rotator angle, LST, and moon information.

    Parameters
    ----------
    visits
        The visit information from cdb_{instrument}.visit1 and
        cdb_{instrument}.visit1_quicklook (if available).
    instrument
        The instrument for the visits.
        Used to calculate the approproximate rotTelPos value.
    cols_from
        Use columns expected from the visit1_quicklook
        table or from the ccdvisit1_quicklook table.
        The difference is whether _median is at the end of the column name.


    Returns
    -------
    visits : `pd.DataFrame`
        The visit information, with additional columns added for
        predicted zeropoint values, sky background in magnitudes,
        an estimated m5 depth (from zeropoint + sky).

    Notes
    -----
    Columns calculated and added:
    lst, HA (via astropy)
    moon_alt, moon_az, moon_RA, moon_dec, moon_distance, moon_illum
    (via the rubin_scheduler almanac)
    fwhm_eff, fwhm_geom, fwhm_500_zenith
    (via something close to the rubin_scheduler SeeingModel (but lambda^-0.2)
    approx_pa, approx_rotTelPos (via rubin_scheduler approx values)
    """
    if not HAS_RUBIN_SCHEDULER:
        logger.info("No rubin_scheduler available, simply returning visits.")
        return visits

    if cols_from.startswith("visit"):
        psf_col = "psf_sigma_median"
        pixel_scale_col = "pixel_scale_median"
    elif cols_from.startswith("ccd"):
        psf_col = "psf_sigma"
        pixel_scale_col = "pixel_scale"
    else:
        raise ValueError(
            "cols_from should indicate either ccd or visit table, and start with 'ccd' or 'visit'",
        )

    # Try to add seeing columns, if "psf_sigma_median" in visits.
    if psf_col in visits.columns:
        seeing_cols = [
            "fwhm_eff",
            "fwhm_geom",
            "fwhm_500_zenith",
        ]
        seeing_df = pd.DataFrame(
            np.zeros((len(visits), len(seeing_cols))), columns=seeing_cols, index=visits.index
        )

        for n in seeing_cols:
            if n in visits.columns:
                visits.drop(labels=n, axis=1, inplace=True)

        # replace PLATESCALE with x.pixel_scale_median when available
        pixel_scale: float | npt.NDArray
        if pixel_scale_col in visits.columns:
            pixel_scale = np.where(
                np.isnan(visits[pixel_scale_col].values), PLATESCALE, visits[pixel_scale_col].values
            )
        else:
            pixel_scale = PLATESCALE
        if psf_col in visits.columns:
            seeing_df["fwhm_eff"] = visits[psf_col] * GAUSSIAN_FWHM_OVER_SIGMA * pixel_scale
            seeing_df["fwhm_geom"] = SeeingModel.fwhm_eff_to_fwhm_geom(seeing_df.fwhm_eff)

        sev = SysEngVals()
        wavelen_corrections = np.zeros(len(visits), float)
        for band in visits.band.unique():
            match = np.where(visits.band.values == band)
            if band not in "ugrizy":
                wavelen_corrections[match] = 1
            else:
                # SeeingModel uses 0.3, but RHL says 0.2
                wavelen_corrections[match] = np.power(500 / sev.eff_wavelengths[band], 0.2)
        # SeeingModel uses 0.6 and RHL agrees
        airmass_corrections = np.power(visits.airmass.values, 0.6)
        fwhm_system = 0.4
        # leave this or not? Does system perform differently with airmass?
        fwhm_atmo = np.sqrt((seeing_df.fwhm_eff / 1.16) ** 2 - fwhm_system**2) / 1.04
        seeing_df["fwhm_500_zenith"] = fwhm_atmo / wavelen_corrections / airmass_corrections

        visits = visits.merge(seeing_df, right_index=True, left_index=True)

    # Add new columns
    new_cols = [
        "lst",
        "HA",
        "approx_pa",
        "approx_rotTelPos",
        "moon_alt",
        "moon_az",
        "moon_RA",
        "moon_Dec",
        "moon_distance",
        "moon_illum",
        "sun_alt",
        "sun_az",
        "sun_RA",
        "sun_Dec",
    ]
    new_df = pd.DataFrame(np.zeros((len(visits), len(new_cols))), columns=new_cols, index=visits.index)

    for n in new_cols:
        if n in visits.columns:
            visits.drop(labels=n, axis=1, inplace=True)

    # Add in physical rotator angle, parallactic angle
    # (these will be added by ConsDB in the future
    lsst_loc = Site("LSST")
    times = Time(visits["obs_start_mjd"], format="mjd", scale="tai", location=lsst_loc.to_earth_location())
    lst = times.sidereal_time("mean").deg
    new_df["lst"] = lst
    new_df["HA"] = (visits["s_ra"] - lst) / 360 * 12 % 24

    almanac = Almanac()

    avals = almanac.get_sun_moon_positions(visits["exp_midpt_mjd"].values)
    new_df["sun_alt"] = np.degrees(avals["sun_alt"])
    new_df["sun_az"] = np.degrees(avals["sun_az"])
    new_df["sun_RA"] = np.degrees(avals["sun_RA"])
    new_df["sun_Dec"] = np.degrees(avals["sun_dec"])
    new_df["moon_alt"] = np.degrees(avals["moon_alt"])
    new_df["moon_az"] = np.degrees(avals["moon_az"])
    new_df["moon_RA"] = np.degrees(avals["moon_RA"])
    new_df["moon_Dec"] = np.degrees(avals["moon_dec"])
    new_df["moon_distance"] = angular_separation(
        new_df["moon_RA"].values, new_df["moon_Dec"].values, visits["s_ra"].values, visits["s_dec"].values
    )
    new_df["moon_illum"] = almanac.get_sun_moon_positions(visits["exp_midpt_mjd"].values)["moon_phase"]

    alt, az = approx_ra_dec2_alt_az(
        visits.s_ra.values,
        visits.s_dec.values,
        lsst_loc.latitude,
        lsst_loc.longitude,
        visits.exp_midpt_mjd.values,
        lmst=None,
    )
    pa = approx_altaz2pa(alt, az, lsst_loc.latitude)
    new_df["approx_pa"] = pa

    if instrument.lower() != "latiss":
        if instrument.lower() == "lsstcomcam":
            tele = "comcam"
        else:
            tele = "rubin"
        rc = rotation_converter(telescope=tele)
        rotTelPos = rc.rotskypos2rottelpos(visits.sky_rotation.values, new_df["approx_pa"].values)
        new_df["approx_rotTelPos"] = rotTelPos

    visits = visits.merge(new_df, right_index=True, left_index=True)
    return visits


def add_model_slew_times(
    visits: pd.DataFrame,
    efd_client: InfluxQueryClient,
    model_settle: float = 1,
    dome_crawl: bool = True,
    ideal_tma: float = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ "Add model (applied tma limits plus FBS-default tma limits) calculated
    slewtimes to visits dataframe, in `slew_model` and `slew_model_ideal`.

    This is only applicable to SimonyiTel at present!

    Parameters
    ----------
    visits
        The visit information. Expected to contain columns of
        s_ra, s_dec, sky_rotation, obs_start_mjd and band for slewtime
        calculation.
    efd_client
        Used to query the EFD for the applied TMA limits at the time
        of the visits.
    model_settle
        The amount of settle time to add to the model_slew.
        This should make the model_slew time match the TMAevent time.
        Might vary over time.
    dome_crawl
        Enable dome crawl when calculating slew times, if True.
    ideal_tma
        Model TMA movement value to use for the ideal model, in percent.

    Returns
    -------
    visits_with_slews, slews : `pd.DataFrame`, `pd.DataFrame`
        Same visit information, with additional columns `slew_model`
        and `slew_model_ideal`.

    Notes
    -----
    Since the slew should be calculated from the previous location on the
    sky, subsets of visits that do not include the starting position may
    have inaccurate first slew estimates. Slews are the model slewtime
    *to* the visit (and compare against `visit_gap` for the same visit).

    """
    if not HAS_RUBIN_SCHEDULER:
        logger.info("No rubin_scheduler available, cannot calculate model slew times.")
        return visits, None

    t_start = Time(visits.obs_start_mjd.min(), format="mjd", scale="tai")
    t_end = Time(visits.obs_start_mjd.max(), format="mjd", scale="tai")
    tma_speeds = get_tma_limits(t_start, t_end, efd_client)

    kinematic_model_ideal = KinemModel(mjd0=t_start.mjd - 0.1)
    # When evaluating slew times between actual images, need to
    # remove delay for closed-loop (the image itself represents the delay)
    kinematic_model_ideal.setup_optics(cl_delay=[0, 0])
    kinematic_model_ideal.setup_telescope(
        **tma_movement(ideal_tma),
        altitude_minpos=15,
        altitude_maxpos=86.5,
        azimuth_minpos=-262,
        azimuth_maxpos=262,
    )
    kinematic_model_ideal.setup_camera(**rotator_movement(100), readtime=3.07)
    kinematic_model_ideal.mount_bands(["u", "g", "r", "i", "z", "y"])

    # Slower kinematic model to modify with actual telescope parameters
    # Set up current kinematic model.
    kinematic_model = KinemModel(mjd0=t_start.mjd - 0.1)
    # When evaluating slew times between actual images, need to
    # remove delay for closed-loop (the image itself represents the delay)
    kinematic_model.setup_optics(cl_delay=[0, 0])
    kinematic_model.setup_camera(band_changetime=140, **rotator_movement(100), readtime=3.07)
    kinematic_model.mount_bands(["u", "g", "r", "i", "z", "y"])

    model_slewtimes = {}  # current performance model
    model_slewtimes_ideal = {}  # ideal performance model

    for dayobs in visits.day_obs.unique():
        night_visits = visits.query("day_obs == @dayobs").sort_values(by="seq_num")
        if len(night_visits) > 0:
            # Park the kinematic models at the start of the night
            kinematic_model.park()
            kinematic_model_ideal.park()
            # Now sequentially slew through visits
            for visitid, v in night_visits.iterrows():
                last_idx = np.where(tma_speeds.index.values - np.datetime64(v.obs_start) < 0)[0][-1]
                tma = dict(tma_speeds.iloc[last_idx])
                tma["settle_time"] = model_settle
                # Change speeds on non-ideal kinematic model
                kinematic_model.setup_telescope(**tma)

                if np.isnan(v.s_ra) | np.isnan(v.s_dec):
                    model_slewtimes[visitid] = np.nan
                    model_slewtimes_ideal[visitid] = np.nan
                else:
                    ra_rad = np.array([np.radians(v.s_ra)])
                    dec_rad = np.array([np.radians(v.s_dec)])
                    sky_angle = np.array([np.radians(v.sky_rotation)])
                    mjd = np.array([v.obs_start_mjd])
                    band = np.array([v.band])
                    slewtime = kinematic_model.slew_times(
                        ra_rad,
                        dec_rad,
                        mjd,
                        rot_sky_pos=sky_angle,
                        bandname=band,
                        lax_dome=dome_crawl,
                        update_tracking=True,
                    )
                    if isinstance(slewtime, float):
                        model_slewtimes[visitid] = slewtime
                    else:
                        model_slewtimes[visitid] = slewtime[0]

                    slewtime = kinematic_model_ideal.slew_times(
                        ra_rad,
                        dec_rad,
                        mjd,
                        rot_sky_pos=sky_angle,
                        bandname=band,
                        lax_dome=dome_crawl,
                        update_tracking=True,
                    )
                    if isinstance(slewtime, float):
                        model_slewtimes_ideal[visitid] = slewtime
                    else:
                        model_slewtimes_ideal[visitid] = slewtime[0]

    slewing = pd.DataFrame(
        [model_slewtimes, model_slewtimes_ideal], index=["slew_model", "slew_model_ideal"]
    ).T
    if "visit_gap" in visits:
        slewing["model_gap"] = visits.visit_gap - slewing.slew_model

    # Add also the distance on the sky between the visits (degrees)
    # This isn't always the slew distance, but it's the best we can do here
    distances = angular_separation(
        visits.s_ra[1:].values, visits.s_dec[1:].values, visits.s_ra[0:-1].values, visits.s_dec[0:-1].values
    )
    # Special - in case there is only one value
    if isinstance(distances, float):
        distances = np.array([distances])
    slewing["slew_distance"] = np.concatenate([np.array([0]), distances])

    visits = visits.merge(slewing, right_index=True, left_index=True)
    return visits, slewing
