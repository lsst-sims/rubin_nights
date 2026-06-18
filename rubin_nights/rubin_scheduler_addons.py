import logging
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.time import Time

from .influx_query import InfluxQueryClient, convert_time_to_tz_datetime
from .observatory_status import get_tma_limits
from .reference_values import PLATESCALE, SIGMA_TO_FWHM

try:
    from rubin_scheduler.scheduler.model_observatory import KinemModel, rotator_movement, tma_movement
    from rubin_scheduler.site_models import Almanac, SeeingModel
    from rubin_scheduler.utils import (
        Site,
        SysEngVals,
        angular_separation,
        approx_altaz2pa,
        approx_ra_dec2_alt_az,
    )

    HAS_RUBIN_SCHEDULER = True
except ModuleNotFoundError:
    HAS_RUBIN_SCHEDULER = False

logger = logging.getLogger(__name__)

SKIPTIME = 600.0 / 60 / 60 / 24  # a big slew in JD/days
CAM_FWHM = 0.207  # arcseconds

__all__ = ["add_rubin_scheduler_cols", "add_model_slew_times"]


def add_rubin_scheduler_cols(
    visits: pd.DataFrame,
    cols_from: str = "visit1_quicklook",
) -> pd.DataFrame:
    """Add columns that require rubin_scheduler (including Almanac)
    parallactic angle and rotator angle, LST, and moon information.

    Parameters
    ----------
    visits
        The visit information from cdb_{instrument}.visit1 and
        cdb_{instrument}.visit1_quicklook (if available).
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
        psf_area_col = "psf_area_median"
        pixel_scale_col = "pixel_scale_median"
    elif cols_from.startswith("ccd"):
        psf_col = "psf_sigma"
        psf_area_col = "psf_area"
        pixel_scale_col = "pixel_scale"
    else:
        raise ValueError(
            "cols_from should indicate either ccd or visit table, and start with 'ccd' or 'visit'",
        )

    # Try to add seeing columns, if psf_sigma column in visits.
    if psf_col in visits.columns:
        # replace PLATESCALE with x.pixel_scale_median when available and good
        pixel_scale: float | npt.NDArray
        if pixel_scale_col in visits.columns:
            pixel_scale = np.where(
                np.isnan(visits[pixel_scale_col].array), PLATESCALE, visits[pixel_scale_col].array
            )
            # Remove nonsense values
            pixel_scale = np.where(visits[pixel_scale_col].array > PLATESCALE * 2.5, PLATESCALE, pixel_scale)
        else:
            pixel_scale = PLATESCALE

        # psf_sigma -> fwhm_geom ("fwhm_second_moment")
        fwhm_geom = visits[psf_col] * SIGMA_TO_FWHM * pixel_scale
        # fwhm_geom -> fwhm_eff .. perhaps incorrect for rubin + PSF model
        fwhm_eff_geom = SeeingModel.fwhm_geom_to_fwhm_eff(fwhm_geom)
        # n_eff -> fwhm_eff
        fwhm_eff = np.sqrt(visits[psf_area_col] / 2.266) * pixel_scale

        sev = SysEngVals()
        wavelen_corrections = np.zeros(len(visits), float)
        for band in visits.band.unique():
            match = np.where(visits.band.values == band)
            if band not in "ugrizy":
                wavelen_corrections[match] = 1
            else:
                # SeeingModel uses 0.3, but summit_utils (+RHL) says 0.2
                wavelen_corrections[match] = np.power(500 / sev.eff_wavelengths[band], 0.2)
        # SeeingModel uses 0.6 and summit_utils agrees
        airmass_corrections = np.power(visits.airmass.to_numpy(), 0.6)
        # Note: these corrections *multiply* the 500nm/zenith to the
        # actual airmass/bandpass values (so divide the actual values to get
        # back to 500nm/zenith). This is the opposite sense to summit_utils.

        if "aos_fwhm" in visits.columns and "donut_blur_fwhm" in visits.columns:
            idiq_aos_cam = np.sqrt(visits.aos_fwhm**2 + CAM_FWHM**2)
            # donut_blur (especially in y band) can be larger than fwhm
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                idiq_donut_blur = np.sqrt(fwhm_geom**2 - visits.donut_blur_fwhm**2 + CAM_FWHM**2)
                atm_fwhm_donut_blur = np.sqrt(visits.donut_blur_fwhm**2 - CAM_FWHM**2)
                atm_fwhm_aos_cam = np.sqrt(fwhm_geom**2 - idiq_aos_cam**2)
            atm_500_zenith = atm_fwhm_aos_cam / wavelen_corrections / airmass_corrections
            atm_500_zenith_donut_blur = atm_fwhm_donut_blur / wavelen_corrections / airmass_corrections
            # Simple quadrature
            fwhm_500_zenith = np.sqrt(atm_500_zenith**2 + idiq_aos_cam**2)
            seeing_df = pd.DataFrame(
                {
                    "fwhm_eff": fwhm_eff,
                    "fwhm_geom": fwhm_geom,
                    "fwhm_eff_geom": fwhm_eff_geom,
                    "fwhm_500_zenith": fwhm_500_zenith,
                    "atm_fwhm": atm_fwhm_aos_cam,
                    "atm_500_zenith": atm_500_zenith,
                    "idiq_aos_cam": idiq_aos_cam,
                    "atm_fwhm_donut_blur": atm_fwhm_donut_blur,
                    "atm_500_zenith_donut_blur": atm_500_zenith_donut_blur,
                    "idiq_donut_blur": idiq_donut_blur,
                    "pixel_scale_est": pixel_scale,
                }
            )

        else:
            seeing_df = pd.DataFrame(
                {"fwhm_eff": fwhm_eff, "fwhm_geom": fwhm_geom, "pixel_scale_est": pixel_scale}
            )

        for col in seeing_df.columns:
            if col in visits.columns:
                visits.drop(labels=col, axis=1, inplace=True)

        visits = visits.merge(seeing_df, right_index=True, left_index=True)

    # Add more columns about sky conditions
    lsst_loc = Site("LSST")
    times = Time(visits.obs_start_mjd, format="mjd", scale="tai", location=lsst_loc.to_earth_location())
    lst = times.sidereal_time("mean").deg
    hour_angle = (visits.s_ra - lst) / 360 * 12 % 24

    if "altitude" in visits and "azimuth" in visits:
        alt = visits.altitude
        az = visits.azimuth
    else:
        alt, az = approx_ra_dec2_alt_az(
            visits.s_ra.array,
            visits.s_dec.array,
            lsst_loc.latitude,
            lsst_loc.longitude,
            visits.exp_midpt_mjd.array,
            lmst=None,
        )
    approx_parallactic = approx_altaz2pa(alt, az, lsst_loc.latitude)

    almanac = Almanac()
    almanac_values = almanac.get_sun_moon_positions(visits.exp_midpt_mjd.array)
    moon_RA = np.degrees(almanac_values["moon_RA"])
    moon_dec = np.degrees(almanac_values["moon_dec"])
    moon_distance = angular_separation(moon_RA, moon_dec, visits.s_ra.to_numpy(), visits.s_dec.to_numpy())
    moon_illum = almanac.get_sun_moon_positions(visits.exp_midpt_mjd.array)["moon_phase"]

    new_df = pd.DataFrame(
        {
            "lst": lst,
            "HA": hour_angle,
            "approx_parallactic": approx_parallactic,
            "sun_alt": np.degrees(almanac_values["sun_alt"]),
            "sun_az": np.degrees(almanac_values["sun_az"]),
            "sun_RA": np.degrees(almanac_values["sun_RA"]),
            "sun_Dec": np.degrees(almanac_values["sun_dec"]),
            "moon_alt": np.degrees(almanac_values["moon_alt"]),
            "moon_az": np.degrees(almanac_values["moon_az"]),
            "moon_RA": moon_RA,
            "moon_Dec": moon_dec,
            "moon_distance": moon_distance,
            "moon_illum": moon_illum,
        },
        index=visits.index,
    )

    for n in new_df.columns:
        if n in visits.columns:
            visits.drop(labels=n, axis=1, inplace=True)

    visits = visits.merge(new_df, right_index=True, left_index=True)
    return visits


def add_model_slew_times(
    visits: pd.DataFrame,
    efd_client: InfluxQueryClient,
    model_settle: float = 1,
    dome_crawl: bool = False,
    slew_while_changing_filter: bool = False,
    ideal_tma: float = 40,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
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
        Might vary over time. In general this is 1s -- however
        the actual visit gap is longer than just TMA.
    dome_crawl
        Enable dome crawl when calculating slew times, if True.
    slew_while_changing_filter
        Slew and change filter at the same time, or if False - sequentially.
    ideal_tma
        Model TMA movement value to use for the ideal model, in percent.

    Returns
    -------
    visits_with_slews, slews : `pd.DataFrame`, `pd.DataFrame` or None
        Same visit information, with additional columns `slew_model`
        and `slew_model_ideal`.
        If rubin_scheduler is not available, `slews` will be None.

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
    readtime = 3.07
    kinematic_model_ideal = KinemModel(mjd0=t_start.mjd - 0.1)
    # When evaluating slew times between actual images, need to
    # remove delay for closed-loop (the image itself represents the delay)
    # also .. we're not adding this delay on-sky.
    kinematic_model_ideal.setup_optics(cl_delay=[0, 0])
    kinematic_model_ideal.setup_telescope(
        **tma_movement(ideal_tma),
        altitude_minpos=15,
        altitude_maxpos=86.5,
        azimuth_minpos=-262,
        azimuth_maxpos=262,
    )
    kinematic_model_ideal.setup_camera(**rotator_movement(100), band_changetime=120, readtime=readtime)
    kinematic_model_ideal.mount_bands(["u", "g", "r", "i", "z", "y"])

    # Slower kinematic model to modify with actual telescope parameters
    # Set up current kinematic model.
    kinematic_model = KinemModel(mjd0=t_start.mjd - 0.1)
    # When evaluating slew times between actual images, need to
    # remove delay for closed-loop (the image itself represents the delay)
    kinematic_model.setup_optics(cl_delay=[0, 0])
    kinematic_model.setup_camera(**rotator_movement(100), band_changetime=120, readtime=readtime)
    kinematic_model.mount_bands(["u", "g", "r", "i", "z", "y"])

    model_slewtimes = {}  # current performance model
    model_slewtimes_ideal = {}  # ideal performance model
    tma_alt_maxv = {}
    tma_az_maxv = {}

    utc_tz_visit_times = convert_time_to_tz_datetime(
        Time(visits.obs_start_mjd, format="mjd", scale="tai").utc
    )
    visits["utc_tz_obs_start"] = utc_tz_visit_times

    for dayobs in visits.day_obs.unique():
        night_visits = visits.query("day_obs == @dayobs").sort_values(by="seq_num")
        if len(night_visits) > 0:
            # Park the kinematic models at the start of the night
            kinematic_model.park()
            kinematic_model_ideal.park()
            # Now sequentially slew through visits
            for visitid, v in night_visits.iterrows():
                last_idx = np.where(tma_speeds.index <= v.utc_tz_obs_start)[0][-1]
                tma = dict(tma_speeds.iloc[last_idx])
                tma["settle_time"] = model_settle
                # Change speeds on non-ideal kinematic model
                kinematic_model.setup_telescope(**tma)
                tma_alt_maxv[visitid] = tma["altitude_maxspeed"]
                tma_az_maxv[visitid] = tma["azimuth_maxspeed"]
                if np.isnan(v.s_ra) | np.isnan(v.s_dec):
                    model_slewtimes[visitid] = np.nan
                    model_slewtimes_ideal[visitid] = np.nan
                else:
                    ra_rad = np.array([np.radians(v.s_ra)])
                    dec_rad = np.array([np.radians(v.s_dec)])
                    sky_angle = np.array([np.radians(v.sky_rotation)])
                    # MJD should be time at start of slew
                    # But we may have large gaps or skipped visits
                    if (v.obs_start_mjd - v.prev_obs_end_mjd) > SKIPTIME:
                        mjd = v.obs_start_mjd
                        min_overhead = 0.0
                    else:
                        mjd = v.prev_obs_end_mjd
                        min_overhead = readtime
                    band = np.array([v.band])
                    slewtime = kinematic_model.slew_times(
                        ra_rad,
                        dec_rad,
                        mjd,
                        rot_sky_pos=sky_angle,
                        bandname=band,
                        lax_dome=dome_crawl,
                        slew_while_changing_filter=slew_while_changing_filter,
                        constant_band_changetime=False,
                        update_tracking=True,
                    )
                    if isinstance(slewtime, float):
                        model_slewtimes[visitid] = max(slewtime, min_overhead)
                    else:
                        model_slewtimes[visitid] = max(slewtime[0], min_overhead)

                    slewtime = kinematic_model_ideal.slew_times(
                        ra_rad,
                        dec_rad,
                        mjd,
                        rot_sky_pos=sky_angle,
                        bandname=band,
                        lax_dome=True,
                        slew_while_changing_filter=slew_while_changing_filter,
                        constant_band_changetime=False,
                        update_tracking=True,
                    )
                    if isinstance(slewtime, float):
                        model_slewtimes_ideal[visitid] = max(slewtime, min_overhead)
                    else:
                        model_slewtimes_ideal[visitid] = max(slewtime[0], min_overhead)

    slewing = pd.DataFrame(
        {
            "slew_model": model_slewtimes,
            "slew_model_ideal": model_slewtimes_ideal,
            "tma_alt_maxv": tma_alt_maxv,
            "tma_ax_maxv": tma_az_maxv,
        },
        index=visits.index,
    )

    if "visit_gap" in visits:
        slewing["model_gap"] = visits.visit_gap - slewing.slew_model

    # Add also the distance on the sky between the visits (degrees)
    # This isn't always the slew distance, but it's the best we can do here
    distances = angular_separation(
        visits.s_ra[1:].to_numpy(),
        visits.s_dec[1:].to_numpy(),
        visits.s_ra[0:-1].to_numpy(),
        visits.s_dec[0:-1].to_numpy(),
    )
    slewing["slew_distance"] = np.concatenate([np.array([0]), distances])

    visits = visits.merge(slewing, right_index=True, left_index=True)
    return visits, slewing
