import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.time import Time

try:
    from rubin_sim.phot_utils import predicted_zeropoint, predicted_zeropoint_hardware

    HAS_RUBIN_SIM = True
except ModuleNotFoundError:
    HAS_RUBIN_SIM = False

from rubin_nights.reference_values import (
    PLATESCALE,
    ZEROPOINT_OFFSETS_LSSTCAM,
    ZEROPOINT_OFFSETS_LSSTCOMCAM,
)

# TODO import from rubin_scheduler instead
SURVEY_START = Time("2025-11-01T12:00:00", scale="tai")

__all__ = ["add_rubin_sim_cols", "consdb_to_opsim"]

logger = logging.getLogger(__name__)


def add_rubin_sim_cols(
    visits: pd.DataFrame,
    instrument: str = "lsstcam",
    predicted_zeropoint_offsets: dict | None = None,
    cols_from: str = "visit1_quicklook",
) -> pd.DataFrame:
    """Add columns that require rubin_sim:
    predicted zeropoint and converted skybackground (mag/sq arcsec).

    Parameters
    ----------
    visits
        The visit information from cdb_{instrument}.visit1 and
        cdb_{instrument}.visit1_quicklook (if available).
    instrument
        The instrument for the visits.
        Used to select the appropriate zeropoint offsets, if not provided.
    predicted_zeropoint_offsets
        Offsets to add to the predicted zeropoint values.
        If None, will pick appropriate defaults based on instrument.
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
    Columns added are:
    zero_point_1s (zero_point[_median] scaled to 1s)
    zero_point_1s_pred (predicted from rubin_sim.predicted_zeropoint)
    clouds (the difference of the above values)
    sky_bg_mag (sky_bg[_median] scaled to mag/arcsecond^2)
    cat_m5 (calculated m5 from zeropoint/sky/readnoise values)
    """
    if not HAS_RUBIN_SIM:
        logger.info("No rubin_sim available, simply returning visits.")
        return visits

    if cols_from.startswith("visit"):
        zero_point_col = "zero_point_median"
        sky_col = "sky_bg_median"
        psf_area_col = "psf_area_median"
        pixel_scale_col = "pixel_scale_median"
    elif cols_from.startswith("ccd"):
        zero_point_col = "zero_point"
        sky_col = "sky_bg"
        psf_area_col = "psf_area"
        pixel_scale_col = "pixel_scale"
    else:
        raise ValueError(
            "cols_from should indicate either ccd or visit table, and start with 'ccd' or 'visit'",
        )

    necessary_cols = [zero_point_col, sky_col]
    for c in necessary_cols:
        if c not in visits.columns:
            logger.error(f"Missing columns for {necessary_cols}. " "Could not add rubin_sim_addons columns.")
            return visits

    # Calculate additional zeropoints and sky columns
    if predicted_zeropoint_offsets is None:
        if instrument.lower() == "lsstcam":
            predicted_zeropoint_offsets = ZEROPOINT_OFFSETS_LSSTCAM
        elif instrument.lower() == "lsstcomcam":
            predicted_zeropoint_offsets = ZEROPOINT_OFFSETS_LSSTCOMCAM
        else:
            predicted_zeropoint_offsets = {"u": 0, "g": 0, "r": 0, "i": 0, "z": 0, "y": 0}

    # Add new columns
    new_cols = [
        "zero_point_1s",
        "zero_point_1s_pred",
        "clouds",
        "sky_bg_mag",
        "cat_m5",
    ]
    new_df = pd.DataFrame(np.zeros((len(visits), len(new_cols))), columns=new_cols, index=visits.index)
    if all(new_cols) in visits.columns:
        logger.debug("All columns already present in visits.")
        return visits
    else:
        for n in new_cols:
            if n in visits.columns:
                visits.drop(labels=n, axis=1, inplace=True)

    visits = visits.merge(new_df, right_index=True, left_index=True)

    # We may have already calculated a "good" pixel scale
    if "pixel_scale_est" not in visits.columns:
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
        visits["pixel_scale_est"] = pixel_scale

    def calc_predicted_zeropoints(x: pd.Series) -> pd.Series:
        if x.exp_time == 0 or np.isnan(x.exp_time) or x.band not in ["u", "g", "r", "i", "z", "y"]:
            # Bail if zero or nan exposure time or not in bandpass dictionary.
            x["zero_point_1s"] = np.nan
            x["zero_point_1s_pred"] = np.nan
            x["sky_bg_mag"] = np.nan
            return x
        # Calculate 1-s 1-e- zeropoints (measured and predicted)
        x["zero_point_1s"] = x[zero_point_col] - 2.5 * np.log10(x.shut_time)
        x["zero_point_1s_pred"] = (
            predicted_zeropoint(x.band, x.airmass, 1) + predicted_zeropoint_offsets[x.band]
        )
        # zp with hardware only would be the expected value for predicting
        # sky counts and for converting sky in electrons to mag
        zp_sky = predicted_zeropoint_hardware(x.band, x.shut_time) + predicted_zeropoint_offsets[x.band]
        x["sky_bg_mag"] = -2.5 * np.log10(x[sky_col] / x.pixel_scale_est**2) + zp_sky
        return x

    visits = visits.apply(calc_predicted_zeropoints, axis=1)
    visits["clouds"] = visits.zero_point_1s_pred - visits.zero_point_1s
    if psf_area_col in visits.columns:
        # psf_area seems like a better choice
        neff = visits[psf_area_col]

        # Calculate predicted m5 with an estimate of readnoise
        noise_instr_sq = 10
        total_noise_sq = neff * (visits[sky_col] + noise_instr_sq)

        snr = 5
        counts_5sigma = (snr**2) / (2) + np.sqrt((snr**4) / (4) + snr**2 * total_noise_sq)
        # We could use the measured zeropoint directly
        # (then in theory should match stats_mag_lim)
        # Or we could use the 'corrected' zeropoint + exposure time
        visits["cat_m5"] = -2.5 * np.log10(counts_5sigma) + visits[zero_point_col]

    return visits


def consdb_to_opsim(consdb_visits: pd.DataFrame) -> pd.DataFrame | None:
    """Minimal conversion from consdb columns to opsim columns.

    Parameters
    ----------
    consdb_visits
        Dataframe of visit + quicklook information from the ConsDB.

    Returns
    -------
    opsim_visits
        Dataframe of visit information reformatted for opsim.
        This is primarily renaming columns.
        The `night` is also added using the first night of the SV
        survey as night=0.
    """
    # Assumes that visits have already been run through augment_visits,
    # with rubin_scheduler and rubin_sim addons available.
    if not HAS_RUBIN_SIM:
        return None

    opsim_mapping = {
        "visit_id": "observationId",
        "s_ra": "fieldRA",
        "s_dec": "fieldDec",
        "sky_rotation": "rotSkyPos",
        "physical_rotator_angle": "rotTelPos",
        "obs_start_mjd": "observationStartMJD",
        "physical_filter": "filter",
        "lst": "observationStartLST",
        "approx_parallactic": "paraAngle",
        "exp_time": "visitExposureTime",
        "dark_time": "visitTime",
        "sky_bg_mag": "skyBrightness",
        "cat_m5": "fiveSigmaDepth",
        "visit_gap": "slewTime",
        "slew_distance": "slewDistance",
        "fwhm_geom": "seeingFwhmGeom",
        "fwhm_eff": "seeingFwhmEff",
        "moon_RA": "moonRA",
        "moon_Dec": "moonDec",
        "moon_alt": "moonAlt",
        "moon_az": "moonAz",
        "moon_distance": "moonDistance",
        "moon_illum": "moonPhase",
        "sun_RA": "sunRA",
        "sun_Dec": "sunDec",
        "sun_alt": "sunAlt",
        "sun_az": "sunAz",
        "atm_500_zenith": "seeingFwhm500",
        "clouds": "cloud_extinction",
    }

    # The critical values for an opsim simulation:
    critical_columns = [
        "s_ra",
        "s_dec",
        "sky_rotation",
        "band",
        "obs_start_mjd",
        "exp_time",
        "scheduler_note",
    ]
    for key in critical_columns:
        if key not in consdb_visits:
            logging.warning("Missing critical columns for opsim input")
            return None

    opsim_visits = consdb_visits.rename(opsim_mapping, axis=1)
    # copy "scheduler_note" to "note" for obs playback at 3.21.0
    # Add a 'night' column aligning with the current default for opsim.
    nexp = np.ones(len(opsim_visits))
    night = np.floor((Time(opsim_visits["observationStartMJD"], format="mjd", scale="tai") - SURVEY_START).jd)
    target_id = np.arange(0, len(opsim_visits), 1)
    dd = pd.DataFrame(
        {"numExposures": nexp, "night": night, "target_id": target_id}, index=opsim_visits.index
    )
    opsim_visits = opsim_visits.merge(dd, left_index=True, right_index=True, how="left")

    return opsim_visits
