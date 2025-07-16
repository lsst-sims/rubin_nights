import logging

import numpy as np
import pandas as pd
from astropy.time import Time

try:
    from rubin_sim.phot_utils import predicted_zeropoint, predicted_zeropoint_hardware

    HAS_RUBIN_SIM = True
except ModuleNotFoundError:
    HAS_RUBIN_SIM = False

__all__ = ["add_rubin_sim_cols", "consdb_to_opsim"]

logger = logging.getLogger(__name__)

EFFTIME_REF_MAGS = {"u": 23.70, "g": 24.97, "r": 24.52, "i": 24.13, "z": 23.56, "y": 22.55}
EFFTIME_TIME = 30
ZEROPOINT_OFFSETS_LSSTCAM = {"u": 0.04, "g": 0.06, "r": 0.11, "i": 0.09, "z": 0.11, "y": 0.08}
# lsstcomcam offsets based on refcats at the time of processing
ZEROPOINT_OFFSETS_LSSTCOMCAM = {"u": 0.26, "g": -0.14, "r": -0.09, "i": -0.10, "z": -0.13, "y": -0.18}
PLATESCALE = 0.2


def add_rubin_sim_cols(
    visits: pd.DataFrame,
    instrument: str = "lsstcam",
    predicted_zeropoint_offsets: dict | None = None,
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

    Returns
    -------
    visits : `pd.DataFrame`
        The visit information, with additional columns added for
        predicted zeropoint values, sky background in magnitudes,
        an estimated m5 depth (from zeropoint + sky).
    """
    if not HAS_RUBIN_SIM:
        logger.info("No rubin_sim available, simply returning visits.")
        return visits

    necessary_cols = ["zero_point_median", "psf_sigma_median", "sky_bg_median"]
    for c in necessary_cols:
        if c not in visits.columns:
            logger.info("Missing columns for psf_sigma_median, zero_point_median or sky_bg_median.")
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
        "sky_bg_median_mag",
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

    def calc_predicted_zeropoints(x):
        if x.exp_time == 0 or np.isnan(x.exp_time) or x.band not in ["u", "g", "r", "i", "z", "y"]:
            # Bail if zero or nan exposure time or not in bandpass dictionary.
            x.zero_point_1s = np.nan
            x.zero_point_1s_pred = np.nan
            x.sky_bg_median_mag = np.nan
            x.cat_m5 = np.nan
            return x
        # Calculate 1-s 1-e- zeropoints (measured and predicted)
        x.zero_point_1s = x.zero_point_median - 2.5 * np.log10(x.exp_time)
        x.zero_point_1s_pred = predicted_zeropoint(x.band, x.airmass, 1) + predicted_zeropoint_offsets[x.band]
        # Convert sky counts/pixel to magnitude/arcsecond^2
        zp_sky = predicted_zeropoint_hardware(x.band, x.shut_time) + predicted_zeropoint_offsets[x.band]
        x.sky_bg_median_mag = -2.5 * np.log10(x.sky_bg_median / PLATESCALE**2) + zp_sky
        return x

    visits = visits.apply(calc_predicted_zeropoints, axis=1)
    visits.clouds = visits.zero_point_1s_pred - visits.zero_point_1s
    # Calculate predicted m5 with an estimate of readnoise
    noise_instr_sq = 13
    total_noise_sq = visits.psf_area_median * (visits.sky_bg_median + noise_instr_sq)
    snr = 5
    counts_5sigma = (snr**2) / (2) + np.sqrt((snr**4) / (4) + snr**2 * total_noise_sq)
    visits.cat_m5 = -2.5 * np.log10(counts_5sigma) + visits.zero_point_median

    return visits


def consdb_to_opsim(visits: pd.DataFrame) -> pd.DataFrame | None:
    """Minimal conversion from consdb columns to opsim columns."""
    # Assumes that visits have already been run through augment_visits,
    # with rubin_scheduler and rubin_sim addons available.
    if not HAS_RUBIN_SIM:
        return None

    opsim_mapping = {
        "visit_id": "observationId",
        "s_ra": "fieldRA",
        "s_dec": "fieldDec",
        "sky_rotation": "rotSkyPos",
        "obs_start_mjd": "observationStartMJD",
        "exp_time": "visitExposureTime",
        "dark_time": "visitTime",
        "sky_bg_median_mag": "skyBrightness",
        "cat_m5": "fiveSigmaDepth",
        "visit_gap": "slewTime",
        "slew_distance": "slewDistance",
        "fwhm_geom": "seeingFwhmGeom",
        "fwhm_eff": "seeingFwhmEff",
        "moon_illum": "moonPhase",
        "fwhm_500_zenith": "FWHM_500",
        "clouds": "cloud_extinction",
    }

    for key in opsim_mapping.keys():
        if key not in visits:
            logging.warning("Run consdb.augment_visits first")
            return None

    opsim = visits.rename(opsim_mapping, axis=1)
    # Appropriate for SV survey
    opsim["nexp"] = 1
    opsim["night"] = np.floor(
        (
            Time(opsim["observationStartMJD"], format="mjd", scale="tai")
            - Time("2025-06-20T12:00:00", scale="tai")
        ).jd
    )


    return opsim
