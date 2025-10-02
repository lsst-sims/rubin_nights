import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.time import Time

try:
    from rubin_sim.phot_utils import calc_neff, predicted_zeropoint, predicted_zeropoint_hardware

    HAS_RUBIN_SIM = True
except ModuleNotFoundError:
    HAS_RUBIN_SIM = False

__all__ = ["add_rubin_sim_cols", "consdb_to_opsim"]

logger = logging.getLogger(__name__)

ZEROPOINT_OFFSETS_LSSTCAM = {"u": 0.12, "g": 0.09, "r": 0.15, "i": 0.14, "z": 0.15, "y": 0.13}
# lsstcomcam offsets based on refcats at the time of processing
ZEROPOINT_OFFSETS_LSSTCOMCAM = {"u": 0.26, "g": -0.14, "r": -0.09, "i": -0.10, "z": -0.13, "y": -0.18}
# lsstcomcam offsets for DP1 are probably 0 although might be
ZEROPOINT_OFFSETS_DP1 = {"u": 0.03, "g": 0.01, "r": 0.00, "i": 0.00, "z": -0.00, "y": 0.01}
# Approximate pixel scale
PLATESCALE = 0.2
GAUSSIAN_FWHM_OVER_SIGMA: float = 2.0 * np.sqrt(2.0 * np.log(2.0))


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
        psf_col = "psf_sigma_median"
        pixel_scale_col = "pixel_scale_median"
    elif cols_from.startswith("ccd"):
        zero_point_col = "zero_point"
        sky_col = "sky_bg"
        psf_col = "psf_sigma"
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

    def calc_predicted_zeropoints(x: pd.Series) -> pd.Series:
        if x.exp_time == 0 or np.isnan(x.exp_time) or x.band not in ["u", "g", "r", "i", "z", "y"]:
            # Bail if zero or nan exposure time or not in bandpass dictionary.
            x.zero_point_1s = np.nan
            x.zero_point_1s_pred = np.nan
            x.sky_bg_mag = np.nan
            x.cat_m5 = np.nan
            return x
        # Calculate 1-s 1-e- zeropoints (measured and predicted)
        x.zero_point_1s = x[zero_point_col] - 2.5 * np.log10(x.exp_time)
        x.zero_point_1s_pred = predicted_zeropoint(x.band, x.airmass, 1) + predicted_zeropoint_offsets[x.band]
        # Convert sky counts/pixel to magnitude/arcsecond^2
        zp_sky = predicted_zeropoint_hardware(x.band, x.shut_time) + predicted_zeropoint_offsets[x.band]
        # replace PLATESCALE with x.pixel_scale_median when available
        if pixel_scale_col in x and not np.isnan(x[pixel_scale_col]):
            pixel_scale = x[pixel_scale_col]
        else:
            pixel_scale = PLATESCALE
        x.sky_bg_mag = -2.5 * np.log10(x[sky_col] / pixel_scale**2) + zp_sky
        return x

    visits = visits.apply(calc_predicted_zeropoints, axis=1)
    visits.clouds = visits.zero_point_1s_pred - visits.zero_point_1s
    if psf_col in visits.columns:
        # Calculate predicted m5 with an estimate of readnoise
        noise_instr_sq = 10
        # psf_area would be good to use but going from fwhm_eff
        # makes us more internally self-consistent
        pixel_scale: float | npt.NDArray
        if pixel_scale_col in visits.columns:
            pixel_scale = np.where(
                np.isnan(visits[pixel_scale_col].values), PLATESCALE, visits[pixel_scale_col].values
            )
        else:
            pixel_scale = PLATESCALE
        fwhm_eff = visits[psf_col] * GAUSSIAN_FWHM_OVER_SIGMA * pixel_scale
        neff = calc_neff(fwhm_eff, pixel_scale)
        total_noise_sq = neff * (visits[sky_col] + noise_instr_sq)

        snr = 5
        counts_5sigma = (snr**2) / (2) + np.sqrt((snr**4) / (4) + snr**2 * total_noise_sq)
        visits.cat_m5 = -2.5 * np.log10(counts_5sigma) + visits[zero_point_col]

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
        "obs_start_mjd": "observationStartMJD",
        "exp_time": "visitExposureTime",
        "dark_time": "visitTime",
        "sky_bg_median_mag": "skyBrightness",
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
        "fwhm_500_zenith": "FWHM_500",
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
    # Appropriate for SV survey
    opsim_visits["nexp"] = 1
    opsim_visits["night"] = np.floor(
        (
            Time(opsim_visits["observationStartMJD"], format="mjd", scale="tai")
            - Time("2025-06-20T12:00:00", scale="tai")
        ).jd
    )

    return opsim_visits
