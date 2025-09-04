import logging
import warnings

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from .rubin_scheduler_addons import add_rubin_scheduler_cols
from .rubin_sim_addons import add_rubin_sim_cols

logger = logging.getLogger(__name__)

BAD_VISITS_LSSTCAM = (
    "https://raw.githubusercontent.com/lsst-dm/excluded_visits/" "refs/heads/main/LSSTCam/bad.ecsv"
)
BAD_VISITS_LSSTCOMCAM = (
    "https://raw.githubusercontent.com/lsst-dm/excluded_visits/" "refs/heads/main/LSSTComCam/bad.ecsv"
)

__all__ = ["augment_visits", "fetch_excluded_visits", "exclude_visits"]


def augment_visits(
    visits: pd.DataFrame,
    instrument: str = "lsstcam",
    skip_rs_columns: bool = False,
    predicted_zeropoint_offsets: dict | None = None,
) -> pd.DataFrame:
    """Add additional columns to the dataframe resulting from
    querying the visit1 + visit1_quicklook tables.
    Calls `add_rubin_scheduler_cols` and `add_rubin_sim_cols`, after
    adding some basic additional information.

    Parameters
    ----------
    visits
        The visit information from cdb_{instrument}.visit1 and
        cdb_{instrument}.visit1_quicklook (if available).
    instrument
        The instrument for the visits.
        Used to calculate the approproximate rotTelPos value.
    skip_rs_columns
        Skip calculation of any columns that require rubin_scheduler
        or rubin_sim, even if those packages are installed.
        This will only sort the visits in time, calculate `visit_gap`,
        and calculate the boresight coordinates in ecliptic and galactic
        coordinates.
    predicted_zeropoint_offsets
        Offsets to add to the predicted zeropoint values.
        If None, will pick appropriate defaults based on instrument.

    Returns
    -------
    visits : `pd.DataFrame`
        The visit information, with additional columns added for
        predicted zeropoint values, sky background in magnitudes,
        an estimated m5 depth (from zeropoint + sky), as well
        as an approximate rotTelPos (likely off by ~1 deg).
        Some columns may be reformatted for dtypes.

    Notes
    -----
    In addition to the columns added by `add_rubin_scheduler_cols` and
    `add_rubin_sim_cols`, this will add the `visit_gap` values as well
    as translations of the coordinates into galactic and ecliptic coordinates.
    Some columns which can occasionally be Object due to None values are
    also converted explicitly back to floats.
    """
    if len(visits) == 0:
        return visits

    # Replace Nones or Nans in important string fields
    values = dict([[e, ""] for e in ["science_program", "target_name", "observation_reason"]])
    visits.fillna(value=values, inplace=True)

    # If no quicklook processing was run, these columns may be object:
    columns_to_floats = [
        "s_ra",
        "s_dec",
        "exp_midpt_mjd",
        "airmass",
        "zero_point_median",
        "zero_point_min",
        "zero_point_max",
        "psf_sigma_median",
        "psf_sigma_min",
        "psf_sigma_max",
        "psf_area_median",
        "psf_area_min",
        "psf_area_max",
        "sky_bg_median",
        "sky_bg_min",
        "sky_bg_max",
        "pixel_scale_median",
    ]
    for col in columns_to_floats:
        if col in visits:
            visits[col] = visits[col].astype("float")

    visits.sort_values(by="exp_midpt_mjd", inplace=True)

    # Add time between visits
    prev_visit_start = np.concatenate([np.array([0]), visits.obs_start_mjd[0:-1]])
    prev_visit_end = np.concatenate([np.array([0]), visits.obs_end_mjd[0:-1]])
    visit_gap = np.concatenate(
        [np.array([0]), (visits.obs_start_mjd[1:].values - visits.obs_end_mjd[:-1].values) * 24 * 60 * 60]
    )  # seconds

    coordinates = SkyCoord(visits.s_ra, visits.s_dec, unit=u.degree, frame="icrs")
    # We get runtime warnings here where nans are present
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ecliptic = coordinates.transform_to("geocentricmeanecliptic")

    new_df = pd.DataFrame(
        [
            prev_visit_start,
            prev_visit_end,
            visit_gap,
            ecliptic.lat.deg,
            ecliptic.lon.deg,
            coordinates.galactic.b.deg,
            coordinates.galactic.l.deg,
        ],
        index=[
            "prev_obs_start_mjd",
            "prev_obs_end_mjd",
            "visit_gap",
            "eclip_lat",
            "eclip_lon",
            "gal_lat",
            "gal_lon",
        ],
        columns=visits.index,
    ).T
    visits = visits.merge(new_df, right_index=True, left_index=True)

    if not skip_rs_columns:
        visits = add_rubin_scheduler_cols(visits, instrument)
        visits = add_rubin_sim_cols(visits, instrument, predicted_zeropoint_offsets)

    return visits


def fetch_excluded_visits(instrument: str = "lsstcam") -> list[str]:
    """Retrieve excluded visit list from the instrument-appropriate
    BAD_VISITS URI at github @ lsst-dm/excluded_visits.

    The bad visit list at this repo is very incomplete.
    See also `targets_and_visits.flag_potential_bad_visits`.

    Parameters
    ----------
    instrument
        Which bad.ecsv file to retrieve.
        The options are lsstcam or lsstcomcam.

    Returns
    -------
    bad_visit_ids : `list` [ `str` ]
        The bad visit_ids from the github repo bad.ecsv file.
    """
    if instrument.lower() == "lsstcam":
        uri = BAD_VISITS_LSSTCAM
    elif instrument.lower() == "lsstcomcam":
        uri = BAD_VISITS_LSSTCOMCAM
    bad_visits = pd.read_csv(uri, comment="#")
    bad_visit_ids = bad_visits.exposure.to_list()
    return bad_visit_ids


def exclude_visits(visits: pd.DataFrame, bad_visit_ids: list[str]) -> pd.DataFrame:
    """Remove the visits_ids in bad_visit_ids from visits.

    Parameters
    ----------
    visits : `pd.DataFrame`
        A dataframe containing visit information, with visit_id values.
    bad_visit_ids : `list` [ `str` ]
        The list of bad visit_ids to remove.
        This could be generated from
        rubin_nights.consdb.fetch_excluded_visits or
        rubin_nights.targets_and_visits.flag_potential_bad_visits
        or any other list of unwanted visit_ids.

    Returns
    -------
    good_visits : `pd.DataFrame`
        The visits dataframe but with bad_visit_ids removed.
    """
    if bad_visit_ids is not None and len(bad_visit_ids) > 0:
        return visits.query("visit_id not in @bad_visit_ids")
