"""Execute queries for the ConsDB."""

import logging
import warnings

import astropy.units as u
import httpx
import numpy as np
import pandas as pd
import pyvo
from astropy.coordinates import SkyCoord
from astropy.time import Time

from .rubin_scheduler_addons import add_rubin_scheduler_cols
from .rubin_sim_addons import add_rubin_sim_cols

logger = logging.getLogger(__name__)

__all__ = ["fetch_excluded_visits", "ConsDbTap", "ConsDbFastAPI"]


GAUSSIAN_FWHM_OVER_SIGMA: float = 2.0 * np.sqrt(2.0 * np.log(2.0))
PLATESCALE = 0.2


BAD_VISITS_LSSTCAM = (
    "https://raw.githubusercontent.com/lsst-dm/excluded_visits/" "refs/heads/main/LSSTCam/bad.ecsv"
)
BAD_VISITS_LSSTCOMCAM = (
    "https://raw.githubusercontent.com/lsst-dm/excluded_visits/" "refs/heads/main/LSSTComCam/bad.ecsv"
)


def fetch_excluded_visits(instrument: str = "lsstcam") -> list[str]:
    """Retrieve excluded visit list from the instrument-appropriate
    BAD_VISITS URI at github @ lsst-dm/excluded_visits.

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


class ConsDb:

    def query(self, query) -> pd.DataFrame:
        """This is implemented in the child classes,
        according to the interface used to access the ConsDB.
        """
        raise NotImplementedError

    def get_visits(
        self,
        instrument: str,
        t_start: Time | None = None,
        t_end: Time | None = None,
        visit_constraint: str | None = None,
        augment_visits: bool = True,
    ) -> pd.DataFrame:
        """ "Fetch visit and quicklook values from the ConsDB.

        Parameters
        ----------
        instrument
            The instrument to search for.
            Typical values would include lsstcomcam, latiss, and lsstcam.
            See https://sdm-schemas.lsst.io/ for more details.
        t_start
            The earliest time to match obs_start.
        t_end
            The latest time to match obs_start.
        visit_constraint
            A constraint to apply to the cdb_{instrument}.visit1 table.
            Example: `"science_program = 'BLOCK-365'"`
        augment_visits
            If True, immediately call consdb.augment_visits after fetching
            visit1 and visit1_quicklook values from the ConsDB.

        Returns
        -------
        visits : `pd.DataFrame`
            The visit information from cdb_{instrument}.visit1 and
            cdb_{instrument}.visit1_quicklook (if available).
            Additional information may be added, such as `visit_gap`.
        """

        query = (
            f"select *, q.* from  cdb_{instrument}.visit1 "
            f"left join cdb_{instrument}.visit1_quicklook as q "
            f"on visit1.visit_id = q.visit_id "
        )
        constraint = []
        if t_start is not None:
            constraint.append(f" obs_start_mjd >= {t_start.mjd} ")
        if t_end is not None:
            constraint.append(f" obs_start_mjd <= {t_end.mjd} ")
        if visit_constraint is not None:
            constraint.append(f" ({visit_constraint}) ")
        constraint = "and".join(constraint)
        if len(constraint) > 0:
            query = query + f" where {constraint}"
        logger.debug(f"Query executed: {query}")
        visits = self.query(query)

        if len(visits) == 0:
            logger.info(f"No visits for {instrument} retrieved from consdb")
            return pd.DataFrame([])

        if augment_visits:
            visits = self.augment_visits(visits, instrument)
        return visits

    def augment_visits(
        self,
        visits: pd.DataFrame,
        instrument: str = "lsstcam",
        predicted_zeropoint_offsets: dict | None = None,
    ) -> pd.DataFrame:
        """Add additional columns to the visits dataframe.

        Parameters
        ----------
        visits
            The visit information from cdb_{instrument}.visit1 and
            cdb_{instrument}.visit1_quicklook (if available).
        instrument
            The instrument for the visits.
            Used to calculate the approproximate rotTelPos value.
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
            "psf_sigma_median",
            "psf_area_median",
            "sky_bg_median",
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

        visits = add_rubin_scheduler_cols(visits, instrument)
        visits = add_rubin_sim_cols(visits, instrument, predicted_zeropoint_offsets)

        return visits

    def exclude_visits(self, visits: pd.DataFrame, bad_visit_ids: list[str]) -> pd.DataFrame:
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


class ConsDbTap(ConsDb):
    """Query the ConsDB TAP service.

    Parameters
    ----------
    api_base : `str`
        Base API for services.
        e.g. https://usdf-rsp.slac.stanford.edu
    token : `str`
        The token for authentication.
    """

    def __init__(self, api_base: str, token: str):
        url = api_base + "/api/consdbtap"
        cred = pyvo.auth.CredentialStore()
        cred.set_password("x-oauth-basic", token)
        self.credential = cred.get("ivo://ivoa.net/sso#BasicAA")
        self.tap = pyvo.dal.TAPService(url, session=self.credential)

    def __repr__(self) -> str:
        return self.tap.baseurl

    def query(self, query) -> pd.DataFrame:
        """Execute TAP ConsDB query.

        Parameters
        ----------
        query : `str`
            SQL query.

        Returns
        -------
        results : `pd.DataFrame`
        """
        try:
            results = self.tap.search(query).to_table().to_pandas()
        except Exception as e:
            logger.warning(e)
            results = pd.DataFrame([])
        return results


class ConsDbFastAPI(ConsDb):
    """Query the ConsDB through the FastAPI interface.

    Parameters
    ----------
    api_base : `str`
        Base API for services.
        e.g. https://usdf-rsp.slac.stanford.edu
    auth : `tuple`
        The username and password for authentication.
    query_timeout : `float`

    """

    def __init__(self, api_base: str, auth: tuple, query_timeout: float = 5 * 60 * 60):
        self.url = api_base + "/consdb/query"
        self.auth = auth
        timeout = httpx.Timeout(timeout=query_timeout, connect=30.0)
        self.httpx_client = httpx.Client(timeout=timeout, auth=self.auth)

    def __repr__(self) -> str:
        return self.url

    def query(self, query) -> pd.DataFrame:
        """Execute FastAPI ConsDB query.

        Parameters
        ----------
        query : `str`
            SQL query.

        Returns
        -------
        results : `pd.DataFrame`
        """
        params = {"query": query}
        try:
            response = self.httpx_client.post(self.url, json=params)
            response.raise_for_status()
        except httpx.RequestError as exc:
            logger.warning(f"An error occurred while requesting {exc.request.url!r}.")
        except httpx.HTTPStatusError as exc:
            logger.warning(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
        if response.status_code != 200:
            messages = []
        else:
            messages = response.json()
        if len(messages) > 0:
            messages = pd.DataFrame(messages["data"], columns=messages["columns"])
            # Check for duplicate columns.
            indices = np.where(pd.Series(messages.columns.duplicated()))[0]
            newcols = messages.columns.to_list()
            for i in indices:
                newcols[i] = newcols[i] + "_duplicate"
            # Have to change only some instances of the duplicates
            messages.columns = newcols
            messages.drop(messages.columns[indices], axis=1, inplace=True)
        return messages
