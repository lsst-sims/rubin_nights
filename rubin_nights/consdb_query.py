"""Execute queries for the ConsDB."""

import logging

import httpx
import numpy as np
import pandas as pd
import pyvo
from astropy.time import Time

try:
    from rubin_scheduler.utils import Site, approx_altaz2pa, approx_ra_dec2_alt_az, rotation_converter

    HAS_RUBIN_SCHEDULER = True
except ModuleNotFoundError:
    HAS_RUBIN_SCHEDULER = False

logger = logging.getLogger(__name__)

__all__ = ["ConsDbTap", "ConsDbFast"]


class ConsDb:
    def query(self, query) -> pd.DataFrame:
        raise NotImplementedError

    def get_visits(self, instrument: str, day_obs_min: str, day_obs_max: str) -> pd.DataFrame:
        """ "Fetch visits from a particular range of day_obs.

        Parameters
        ----------
        instrument : `str`
            The instrument to search for.
            Typical values would include lsstcomcam, latiss, and lsstcam.
            See https://sdm-schemas.lsst.io/ for more details.
        day_obs_min : `str`
            The minimum day_obs for visits.
            Format YYYY-MM-DD.
        day_obs_max : `str`
            The maximum day_obs for visits.
            Format YYYY-MM-DD.

        Returns
        -------
        visits : `pd.DataFrame`
            The visit information from cdb_{instrument}.visit1 and
            cdb_{instrument}.visit1_quicklook (if available).
            Additional information may be added, such as `visit_gap`.

        Notes
        -----
        This is useful for gathering all visits from a given range of time.
        For visits from a particular science survey, do a direct query.
        """
        day_obs_int_min = int(day_obs_min.replace("-", ""))
        day_obs_int_max = int(day_obs_max.replace("-", ""))

        # Querying separately and joining in pandas works.
        # Otherwise, duplicate columns are a problem for FastAPI (but not TAP).
        visit_query = f"""
            SELECT *
            FROM cdb_{instrument}.visit1
             WHERE day_obs >= {day_obs_int_min}
             and day_obs  <= {day_obs_int_max}
        """

        quicklook_query = f"""
            SELECT q.*  FROM cdb_{instrument}.visit1_quicklook as q,
            cdb_{instrument}.visit1 as v
             WHERE q.visit_id = v.visit_id and
             v.day_obs >= {day_obs_int_min}
             and v.day_obs <= {day_obs_int_max}
        """

        visits = self.query(visit_query)
        if len(visits) == 0:
            logger.info(
                f"No visits for {instrument} between {day_obs_int_min} to "
                f"{day_obs_int_max} retrieved from consdb"
            )

        visits.set_index("visit_id", inplace=True)

        quicklook = self.query(quicklook_query)

        if len(quicklook) > 0:
            quicklook.set_index("visit_id", inplace=True)
            visits = visits.join(quicklook, lsuffix="", rsuffix="_q")

        visits = self.augment_visits(visits, instrument)
        return visits

    def augment_visits(self, visits: pd.DataFrame, instrument: str = "lsstcam") -> pd.DataFrame:

        values = dict([[e, ""] for e in ["science_program", "target_name", "observation_reason"]])
        visits.fillna(value=values, inplace=True)

        columns_to_floats = [
            "s_ra",
            "s_dec",
            "exp_midpt_mjd",
            "airmass",
            "zero_point_median",
            "psf_sigma_median",
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
        visits["prev_obs_start_mjd"] = prev_visit_start
        visits["prev_obs_end_mjd"] = prev_visit_end
        visits["visit_gap"] = visit_gap

        if HAS_RUBIN_SCHEDULER:
            # Add in physical rotator angle, parallactic angle
            # (these will be added by ConsDB in the future
            lsst_loc = Site("LSST")
            times = Time(
                visits["obs_start_mjd"], format="mjd", scale="tai", location=lsst_loc.to_earth_location()
            )
            lst = times.sidereal_time("mean").deg
            visits["lst"] = lst
            visits["HA"] = (visits["s_ra"] - lst) / 360 * 12 % 24

            alt, az = approx_ra_dec2_alt_az(
                visits.s_ra.values,
                visits.s_dec.values,
                lsst_loc.latitude,
                lsst_loc.longitude,
                visits.exp_midpt_mjd.values,
                lmst=None,
            )
            pa = approx_altaz2pa(alt, az, lsst_loc.latitude)
            visits["approx_pa"] = pa

            if instrument.lower() != "latiss":
                if instrument.lower() == "lsstcomcam":
                    tele = "comcam"
                else:
                    tele = "rubin"
                rc = rotation_converter(telescope=tele)
                rotTelPos = rc.rotskypos2rottelpos(visits.sky_rotation.values, visits["approx_pa"].values)
                visits["approx_rotTelPos"] = rotTelPos

        return visits


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

    def __repr__(self):
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
            results = self.tap.search(query)
            if len(results) == 0:
                results = []
            results = pd.DataFrame(results)
        except Exception as e:
            logger.warning(e)
            results = pd.DataFrame([])
        return results


class ConsDbFast(ConsDb):
    """Query the ConsDB through the FastAPI interface.

    Parameters
    ----------
    api_base : `str`
        Base API for services.
        e.g. https://usdf-rsp.slac.stanford.edu
    auth : `tuple`
        The username and password for authentication.
    """

    def __init__(self, api_base: str, auth: tuple):
        self.url = api_base + "/consdb/query"
        self.auth = auth

    def __repr__(self):
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
        # Some requests from the logging endpoints fail the first time.
        response = httpx.post(self.url, auth=self.auth, json=params)
        # So, try twice (but twice should succeed)
        if response.status_code != 200:
            try:
                response = httpx.post(self.url, auth=self.auth, json=params)
                response.raise_for_status()
            except httpx.RequestError as exc:
                logger.warning(f"An error occurred while requesting {exc.request.url!r}.")
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    f"Error response {exc.response.status_code} while requesting {exc.request.url!r}."
                )
        if response.status_code != 200:
            messages = []
        else:
            messages = response.json()
        if len(messages) > 0:
            messages = pd.DataFrame(messages["data"], columns=messages["columns"])
        return messages
