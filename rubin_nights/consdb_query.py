"""Execute queries for the ConsDB."""

import logging

import astropy.units as u
import httpx
import numpy as np
import pandas as pd
import pyvo
from astropy.coordinates import SkyCoord
from astropy.time import Time

try:
    from rubin_scheduler.utils import Site, approx_altaz2pa, approx_ra_dec2_alt_az, rotation_converter

    HAS_RUBIN_SCHEDULER = True
except ModuleNotFoundError:
    HAS_RUBIN_SCHEDULER = False

try:
    from rubin_sim.phot_utils import predicted_zeropoint, predicted_zeropoint_hardware

    HAS_RUBIN_SIM = True
except ModuleNotFoundError:
    HAS_RUBIN_SIM = False

logger = logging.getLogger(__name__)

__all__ = ["ConsDbTap", "ConsDbFastAPI"]


GAUSSIAN_FWHM_OVER_SIGMA: float = 2.0 * np.sqrt(2.0 * np.log(2.0))
PLATESCALE = 0.2
ZEROPOINT_OFFSETS = {"u": 0, "g": 0, "r": 0, "i": 0, "z": 0, "y": 0}
ZEROPOINT_OFFSETS_LSSTCOMCAM = {"u": 0.26, "g": -0.14, "r": -0.09, "i": -0.10, "z": -0.13, "y": -0.18}

BAD_VISITS_LSSTCAM = (
    "https://raw.githubusercontent.com/lsst-dm/excluded_visits/" "refs/heads/main/LSSTCam/bad.ecsv"
)
BAD_VISITS_LSSTCOMCAM = (
    "https://raw.githubusercontent.com/lsst-dm/excluded_visits/" "refs/heads/main/LSSTComCam/bad.ecsv"
)


class ConsDb:

    def query(self, query) -> pd.DataFrame:
        raise NotImplementedError

    def get_visits(
        self, instrument: str, t_start: Time, t_end: Time, augment_visits: bool = True
    ) -> pd.DataFrame:
        """ "Fetch visits from a particular range of times.

        Parameters
        ----------
        instrument : `str`
            The instrument to search for.
            Typical values would include lsstcomcam, latiss, and lsstcam.
            See https://sdm-schemas.lsst.io/ for more details.
        t_start : `Time`
            The earliest time to match obs_start.
        t_end : `Time`
            The latest time to match obs_start.
        augment_visits : `boolean
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
            f"select v.*, q.* from  cdb_{instrument}.visit1 as v "
            f"left join cdb_{instrument}.visit1_quicklook as q "
            f"on v.visit_id = q.visit_id "
            f"where obs_start_mjd >= {t_start.mjd} and obs_start_mjd <= {t_end.mjd}"
        )

        visits = self.query(query)

        if len(visits) == 0:
            logger.info(
                f"No visits for {instrument} between {t_start.iso} to " f"{t_end.iso} retrieved from consdb"
            )
            return pd.DataFrame([])

        if augment_visits:
            visits = self.augment_visits(visits, instrument)
        return visits

    def augment_visits(self, visits: pd.DataFrame, instrument: str = "lsstcam") -> pd.DataFrame:
        """Add additional columns to the visits dataframe.

        Parameters
        ----------
        visits : `pd.DataFrame`
            The visit information from cdb_{instrument}.visit1 and
            cdb_{instrument}.visit1_quicklook (if available).
        instrument : `str`
            The instrument for the visits.
            Used to calculate the approproximate rotTelPos value.

        Returns
        -------
        visits : `pd.DataFrame`
            The visit information, with additional columns added for
            predicted zeropoint values, sky background in magnitudes,
            an estimated m5 depth (from zeropoint + sky), as well
            as an approximate rotTelPos (likely off by ~1 deg).
            Some columns may be reformatted for dtypes.
        """
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

        if HAS_RUBIN_SIM:
            # Add predicted 1s zeropoint
            new_cols = ["zero_point_1s", "zero_point_1s_pred", "sky_bg_median_mag", "cat_m5"]
            new_df = pd.DataFrame(
                np.zeros((len(visits), len(new_cols))), columns=new_cols, index=visits.index
            )
            visits = visits.merge(new_df, right_index=True, left_index=True)

            def calc_predicted_zeropoints(x):
                if x.shut_time == 0 or np.isnan(x.shut_time):
                    x.zero_point_1s = np.nan
                    x.zero_point_1s_pred = np.nan
                    x.sky_bg_median_mag = np.nan
                    x.cat_m5 = np.nan
                    return x
                try:
                    x.zero_point_1s = x.zero_point_median - 2.5 * np.log10(x.shut_time)
                    x.zero_point_1s_pred = (
                        predicted_zeropoint(x.band, x.airmass, 1) + self.predicted_zeropoint_offsets[x.band]
                    )
                    # Convert sky counts/pixel to magnitude/arcsecond^2
                    zp_sky = (
                        predicted_zeropoint_hardware(x.band, x.shut_time)
                        + self.predicted_zeropoint_offsets[x.band]
                    )
                    x.sky_bg_median_mag = -2.5 * np.log10(x.sky_bg_median / PLATESCALE**2) + zp_sky
                    # Do an approximation for the instrumental noise (in e-)
                    noise_instr_sq = 13
                    total_noise_sq = x.psf_area_median * (x.sky_bg_median + noise_instr_sq)
                    counts_5sigma = np.sqrt(total_noise_sq) * 5
                    x.cat_m5 = -2.5 * np.log10(counts_5sigma) + x.zero_point_median
                except KeyError:
                    # Some bands aren't in the lookup (such as pinhole)
                    # And some visits
                    pass
                return x

            try:
                visits = visits.apply(calc_predicted_zeropoints, axis=1)
            except AttributeError:
                # Missing quicklook columns for psf or zeropoint or sky
                logger.debug("Missing columns for psf_sigma_median, zero_point_median or sky_bg_median.")
                pass

        return visits

    def exclude_visits(
        self, visits: pd.DataFrame, bad_visit_list: list[int] | None = None, instrument: str = "lsstcam"
    ) -> pd.DataFrame:
        """Remove a list of bad visit_id values.

        Parameters
        ----------
        bad_visit_list : `list` [`str`] or `None`
            A list of bad visit_ids.
            The default of None will download the bad visits from
            the instrument-appropriate BAD_VISITS URI in
            github @ lsst-dm/excluded_visits.
        ins
        """
        # Download bad visit information from github if needed.
        if bad_visit_list is None:
            if instrument.lower() == "lsstcam":
                uri = BAD_VISITS_LSSTCAM
            elif instrument.lower() == "lsstcomcam":
                uri = BAD_VISITS_LSSTCOMCAM
            bad_visits = pd.read_csv(uri, comment="#")
            bad_visit_list = bad_visits.exposure.to_list()
        if bad_visit_list is None:
            logging.warning("No bad_visit_list provided and could not find default match.")
            return visits
        # Drop the bad visits
        visits = visits.query("visit_id not in @bad_visit_list")
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
        self.predicted_zeropoint_offsets = ZEROPOINT_OFFSETS

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
            results = self.tap.search(query)
            if len(results) == 0:
                results = []
            results = pd.DataFrame(results)
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
        self.predicted_zeropoint_offsets = ZEROPOINT_OFFSETS

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
