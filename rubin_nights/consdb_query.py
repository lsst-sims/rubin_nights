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

try:
    from rubin_scheduler.site_models import Almanac
    from rubin_scheduler.utils import (
        Site,
        angular_separation,
        approx_altaz2pa,
        approx_ra_dec2_alt_az,
        rotation_converter,
    )

    HAS_RUBIN_SCHEDULER = True
except ModuleNotFoundError:
    HAS_RUBIN_SCHEDULER = False

try:
    from rubin_sim.phot_utils import predicted_zeropoint, predicted_zeropoint_hardware

    HAS_RUBIN_SIM = True
except ModuleNotFoundError:
    HAS_RUBIN_SIM = False

logger = logging.getLogger(__name__)

__all__ = ["fetch_excluded_visits", "ConsDbTap", "ConsDbFastAPI"]


GAUSSIAN_FWHM_OVER_SIGMA: float = 2.0 * np.sqrt(2.0 * np.log(2.0))
PLATESCALE = 0.2
ZEROPOINT_OFFSETS_LSSTCAM = {"u": 0.0279, "g": 0.048, "r": 0.109, "i": 0.0919, "z": 0.0959, "y": 0.0383}
ZEROPOINT_OFFSETS_LSSTCOMCAM = {"u": 0.26, "g": -0.14, "r": -0.09, "i": -0.10, "z": -0.13, "y": -0.18}

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
    instrument : `str`
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
        instrument : `str`
            The instrument to search for.
            Typical values would include lsstcomcam, latiss, and lsstcam.
            See https://sdm-schemas.lsst.io/ for more details.
        t_start : `Time` or None
            The earliest time to match obs_start.
        t_end : `Time` or None
            The latest time to match obs_start.
        visit_constraint : `str` or None
            A constraint to apply to the cdb_{instrument}.visit1 table.
            Example: `"science_program = 'BLOCK-365'"`
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

    def add_rubin_sim_cols(
        self,
        visits: pd.DataFrame,
        instrument: str = "lsstcam",
        predicted_zeropoint_offsets: dict | None = None,
    ) -> pd.DataFrame:
        """Add columns that require rubin_sim:
        predicted zeropoint and converted skybackground (mag/sq arcsec).

        Parameters
        ----------
        visits : `pd.DataFrame`
            The visit information from cdb_{instrument}.visit1 and
            cdb_{instrument}.visit1_quicklook (if available).
        instrument : `str`
            The instrument for the visits.
            Used to select the appropriate zeropoint offsets, if not provided.
        predicted_zeropoint_offsets : `dict` { `str`: `float` }
            Offsets to add to the predicted zeropoint values.
            If None, will pick appropriate defaults based on instrument.

        Returns
        -------
        visits : `pd.DataFrame`
            The visit information, with additional columns added for
            predicted zeropoint values, sky background in magnitudes,
            an estimated m5 depth (from zeropoint + sky).
        """
        if predicted_zeropoint_offsets is None:
            if instrument.lower() == "lsstcam":
                predicted_zeropoint_offsets = ZEROPOINT_OFFSETS_LSSTCAM
            elif instrument.lower() == "lsstcomcam":
                predicted_zeropoint_offsets = ZEROPOINT_OFFSETS_LSSTCOMCAM
            else:
                predicted_zeropoint_offsets = {"u": 0, "g": 0, "r": 0, "i": 0, "z": 0, "y": 0}

        # Add new columns
        new_cols = ["zero_point_1s", "zero_point_1s_pred", "clouds", "sky_bg_median_mag", "cat_m5"]
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
            if x.shut_time == 0 or np.isnan(x.shut_time):
                x.zero_point_1s = np.nan
                x.zero_point_1s_pred = np.nan
                x.sky_bg_median_mag = np.nan
                x.cat_m5 = np.nan
                return x
            try:
                x.zero_point_1s = x.zero_point_median - 2.5 * np.log10(x.shut_time)
                x.zero_point_1s_pred = (
                    predicted_zeropoint(x.band, x.airmass, 1) + predicted_zeropoint_offsets[x.band]
                )
                x.clouds = x.zero_point_1s - x.zero_point_1s_pred
                # Convert sky counts/pixel to magnitude/arcsecond^2
                zp_sky = (
                    predicted_zeropoint_hardware(x.band, x.shut_time) + predicted_zeropoint_offsets[x.band]
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

    def add_rubin_scheduler_cols(self, visits: pd.DataFrame, instrument: str = "lsstcam") -> pd.DataFrame:
        """Add columns that require rubin_scheduler:
        parallactic angle and rotator angle,

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
            an estimated m5 depth (from zeropoint + sky).
        """
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
        ]
        new_df = pd.DataFrame(np.zeros((len(visits), len(new_cols))), columns=new_cols, index=visits.index)

        if all(new_cols) in visits.columns:
            logger.debug("All columns already present in visits.")
            return visits
        else:
            for n in new_cols:
                if n in visits.columns:
                    visits.drop(labels=n, axis=1, inplace=True)

        # Add in physical rotator angle, parallactic angle
        # (these will be added by ConsDB in the future
        lsst_loc = Site("LSST")
        times = Time(
            visits["obs_start_mjd"], format="mjd", scale="tai", location=lsst_loc.to_earth_location()
        )
        lst = times.sidereal_time("mean").deg
        new_df["lst"] = lst
        new_df["HA"] = (visits["s_ra"] - lst) / 360 * 12 % 24

        almanac = Almanac()

        avals = almanac.get_sun_moon_positions(visits["exp_midpt_mjd"].values)
        new_df["moon_alt"], new_df["moon_az"] = np.degrees([avals["moon_alt"][0], avals["moon_az"][0]])
        new_df["moon_RA"], new_df["moon_Dec"] = np.degrees([avals["moon_RA"][0], avals["moon_dec"][0]])
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

    def augment_visits(
        self,
        visits: pd.DataFrame,
        instrument: str = "lsstcam",
        predicted_zeropoint_offsets: dict | None = None,
    ) -> pd.DataFrame:
        """Add additional columns to the visits dataframe.

        Parameters
        ----------
        visits : `pd.DataFrame`
            The visit information from cdb_{instrument}.visit1 and
            cdb_{instrument}.visit1_quicklook (if available).
        instrument : `str`
            The instrument for the visits.
            Used to calculate the approproximate rotTelPos value.
        predicted_zeropoint_offsets : `dict` { `str`: `float` }
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

        if HAS_RUBIN_SCHEDULER:
            visits = self.add_rubin_scheduler_cols(visits, instrument)

        if HAS_RUBIN_SIM:
            visits = self.add_rubin_sim_cols(visits, instrument, predicted_zeropoint_offsets)

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
