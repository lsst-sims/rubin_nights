import getpass
import logging
from datetime import datetime
from functools import cache
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd
from astropy.time import Time

from .reference_values import API_ENDPOINTS

logger = logging.getLogger(__name__)

__all__ = [
    "InfluxQueryClient",
    "day_obs_from_efd_index",
    "day_obs_from_efd_index_array",
    "convert_time_to_tz_datetime",
]


TZ_UTC = ZoneInfo("UTC")

# Quiet all warnings and errors related to repertoire vs segwarides
REPERTOIRE_DEV = True


def day_obs_from_efd_index(x: pd.Series) -> int:
    """Use with pandas apply(efd_values, axis=1) to get dayobs.

    To be deprecated:
    Use day_obs_from_efd_index_series for faster conversion and safety
    in case of zero-length dataframes.
    """
    dayobs_time = Time(np.floor(Time(x.name, scale="utc").tai.mjd - 0.5), format="mjd", scale="tai")
    return int(dayobs_time.isot.split("T")[0].replace("-", ""))


def day_obs_from_efd_index_array(utc_time_format: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Convert an array or series of astropy-readable times to dayobs ints."""
    if len(utc_time_format) == 0:
        return np.array([], dtype=int)
    # convert to the MJD DayObs time (Noon UTC)
    dayobs_mjd = np.floor(Time(utc_time_format, scale="utc").tai.mjd - 0.5)
    dayobs_times = Time(dayobs_mjd, format="mjd", scale="tai")
    # Truncate each isot string to "YYYY-MM-DD" then strip hyphens
    dayobs = np.char.replace(dayobs_times.isot.astype("U10"), "-", "").astype(int)
    return dayobs


def convert_time_to_tz_datetime(time: Time) -> datetime:
    """Convert astropy Time to datetime.datetime with UTC timezone.

    Useful for when you want to select EFD entries based on a Time,
    as the EFD indexes are timezone-aware datetimes.
    """
    return time.to_datetime(timezone=TZ_UTC)


class RepertoireCredsError(Exception):
    pass


class InfluxQueryClient:
    """Query for InfluxDB data such as EFD.

    Parameters
    ----------
    site
        The site to use for the influx db, e.g. usdf, summit, base.
        Note that influxdb sites can be special: e.g. currently
        the EFD at USDF is only on usdf-rsp (usdf, not -dev) and
        the creds for the EFD at USDF are also served at summit.
    db_name
        The database to query.
        Default is "efd".
    repertoire_site
        The site to use for repertoire discovery for influx credentials.
        Does not necessarily have to be the same as 'site' for the
        influx db itself, but should match the auth token.
        If None, will match 'site'.
    auth
        The username and password for authentication to repertoire.
        Note that *repertoire* auth is site-specific, even though
        influx databases can be only available at one site (thus you may
        need cross-site authentication). Also only usdf-rsp currently works.
        If None, will fallback to segwarides.
    id_tag
        Add the service name or user name as a comment to the query.
        This aids in tracking query sources in EFD logs.
        If None, will be set to username.
    query_timeout
        Time (in seconds) to wait for query to return.
    """

    def __init__(
        self,
        site: str = "usdf",
        db_name: str = "efd",
        repertoire_site: str | None = None,
        auth: tuple | None = None,
        id_tag: str | None = None,
        query_timeout: float = 5 * 60,
    ) -> None:

        # Site should match general user-expectations and keys in API_ENDPOINTS
        self.site = site.lower()
        # db_name is the identifier when sending query params to the RESTAPI
        # Will be used in gathering credentials from repertoire (probably?)
        self.db_name = db_name
        # influx_db will be the name in repertoire for the influxdb.
        self.influx_db = f"{self.site.replace('-', '')}" + "_"
        # this part is a guess...
        self.influx_db += f"{db_name.lower().replace('lsst.', '')}"

        if repertoire_site is None:
            self.repertoire_site = self.site
        else:
            self.repertoire_site = repertoire_site

        self.null_result = pd.DataFrame([])

        if id_tag is None:
            id_tag = getpass.getuser()
        self.query_tag = f" /* source: {id_tag} via rubin_nights.InfluxQueryClient */"

        # For user convenience, save the last query.
        self.last_query = "No query issued yet."

        # Fetch the influxdb credentials.
        if auth is not None:
            try:
                self.url, influx_auth = self._fetch_credentials_repertoire(auth)
            except RepertoireCredsError:
                if not REPERTOIRE_DEV:
                    logger.warning("Failed to fetch credentials from repertoire. Trying segwarides.")
                self.url, influx_auth = self._fetch_credentials_segwarides()
        else:
            if not REPERTOIRE_DEV:
                logger.warning(
                    "Fetching influx credentials from segwarides "
                    "will soon be deprecated. Please add an auth token. "
                )
            self.url, influx_auth = self._fetch_credentials_segwarides()

        # Set up connections to influx database RestAPI endpoint.
        timeout = httpx.Timeout(query_timeout, connect=10.0)
        self.httpx_client = httpx.Client(base_url=self.url, timeout=timeout, auth=influx_auth)
        self.async_client = httpx.AsyncClient(base_url=self.url, timeout=timeout, auth=influx_auth)

    def _fetch_credentials_repertoire(self, auth: tuple) -> tuple[str, tuple[str, bytes]]:
        """Fetch the credentials via repertoire.

        Parameters
        ----------
        auth
            The username and password for authentication to repertoire
            (RSP/gaefaelfwr token).
        """
        creds_service = (
            f"{API_ENDPOINTS[self.repertoire_site]}/repertoire/discovery/influxdb/{self.influx_db}"
        )
        logger.debug(f"Attempting to fetch credentials from {creds_service}")
        try:
            response = httpx.get(creds_service, auth=auth)
            response.raise_for_status()
        except Exception as e:
            logger.debug(e)
            if not REPERTOIRE_DEV:
                logger.error(f"Could not fetch credentials from repertoire at {creds_service}.")
            else:
                logger.debug(f"Could not fetch credentials from repertoire at {creds_service}.")
            raise RepertoireCredsError

        # Parse the influx db credentials.
        try:
            influx_creds = response.json()
        except Exception as e:
            logger.debug(e)
            logger.error(f"Could not parse credentials from repertoire at {creds_service}.")
            raise RepertoireCredsError

        auth = (influx_creds["username"], influx_creds["password"])
        url = influx_creds["url"]
        logger.info(f"Fetched credentials from repertoire for {self.influx_db}.")
        self.creds_from = "repertoire"
        return url, auth

    def _fetch_credentials_segwarides(self) -> tuple[str, tuple[str, bytes]]:
        "Fetch the credentials via segwarides (to be deprecated)."
        creds_service = f"https://roundtable.lsst.codes/segwarides/creds/{self.site.replace('-', '')}_efd"
        try:
            response = httpx.get(creds_service)
            response.raise_for_status()
        except Exception as e:
            logger.error(
                f"Could not fetch credentials from segwarides for {self.influx_db} using {creds_service}."
            )
            logger.error(e)
            raise e
        # Parse the creds.
        logger.info(f"Fetched credentials from segwarides for {self.influx_db}.")
        influx_creds = response.json()
        auth = (influx_creds["username"], influx_creds["password"])
        url = "https://" + influx_creds["host"] + influx_creds["path"].rstrip("/")
        self.creds_from = "segwarides"
        return url, auth

    def __repr__(self) -> str:
        return f"{self.influx_db} at {self.url}"

    def _to_dataframe(self, response: dict) -> pd.DataFrame:
        """Convert an InfluxDB response to a dataframe.

        Parameters
        ----------
        response
            The JSON response from the InfluxDB API.
        """
        # One InfluxQL query is submitted at a time
        statement = response["results"][0]
        if "series" not in statement:
            # zero results
            return pd.DataFrame([])
        # One InfluxDB measurement/topic queried at a time
        series = statement["series"][0]
        result = pd.DataFrame(series.get("values", []), columns=series["columns"])
        if "time" not in result.columns:
            return result

        # Set time index.
        time_index = pd.DatetimeIndex(pd.to_datetime(result["time"], format="ISO8601"))
        if time_index.tzinfo is None:
            time_index = time_index.tz_localize("UTC")
        result = result.set_index(time_index).drop("time", axis=1)
        # Fill other data.
        if "tags" in series:
            for k, v in series["tags"].items():
                result[k] = v
        if "name" in series:
            result.attrs["name"] = series["name"]
        return result

    @staticmethod
    def build_influxdb_query(
        measurement: str,
        fields: list[str] | str = "*",
        time_range: tuple[Time, Time] | None = None,
        filters: list[tuple[str, str]] | None = None,
    ) -> str:
        """Build an influx DB query for `fields` from `measurement` (topic),
        usually over a time range.

        Parameters
        ----------
        measurement
            The name of the topic / measurement.
        fields
            List of fields to return from the topic.
            Default `*` returns all fields.
        time_range
            The time window (in astropy.time.Time) to query.
        filters
            The additional conditions to match for the query.
            e.g. ('salIndex', 1) would add salIndex=1 to the query.

        Returns
        -------
        query : `str`
        """
        if isinstance(fields, str):
            fields = [fields]
        fields = ", ".join(fields) if fields else "*"

        query = f'SELECT {fields} FROM "{measurement}"'

        conditions = []

        if time_range:
            t_start, t_end = time_range
            conditions.append(f"time >= '{t_start.utc.isot}Z' AND time <= '{t_end.utc.isot}Z'")

        if filters:
            for key, value in filters:
                conditions.append(f"{key} = {value}")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        return query

    @staticmethod
    def build_influxdb_top_n_query(
        measurement: str,
        fields: list[str] | str = "*",
        num: int = 10,
        time_cut: Time | None = None,
        filters: list[tuple[str, str]] | None = None,
    ) -> str:
        """Build an influx DB query for `fields` from `measurement`,
        restricted to `num` records.

        Parameters
        ----------
        measurement
            The name of the topic / measurement.
        fields
            List of fields to return from the topic.
            Default `*` will return all fields.
        num
            The maximum number of records to return.
        time_cut
            Search for only records at or before this time.
        filters
            The additional conditions to match for the query.
            e.g. ('salIndex', 1) would add salIndex=1 to the query.

        Returns
        -------
        query : `str`
        """
        if isinstance(fields, str):
            fields = [fields]
        fields = ", ".join(fields) if fields else "*"

        query = f'SELECT {fields} FROM "{measurement}"'

        conditions = []

        if time_cut:
            conditions.append(f"time <= '{time_cut.utc.isot}Z'")

        if filters:
            for key, value in filters:
                conditions.append(f"{key} = {value}")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        limit = f" GROUP BY * ORDER BY DESC LIMIT {num}"
        query += limit

        return query

    def query(self, query: str) -> pd.DataFrame:
        """Send and receive results from the InfluxDB API,
        with a synchronous query.

        Parameters
        ----------
        query
            The query to send to the InfluxDb.

        Returns
        -------
        result : `pd.DataFrame`
        """
        # If requested topic is not in known topics,
        # the query can arrive here as "".
        if query is None or len(query) == 0:
            return self.null_result
        # Add an identifier string to the query
        if "show" in query:
            # show queries only work with the comment first.
            query = self.query_tag + query
        else:
            query = query + self.query_tag
        params = {"db": self.db_name, "q": query}
        self.last_query = query

        try:
            response = self.httpx_client.get(
                "/query",
                params=params,
            )
            logger.debug(f"Issued query: {params['q']}")
            response.raise_for_status()
        except Exception as e:
            logger.warning(e)
            response = None

        if response:
            result = self._to_dataframe(response.json())
        else:
            result = self.null_result

        return result

    async def async_query(self, query: str) -> pd.DataFrame:
        """Send and receive results from the InfluxDB API,
        with an asynchronous query.

        Parameters
        ----------
        query
            The query to send to the InfluxDb.

        Returns
        -------
        result : `pd.DataFrame`
        """
        # Add an identifier string to the query
        if "show" in query:
            # show queries only work with the comment first.
            query = self.query_tag + query
        else:
            query = query + self.query_tag
        params = {"db": self.db_name, "q": query}
        self.last_query = query

        try:
            response = await self.async_client.get(
                "/query",
                params=params,
            )
            response.raise_for_status()
            logger.debug(f"Issued query: {params['q']}")
        except Exception as e:
            logger.warning(e)
            response = None

        if response:
            result = self._to_dataframe(response.json())
        else:
            result = self.null_result

        return result

    @cache
    def get_topics(self) -> list[str]:
        """Find all available topics."""
        # Just use sync query. It runs once.
        r = self.query("show measurements")
        self.topics = list(r["name"].values)
        return self.topics

    def get_fields(self, measurement: str) -> pd.DataFrame:
        """Query the list of field names for a topic.

        Parameters
        ----------
        measurement
            Name of measurement/topic to query for field names.

        Returns
        -------
        fields : `pd.DataFrame`
            fieldKey / fieldType values.
        """
        query = f'show field keys from "{measurement}"'
        return self.query(query)

    async def async_get_fields(self, measurement: str) -> pd.DataFrame:
        """Query the list of field names for a topic.

        Parameters
        ----------
        measurement
            Name of measurement/topic to query for field names.

        Returns
        -------
        fields : `pd.DataFrame`
            fieldKey / fieldType values.
        """
        query = f'show field keys from "{measurement}"'
        return await self.async_query(query)

    def _time_series_query(
        self,
        topic_name: str,
        fields: str | list[str],
        t_start: Time,
        t_end: Time,
        index: int | None = None,
    ) -> str:
        """Build specific query between t_start and t_end.

        Adds some logging and checks around build_influxdb_query.

        Parameters
        ----------
        topic_name
            The name of the topic or measurement to query.
        fields
            The field or fields to return from topic_name.
            The entry '*' will return all fields.
        t_start
            The start of the time window for the query.
        t_end
            The end of the time window for the query.

        Returns
        -------
        query: `str`
        """
        if topic_name not in self.get_topics():
            logger.error(f"{topic_name} not in {self.db_name} topics.")
            return ""
        if index:
            filters = [("salIndex", str(index))]
        else:
            filters = None
        query = self.build_influxdb_query(
            topic_name, fields=fields, time_range=(t_start, t_end), filters=filters
        )
        return query

    def select_time_series(
        self,
        topic_name: str,
        fields: str | list[str],
        t_start: Time,
        t_end: Time,
        index: int | None = None,
    ) -> pd.DataFrame:
        """Sync query to return data from `topic_name`
        between `t_start` and `t_end`.

        Parameters
        ----------
        topic_name
            The name of the topic or measurement to query.
        fields
            The field or fields to return from topic_name.
            The entry '*' will return all fields.
        t_start
            The start of the time window for the query.
        t_end
            The end of the time window for the query.

        Returns
        -------
        query_results: `pd.DataFrame`
            The result of the query.
        """
        query = self._time_series_query(
            topic_name=topic_name, fields=fields, t_start=t_start, t_end=t_end, index=index
        )
        return self.query(query)

    async def async_select_time_series(
        self,
        topic_name: str,
        fields: str | list[str],
        t_start: Time,
        t_end: Time,
        index: int | None = None,
    ) -> pd.DataFrame:
        """Async query to return data from `topic_name`
        between `t_start` and `t_end`.

        Parameters
        ----------
        topic_name
            The name of the topic or measurement to query.
        fields
            The field or fields to return from topic_name.
            The entry '*' will return all fields.
        t_start
            The start of the time window for the query.
        t_end
            The end of the time window for the query.

        Returns
        -------
        query_results: `pd.DataFrame`
            The result of the query.
        """
        query = self._time_series_query(
            topic_name=topic_name, fields=fields, t_start=t_start, t_end=t_end, index=index
        )
        return await self.async_query(query)

    def _top_n_query(
        self,
        topic_name: str,
        fields: str | list[str],
        num: int,
        time_cut: Time = None,
        index: int | None = None,
    ) -> str:
        """Build specific query for most recent `num` records.

        Adds some logging and checks around build_influxdb_top_n_query.

        Parameters
        ----------
        topic_name
            The name of the topic or measurement to query.
        fields
            The field or fields to return from topic_name.
            The entry '*' will return all fields.
        num
            The number of records to return.
        time_cut
            If not None, looks for records prior to this time.

        Returns
        -------
        query: `str`
        """
        if topic_name not in self.get_topics():
            logger.error(f"{topic_name} not in {self.db_name} topics.")
            return ""
        if index:
            filters = [("salIndex", str(index))]
        else:
            filters = None
        query = self.build_influxdb_top_n_query(
            topic_name, fields=fields, num=num, time_cut=time_cut, filters=filters
        )
        return query

    def select_top_n(
        self,
        topic_name: str,
        fields: str | list[str],
        num: int,
        time_cut: Time = None,
        index: int | None = None,
    ) -> pd.DataFrame:
        """Sync query to return `num` records from `topic_name`.

        Parameters
        ----------
        topic_name
            The name of the topic or measurement to query.
        fields
            The field or fields to return from topic_name.
            The entry '*' will return all fields.
        num
            The number of records to return.
        time_cut
            If not None, looks for records prior to this time.

        Returns
        -------
        query_results: `pd.DataFrame`
            The result of the query.
        """
        query = self._top_n_query(
            topic_name=topic_name, fields=fields, num=num, time_cut=time_cut, index=index
        )
        return self.query(query)

    async def async_select_top_n(
        self,
        topic_name: str,
        fields: str | list[str],
        num: int,
        time_cut: Time = None,
        index: int | None = None,
    ) -> pd.DataFrame:
        """Async query to return `num` records from `topic_name`.

        Parameters
        ----------
        topic_name
            The name of the topic or measurement to query.
        fields
            The field or fields to return from topic_name.
            The entry '*' will return all fields.
        num
            The number of records to return.
        time_cut
            If not None, looks for records prior to this time.

        Returns
        -------
        query_results: `pd.DataFrame`
            The result of the query.
        """
        query = self._top_n_query(
            topic_name=topic_name, fields=fields, num=num, time_cut=time_cut, index=index
        )
        return await self.async_query(query)
