from __future__ import annotations

import getpass
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Optional, Type
from zoneinfo import ZoneInfo

import httpx2
import numpy as np
import pandas as pd
from astropy.time import Time

from .reference_values import API_ENDPOINTS

logger = logging.getLogger(__name__)

__all__ = [
    "InfluxQueryClient",
    "AsyncInfluxQueryClient",
    "day_obs_from_efd_index",
    "day_obs_from_efd_index_array",
    "convert_time_to_tz_datetime",
]


TZ_UTC = ZoneInfo("UTC")


def day_obs_from_efd_index(x: pd.Series) -> int:
    """Use with pandas apply(efd_values, axis=1) to get dayobs.

    To be deprecated:
    Use day_obs_from_efd_index_array for faster conversion and safety
    in case of zero-length dataframes.
    """
    dayobs_time = Time(np.floor(Time(x.name, scale="utc").tai.mjd - 0.5), format="mjd", scale="tai")
    return int(dayobs_time.isot.split("T")[0].replace("-", ""))


def day_obs_from_efd_index_array(utc_time_format: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Convert an array or series of astropy-readable times to dayobs ints.

    >>> day_obs = day_obs_from_efd_index_array(efd_dataframe.index)
    """
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


class InfluxQueryClientBase:
    """Base class for sync and async InfluxDb client.

    Parameters
    ----------
    site
        The site to use for the influx db, e.g. usdf, summit, base.
    db_name
        The database to query.
        Default is "efd".
    repertoire_site
        The site to use for repertoire discovery for influx credentials -
        must match one of the keys of the API_ENDPOINTS dictionary.
        Does not necessarily have to be the same as 'site' for the
        influx db itself, but should match the auth token.
        If None, will match 'site'.
    auth
        The username and password for authentication to repertoire.
        Note that *repertoire* auth is site-specific.
    influx_auth_file
        Path to a JSON file containing a mapping of aliases (e.g. "usdf_efd")
        to InfluxDB connection information, in the format returned by the
        repertoire /influxdb endpoint (including "url", "username" and
        "password"). Used only if `repertoire_auth` is None.
        If this is also None, the EFDAUTH environment variable is checked
        for the path to such a file (to match the lsst_efd_client option).
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
        influx_auth_file: str | Path | None = None,
        id_tag: str | None = None,
        query_timeout: float = 5 * 60,
    ) -> None:

        # Location of influx database (usdf, usdf-dev, summit, etc).
        self.site = site.lower()
        # db_name is the database identifier when sending query.
        self.db_name = db_name

        # influx_db will match the name in repertoire for the influxdb.
        # Note that usdf-dev is labelled usdfdev for influx_db.
        self.influx_db = f"{self.site.replace('-', '')}" + "_"
        # this part is a guess (for the future).
        # self.influx_db += f"{db_name.lower().replace('lsst.', '')}"
        # At present, the only database listed in repertoire is 'efd'.
        self.influx_db += "efd"

        self.null_result = pd.DataFrame([])

        if id_tag is None:
            id_tag = getpass.getuser()
        self.query_tag = f" /* source: {id_tag} via rubin_nights.InfluxQueryClient */"

        # For user convenience, save the last query.
        self.last_query = "No query issued yet."
        self.topics = [""]

        # Resolve the influx url and credentials.
        # First: fetch the credentials from repertoire.
        if auth is not None:
            if repertoire_site is None:
                repertoire_site = self.site
            self.url, influx_auth = self._fetch_credentials(repertoire_site, auth)
        # Second: an explicit auth file.
        elif influx_auth_file is not None:
            self.url, influx_auth = self._read_credentials_file(influx_auth_file)
        # Last: the EFDAUTH environment variable.
        elif os.getenv("EFDAUTH"):
            self.url, influx_auth = self._read_credentials_file(os.environ["EFDAUTH"])
        else:
            raise ValueError(
                "No influx credentials available: provide auth or influx_auth_file, or set EFDAUTH."
            )
        self.timeout = httpx2.Timeout(query_timeout, connect=10.0)
        self.setup_client(influx_auth)

    def _read_credentials_file(self, auth_file: str | Path) -> tuple[str, tuple[str, str]]:
        """Read the influx credentials for `self.influx_db` from a JSON file.

        Parameters
        ----------
        auth_file
            Path to a JSON file mapping aliases (such as "usdf_efd") to
            InfluxDB connection information, as returned by repertoire.
        """
        auth_path = Path(auth_file)
        auth_data = json.loads(auth_path.read_text())
        auth_creds = auth_data.get(self.influx_db)
        if not auth_creds:
            raise ValueError(
                f"No credentials available for {self.influx_db} in {auth_path}. "
                f"Available keys: {list(auth_data.keys())}"
            )
        auth = (auth_creds["username"], auth_creds["password"])
        logger.info(f"Read credentials for {self.influx_db} from {auth_path}.")
        self.creds_from = f"file @ {auth_path}"
        return auth_creds["url"], auth

    def _fetch_credentials(self, site: str, auth: tuple) -> tuple[str, tuple[str, str]]:
        """Fetch the credentials via repertoire.

        Parameters
        ----------
        site
            The repertoire site to query (must match auth token).
        auth
            The username and password for authentication to repertoire
            (RSP/gaefaelfwr token).
        """
        creds_service = f"{API_ENDPOINTS[site]}/repertoire/discovery/influxdb"
        logger.debug(f"Attempting to fetch credentials from {creds_service}")

        try:
            with httpx2.Client() as client:
                response = client.get(creds_service, auth=auth)
                response.raise_for_status()
        except (httpx2.HTTPStatusError, httpx2.RequestError) as exc:
            logger.error(f"Could not reach repertoire discovery at {creds_service}.")
            raise exc

        # Parse the influx db credentials.
        try:
            influx_creds = response.json()
        except Exception as exc:
            logger.error(f"Could not parse credentials from repertoire at {creds_service}.")
            raise exc

        # There could be multiple options - find self.influx_db
        if self.influx_db not in influx_creds:
            raise ValueError(f"No credentials available for {self.influx_db}.")

        db_creds = influx_creds[self.influx_db]
        auth = (db_creds["username"], db_creds["password"])
        url = db_creds["url"]
        logger.info(f"Fetched credentials from repertoire for {self.influx_db}.")
        self.creds_from = f"repertoire @ {site}"
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
        if index:
            filters = [("salIndex", str(index))]
        else:
            filters = None
        query = self.build_influxdb_query(
            topic_name, fields=fields, time_range=(t_start, t_end), filters=filters
        )
        return query

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
        if index:
            filters = [("salIndex", str(index))]
        else:
            filters = None
        query = self.build_influxdb_top_n_query(
            topic_name, fields=fields, num=num, time_cut=time_cut, filters=filters
        )
        return query


class InfluxQueryClient(InfluxQueryClientBase):
    def __init__(
        self,
        site: str = "usdf",
        db_name: str = "efd",
        influx_auth_file: str | Path | None = None,
        repertoire_site: str | None = None,
        auth: tuple | None = None,
        id_tag: str | None = None,
        query_timeout: float = 5 * 60,
    ) -> None:
        super().__init__(
            site=site,
            db_name=db_name,
            repertoire_site=repertoire_site,
            auth=auth,
            influx_auth_file=influx_auth_file,
            id_tag=id_tag,
            query_timeout=query_timeout,
        )

    def setup_client(self, influx_auth: tuple[str, str]) -> None:
        self.httpx_client = httpx2.Client(base_url=self.url, timeout=self.timeout, auth=influx_auth)

    def close(self) -> None:
        self.httpx_client.close()

    def __enter__(self) -> InfluxQueryClient:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def query(self, query: str) -> pd.DataFrame:
        """Send and receive results from the InfluxDB API.

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

    def get_topics(self) -> list[str]:
        """Find all available topics."""
        if self.topics == [""]:
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
        if topic_name not in self.get_topics():
            logger.error(f"{topic_name} not in {self.db_name} topics.")
            return pd.DataFrame([])
        query = self._time_series_query(
            topic_name=topic_name, fields=fields, t_start=t_start, t_end=t_end, index=index
        )
        return self.query(query)

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
        if topic_name not in self.get_topics():
            logger.error(f"{topic_name} not in {self.db_name} topics.")
            return pd.DataFrame([])
        query = self._top_n_query(
            topic_name=topic_name, fields=fields, num=num, time_cut=time_cut, index=index
        )
        return self.query(query)


class AsyncInfluxQueryClient(InfluxQueryClientBase):
    def __init__(
        self,
        site: str = "usdf",
        db_name: str = "efd",
        influx_auth_file: str | Path | None = None,
        repertoire_site: str | None = None,
        auth: tuple | None = None,
        id_tag: str | None = None,
        query_timeout: float = 5 * 60,
    ) -> None:
        super().__init__(
            site=site,
            db_name=db_name,
            repertoire_site=repertoire_site,
            auth=auth,
            influx_auth_file=influx_auth_file,
            id_tag=id_tag,
            query_timeout=query_timeout,
        )

    def setup_client(self, influx_auth: tuple[str, str]) -> None:
        self.async_client = httpx2.AsyncClient(base_url=self.url, timeout=self.timeout, auth=influx_auth)

    async def aclose(self) -> None:
        await self.async_client.aclose()

    async def __aenter__(self) -> AsyncInfluxQueryClient:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.aclose()

    async def query(self, query: str) -> pd.DataFrame:
        """Send and receive results from the InfluxDB API, async.

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

    async def get_topics(self) -> list[str]:
        """Find all available topics."""
        if self.topics == [""]:
            r = await self.query("show measurements")
            self.topics = list(r["name"].values)
        return self.topics

    async def get_fields(self, measurement: str) -> pd.DataFrame:
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
        return await self.query(query)

    async def select_time_series(
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
        if topic_name not in await self.get_topics():
            logger.error(f"{topic_name} not in {self.db_name} topics.")
            return pd.DataFrame([])
        query = self._time_series_query(
            topic_name=topic_name, fields=fields, t_start=t_start, t_end=t_end, index=index
        )
        return await self.query(query)

    async def select_top_n(
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
        if topic_name not in await self.get_topics():
            logger.error(f"{topic_name} not in {self.db_name} topics.")
            return pd.DataFrame([])
        query = self._top_n_query(
            topic_name=topic_name, fields=fields, num=num, time_cut=time_cut, index=index
        )
        return await self.query(query)
