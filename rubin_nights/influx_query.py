import logging

import httpx
import numpy as np
import pandas as pd
from astropy.time import Time

logger = logging.getLogger(__name__)

__all__ = ["InfluxQueryClient", "day_obs_from_efd_index"]


def day_obs_from_efd_index(x: pd.Series) -> int:
    """Use with pandas apply(efd_values, axis=1) to get dayobs."""
    dayobs_time = Time(np.floor(Time(x.name, scale="utc").tai.mjd - 0.5), format="mjd", scale="tai")
    return int(dayobs_time.isot.split("T")[0].replace("-", ""))


class InfluxQueryClient:
    """Query for InfluxDB data such as EFD.

    Parameters
    ----------
    site
        The site to use for the EFD.
        Note: `usdf-dev` does not exist, and will be replaced with `usdf`.
        Summit is untested.
    db_name
        The database to query.
        Default is "efd".
    results_as_dataframe
        If True, convert query results into a pandas DataFrame.
        If False, results are returned as a list of dictionaries.
    """

    def __init__(
        self,
        site: str = "usdf",
        db_name: str = "efd",
        query_timeout: float = 5 * 60,
        results_as_dataframe: bool = True,
    ) -> None:
        if site == "usdf-dev":
            site = "usdf"
        self.site = site + "_efd"
        self.url, auth = self._fetch_credentials()
        self.db_name = db_name
        self.results_as_dataframe = results_as_dataframe
        timeout = httpx.Timeout(query_timeout, connect=10.0)
        self.httpx_client = httpx.Client(base_url=self.url, timeout=timeout, auth=auth)

    def _fetch_credentials(self) -> tuple[str, tuple[str, bytes]]:
        creds_service = f"https://roundtable.lsst.codes/segwarides/creds/{self.site}"
        try:
            efd_creds = httpx.get(creds_service)
        except Exception as e:
            logger.error(f"Could not fetch credentials for {self.site}")
            logger.error(e)
            efd_creds.raise_for_status()
        efd_creds = efd_creds.json()
        auth = (efd_creds["username"], efd_creds["password"])
        url = "https://" + efd_creds["host"] + efd_creds["path"].rstrip("/")
        return url, auth

    def __repr__(self) -> str:
        return f"{self.db_name} at {self.url}"

    def query(self, query: str) -> dict | pd.DataFrame:
        """Send a query to the InfluxDB API."""
        params = {"db": self.db_name, "q": query}
        try:
            response = self.httpx_client.get(
                "/query",
                params=params,
            )
            response.raise_for_status()
        except Exception as e:
            logger.warning(e)
            response = None

        if response:
            if self.results_as_dataframe:
                result = self._to_dataframe(response.json())
            else:
                result = response.json()
        else:
            result = []
            if self.results_as_dataframe:
                result = pd.DataFrame(result)
        if len(result) == 0:
            logging.debug(f"Query {query} produced no results.")

        return result

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
        # One InfluxDB measurement queried at a time
        series = statement["series"][0]
        result = pd.DataFrame(series.get("values", []), columns=series["columns"])
        if "time" not in result.columns:
            return result
        result = result.set_index(pd.to_datetime(result["time"], format="ISO8601")).drop("time", axis=1)
        if result.index.tzinfo is None:
            result.index = result.index.tz_localize("UTC")
        if "tags" in series:
            for k, v in series["tags"].items():
                result[k] = v
        if "name" in series:
            result.name = series["name"]
        return result

    def get_topics(self) -> list[str]:
        """Find all available topics."""
        topics = self.query("show measurements")["name"].to_list()
        return topics

    def get_fields(self, measurement: str) -> pd.DataFrame:
        """Query the list of field names for a topic.

        Parameters
        ----------
        measurement
            Name of measurement/topic to query for field names.

        Returns
        -------
        fields : `pd.DataFrame`
            DataFrame with fieldKey / fieldType columns.
        """
        query = f'show field keys from "{measurement}"'
        return self.query(query)

    @staticmethod
    def build_influxdb_query(
        measurement: str,
        fields: list[str] | str = "*",
        time_range: tuple[Time, Time] | None = None,
        filters: list[tuple[str, str]] | None = None,
    ) -> str:
        """Build an influx DB query.

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
        """Build an influx DB query.

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

    def select_time_series(
        self,
        topic_name: str,
        fields: str | list[str],
        t_start: Time,
        t_end: Time,
        index: int | None = None,
    ) -> pd.DataFrame:
        if index:
            filters = [("salIndex", str(index))]
        else:
            filters = None
        query = self.build_influxdb_query(
            topic_name, fields=fields, time_range=(t_start, t_end), filters=filters
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
        if index:
            filters = [("salIndex", str(index))]
        else:
            filters = None
        query = self.build_influxdb_top_n_query(
            topic_name, fields=fields, num=num, time_cut=time_cut, filters=filters
        )
        return self.query(query)
