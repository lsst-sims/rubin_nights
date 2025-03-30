import logging

import httpx
import pandas as pd
from astropy.time import Time

logger = logging.getLogger(__name__)

__all__ = ["EfdQueryClient", "build_influxdb_query", "build_influxdb_top_n_query"]


def build_influxdb_query(
    measurement: str,
    fields: list[str] | None = None,
    time_range: tuple[Time, Time] | None = None,
    filters: list[tuple[str, str]] | None = None,
) -> str:
    """Build an influx DB query.

    Parameters
    ----------
    measurement : `str`
        The name of the topic / measurement.
    fields : `list` [`str`] or None
        List of fields to return from the topic.
        Default None uses `*` (all fields).
    time_range : `tuple` (`Time`, `Time`) or None
        The time window (in astropy.time.Time) to query.
    filters : `list` (`str`, `str`) or None
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


def build_influxdb_top_n_query(
    measurement: str,
    fields: list[str] | None = None,
    num: int = 10,
    time_cut: Time | None = None,
    filters: list[tuple[str, str]] | None = None,
) -> str:
    """Build an influx DB query.

    Parameters
    ----------
    measurement : `str`
        The name of the topic / measurement.
    fields : `list` [`str`] or None
        List of fields to return from the topic.
        Default None uses `*` (all fields).
    num : `int`
        The maximum number of records to return.
    time_cut : `Time` or None
        Search for only records at or before this time.
    filters : `list` (`str`, `str`) or None
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


class EfdQueryClient:
    """Query for EFD InfluxDB data.

    Parameters
    ----------
    site : `str`, optional
        The site to use for the EFD.
        Note: `usdf-dev` does not work, and will be replaced with `usdf`.
        Summit is untested.
    db_name : `str`, optional
        The database to query.
        Default is "efd".
    results_as_dataframe : `bool`
        If True, convert query results into a pandas DataFrame.
        If False, results are returned as a list of dictionaries.
    """

    def __init__(self, site: str = "usdf", db_name: str = "efd", results_as_dataframe: bool = True):
        if site == "usdf-dev":
            site = "usdf"
        self.site = site + "_efd"
        self._fetch_credentials()
        self.db_name = db_name
        self.results_as_dataframe = results_as_dataframe

    def _fetch_credentials(self):
        creds_service = f"https://roundtable.lsst.codes/segwarides/creds/{self.site}"
        efd_creds = httpx.get(creds_service)
        efd_creds = efd_creds.json()
        self.auth = (efd_creds["username"], efd_creds["password"])
        self.url = "https://" + efd_creds["host"] + efd_creds["path"] + "query"

    def __repr__(self):
        return f"{self.db_name} at {self.url}"

    def query(self, query: str) -> dict | pd.DataFrame:
        """Send a query to the InfluxDB API."""
        params = {"db": self.db_name, "q": query}
        try:
            response = httpx.get(
                self.url,
                auth=self.auth,
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
            logging.warning(f"Query {query} produced no results.")

        return result

    def _to_dataframe(self, response: dict) -> pd.DataFrame:
        """Convert an InfluxDB response to a dataframe.

        Parameters
        ----------
        response : dict
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
        result = result.set_index(pd.to_datetime(result["time"])).drop("time", axis=1)
        if result.index.tzinfo is None:
            result.index = result.index.tz_localize("UTC")
        if "tags" in series:
            for k, v in series["tags"].items():
                result[k] = v
        if "name" in series:
            result.name = series["name"]
        return result

    def select_time_series(
        self,
        topic_name,
        fields,
        t_start,
        t_end,
        index=None,
    ):
        if index:
            filters = [("salIndex", index)]
        else:
            filters = None
        query = build_influxdb_query(topic_name, fields=fields, time_range=(t_start, t_end), filters=filters)
        return self.query(query)

    def select_top_n(self, topic_name, fields, num, time_cut=None, index=None):
        if index:
            filters = [("salIndex", index)]
        else:
            filters = None
        query = build_influxdb_top_n_query(
            topic_name, fields=fields, num=num, time_cut=time_cut, filters=filters
        )
        return self.query(query)
