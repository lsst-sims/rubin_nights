import logging
import re
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta

from .dayobs_utils import day_obs_to_time

__all__ = ["NightReportClient", "NarrativeLogClient", "ExposureLogClient"]


logger = logging.getLogger(__name__)


def logtime_to_datetime(x: pd.Series, column: str) -> str:
    """Convert the times reported in the narrative or exposure log (TAI)
    into datetimes which can be merged with EFD indexes (UTC).

    Use as pd.apply function like so:

    .. code-block:: python

        messages["time"] = messages.apply(logtime_to_datetime,
                                          args=("date_added",), axis=1)
        messages.set_index("time", inplace=True)
        messages.index = messages.index.tz_localize("UTC")

    """
    return Time(x[column], format="isot", scale="tai").utc.datetime


class LoggingServiceClient:
    """Query the logging services.

    Parameters
    ----------
    url
        Endpoint URL for the night report.
    auth
        The username and password for authentication.
    results_as_dataframe
        If True, convert query results into a pandas DataFrame.
        If False, results are returned as a list of dictionaries.
    """

    def __init__(self, url: str, auth: tuple, results_as_dataframe: bool = True):
        self.url = url
        self.results_as_dataframe = results_as_dataframe
        timeout = httpx.Timeout(120, connect=30)
        self.httpx_client = httpx.Client(timeout=timeout, auth=auth)
        self.async_client = httpx.AsyncClient(timeout=timeout, auth=auth)

    def __repr__(self) -> str:
        return self.url

    async def close_async(self) -> None:
        """Close async connection client."""
        await self.async_client.aclose()

    def close(self) -> None:
        """Close sync connection client."""
        self.httpx_client.close()

    def query(self, params: dict) -> list[dict] | pd.DataFrame:
        """Execute sync query to logging services API, for any params.

        Parameters
        ----------
        params
            Dictionary of parameters for the REST API query.
            See docs for each service for more details.

        Returns
        -------
        messages : `list` [`dict`] or `pd.DataFrame`
            The returned log messages (if any available).
            If `self.results_as_dataframe` is True, this will be
            transformed to a pandas DataFrame.
        """
        # This is a stupid simple retry - because the logging services
        # often drop the first request, but are ok after that.
        response = self.httpx_client.get(self.url, params=params)
        if response.status_code != 200:
            try:
                response = self.httpx_client.get(self.url, params=params)
                response.raise_for_status()
            except httpx.RequestError as exc:
                logger.warning(f"An error occurred while requesting {exc.request.url!r}.")
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    f"Error response {exc.response.status_code} while requesting {exc.request.url!r}."
                )
        # If query was successful, decode and dataframe
        if response.status_code == 200:
            messages = response.json()
        else:
            messages = []
        if self.results_as_dataframe:
            messages = pd.DataFrame(messages)
        return messages

    async def async_query(self, params: dict) -> list[dict] | pd.DataFrame:
        """Execute async query to logging services API, for any params.

        Parameters
        ----------
        params
            Dictionary of parameters for the REST API query.
            See docs for each service for more details.

        Returns
        -------
        messages : `list` [`dict`] or `pd.DataFrame`
            The returned log messages (if any available).
            If `self.results_as_dataframe` is True, this will be
            transformed to a pandas DataFrame.
        """
        # This is a stupid simple retry - because the logging services
        # often drop the first request, but are ok after that.
        response = await self.async_client.get(self.url, params=params)
        if response.status_code != 200:
            try:
                response = await self.async_client.get(self.url, params=params)
                response.raise_for_status()
            except httpx.RequestError as exc:
                logger.warning(f"An error occurred while requesting {exc.request.url!r}.")
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    f"Error response {exc.response.status_code} while requesting {exc.request.url!r}."
                )
        # If query was successful, decode and dataframe
        if response.status_code == 200:
            messages = response.json()
        else:
            messages = []
        if self.results_as_dataframe:
            messages = pd.DataFrame(messages)
        return messages


class NightReportClient(LoggingServiceClient):
    """Query for the night report log.

    Parameters
    ----------
    api_base
        Base API for services.
        e.g. https://usdf-rsp.slac.stanford.edu
    auth
        The username and password for authentication.
    """

    def __init__(self, api_base: str, auth: tuple) -> None:
        url = api_base + "/nightreport/reports"
        super().__init__(url=url, auth=auth, results_as_dataframe=False)

    @staticmethod
    def query_params(day_obs: str | int) -> dict[str, Any]:
        """Set query parameters for night report query.

        Parameters
        ----------
        day_obs
            The day_obs for the night report.

        Returns
        -------
        params : `dict`
        """
        if isinstance(day_obs, str):
            try:
                int(day_obs)
            except ValueError:
                day_obs = int(day_obs.replace("-", ""))

        next_day_obs = day_obs_to_time(day_obs) + TimeDelta(1, format="jd")
        next_day_obs = next_day_obs.isot.split("T")[0].replace("-", "")

        params = {
            "min_day_obs": day_obs,
            "max_day_obs": next_day_obs,
            "is_valid": "true",
        }
        return params

    def query_night_report(
        self,
        day_obs: str | int,
        telescope: Literal["AuxTel", "Simonyi"] | None = None,
        return_html: bool = True,
    ) -> tuple[list[dict], str]:
        """Fetch the night report logs.

        Parameters
        ----------
        day_obs
            The day_obs of the night report.
            Format YYYY-MM-DD (str) or YYYYMMDD (int).
        telescope
            Format the night report logs for this telescope.
            Options: AuxTel, Simonyi or None (None will return both).
            The night_report now returns both telescope's summary reports.
        return_html
            Send back an HTML formatted version of the first night report log,
            optionally for a given telescope only.

        Returns
        -------
        night_reports : `list` {`dict`}
            The night report logs, which are a list
            (often a single-element list, but can be multiple during the night)
            of dictionary key:value pairs describing the night report.
        html : `str` (optional)
            If `return_html` is True, also return an HTML formatted version
            of the night report, potentially for a given telescope only.
        """
        params = self.query_params(day_obs)

        night_reports = self.query(params=params)

        if len(night_reports) == 0:
            logger.warning(f"No night report available for {day_obs}")

        if return_html:
            html = self.format_night_report(night_reports, telescope=telescope)
        else:
            html = ""

        return night_reports, html

    async def async_query_night_report(
        self,
        day_obs: str | int,
        telescope: Literal["AuxTel", "Simonyi"] | None = None,
        return_html: bool = True,
    ) -> tuple[list[dict], str]:
        """Fetch the night report logs.

        Parameters
        ----------
        day_obs
            The day_obs of the night report.
            Format YYYY-MM-DD (str) or YYYYMMDD (int).
        telescope
            Format the night report logs for this telescope.
            Options: AuxTel, Simonyi or None (None will return both).
            The night_report now returns both telescope's summary reports.
        return_html
            Send back an HTML formatted version of the first night report log,
            optionally for a given telescope only.

        Returns
        -------
        night_reports : `list` {`dict`}
            The night report logs, which are a list
            (often a single-element list, but can be multiple during the night)
            of dictionary key:value pairs describing the night report.
        html : `str` (optional)
            If `return_html` is True, also return an HTML formatted version
            of the night report, potentially for a given telescope only.
        """
        params = self.query_params(day_obs)

        night_reports = await self.async_query(params=params)

        if len(night_reports) == 0:
            logger.warning(f"No night report available for {day_obs}")

        if return_html:
            html = self.format_night_report(night_reports, telescope=telescope)
        else:
            html = ""

        return night_reports, html

    @staticmethod
    def format_night_report(night_reports: list[dict], telescope: str | None = None) -> str:
        # Accept a few varieties of "telescope"
        if telescope is not None:
            if telescope.lower().startswith("aux"):
                telescope = "AuxTel"
            elif telescope.lower().startswith("main"):
                telescope = "Simonyi"
            elif telescope.lower().startswith("simonyi"):
                telescope = "Simonyi"
            else:
                telescope = None
        if isinstance(night_reports, list):
            log = night_reports[0]
        else:
            log = night_reports
        html = ""
        # observing crew
        html += f"<p> <strong>Observing crew: </strong> {log['observers_crew']} <br>"
        # night plan
        night_plan_block = "BLOCK" + urlparse(log["confluence_url"]).fragment.split("BLOCK")[-1]
        if night_plan_block == "BLOCK":
            night_plan_block = log["confluence_url"]
        night_url = log["confluence_url"]
        # The night plan isn't generally being populated now.
        if len(night_url) > 0:
            html += (
                f"<p> <strong>Night plan: </strong> <a href='{night_url}' "
                f"target='_blank' ref='noreferrer noopener'>"
            )
            html += f"{night_plan_block}</a> <br>"
        # summary
        html += "<p> <strong>Summary:</strong><br>"
        summary = re.sub(r"[\n]{2,}", "\n", log["summary"]).replace("\n", "<br>")
        html += f"{summary}"
        if telescope is None:
            extra_summary_keys = ["maintel_summary", "auxtel_summary"]
        else:
            if telescope.lower() == "simonyi":
                extra_summary_keys = ["maintel_summary"]
            elif telescope.lower() == "auxtel":
                extra_summary_keys = ["auxtel_summary"]
            else:
                extra_summary_keys = ["maintel_summary", "auxtel_summary"]
        # Add summary for relevant telescope
        for key in extra_summary_keys:
            if key in log.keys():
                logvals = log[key]
                if logvals is not None:
                    html += f"<p> <strong> {key.replace('_', ' ')}: </strong><br>"
                    summary = re.sub(r"[\n]{2,}", "\n", logvals).replace("\n", "<br>")
                    html += f"{summary}"
        if "telescope_status" in log:
            html += "<p> <strong>Status:</strong><br>"
            status = log["telescope_status"].replace("\n", "<br>")
            html += f"{status}"
        return html


class NarrativeLogClient(LoggingServiceClient):
    """Query for the narrative log.

    Parameters
    ----------
    api_base
        Base API for services.
        e.g. https://usdf-rsp.slac.stanford.edu
    auth
        The username and password for authentication.
    """

    def __init__(self, api_base: str, auth: tuple) -> None:
        url = api_base + "/narrativelog/messages"
        super().__init__(url=url, auth=auth, results_as_dataframe=True)

    @staticmethod
    def query_params(t_start: Time, t_end: Time, user_params: dict | None = None) -> dict[str, Any]:
        """Set query parameters for narrative log query.

        Parameters
        ----------
        t_start
            Time of start of narrative log query.
        t_end
            Time of end of narrative log query.
        user_params
            Additional parameters to add or override defaults.
            Passing `{'limit': int}` can override the default limit.

        Returns
        -------
        params : `dict`
        """
        log_limit = 50000
        params = {
            "is_human": "either",
            "is_valid": "true",
            "has_date_begin": True,
            "min_date_begin": t_start.to_datetime(),
            "max_date_begin": t_end.to_datetime(),
            "order_by": "date_begin",
            "limit": log_limit,
        }
        if user_params is not None:
            params.update(user_params)
        return params

    @staticmethod
    def _reformat_narrative_log(messages: pd.DataFrame) -> pd.DataFrame:
        # Define pd.apply functions relevant for narrative log only
        def strip_rns(x: pd.Series) -> str:
            """Remove excessive returns from narrative log messages."""
            return x.message_text.replace("\r\n", "\n").replace("\n\n", "\n").rstrip("\n")

        # join log components for compactness
        def simplify_log(x: pd.Series, column: str) -> str:
            if column == "components_json":
                # Then x[column] will be a dictionary
                if x[column] is None or x[column] == {}:
                    component = "Log"
                else:

                    def findnames(testvalue: str | list | dict) -> str:
                        if isinstance(testvalue, str):
                            return testvalue
                        else:
                            if isinstance(testvalue, list):
                                testvalue = testvalue[-1]
                            elif isinstance(testvalue, dict):
                                testvalue = testvalue["name"]
                        return findnames(testvalue)

                    component = "Log " + findnames(x[column])
            else:
                if x[column] is None:
                    component = "Log"
                else:
                    component = "Log " + " ".join(x[column])
            return component

        # Strip excessive \r\n and \n\n from messages
        messages["message_text"] = messages.apply(strip_rns, axis=1)
        # Add a time index
        time_start = messages.apply(logtime_to_datetime, args=("date_begin",), axis=1)
        time_end = messages.apply(logtime_to_datetime, args=("date_end",), axis=1)
        time = np.where(messages.time_lost > 0, time_end, time_start)
        messages["time"] = time

        messages.set_index("time", inplace=True)
        messages.index = messages.index.tz_localize("UTC")
        # Join the components and add "Log" explicitly
        # Choose between 'components' and 'components_json'
        if np.all(messages["components_json"] == None):  # noqa: E711
            key = "components"
        else:
            key = "components_json"
        messages["component"] = messages.apply(simplify_log, args=(key,), axis=1)
        return messages

    def query_log(self, t_start: Time, t_end: Time, user_params: dict | None = None) -> pd.DataFrame:
        """Get narrative log entries over a specified timespan.

        Parameters
        ----------
        t_start
            Time of start of narrative log query.
        t_end
            Time of end of narrative log query.
        user_params
            Additional parameters to add or override defaults.
            Passing `{'limit': int}` can override the default limit.

        Returns
        -------
        messages : `pd.DataFrame`
            Narrative log messages.

        Notes
        -----
        Some modifications are made to the raw narrative logs.
        Extra space is stripped out and a simple "Log <component>" key
        is added to the dataframe (identifying Simonyi/Auxtel specific issues).
        The index is replaced by a time, in order to insert the narrative
        log values into other events at the telescope.
        """
        params = self.query_params(t_start=t_start, t_end=t_end, user_params=user_params)

        messages: pd.DataFrame = self.query(params=params)
        if len(messages) == params["limit"]:
            logger.warning(f"Narrative log messages hit log_limit ({params['limit']})")
        if len(messages) > 0:
            messages = self._reformat_narrative_log(messages)
        return messages

    async def async_query_log(
        self, t_start: Time, t_end: Time, user_params: dict | None = None
    ) -> pd.DataFrame:
        """Async get narrative log entries over a specified timespan.

        Parameters
        ----------
        t_start
            Time of start of narrative log query.
        t_end
            Time of end of narrative log query.
        user_params
            Additional parameters to add or override defaults.
            Passing `{'limit': int}` can override the default limit.

        Returns
        -------
        messages : `pd.DataFrame`
            Narrative log messages.

        Notes
        -----
        Some modifications are made to the raw narrative logs.
        Extra space is stripped out and a simple "Log <component>" key
        is added to the dataframe (identifying Simonyi/Auxtel specific issues).
        The index is replaced by a time, in order to insert the narrative
        log values into other events at the telescope.
        """
        params = self.query_params(t_start=t_start, t_end=t_end, user_params=user_params)

        messages: pd.DataFrame = await self.async_query(params=params)
        if len(messages) == params["limit"]:
            logger.warning(f"Narrative log messages hit log_limit ({params['limit']})")
        if len(messages) > 0:
            messages = self._reformat_narrative_log(messages)
        return messages


class ExposureLogClient(LoggingServiceClient):
    """Query for the exposure log.

    Parameters
    ----------
    api_base
        Base API for services.
        e.g. https://usdf-rsp.slac.stanford.edu
    auth
        The username and password for authentication.
    """

    def __init__(self, api_base: str, auth: tuple):
        url = api_base + "/exposurelog/messages"
        super().__init__(url=url, auth=auth, results_as_dataframe=True)

    @staticmethod
    def query_params(t_start: Time, t_end: Time, user_params: dict | None = None) -> dict[str, Any]:
        """Set query parameters for exposure log query.

        Parameters
        ----------
        t_start
            Time of start of exposure log query.
        t_end
            Time of end of exposure log query.
        user_params
            Additional parameters to add or override defaults.
            Passing `{'limit': int}` can override the default limit.

        Returns
        -------
        params : `dict`
        """
        log_limit = 50000
        params = {
            "is_human": "either",
            "is_valid": "true",
            "min_date_added": t_start.to_datetime(),
            "max_date_added": t_end.to_datetime(),
            "limit": log_limit,
        }
        if user_params is not None:
            params.update(user_params)
        return params

    def query_log(self, t_start: Time, t_end: Time, user_params: dict | None = None) -> pd.DataFrame:
        """Get exposure log message entries over a specified timespan.

        Parameters
        ----------
        t_start
            Time of start of exposure log query.
        t_end
            Time of end of exposure log query.
        user_params
            Additional parameters to add or override defaults.
            Passing `{'limit': int}` can override the default limit.

        Returns
        -------
        messages : `pd.DataFrame`
            Exposure log messages.
        """
        params = self.query_params(t_start=t_start, t_end=t_end, user_params=user_params)

        messages = self.query(params=params)
        if len(messages) == params["limit"]:
            logger.warning(f"Exposure log messages hit log_limit ({params['limit']})")

        return messages

    async def async_query_log(
        self, t_start: Time, t_end: Time, user_params: dict | None = None
    ) -> pd.DataFrame:
        """Get exposure log message entries over a specified timespan.

        Parameters
        ----------
        t_start
            Time of start of exposure log query.
        t_end
            Time of end of exposure log query.
        user_params
            Additional parameters to add or override defaults.
            Passing `{'limit': int}` can override the default limit.

        Returns
        -------
        messages : `pd.DataFrame`
            Exposure log messages.
        """
        params = self.query_params(t_start=t_start, t_end=t_end, user_params=user_params)

        messages = await self.async_query(params=params)
        if len(messages) == params["limit"]:
            logger.warning(f"Exposure log messages hit log_limit ({params['limit']})")

        return messages
