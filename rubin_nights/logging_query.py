"""Execute queries for logging services."""

import logging
from typing import Literal
from urllib.parse import urlparse

import httpx
import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta

__all__ = ["LoggingServiceClient", "NightReportClient", "ExposureLogClient"]


logger = logging.getLogger(__name__)


class LoggingServiceClient:
    """Query the logging services.

    Parameters
    ----------
    url : `str`
        Endpoint URL for the night report.
    auth : `tuple`
        The username and password for authentication.
    results_as_dataframe : `bool`
        If True, convert query results into a pandas DataFrame.
        If False, results are returned as a list of dictionaries.
    """

    def __init__(self, url: str, auth: tuple, results_as_dataframe: bool = True):
        self.url = url
        self.auth = auth
        self.results_as_dataframe = results_as_dataframe

    def __repr__(self):
        return self.url

    def query(self, params: dict) -> list[dict] | pd.DataFrame:
        """Execute query to logging services API, for any params.

        Parameters
        ----------
        params : `dict`
            Dictionary of parameters for the REST API query.
            See docs for each service for more details.

        Returns
        -------
        messages : `list` [`dict`] or `pd.DataFrame`
            The returned log messages (if any available).
            If `self.results_as_dataframe` is True, this will be
            transformed to a pandas DataFrame.
        """
        # Some requests from the logging endpoints fail the first time.
        response = httpx.get(self.url, auth=self.auth, params=params)
        # So, try twice (but twice should succeed)
        if response.status_code != 200:
            try:
                response = httpx.get(self.url, auth=self.auth, params=params)
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
    api_base : `str`
        Base API for services.
        e.g. https://usdf-rsp.slac.stanford.edu
    auth : `tuple`
        The username and password for authentication.
    """

    def __init__(self, api_base: str, auth: tuple):
        url = api_base + "/nightreport/reports"
        super().__init__(url=url, auth=auth, results_as_dataframe=False)

    def query_night_report(
        self, day_obs: str, telescope: Literal["AuxTel", "Simonyi"], display_report: bool = True
    ) -> list[dict]:
        """Fetch the night report logs.

        Parameters
        ----------
        day_obs :  `str`
            The day_obs of the night report. Format YYYY-MM-DD.
        telescope : `str`
            Fetch the night report logs for this telescope (AuxTel or Simonyi).
        display_report : `bool`
            Display the night report logs immediately.

        Returns
        -------
        night_reports : `list` {`dict`}
            The night report logs for this telescope, which are a list
            (often a single-element list, but can be multiple during the night)
            of dictionary key:value pairs describing the night report.
        """
        # convert day_obs YYYY-MM-DD into int for log request
        this_dayobs = day_obs.replace("-", "")
        next_dayobs = (Time(day_obs, format="iso") + TimeDelta(1, format="jd")).iso[0:10].replace("-", "")

        if telescope.lower().startswith("aux"):
            tel_nr = "AuxTel"
        else:
            tel_nr = "Simonyi"

        params = {
            "telescopes": tel_nr,
            "min_day_obs": this_dayobs,
            "max_day_obs": next_dayobs,
            "is_valid": "true",
        }

        night_reports = self.query(params=params)

        if len(night_reports) == 0:
            logger.warning(f"No night report available for {day_obs}")

        if display_report:
            self.display_night_report(night_reports)

        return night_reports

    @staticmethod
    def display_night_report(night_reports: list[dict]):
        if isinstance(night_reports, list):
            log = night_reports[0]
        else:
            log = night_reports
        try:
            from IPython.display import Markdown, display

            display(Markdown(f"Observing crew : {log['observers_crew']}"))
            night_plan_block = "BLOCK" + urlparse(log["confluence_url"]).fragment.split("BLOCK")[-1]
            if night_plan_block == "BLOCK":
                night_plan_block = log["confluence_url"]
            url = log["confluence_url"]
            display(
                Markdown(
                    f'Night plan : <a href="{url}" target="_blank" rel="noreferrer noopener">'
                    f"{night_plan_block}</a>"
                )
            )
            display(Markdown("<strong>Summary</strong>"))
            display(Markdown(log["summary"]))
            display(Markdown("<strong>Status</strong>"))
            display(Markdown(log["telescope_status"]))
        except ModuleNotFoundError:
            print(f"Observing crew : {log['observers_crew']}")
            night_plan_block = "BLOCK" + urlparse(log["confluence_url"]).fragment.split("BLOCK")[-1]
            if night_plan_block == "BLOCK":
                night_plan_block = log["confluence_url"]
            url = log["confluence_url"]
            print(
                f'Night plan : <a href="{url}" target="_blank" rel="noreferrer noopener">'
                f"{night_plan_block}</a>"
            )
            print("Summary:")
            print(log["summary"])
            print("Status:")
            print(log["telescope_status"])


class NarrativeLogClient(LoggingServiceClient):
    """Query for the narrative log.

    Parameters
    ----------
    api_base : `str`
        Base API for services.
        e.g. https://usdf-rsp.slac.stanford.edu
    auth : `tuple`
        The username and password for authentication.
    """

    def __init__(self, api_base: str, auth: tuple):
        url = api_base + "/narrativelog/messages"
        super().__init__(url=url, auth=auth, results_as_dataframe=True)

    def query_log(self, t_start: Time, t_end: Time, user_params: dict | None = None) -> pd.DataFrame:
        """Get narrative log entries over a specified timespan.

        Parameters
        ----------
        t_start : `Time`
            Time of start of narrative log query.
        t_end : `Time`
            Time of end of narrative log query.
        user_params : `dict`, optional
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

        messages = self.query(params=params)
        if len(messages) == log_limit:
            logger.warning(f"Narrative log messages hit log_limit ({log_limit})")
        if len(messages) > 0:
            # Strip out excessive \r\n values
            def strip_rns(x):
                return x.message_text.replace("\r\n", "\n").replace("\n\n", "\n").rstrip("\n")

            # Convert string time to datetime
            def make_time(x, column):
                return Time(x[column], format="isot", scale="tai").utc.datetime

            # join log components for compactness
            def clarify_log(x, column):
                if column == "components_json":
                    if x[column] is None:
                        component = "Log"
                    elif x[column].values() is None:
                        component = "Log"
                    else:
                        component = "Log " + " ".join(x[column].values())
                else:
                    if x[column] is None:
                        component = "Log"
                    else:
                        component = "Log " + " ".join(x[column])
                return component

            # Strip excessive \r\n and \n\n from messages
            messages["message_text"] = messages.apply(strip_rns, axis=1)
            # Add a time index -
            # date_added seems to align best with remainder of scriptqueue
            messages["time"] = messages.apply(make_time, args=("date_added",), axis=1)
            messages.set_index("time", inplace=True)
            messages.index = messages.index.tz_localize("UTC")
            # Join the components and add "Log" explicitly
            # Choose between 'components' and 'components_json'
            if np.all(messages["components_json"] == None):  # noqa: E711
                key = "components"
            else:
                key = "components_json"
            messages["component"] = messages.apply(clarify_log, args=(key,), axis=1)
        return messages


class ExposureLogClient(LoggingServiceClient):
    """Query for the exposure log.

    Parameters
    ----------
    api_base : `str`
        Base API for services.
        e.g. https://usdf-rsp.slac.stanford.edu
    auth : `tuple`
        The username and password for authentication.
    """

    def __init__(self, api_base: str, auth: tuple):
        url = api_base + "/exposurelog/messages"
        super().__init__(url=url, auth=auth, results_as_dataframe=True)

    def query_log(self, t_start: Time, t_end: Time, user_params: dict | None = None) -> pd.DataFrame:
        """Get exposure log entries over a specified timespan.

        Parameters
        ----------
        t_start : `Time`
            Time of start of narrative log query.
        t_end : `Time`
            Time of end of narrative log query.
        user_params : `dict`, optional
            Additional parameters to add or override defaults.
            Passing `{'limit': int}` can override the default limit.

        Returns
        -------
        messages : `pd.DataFrame`
            Exposure log messages.
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

        messages = self.query(params=params)
        if len(messages) == log_limit:
            logger.warning(f"Narrative log messages hit log_limit ({log_limit})")

        return messages
