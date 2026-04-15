import logging

import pandas as pd
from astropy.time import Time

from .influx_query import InfluxQueryClient

logger = logging.getLogger(__name__)

__all__ = [
    "diasource_visit_summaries",
]


def diasource_visit_summaries(
    t_start: Time, t_end: Time, influx_client: InfluxQueryClient | None = None
) -> pd.DataFrame:
    """Query lsst.prompt.prod topics to retrieve pipelines metrics outputs.

    Parameters
    ----------
    t_start
        Time of the start of the events.
    t_end
        Time of the end of the events.
    influx_client
        InfluxQueryClient with prompt processing metric outputs.
        If None, appropriate connection at USDF will be used.
        This should be equivalent to the endpoints['sasquatch'] client.
    """
    if influx_client is None:
        influx_client = InfluxQueryClient("usdf-dev", db_name="lsst.prompt")

    # DiaSources
    topic = "lsst.prompt.prod.numDiaSourcesGood"
    dia_det: pd.DataFrame = influx_client.select_time_series(
        topic, ["visit", "detector", "numAllDiaSources", "numGoodDiaSources", "run"], t_start, t_end
    )
    logger.info(f"Retrieved {len(dia_det)} records from {topic}, from {len(dia_det.run.unique())} runs.")

    if len(dia_det) > 0:
        alertsum = dia_det.groupby("visit").agg(
            {"numAllDiaSources": "sum", "numGoodDiaSources": ("sum", "median"), "detector": "count"}
        )
        alertsum.index = alertsum.index.astype(int)
        cols = [f"{c[0]}_{c[1]}" for c in alertsum.columns]
        alertsum = alertsum.droplevel(level=0, axis=1)
        alertsum.columns = cols
        alertsum.rename({"detector_count": "nDiaDetectors_count"}, axis=1, inplace=True)
    else:
        alertsum = pd.DataFrame(
            [],
            columns=[
                "numAllDiaSources_sum",
                "numGoodDiaSources_sum",
                "numGoodDiaSources_median",
                "nDiaDetectors_count",
            ],
        )
        logger.warning(f"No records in {topic}.")

    # Solar system alerts
    topic = "lsst.prompt.prod.numSsObjects"
    sso_det: pd.DataFrame = influx_client.select_time_series(
        topic, ["visit", "detector", "NumSsObjectsMetric", "run"], t_start, t_end
    )
    logger.info(f"Retrieved {len(sso_det)} records from {topic}, from {len(sso_det.run.unique())} runs.")

    if len(sso_det) > 0:
        ssosum = sso_det.groupby("visit").agg({"NumSsObjectsMetric": "sum", "detector": "count"})
        ssosum.index = ssosum.index.astype(int)
        ssosum.rename(
            {"detector": "nSsDetectors_count", "NumSsObjectsMetric": "numSsObjects_sum"}, axis=1, inplace=True
        )
    else:
        ssosum = pd.DataFrame([], columns=["numSsObjects_sum", "nSsDetectors_count"])
        logger.warning(f"No records from {topic}")

    # And direct solar system associations (not alerts)
    topic = "lsst.prompt.prod.numDirectSsObjects"
    asso_det: pd.DataFrame = influx_client.select_time_series(
        topic, ["visit", "detector", "NumSsObjectsMetric", "run"], t_start, t_end
    )
    logger.info(f"Retrieved {len(asso_det)} records from {topic}, from {len(asso_det.run.unique())} runs.")

    if len(asso_det) > 0:
        assosum = asso_det.groupby("visit").agg({"NumSsObjectsMetric": "sum", "detector": "count"})
        assosum.index = assosum.index.astype(int)
        assosum.rename(
            {"detector": "nDSsDetectors_count", "NumSsObjectsMetric": "numDirectSsObjects_sum"},
            axis=1,
            inplace=True,
        )
    else:
        assosum = pd.DataFrame([], columns=["numDirectSsObjects_sum", "nDSsDetectors_count"])
        logger.warning(f"No records from {topic}")

    alertsum = pd.merge(ssosum, alertsum, how="outer", left_index=True, right_index=True)
    alertsum = pd.merge(assosum, alertsum, how="outer", left_index=True, right_index=True)

    return alertsum
