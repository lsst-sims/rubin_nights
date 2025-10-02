import logging

import numpy as np
import pandas as pd
from astropy.time import Time

from .influx_query import InfluxQueryClient, day_obs_from_efd_index

# from lsst.ts.xml.sal_enums import State as CSCState


__all__ = ["get_dome_open_close", "mtm1m3_slewflag_times", "get_rotator_limits", "get_tma_limits"]

logger = logging.getLogger(__name__)


def get_dome_open_close(t_start: Time, t_end: Time, efd_client: InfluxQueryClient) -> pd.DataFrame:
    """Dataframe containing the open and close times for Simonyi dome,
    from ~50% positionActual shutter values `lsst.sal.MTDome.apertureShutter`.

    Parameters
    ----------
    t_start
        Time of the start of the events.
    t_end
        Time of the end of the events.
    efd_client
        Sync EFD client.

    Returns
    -------
    dome_open_close : `pd.DataFrame`
        Dataframe containing pairs of open/close datetimes + elapsed time
        for each dome-open period in each day_obs.

    Notes
    -----
    If the query time does not include the open or close time
    for a particular period, the result will either be a negative
    elapsed open dome time or no opening/closing value reported.
    """
    # Get dome open/close information
    query = (
        "SELECT positionActual0, positionActual1, "
        "positionCommanded0, positionCommanded1 FROM "
        '"lsst.sal.MTDome.apertureShutter" WHERE '
        f"time >= '{t_start.isot}Z' AND time <= '{t_end.isot}Z' "
        "AND (abs(positionActual0) >= 48 and abs(positionActual0) <= 55) "
        "and (abs(positionActual1) >= 48 and abs(positionActual1) <= 55)"
    )
    # this should come from lsst.sal.MTDome.logevent_shutterMotion
    # instead, once logevent_shutterMotion becomes reliable.
    dome_shutter: pd.DataFrame = efd_client.query(query)
    if len(dome_shutter) == 0:
        # Make and return an empty data frame with the expected columns
        dome_shutter = pd.DataFrame([], columns=["day_obs", "open_time", "close_time", "open_hours"])
        return dome_shutter

    # Add day_obs
    dome_shutter["day_obs"] = dome_shutter.apply(day_obs_from_efd_index, axis=1)
    # Find open/close times in each day_obs
    # Fails on day_obs 20250809 -- TODO
    dome_open = []
    for day_obs in dome_shutter.day_obs.unique():
        # dome open/close events
        dd = dome_shutter.query("day_obs == @day_obs")
        opening = dd.query("positionCommanded0 == 100 or positionCommanded1 == 100")
        if len(opening) > 0:
            gaps = (
                np.concatenate(
                    [np.array([-1]), np.where((np.diff(opening.index) / pd.Timedelta(1, "s")) > 5 * 60)[0]]
                )
                + 1
            )
            open_start = opening.iloc[gaps].index.values
            closing = dd.query("positionCommanded0 == 0 or positionCommanded1 == 0")
            if len(closing) > 0:
                # This means for an in-progress night with the dome open,
                # or if the query doesn't include the closing time,
                # we won't record the last open/close.
                gaps = (
                    np.concatenate(
                        [
                            np.array([-1]),
                            np.where((np.diff(closing.index) / pd.Timedelta(1, "s")) > 5 * 60)[0],
                        ]
                    )
                    + 1
                )
                close_start = closing.iloc[gaps].index.values
                for os, cs in zip(open_start, close_start):
                    open_time = Time(os).tai
                    close_time = Time(cs).tai
                    open_hours = (close_time - open_time).jd * 24
                    dome_open.append([day_obs, open_time, close_time, open_hours])
    dome_open = pd.DataFrame(dome_open, columns=["day_obs", "open_time", "close_time", "open_hours"])
    return dome_open


def mtm1m3_slewflag_times(t_start: Time, t_end: Time, efd_client: InfluxQueryClient) -> pd.DataFrame:
    """Dataframe containing slew times calculated
    from the mtm1m3 clear/set SlewFlags, and linked to groupId using nextVisit.

    Parameters
    ----------
    t_start
        Time of the start of the events.
    t_end
        Time of the end of the events.
    efd_client
        Sync EFD client.

    Returns
    -------
    mt_slews : `pd.DataFrame`
        Dataframe containing groupId, scriptSalIndex, and mt_slew_time.
    """
    # Get MTM1M3 slew flags
    slew_start = efd_client.select_time_series(
        "lsst.sal.MTM1M3.command_setSlewFlag", ["private_identity"], t_start, t_end
    )
    slew_end = efd_client.select_time_series(
        "lsst.sal.MTM1M3.command_clearSlewFlag", ["private_identity"], t_start, t_end
    )
    # We can only match the "Script:" entries (with script salindex values).
    slew_start = slew_start.query("private_identity.str.contains('Script:')")
    slew_end = slew_end.query("private_identity.str.contains('Script:')")
    slew_start["scriptSalIndex"] = slew_start.private_identity.str.strip("Script:").astype(int)
    slew_end["scriptSalIndex"] = slew_end.private_identity.str.strip("Script:").astype(int)

    # Check which queues to check for restarts (probably just 1)
    # queue_indexes = np.unique(np.floor(slew_start.scriptSalIndex.values/1e5))

    # ScriptQueue restarts -- should do this
    # slew_start_idx = []
    # slew_end_idx = []
    # for queue_index in queue_indexes:
    #     enabled_state = CSCState.ENABLED.value
    #     topic = "lsst.sal.ScriptQueue.logevent_summaryState"
    #     fields = ["summaryState"]
    #     dd = efd_client.select_time_series(topic, fields,
    #     t_start, t_end, index=int(queue_index))
    #     if len(dd) > 0:
    #         # Identify re-enable times
    #         restarts = dd.query("summaryState == @enabled_state")
    #         slew_start_idx.append(np.searchsorted(slew_start.index.values,
    #         restarts.index.values))
    #         slew_end_idx.append(np.searchsorted(slew_end.index.values,
    #         restarts.index.values))

    slew_start = slew_start.reset_index().groupby("scriptSalIndex").agg({"time": "first"}).reset_index()
    slew_end = slew_end.reset_index().groupby("scriptSalIndex").agg({"time": "last"}).reset_index()

    mt_slew = pd.merge(
        slew_start,
        slew_end,
        how="outer",
        left_on="scriptSalIndex",
        right_on="scriptSalIndex",
        suffixes=["_start", "_end"],
    )
    mt_slew["mt_slew_time"] = (mt_slew["time_end"] - mt_slew["time_start"]) / np.timedelta64(1, "s")

    missing = set(slew_start.scriptSalIndex.values).symmetric_difference(set(slew_end.scriptSalIndex.values))
    logging.debug(
        f"Found {len(slew_start)} slew starts and {len(slew_end)} slew ends, with "
        f"{len(slew_start.scriptSalIndex.unique())} and {len(slew_end.scriptSalIndex.unique())} "
        f"unique script salIndexes each."
    )
    logging.debug(f"Differences include {missing} script salIndex")

    # Get nextVisit events as well, to get groupId.
    topic = "lsst.sal.ScriptQueue.logevent_nextVisit"
    nextvisits = efd_client.select_time_series(topic, "*", t_start, t_end, index=1)
    # Multiple next visit events can be issued for the same target, so
    # group next visit events on script salindex if the target is the same.
    # Only the last groupId will be the acquired exposure.
    nextvisits = (
        nextvisits.reset_index()
        .groupby(["scriptSalIndex", "position0", "position1", "cameraAngle"])
        .last()
        .reset_index()
    )
    nextvisits = nextvisits.set_index("time")

    mt_slew = pd.merge(nextvisits[["groupId", "scriptSalIndex"]], mt_slew, how="left", on="scriptSalIndex")
    return mt_slew


def get_rotator_limits(t_start: Time, t_end: Time, efd_client: InfluxQueryClient) -> pd.DataFrame:
    # Get rotator limit information
    topic = "lsst.sal.MTRotator.logevent_configuration"
    rot_mapping = {
        "positionAngleLowerLimit": "rotator_min",
        "positionAngleUpperLimit": "rotator_max",
        "velocityLimit": "maxspeed",
        "accelerationLimit": "accel",
        "emergencyJerkLimit": "jerk",
        "drivesEnabled": "drivesEnabled",
    }
    fields = list(rot_mapping.keys())
    rot_start = efd_client.select_top_n(topic, fields, num=1, time_cut=t_start)
    rot = efd_client.select_time_series(topic, fields, t_start, t_end)
    rot = pd.concat([rot_start, rot])
    rot.query("drivesEnabled == 1.0", inplace=True)
    rot.rename(rot_mapping, axis=1, inplace=True)
    # Make sure a value is in place for t_start
    index_edges = [
        pd.to_datetime(t_start.utc.datetime).tz_localize("UTC"),
        pd.to_datetime(t_end.utc.datetime).tz_localize("UTC"),
    ]
    rot_edges = pd.DataFrame(np.nan, index=index_edges, columns=list(rot_mapping.values()))
    rot = pd.concat([rot, rot_edges])
    rot.sort_index(inplace=True)
    rot.drop("drivesEnabled", axis=1, inplace=True)
    rot.ffill(axis=0, inplace=True)
    rot.query("index >= @index_edges[0] and index <= @index_edges[1]", inplace=True)
    return rot


def get_tma_limits(t_start: Time, t_end: Time, efd_client: InfluxQueryClient) -> pd.DataFrame:
    # Get elevation limits
    topic = "lsst.sal.MTMount.logevent_elevationControllerSettings"
    el_mapping = {
        "minL1Limit": "altitude_minpos",
        "maxL1Limit": "altitude_maxpos",
        "maxMoveVelocity": "altitude_maxspeed",
        "maxMoveAcceleration": "altitude_accel",
        "maxMoveJerk": "altitude_jerk",
    }
    fields = list(el_mapping.keys())
    elevation_start = efd_client.select_top_n(topic, fields, num=1, time_cut=t_start)
    elevation = efd_client.select_time_series(topic, fields, t_start, t_end)
    elevation = pd.concat([elevation_start, elevation])
    elevation.rename(el_mapping, axis=1, inplace=True)
    # Get azimuth limits
    topic = "lsst.sal.MTMount.logevent_azimuthControllerSettings"
    az_mapping = {
        "minL1Limit": "azimuth_minpos",
        "maxL1Limit": "azimuth_maxpos",
        "maxMoveVelocity": "azimuth_maxspeed",
        "maxMoveAcceleration": "azimuth_accel",
        "maxMoveJerk": "azimuth_jerk",
    }
    fields = list(az_mapping.keys())
    azimuth_start = efd_client.select_top_n(topic, fields, num=1, time_cut=t_start)
    azimuth = efd_client.select_time_series(topic, fields, t_start, t_end)
    azimuth = pd.concat([azimuth_start, azimuth])
    azimuth.rename(az_mapping, axis=1, inplace=True)
    # First be sure we can come up with a value in place for t_start
    index_edges = [
        pd.to_datetime(t_start.utc.datetime).tz_localize("UTC"),
        pd.to_datetime(t_end.utc.datetime).tz_localize("UTC"),
    ]
    elevation_edges = pd.DataFrame(np.nan, index=index_edges, columns=list(el_mapping.values()))
    azimuth_edges = pd.DataFrame(np.nan, index=index_edges, columns=list(az_mapping.values()))
    elevation = pd.concat([elevation, elevation_edges])
    elevation.sort_index(inplace=True)
    # Fill nans with previous values
    elevation.ffill(axis=0, inplace=True)
    azimuth = pd.concat([azimuth, azimuth_edges])
    azimuth.sort_index(inplace=True)
    # Fill nans with previous values
    azimuth.ffill(axis=0, inplace=True)
    # Merge these together with a 10 second tolerance
    match_range = pd.Timedelta(10, unit="second")
    tma = pd.merge_asof(
        left=elevation,
        right=azimuth,
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=match_range,
    )
    # And another fill, where azimuth was updated without altitude, etc.
    tma.ffill(axis=0, inplace=True)
    tma.query("index >= @index_edges[0] and index <= @index_edges[1]", inplace=True)
    return tma
