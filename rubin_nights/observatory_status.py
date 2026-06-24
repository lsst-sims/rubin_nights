import logging

import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta

from .dayobs_utils import (
    day_obs_list,
    day_obs_sunset_sunrise,
    day_obs_sunset_sunrise_df,
    day_obs_to_time,
    time_to_day_obs,
)
from .influx_query import InfluxQueryClient, day_obs_from_efd_index_array

__all__ = [
    "get_dome_open_close",
    "mtm1m3_slewflag_times",
    "get_rotator_limits",
    "get_tma_limits",
    "get_mounted_bandpasses",
    "get_obs_status_messages",
    "_obs_status_state_changes",
    "get_observatory_state_times",
    "_count_contribution",
    "count_observatory_states",
]

logger = logging.getLogger(__name__)


def get_dome_open_close(
    t_start: Time, t_end: Time, efd_client: InfluxQueryClient, with_sunset_sunrise: bool = True
) -> pd.DataFrame:
    """Dataframe containing the open and close times for Simonyi dome,
    from positionCommanded and positionActual shutter values in
    `lsst.sal.MTDome.apertureShutter`.

    Parameters
    ----------
    t_start
        Time of the start of the events.
        Note that this will be corrected to the start of the dayobs,
        to avoid missing dome-open events at the start of the night.
    t_end
        Time of the end of the events.
        Note that this will be corrected to the end of the dayobs,
        to avoid missing dome-close events later in the night.
    efd_client
        Sync EFD client.
    with_sunrise_sunset
        If True (default), add -12 degree sunset and sunrise columns.

    Returns
    -------
    dome_open_close : `pd.DataFrame`
        Dataframe containing pairs of open/close datetimes + elapsed time
        for each dome-open period in each day_obs.
        Note that the dome open/close times as well as sunset/sunrise
        are in UTC, including utc timescale.

    Notes
    -----
    This primarily returns dome open + close pairs.
    However, a dome open event without a later close will be returned
    as simply a dome open.
    """
    if t_end <= t_start:
        logger.warning(f"t_end {t_end.isot} should be larger than t_start {t_start.isot}")
        return pd.DataFrame([])

    # Adjust t_start and t_end to the start and end of their relevant dayobs.
    t_start = day_obs_to_time(time_to_day_obs(t_start))
    t_end = day_obs_to_time(time_to_day_obs(t_end))
    if t_end == t_start:
        t_end += TimeDelta(0.999, format="jd")

    logger.debug(
        f"Getting dome open/close times for {t_start.isot} to {t_end.isot} " "(corrected to start of dayobs)."
    )

    # Get dome open/close information.
    # this should likely come from lsst.sal.MTDome.logevent_shutterMotion
    # instead, once logevent_shutterMotion becomes reliable.
    open_query = (
        "SELECT positionActual0, positionActual1, "
        "positionCommanded0, positionCommanded1 FROM "
        '"lsst.sal.MTDome.apertureShutter" WHERE '
        f"time >= '{t_start.utc.isot}Z' AND time <= '{t_end.utc.isot}Z' "
        "AND (abs(positionCommanded0) = 100 and abs(positionCommanded1) = 100) "
        "AND (abs(positionActual0) >= 25 and abs(positionActual0) <= 85) "
        "and (abs(positionActual1) >= 25 and abs(positionActual1) <= 85)"
    )
    dome_shutter_open: pd.DataFrame = efd_client.query(open_query)

    # dome closes a bit faster than it opens, so having a wider range of
    # positionActual values is helpful.
    close_query = (
        "SELECT positionActual0, positionActual1, "
        "positionCommanded0, positionCommanded1 FROM "
        '"lsst.sal.MTDome.apertureShutter" WHERE '
        f"time >= '{t_start.utc.isot}Z' AND time <= '{t_end.utc.isot}Z' "
        "AND (abs(positionCommanded0) = 0 and abs(positionCommanded1) = 0) "
        "AND (abs(positionActual0) >= 25 and abs(positionActual0) <= 85) "
        "and (abs(positionActual1) >= 25 and abs(positionActual1) <= 85)"
    )
    dome_shutter_close: pd.DataFrame = efd_client.query(close_query)

    # If there were no opening events during the timespan:
    if len(dome_shutter_open) == 0:
        # Add 0 open/close times for each day_obs.
        dome_open = [
            [
                pd.NaT,
                pd.NaT,
                0,
                day_obs,
            ]
            for day_obs in day_obs_list(t_start, t_end)
        ]

    else:
        # Add day_obs
        dome_shutter_open["day_obs"] = day_obs_from_efd_index_array(dome_shutter_open.index.to_series())
        if len(dome_shutter_close) > 0:
            dome_shutter_close["day_obs"] = day_obs_from_efd_index_array(dome_shutter_close.index.to_series())

        # Find open/close times in each day_obs, including no-event day_obs
        dome_open = []
        for day_obs in day_obs_list(t_start, t_end):
            # dome open/close events
            opening = dome_shutter_open.query("day_obs == @day_obs")
            open_start = None
            if len(opening) > 0:
                # There are many 'opening' lines in dome_shutter_open;
                # Find the first in each 5 minute interval.
                gaps = np.where((np.diff(opening.index) / pd.Timedelta(1, "s")) > 5 * 60)[0]
                # And add 1 because np.diff gives you the previous index.
                gaps += 1
                # And add an index for the very first dome_open index,
                # which doesn't have a previous 5 minute interval
                # (so misses diff).
                gaps = np.concatenate([np.array([0]), gaps])
                open_start = Time(opening.iloc[gaps].index.values, scale="utc").utc.datetime

            if "day_obs" in dome_shutter_close.columns:
                closing = dome_shutter_close.query("day_obs == @day_obs")
            else:
                closing = pd.DataFrame([])
            close_start = None
            if len(closing) > 0:
                # Pick out the dome closing events that are first in each
                # 3 minute interval (separate dome closing events).
                gaps = np.where((np.diff(closing.index) / pd.Timedelta(1, "s")) > 3 * 60)[0]
                gaps += 1
                gaps = np.concatenate([np.array([0]), gaps])
                close_start = Time(closing.iloc[gaps].index.values, scale="utc").utc.datetime

            # Sometimes telemetry is unexpected, so don't just zip.
            # Look through open and close and match them up.
            if open_start is not None:
                for i in range(len(open_start)):
                    open_time = open_start[i]
                    if close_start is not None:
                        # Find the possible close times for this open_time.
                        close_idx = np.where(close_start >= open_time)[0]
                        # If there are any - pick the first one.
                        if len(close_idx) > 0:
                            close_time = close_start[close_idx[0]]
                        else:
                            close_time = pd.NaT
                    else:
                        # No close_start times at all
                        close_time = pd.NaT
                    if pd.isna(close_time):
                        # Let's give a cumulative "up to now" time,
                        # if there was an open time but no close.
                        # (remember day_obs_sunset_sunrise caches)
                        sunset, sunrise = day_obs_sunset_sunrise(day_obs, sun_alt=-12)
                        count_stop = np.min([Time.now().utc.datetime, sunrise.utc.datetime])
                        open_hours = (count_stop - open_time) / np.timedelta64(3600, "s")
                    else:
                        # Calculate the actual dome-shutter-open time.
                        open_hours = (close_time - open_time) / np.timedelta64(3600, "s")

                    dome_open.append([open_time, close_time, open_hours, day_obs])
            else:
                # No open start times at all - append an empty line
                # (but check that we're not looking at a day in the future).
                if Time.now() > day_obs_to_time(day_obs):
                    dome_open.append(
                        [
                            pd.NaT,
                            pd.NaT,
                            0,
                            day_obs,
                        ]
                    )

    dome_open_df = pd.DataFrame(
        dome_open,
        columns=[
            "open_time",
            "close_time",
            "dome_hours",
            "day_obs",
        ],
    )

    if with_sunset_sunrise:
        # Add sunrise/sunset/night open hours information to the dataframe.

        def _apply_night_hours(x: pd.Series) -> pd.Series:
            """Add the sunrise/sunset and overlap with dome open hours.
            A pandas apply method for the dome_open_close function.
            """
            sunset, sunrise = day_obs_sunset_sunrise(x["day_obs"], sun_alt=-12)
            sunset12 = sunset.utc.datetime
            sunrise12 = sunrise.utc.datetime
            night_hours = (sunrise12 - sunset12) / pd.Timedelta(1, "h")

            if pd.isna(x["open_time"]) and pd.isna(x["close_time"]):
                # Dome did not open at all.
                start = sunrise12
                end = sunrise12
            elif pd.isna(x["close_time"]):
                # Only close time is NaT (dome is still open).
                start = np.max([x["open_time"], sunset12])
                end = np.min([Time.now().utc.datetime, x["sunrise12"]])
            else:
                # Dome opened and closed. Account for nighttime.
                # Don't count open time before sunset.
                start = np.max([x["open_time"], sunset12])
                # Don't count open time beyond sunrise.
                end = np.min([x["close_time"], sunrise12])

                # If the dome opened and closed during the daytime, disregard.
                # Afternoon :
                if x["close_time"] < sunset12:
                    end = start
                # Morning:
                if x["open_time"] > sunrise12:
                    end = start

            open_hours = (end - start) / pd.Timedelta(1, "h")
            x = pd.Series(
                [sunset12, sunrise12, night_hours, open_hours],
                index=["sunset12", "sunrise12", "night_hours", "open_hours"],
            )
            return x

        night_df = dome_open_df.apply(_apply_night_hours, axis=1)
        dome_open_df = pd.merge(dome_open_df, night_df, how="outer", left_index=True, right_index=True)

    return dome_open_df


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
    slew_start: pd.DataFrame = efd_client.select_time_series(
        "lsst.sal.MTM1M3.command_setSlewFlag", ["private_identity"], t_start, t_end
    )
    slew_end: pd.DataFrame = efd_client.select_time_series(
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
    mt_slew.rename({"time_end": "mtm1m3_clear", "time_start": "mtm1m3_set"}, axis=1, inplace=True)
    mt_slew["mt_slew_time"] = (mt_slew["mtm1m3_clear"] - mt_slew["mtm1m3_set"]) / pd.Timedelta(1, "s")  # type: ignore[operator]

    missing = set(slew_start.scriptSalIndex.values).symmetric_difference(set(slew_end.scriptSalIndex.values))
    logging.debug(
        f"Found {len(slew_start)} slew starts and {len(slew_end)} slew ends, with "
        f"{len(slew_start.scriptSalIndex.unique())} and {len(slew_end.scriptSalIndex.unique())} "
        f"unique script salIndexes each."
    )
    logging.debug(f"Differences include {missing} script salIndex")

    # Get nextVisit events as well, to get groupId.
    topic = "lsst.sal.ScriptQueue.logevent_nextVisit"
    nextvisits: pd.DataFrame = efd_client.select_time_series(topic, "*", t_start, t_end, index=1)
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
    rot_start: pd.DataFrame = efd_client.select_top_n(topic, fields, num=1, time_cut=t_start)
    rot: pd.DataFrame = efd_client.select_time_series(topic, fields, t_start, t_end)
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
    elevation_start: pd.DataFrame = efd_client.select_top_n(topic, fields, num=1, time_cut=t_start)
    elevation: pd.DataFrame = efd_client.select_time_series(topic, fields, t_start, t_end)
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


def get_mounted_bandpasses(t_start: Time, t_end: Time, efd_client: InfluxQueryClient) -> pd.DataFrame:
    topic = "lsst.sal.MTCamera.logevent_availableFilters"
    # Find starting value
    bands_start = efd_client.select_top_n(topic, ["filterTypes"], num=1, time_cut=t_start)
    bands_during = efd_client.select_time_series(topic, ["filterTypes"], t_start, t_end)
    bands = pd.concat([bands_start, bands_during])
    bands.rename(columns={"filterTypes": "available_bands"}, inplace=True)
    bands.sort_index(inplace=True)

    # Reformat string of bandpass names to list, dropping "none"
    def parse_available_bands(x: pd.Series) -> list[str]:
        return [
            f'"{band.strip()}"'
            for band in x.available_bands.split(",")
            if band.strip() and band.strip().lower() != "none"
        ]

    bands["available_bands"] = bands.apply(parse_available_bands, axis=1)
    return bands


def get_obs_status_messages(t_start: Time, t_end: Time, efd_client: InfluxQueryClient) -> pd.DataFrame:
    """Query observatory status for Simonyi MTScheduler only.

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
    obs_status_messages
        DataFrame with observatory status messages.
        WEATHER messages with no other state are updated to IDLE | WEATHER
        and day_obs is added as a column to the dataframe.
    """
    topic = "lsst.sal.Scheduler.logevent_observatoryStatus"
    fields = ["status", "note", "statusLabels"]
    obs_status_messages: pd.DataFrame = efd_client.select_time_series(topic, fields, t_start, t_end, index=1)
    if len(obs_status_messages) == 0:
        obs_status_messages = pd.DataFrame([], columns=fields)
    obs_status_messages["day_obs"] = day_obs_from_efd_index_array(obs_status_messages.index.to_series())
    # WEATHER or DOWNTIME alone should match with IDLE
    idx = obs_status_messages.query("statusLabels == 'WEATHER'").index
    obs_status_messages.loc[idx, "statusLabels"] = "IDLE | WEATHER"
    idx = obs_status_messages.query("statusLabels == 'DOWNTIME'").index
    obs_status_messages.loc[idx, "statusLabels"] = "IDLE | DOWNTIME"
    return obs_status_messages


def _obs_status_state_changes(obs_status_messages: pd.DataFrame, status_type: str) -> pd.DataFrame:
    """Find start and end of state changes for `status_type`.

    Assumes that WEATHER and DOWNTIME are descriptive and exist
    together with other states.
    However OPERATIONAL, FAULT, IDLE and UNKNOWN are 'exclusive'.

    Parameters
    ----------
    obs_status_messages
        The dataframe containing the observatory status messages.
    status_type
        The state change (WEATHER, FAULT, DOWNTIME) to check for.
        'UNKNOWN' should be handled by special case _obs_status_time_unknown.

    Returns
    -------
    down_summary
        The dataframe containing the summary of downtime periods.

    Notes
    -----
    Unless the status_type is specified as UNKNOWN or OPERATIONAL,
    this method will 'skip' the unknown updates,
    which essentially treats these as unwelcome interruptions to what would
    otherwise be a constant state of FAULT or DOWNTIME or WEATHER.
    This means a WEATHER period which would be interrupted by UNKNOWN will
    include the UNKNOWN period (likewise for FAULT or DOWNTIME).
    """
    status_type = status_type.upper()
    # If in one of the descriptive states, drop UNKNOWN.
    if status_type in ("WEATHER", "DOWNTIME"):
        o = obs_status_messages.query("statusLabels != 'UNKNOWN'")
    else:
        # Do not drop UNKNOWN. We will override these later
        # when counting up contributed hours.
        o = obs_status_messages.copy()
    o.reset_index(inplace=True)

    # Are there no records with this type at all?
    if len(o.query("statusLabels.str.contains(@status_type)")) == 0:
        down_summary = pd.DataFrame([], columns=["day_obs", "sunset12", "sunrise12", "start", "end", "hours"])
        return down_summary

    # Select the previous records to those with 'status_type'
    idx = o.query("statusLabels.str.contains(@status_type)").index.values - 1
    idx = idx[np.where((idx >= 0) & (idx <= len(o)))]
    # Then select the previous records which were not 'status_type', to see
    # where the starting point of the 'status_type' really was.
    idx = o.iloc[idx].query("not statusLabels.str.contains(@status_type)").index.values + 1
    idx = idx[np.where((idx >= 0) & (idx <= len(o)))]
    down_starts = o.iloc[idx].copy()
    down_starts["start"] = True
    # Now select the records prior to those which were not status_type
    idx = o.query("not statusLabels.str.contains(@status_type)").index.values - 1
    idx = idx[np.where((idx >= 0) & (idx <= len(o)))]
    # And check which of those were actually downtime
    # (signalling the last record of status_type)
    idx = o.iloc[idx].query("statusLabels.str.contains(@status_type)").index.values + 1
    idx = idx[np.where((idx >= 0) & (idx <= len(o)))]
    down_ends = o.iloc[idx].copy()
    down_ends["start"] = False
    down_edges = pd.concat([down_starts, down_ends]).sort_values("time")
    # Summarize closures.
    closure = []
    tnow = Time.now().utc
    for day_obs in obs_status_messages.query("statusLabels.str.contains(@status_type)").day_obs.unique():
        ws = down_edges.query("day_obs == @day_obs and start")
        we = down_edges.query("day_obs == @day_obs and not start")
        sunset12, sunrise12 = day_obs_sunset_sunrise(day_obs, -12)
        sunset12 = sunset12.utc
        sunrise12 = sunrise12.utc
        if tnow > sunset12 and tnow < sunrise12:
            # Special case of querying within an ongoing night.
            # Let's just update sunrise for now, to cut 'end' times down.
            sunrise12 = tnow
        # If all messages in this night and adjacent nights are DOWN,
        # they don't show up in down_edges so mark all as down.
        if len(ws) == 0 and len(we) == 0:
            closure.append(
                [
                    day_obs,
                    sunset12.datetime,
                    sunrise12.datetime,
                    sunset12.datetime,
                    sunrise12.datetime,
                    (sunrise12 - sunset12).jd * 24,
                ]
            )
        # If the fault ended during this dayobs but did not start:
        elif len(ws) == 0:
            if len(we) > 1:
                raise ValueError(f"Too many fault ends in dayobs {day_obs}")
            end = Time(we.time.values[0], scale="utc")
            # Did it end after sunset? - down from sunset to end
            if end > sunset12:
                closure.append(
                    [
                        day_obs,
                        sunset12.datetime,
                        sunrise12.datetime,
                        sunset12.datetime,
                        end.datetime,
                        (end - sunset12).jd * 24,
                    ]
                )
        # If the fault started in this dayobs but did not end:
        elif len(we) == 0:
            if len(ws) > 1:
                raise ValueError(f"Too many fault starts in dayobs {day_obs}")
            start = Time(ws.time.values[0], scale="utc")
            # Set start to sunset12 at least.
            if start < sunset12:
                start = sunset12
            # Did it start before sunrise? - down from start to sunrise.
            if start < sunrise12:
                closure.append(
                    [
                        day_obs,
                        sunset12.datetime,
                        sunrise12.datetime,
                        start.datetime,
                        sunrise12.datetime,
                        (sunrise12 - start).jd * 24,
                    ]
                )
        else:
            # Both start and end of down within dayobs.
            starts = Time(ws.time.values, scale="utc")
            # Only deal with faults that start before sunrise.
            starts = starts[starts < sunrise12]
            ends = Time(we.time.values, scale="utc")
            for i in range(len(starts)):
                start_time = starts[i]
                # Find any fault end times after start
                end_time = np.where(ends > start_time)[0]
                if len(end_time) > 0:
                    # Pick the first one.
                    end_time = ends[end_time[0]]
                else:
                    # No match, but it should end at sunrise.
                    end_time = sunrise12
                if start_time < sunset12:
                    start_time = sunset12
                if end_time > sunset12:
                    if end_time > sunrise12:
                        end_time = sunrise12
                    if start_time != end_time:
                        closure.append(
                            [
                                day_obs,
                                sunset12.datetime,
                                sunrise12.datetime,
                                start_time.datetime,
                                end_time.datetime,
                                (end_time - start_time).jd * 24,
                            ]
                        )
    down_summary = pd.DataFrame(
        closure, columns=["day_obs", "sunset12", "sunrise12", "start", "end", "hours"]
    )
    return down_summary


def get_observatory_state_times(t_start: Time, t_end: Time, efd_client: InfluxQueryClient) -> pd.DataFrame:
    """Get observatory status information, linked into state changes,
    e.g. the start and end of a DOWNTIME or WEATHER period.

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
    obs_status_periods : `pd.DataFrame`
        A dataframe containing the start and end times of WEATHER, DOWNTIME,
        IDLE, UNKNOWN, FAULT and OPERATIONAL periods,
        limited by -12 to -12 twilight.
    """
    obs_status_messages = get_obs_status_messages(t_start, t_end, efd_client)

    weather = _obs_status_state_changes(obs_status_messages, "WEATHER")
    weather["type"] = "WEATHER"
    downtime = _obs_status_state_changes(obs_status_messages, "DOWNTIME")
    downtime["type"] = "DOWNTIME"
    fault = _obs_status_state_changes(obs_status_messages, "FAULT")
    fault["type"] = "FAULT"
    operational = _obs_status_state_changes(obs_status_messages, "OPERATIONAL")
    operational["type"] = "OPERATIONAL"
    idle = _obs_status_state_changes(obs_status_messages, "IDLE")
    idle["type"] = "IDLE"
    unknown = _obs_status_state_changes(obs_status_messages, "UNKNOWN")
    unknown["type"] = "UNKNOWN"
    obs_status_periods = pd.concat([weather, downtime, fault, idle, unknown, operational]).sort_values(
        "start", ignore_index=True
    )
    return obs_status_periods


def _count_contribution(x: pd.Series, obs_status_periods: pd.DataFrame) -> float:
    """Apply priorities to how to count periods of downtime."""
    # Weather is always just weather and downtime trumps all.
    if x.type == "WEATHER" or x.type == "DOWNTIME":
        return x.hours

    count_time = x.hours

    # Downtime will override an exclusive state.
    # Otherwise must just count up unknown, idle, fault, and operational.
    downtime = obs_status_periods.query("type == 'DOWNTIME' and day_obs == @x.day_obs")
    if len(downtime) > 0:
        # Did this period end before (any) downtime on this dayobs started?
        # Or start after (any) downtime on this dayobs started?
        if (x.end <= downtime.start.min()) or (x.start >= downtime.end.max()):
            count_time = x.hours
        else:
            # Now this period could have overlapped a downtime
            # Start with the full down period, subtract off DOWNTIME overlaps.
            count_time = x.hours
            for i, down in downtime.iterrows():
                # Does it overlap this DOWNTIME at all?
                if x.start <= down.end and x.end >= down.start:
                    # Subtract off the period of downtime that overlaps.
                    discount_start = max(x.start, down.start)
                    discount_end = min(x.end, down.end)
                    # From pandas timestamps, convert to float hours.
                    count_time -= (discount_end - discount_start) / pd.Timedelta(hours=1)

    return count_time


def count_observatory_states(
    obs_status_periods: pd.DataFrame, dome_open: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add a column which 'counts' the contribution of a given observatory
    state to the overall nightly reporting.

    This could vary depending on goals.
    Currently: this method only discounts 'fault' periods that occur
    during DOWNTIME.
    """
    # Apply priorities for 'contributed' hours to downtime periods
    contributed_hours = obs_status_periods.apply(
        _count_contribution, obs_status_periods=obs_status_periods, axis=1
    )
    # Round contributions < .2 sec down to 0.
    contributed_hours[contributed_hours < 0.2 / 60 / 60] = 0
    obs_status_periods["contributed_hours"] = contributed_hours

    # Create day_obs summaries.
    w = (
        obs_status_periods.query("type == 'WEATHER'")
        .groupby("day_obs")
        .agg({"contributed_hours": "sum"})
        .rename({"contributed_hours": "weather_down"}, axis=1)
    )
    d = (
        obs_status_periods.query("type == 'DOWNTIME'")
        .groupby("day_obs")
        .agg({"contributed_hours": "sum"})
        .rename({"contributed_hours": "downtime_down"}, axis=1)
    )
    f = (
        obs_status_periods.query("type == 'FAULT'")
        .groupby("day_obs")
        .agg({"contributed_hours": "sum"})
        .rename({"contributed_hours": "fault_down"}, axis=1)
    )
    o = (
        obs_status_periods.query("type == 'OPERATIONAL'")
        .groupby("day_obs")
        .agg({"contributed_hours": "sum"})
        .rename({"contributed_hours": "operational_hours"}, axis=1)
    )
    i = (
        obs_status_periods.query("type == 'IDLE'")
        .groupby("day_obs")
        .agg({"contributed_hours": "sum"})
        .rename({"contributed_hours": "idle_down"}, axis=1)
    )
    u = (
        obs_status_periods.query("type == 'UNKNOWN'")
        .groupby("day_obs")
        .agg({"contributed_hours": "sum"})
        .rename({"contributed_hours": "unknown_down"}, axis=1)
    )

    if dome_open is not None:
        dome = dome_open.groupby("day_obs").agg(
            {"sunset12": "first", "sunrise12": "first", "night_hours": "first", "open_hours": "sum"}
        )
        summary = pd.merge(dome, w, how="outer", left_index=True, right_index=True)
    else:
        summary = day_obs_sunset_sunrise_df(
            obs_status_periods.day_obs.min(), obs_status_periods.day_obs.max()
        )
        summary.set_index("day_obs", inplace=True)
        summary = pd.merge(summary, w, how="outer", left_index=True, right_index=True)
    summary = pd.merge(summary, d, how="outer", left_index=True, right_index=True)
    summary = pd.merge(summary, f, how="outer", left_index=True, right_index=True)
    summary = pd.merge(summary, o, how="outer", left_index=True, right_index=True)
    summary = pd.merge(summary, i, how="outer", left_index=True, right_index=True)
    summary = pd.merge(summary, u, how="outer", left_index=True, right_index=True)
    summary = summary.fillna(0)
    return obs_status_periods, summary
