import logging
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from astropy.time import Time

from .dayobs_utils import day_obs_list, day_obs_sunset_sunrise, day_obs_to_time
from .influx_query import InfluxQueryClient, day_obs_from_efd_index

__all__ = [
    "get_dome_open_close",
    "mtm1m3_slewflag_times",
    "get_rotator_limits",
    "get_tma_limits",
    "get_mounted_bandpasses",
    "obs_status_state_changes",
    "parse_observatory_status",
]

logger = logging.getLogger(__name__)

tz_utc = ZoneInfo("UTC")


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
    t_end
        Time of the end of the events.
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

    if len(dome_shutter_open) == 0:
        # Make and return an empty data frame with the expected columns.
        dome_shutter = pd.DataFrame([], columns=["day_obs", "open_time", "close_time", "open_hours"])
        return dome_shutter

    # Add day_obs
    dome_shutter_open["day_obs"] = dome_shutter_open.apply(day_obs_from_efd_index, axis=1)
    if len(dome_shutter_close) > 0:
        dome_shutter_close["day_obs"] = dome_shutter_close.apply(day_obs_from_efd_index, axis=1)

    # Find open/close times in each day_obs, including no-event day_obs
    dome_open = []
    for day_obs in day_obs_list(t_start, t_end):
        # dome open/close events
        opening = dome_shutter_open.query("day_obs == @day_obs")
        open_start = None
        if len(opening) > 0:
            # There are many 'opening' lines in dome_shutter_open;
            # pick out the ones which are the first in each 5 minute interval
            # This should separate dome opening events (which are <5 minutes).
            gaps = np.where((np.diff(opening.index) / pd.Timedelta(1, "s")) > 5 * 60)[0]
            # And add 1 because np.diff gives you the previous index.
            gaps += 1
            # And add an index for the very first dome_open index,
            # which doesn't have a previous 5 minute interval (so misses diff).
            gaps = np.concatenate([np.array([0]), gaps])
            open_start = Time(opening.iloc[gaps].index.values, scale="utc").utc.datetime

        closing = dome_shutter_close.query("day_obs == @day_obs")
        close_start = None
        if len(closing) > 0:
            # Pick out the dome closing events that are first in each
            # 3 minute interval (separate dome closing events).
            gaps = np.where((np.diff(closing.index) / pd.Timedelta(1, "s")) > 3 * 60)[0]
            gaps += 1
            gaps = np.concatenate([np.array([0]), gaps])
            close_start = Time(closing.iloc[gaps].index.values, scale="utc").utc.datetime

        # Sometimes telemetry is weird .. can't just zip these.
        # Look through open and close and match them up.
        if open_start is not None:
            for i in range(len(open_start)):
                open_time = open_start[i]
                if close_start is not None:
                    # Find the possible close times for this open_time.
                    close_time = np.where(close_start >= open_time)[0]
                    # If there are any - pick the first one.
                    if len(close_time) > 0:
                        close_time = close_start[close_time[0]]
                    else:
                        close_time = pd.NaT
                else:
                    # No close_start times at all
                    close_time = pd.NaT
                if not pd.isna(close_time):
                    open_hours = (close_time - open_time) / np.timedelta64(3600, "s")
                else:
                    open_hours = np.nan

                dome_open.append([day_obs, open_time, close_time, open_hours])
        else:
            # No open start times at all.
            # (but check that we're not looking at a day in the future).
            if Time.now() > day_obs_to_time(day_obs):
                dome_open.append([day_obs, pd.NaT, pd.NaT, 0])

    dome_open = pd.DataFrame(dome_open, columns=["day_obs", "open_time", "close_time", "dome_hours"])

    if with_sunset_sunrise:
        # Add sunrise/sunset/night open hours information to the dataframe.
        cols = ["sunset12", "sunrise12", "night_hours", "open_hours"]
        night_info = pd.DataFrame(
            [
                np.array([pd.Timestamp(0)] * len(dome_open)),
                np.array([pd.Timestamp(0)] * len(dome_open)),
                np.zeros(len(dome_open)),
                np.zeros(len(dome_open)),
            ],
            index=cols,
            columns=dome_open.index.copy(),
        ).T
        dome_open = dome_open.join(night_info)

        def apply_night_hours(x: pd.Series) -> pd.Series:
            sunset, sunrise = day_obs_sunset_sunrise(x.day_obs, sun_alt=-12)
            x.sunset12 = sunset.utc.datetime
            x.sunrise12 = sunrise.utc.datetime
            x.night_hours = (x.sunrise12 - x.sunset12) / pd.Timedelta(1, "h")
            # Don't count open time before sunset.
            if not pd.isna(x.open_time):
                start = np.max([x.open_time, x.sunset12])
            else:
                # Put in a value that will result in 0 open time
                start = x.sunrise12
            if not pd.isna(x.close_time):
                # Don't count open time beyond sunrise.
                end = np.min([x.close_time, x.sunrise12])
            else:
                # If we have not closed the dome yet .. choose sunrise?
                end = x.sunrise12
            # If the dome opened and closed during the daytime, disregard.
            # Open and close in the afternoon.
            if x.close_time < x.sunset12:
                end = start
            # Open and close in the morning.
            if x.open_time > x.sunrise12:
                end = start
            x.open_hours = (end - start) / pd.Timedelta(1, "h")
            return x

        # dome_open['night_hours'] = dome_open.apply(apply_night_hours, axis=1)
        dome_open = dome_open.apply(apply_night_hours, axis=1)

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
    mt_slew["mt_slew_time"] = (mt_slew["mtm1m3_clear"] - mt_slew["mtm1m3_set"]) / np.timedelta64(1, "s")

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
    def parse_available_bands(x: pd.Series) -> pd.Series:
        return [
            f'"{band.strip()}"'
            for band in x.available_bands.split(",")
            if band.strip() and band.strip().lower() != "none"
        ]

    bands["available_bands"] = bands.apply(parse_available_bands, axis=1)
    return bands


def obs_status_state_changes(
    obs_status_messages: pd.DataFrame, status_type: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find start and end of state changes for `status_type`,
    then consolidate into periods of downtime.

    Parameters
    ----------
    obs_status_messages
        The dataframe containing the observatory status messages.
    status_type
        The state change (WEATHER, FAULT, DOWNTIME) to check for.

    Returns
    -------
    down_summary
        The dataframe containing the summary of downtime periods.
    down_edges
        The dataframe containing the messages identified as state changes.
    """
    status_type = status_type.upper()
    if status_type != "UNKNOWN":
        o = obs_status_messages.query("statusLabels != 'UNKNOWN'")
    else:
        # You could run this for UNKNOWN. It's not obvious the implications,
        # since unknown can occur in the middle of other downtimes,
        # and by skipping those messages (above) we've folded the unknown
        # into accounting for the other state.
        # May be reasonable to consider UNKNOWN as FAULT if during the night.
        o = obs_status_messages.copy()
    o.reset_index(inplace=True)
    # Select the previous records to those with 'status_type'
    idx = o.query("statusLabels.str.contains(@status_type)").index.values - 1
    idx = idx[np.where((idx >= 0) & (idx <= len(o)))]
    # Then select the previous records which were not 'status_type'
    idx = o.iloc[idx].query("not statusLabels.str.contains(@status_type)").index.values + 1
    idx = idx[np.where((idx >= 0) & (idx <= len(o)))]
    down_starts = o.iloc[idx]
    down_starts["start"] = True
    # Now select the records prior to those which were not status_type
    idx = o.query("not statusLabels.str.contains(@status_type)").index.values - 1
    idx = idx[np.where((idx >= 0) & (idx <= len(o)))]
    # And check which of those were actually downtime
    # (signalling the last record of status_type)
    idx = o.iloc[idx].query("statusLabels.str.contains(@status_type)").index.values + 1
    idx = idx[np.where((idx >= 0) & (idx <= len(o)))]
    down_ends = o.iloc[idx]
    down_ends["start"] = False
    down_edges = pd.concat([down_starts, down_ends]).sort_values("time")
    # Summarize closures.
    closure = []
    for day_obs in obs_status_messages.query("statusLabels.str.contains(@status_type)").day_obs.unique():
        ws = down_edges.query("day_obs == @day_obs and start")
        we = down_edges.query("day_obs == @day_obs and not start")
        sunset12, sunrise12 = day_obs_sunset_sunrise(day_obs, -12)
        sunset12 = sunset12.utc
        sunrise12 = sunrise12.utc
        # # Include only starts and ends after sunset and before sunrise.
        # sunset12_tzaware = sunset12.to_datetime(timezone=tz_utc)
        # ws = ws.query("time >= @sunset12_tzaware")
        # we = we.query("time >= @sunset12_tzaware")
        # sunrise12_tzaware = sunrise12.to_datetime(timezone=tz_utc)
        # ws = ws.query("time >= @sunrise12_tzaware")
        # we = we.query("time >= @sunrise12_tzaware")
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
            # Only deal with faults that start before sunrise..
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
    down_summary = pd.DataFrame(closure, columns=["day_obs", "sunset12", "sunrise12", "start", "end", "down"])
    return down_summary, down_edges


def parse_observatory_status(t_start: Time, t_end: Time, efd_client: InfluxQueryClient) -> pd.DataFrame:
    """Get observatory status information."""
    # Fetch the messages.
    topic = "lsst.sal.Scheduler.logevent_observatoryStatus"
    fields = ["status", "note", "statusLabels"]
    obs_status_messages: pd.DataFrame = efd_client.select_time_series(topic, fields, t_start, t_end)
    if len(obs_status_messages) == 0:
        obs_status_messages = pd.DataFrame([], columns=fields)
    obs_status_messages["day_obs"] = obs_status_messages.apply(day_obs_from_efd_index, axis=1)

    weather, weather_edges = obs_status_state_changes(obs_status_messages, "WEATHER")
    weather["type"] = "WEATHER"
    fault, fault_edges = obs_status_state_changes(obs_status_messages, "FAULT")
    fault["type"] = "FAULT"
    downtime, downtime_edges = obs_status_state_changes(obs_status_messages, "DOWNTIME")
    downtime["type"] = "DOWNTIME"
    return pd.concat([weather, fault, downtime]).sort_values("start", ignore_index=True)
