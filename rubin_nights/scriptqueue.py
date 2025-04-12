import logging
from enum import Enum

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta
from lsst.ts.xml.enums.Script import ScriptState
from lsst.ts.xml.enums.ScriptQueue import SalIndex
from lsst.ts.xml.sal_enums import State as CSCState

from .connections import get_clients
from .influx_query import EfdQueryClient
from .logging_query import ExposureLogClient, NarrativeLogClient

# To generate a tiny gap in time
EPS_TIME = np.timedelta64(1, "ms")
TIMESTAMP_ZERO = Time(0, format="unix_tai").utc.datetime


# Run as 'apply' per row (axis=1)
def apply_enum(x: pd.Series, column: str, enumvals: Enum) -> str:
    return enumvals(x[column]).name


def make_time(x, column):
    return Time(x[column], format="isot", scale="tai").utc.datetime


logger = logging.getLogger(__name__)

__all__ = [
    "get_scheduler_configs",
    "get_script_stream",
    "get_script_state",
    "get_script_status",
    "get_error_codes",
    "get_tracebacks",
    "get_narrative_and_errors",
    "get_exposure_info",
    "get_consolidated_messages",
]


def get_scheduler_configs(
    t_start: Time,
    t_end: Time,
    efd_client: EfdQueryClient,
    obsenv_client: EfdQueryClient,
    queueIndex: int | None = None,
) -> pd.DataFrame:
    """Return information needed to recreate FBS configuration.

    This requires checking the obsenv (`lsst.obsenv.summary`)
    to find the version of ts_config_ocs in use,
    the EFD (`lsst.sal.Scheduler.logevent_dependenciesVersions`)
    to find the version of rubin_scheduler and dependencies,
    and the EFD (`lsst.sal.Scheduler.logevent_configureApplied`)
    to find the specific FBS configuration file in use.

    Searches both the time within t_start to t_end, as well as the last
    configuration applied before this time period.

    Defining queueIndex will search dependencies and configurations for
    that queue only.

    Parameters
    ----------
    t_start : `astropy.Time`
        The time of the start of the period.
    t_end : `astropy.Time`
        The time at the end of the period.
    efd_client : `EfdQueryClient`
        An EFD client pointed to the standard EFD database.
    obsenv_client : `EfdQueryClient`
        An EFD client pointed to the obsenv database.
    queueIndex : `int` or `None`
        The salIndex of a specific queue (1=Simonyi, 2=Auxtel, 3=OCS).
        If None, queries all queues, but the initial state may be missed.

    Returns
    -------
    sched_config : `pd.DataFrame`
        A dataframe carrying the configuration information.
        Some columns are compacted into single strings, so
        the entire dataframe can fit into a limited set of columns.
    """
    # First find the obsenv to find the version of ts_config_ocs
    topic = "lsst.obsenv.summary"
    fields = ["summit_extras", "summit_utils", "ts_standardscripts", "ts_externalscripts", "ts_config_ocs"]
    obsenv_start = obsenv_client.select_top_n(topic, fields, num=1, time_cut=t_start)
    obsenv = obsenv_client.select_time_series(topic, fields, t_start, t_end)
    obsenv = pd.concat([obsenv_start, obsenv])
    if len(obsenv) == 0:
        logging.warning("Could not find obsenv values.")
        # This shouldn't happen, but could before obsenv was implemented.
        # We need something to fill in for work below.
        bad_obsenv0 = [(t_start - TimeDelta(1, format="mjd") * 3).utc.datetime] + ["unknown" for f in fields]
        bad_obsenv1 = [t_start.utc.datetime] + ["unknown" for f in fields]
        obsenv = pd.DataFrame([bad_obsenv0, bad_obsenv1], columns=["time"] + fields)
        obsenv.set_index("time", inplace=True)
        obsenv.index = obsenv.index.tz_localize("UTC")

    # Label whether it was an obsenv *update* (i.e. changed ts_config_ocs, etc)
    # Or just an obsenv *check* without update
    # (obsenv entries are triggered by a command at the summit,
    # which could be either of these jobs)
    check = np.all((obsenv[fields][1:].values == obsenv[fields][:-1].values), axis=1)
    classname = np.where(check, "Obsenv Check", "Obsenv Update")
    obsenv["classname"] = np.concatenate([np.array(["Obsenv"]), classname])
    # Reconfigure some of the values to match the dataframe shape for logs
    obsenv["description"] = "ts_config_ocs: " + obsenv["ts_config_ocs"]
    # Build compact config string
    obsenv["config"] = (
        "ts_standardscripts: "
        + obsenv["ts_standardscripts"]
        + "; ts_externalscripts: "
        + obsenv["ts_externalscripts"]
        + "; summit_utils: "
        + obsenv["summit_utils"]
        + "; summit_extras: "
        + obsenv["summit_extras"]
    )
    # The obsenv is shared across all scriptqueues.
    # The salIndex has to apply to all.
    obsenv["salIndex"] = 0
    obsenv["script_salIndex"] = -1

    # Scheduler dependency information - updated independently of obsenv.
    topic = "lsst.sal.Scheduler.logevent_dependenciesVersions"
    fields = [
        "cloudModel",
        "downtimeModel",
        "seeingModel",
        "skybrightnessModel",
        "observatoryLocation",
        "observatoryModel",
        "scheduler",
        "salIndex",
        "version",
    ]
    deps_start = efd_client.select_top_n(topic, fields, num=1, time_cut=t_start, index=queueIndex)
    deps = efd_client.select_time_series(topic, fields, t_start, t_end, index=queueIndex)
    deps = pd.concat([deps_start, deps])
    if len(deps) == 0:
        logging.warning("Could not find scheduler dependencies.")
        bad_deps = [t_start.utc.datetime] + ["unknown" for f in fields]
        deps = pd.DataFrame(bad_deps, columns=["time"] + fields)
        deps.set_index("time", inplace=True)
        deps.index = deps.index.tz_localize("UTC")

    # Reconfigure output to fit into script_status fields
    deps["classname"] = "Scheduler dependencies"
    # FBS version information isn't propagated - use seeingModel
    def fbs_version(x):
        return f"{x.scheduler} {x.seeingModel}"
    deps['description'] = deps.apply(fbs_version, axis=1)
    models = [c for c in deps.columns if "observatory" in c or "Model" in c]

    def build_compact_config_string(x, models):
        dep_string = ""
        for m in models:
            dep_string += f"{m}: {x[m]}, "
        dep_string = dep_string[:-2]
        return dep_string

    deps["config"] = deps.apply(build_compact_config_string, args=[models], axis=1)
    deps["script_salIndex"] = -1

    # The configurationApplied should happen with every scheduler update
    topic = "lsst.sal.Scheduler.logevent_configurationApplied"
    fields = ["SchedulerId", "configurations", "salIndex", "schemaVersion", "url", "version"]
    conf_start = efd_client.select_top_n(topic, fields, num=1, time_cut=t_start, index=queueIndex)
    conf = efd_client.select_time_series(topic, fields, t_start, t_end, index=queueIndex)
    conf = pd.concat([conf_start, conf])
    if len(conf) == 0:
        logging.warning("Could not find scheduler configuration.")
        bad_conf = [t_start.utc.datetime] + ["unknown" for f in fields]
        conf = pd.DataFrame(bad_conf, columns=["time"] + fields)
        conf.set_index("time", inplace=True)
        conf.index = conf.index.tz_localize("UTC")

    conf["classname"] = "Scheduler configuration"
    # To get the scheduler relevant info in a single line,
    # pull in ts_config_ocs to the configuration information.
    ts_config_ocs_in_place = []
    for time in conf.index:
        prev_obsenv = obsenv.query("index < @time")
        if len(prev_obsenv) == 0:
            ts_config_ocs_in_place.append("Unknown")
        else:
            ts_config_ocs_in_place.append(prev_obsenv.iloc[-1]["ts_config_ocs"])
    conf["ts_config_ocs"] = ts_config_ocs_in_place

    def build_link_to_config(x):
        desc_string = x.configurations.split(",")[-1] + "<br> ts_config_ocs " + x.ts_config_ocs
        link = f"https://github.com/lsst-ts/ts_config_ocs/tree/{x.ts_config_ocs}/Scheduler/feature_scheduler"
        url = f'<a href="{link}" target="_blank" rel="noreferrer noopener">{desc_string}</a>'
        return url

    conf["description"] = conf.apply(build_link_to_config, axis=1)
    conf.rename({"configurations": "config"}, axis=1, inplace=True)
    conf["script_salIndex"] = -1

    # Combine results
    sched_config = pd.concat([deps, conf, obsenv])

    # Drop columns, add timestamps and state
    cols = ["classname", "description", "config", "salIndex", "script_salIndex"]
    drop_cols = [c for c in sched_config.columns if c not in cols]
    sched_config.drop(drop_cols, axis=1, inplace=True)
    sched_config.sort_index(inplace=True)
    sched_config["timestampProcessStart"] = (
        sched_config.index.copy().tz_localize(None).astype("datetime64[ns]")
    )
    sched_config["finalScriptState"] = "Configuration"
    print(f"Found {len(sched_config)} scheduler configuration records")
    return sched_config


def get_script_stream(t_start: Time, t_end: Time, efd_client: EfdQueryClient) -> pd.DataFrame:
    """Get script description and configuration from
    lsst.sal.Script.logevent_description and lsst.sal.Script.command_configure
    topics.

    Parameters
    ----------
    t_start : `astropy.Time`
        The time to start searching for script events.
    t_end : `astropy.Time`
        The time at which to end searching for script events.
    efd_client : `EfdQueryClient`
        EfdClient to query the efd.

    Returns
    -------
    script_stream : `pd.DataFrame`
        DataFrame containing script description and configuration.

    Note
    ----
    Note that these do not explicitly carry the scriptqueue salindex
    information. The "salIndex" in these topics is the script_salIndex.
    """
    # Script will find information about how scripts are configured.
    # The description topic gives a more succinct human name to the scripts
    topic = "lsst.sal.Script.logevent_description"
    fields = ["classname", "description", "salIndex"]
    scriptdescription = efd_client.select_time_series(topic, fields, t_start, t_end)
    scriptdescription.rename({"salIndex": "script_salIndex"}, axis=1, inplace=True)

    # This gets us more information about the script parameters,
    # how they were configured
    topic = "lsst.sal.Script.command_configure"
    fields = ["blockId", "config", " executionId", "salIndex"]
    # note blockId is only filled for JSON BLOCK activities
    scriptconfig = efd_client.select_time_series(topic, fields, t_start, t_end)
    scriptconfig.rename({"salIndex": "script_salIndex"}, axis=1, inplace=True)

    # Merge these together on script_salIndex which is unique over tinterval
    # Found that (command_configure - script description) index time is
    # mostly << 1 second for each script and < 1 second over a night
    if len(scriptconfig) == 0 or len(scriptdescription) == 0:
        logger.info(
            f"Length of scriptdescription ({len(scriptdescription)}) "
            f"and scriptconfig ({len(scriptconfig)}) in "
            f"time period {t_start.utc.iso} to {t_end.utc.iso}"
        )
        script_stream = pd.DataFrame([])
    else:
        script_stream = pd.merge(scriptdescription, scriptconfig, on="script_salIndex", suffixes=["_d", "_r"])
    return script_stream


def get_script_state(
    t_start: Time, t_end: Time, queueIndex: int | None, efd_client: EfdQueryClient
) -> pd.DataFrame:
    """Get script status from lsst.sal.ScriptQueue.logevent_script topic.

    Parameters
    ----------
    t_start : `astropy.Time`
        The time to start searching for script events.
    t_end : `astropy.Time`
        The time at which to end searching for script events.
    efd_client : `EfdQueryClient`
        EfdClient to query the efd.

    Returns
    -------
    script_state : `pd.DataFrame`
        DataFrame containing timing information and states.


    Note
    ----
    The scriptqueue is explicit here, in the salIndex.
    From here, these can be tied to the running of individual scripts,
    within a single restart of the scriptqueue only.
    """
    # The status of each of these scripts is stored
    # in scriptQueue.logevent_script
    # so find the status of each of these scripts
    # (this is status at individual stages).
    topic = "lsst.sal.ScriptQueue.logevent_script"
    fields = [
        "blockId",
        "path",
        "processState",
        "scriptState",
        "salIndex",
        "scriptSalIndex",
        "timestampProcessStart",
        "timestampConfigureStart",
        "timestampConfigureEnd",
        "timestampRunStart",
        "timestampProcessEnd",
    ]
    # Providing an integer salIndex will restrict this query to a single queue,
    # but None will query all queues.
    scripts = efd_client.select_time_series(topic, fields, t_start, t_end, index=queueIndex)
    scripts.rename({"scriptSalIndex": "script_salIndex"}, axis=1, inplace=True)
    if len(scripts) == 0:
        print(f"Found 0 script events in {t_start.utc.iso} to {t_end.utc.iso}.")
        script_status = pd.DataFrame([])

    else:
        # Group scripts on 'script_salIndex' to consolidate the information
        # about its status stages
        # Make a new column which we will fill with the max script state
        # (== final state, given enum)
        # (new column so we don't have to deal with multi-indexes from
        # multiple aggregation methods)
        scripts["finalScriptState"] = scripts["scriptState"]
        script_status = scripts.groupby("script_salIndex").agg(
            {
                "path": "first",
                "salIndex": "max",
                "finalScriptState": "max",
                "scriptState": "unique",
                "processState": "unique",
                "timestampProcessStart": "min",
                "timestampConfigureStart": "min",
                "timestampConfigureEnd": "max",
                "timestampRunStart": "max",
                "timestampProcessEnd": "max",
            }
        )
        # Convert timestamp columns from unix_tai timestamps for readability.
        # Yes, these timestamps really are unix_tai.
        for col in [c for c in script_status.columns if c.startswith("timestamp")]:
            script_status[col] = Time(script_status[col], format="unix_tai").utc.datetime
        # Apply ScriptState enum for readability of final state
        script_status["finalScriptState"] = script_status.apply(
            apply_enum, args=["finalScriptState", ScriptState], axis=1
        )
        # Will apply 'best time' index after merge with script_stream
    return script_status


def get_script_status(t_start: Time, t_end: Time, efd_client: EfdQueryClient) -> pd.DataFrame:
    """Given a start and end time, appropriately query each ScriptQueue to find
    script descriptions, configurations and status.

    Parameters
    ----------
    t_start : `astropy.Time`
        The time to start searching for script events.
    t_end : `astropy.Time`
        The time at which to end searching for script events.
    efd_client : `EfdQueryClient`
        EfdClient to query the efd.
    obsenv_client: `EfdQueryClient`
        EfdClient to query the obsenv (different database).

    Returns
    -------
    script_status : `pd.DataFrame`
        DataFrame containing script description, configuration,
        timing information and states.


    Note
    ----
    The index of the returned dataframe is chosen from the timestamps
    recorded for the script. In order to best place the script message
    inline with other events such as acquired images, the time used is the
    `timestampRunStart` if available, `timestampConfigureEnd` next, and
    then falls back to `timestampConfigureStart` or `timestampProcessStart`
    if those are also not available.
    """

    # The script_salIndex is ONLY unique during the time that a particular
    # queue remains not OFFLINE
    # However, each queue can go offline independently, so the time intervals
    # that are required for each queue
    # can be different, and requires inefficient querying of the
    # lsst.sal.Script topics (which don't include  the queue identification
    # explicitly). Furthermore, the downtime is infrequent, so probably we'd
    # most of the time prefer to do the efficient thing and query everything
    # all at once.

    # So first - see if that's possible.
    topic = "lsst.sal.ScriptQueue.logevent_summaryState"
    fields = ["salIndex", "summaryState"]
    # Were there breaks in this queue?
    dd = efd_client.select_time_series(topic, fields, t_start, t_end)
    if len(dd) == 0:
        offline_events = 0
    else:
        offline_state = CSCState.OFFLINE.value  # noqa: F841
        offline_events = len(dd.query("summaryState == @offline_state"))

    if offline_events == 0:
        print(f"No OFFLINE events during time interval {t_start} to {t_end} for any queue.")
        # So then go ahead and just do a single big query.
        script_stream = get_script_stream(t_start, t_end, efd_client)
        script_status = get_script_state(t_start, t_end, None, efd_client)
        if len(script_stream) == 0 or len(script_status) == 0:
            logger.info(
                f"Zero-length script queue description ({len(script_stream)}) "
                f"or script queue status ({len(script_status)}) in "
                f"time period {t_start.utc.iso} to {t_end.utc.iso}"
            )
        else:
            script_status = pd.merge(
                script_stream, script_status, left_on="script_salIndex", right_index=True, suffixes=["", "_s"]
            )

    else:
        # The ScriptQueues can be started and stopped independently,
        # so run needs to run per-scriptqueue, per-uptime
        script_status = []
        for queue in SalIndex:
            topic = "lsst.sal.ScriptQueue.logevent_summaryState"
            fields = ["salIndex", "summaryState"]
            # Were there breaks in this particular queue?
            dd = efd_client.select_time_series(topic, fields, t_start, t_end, index=queue)
            if len(dd) == 0:
                tstops = []
                tintervals = [[t_start, t_end]]
            else:
                dd["state"] = dd.apply(apply_enum, args=["summaryState", CSCState], axis=1)
                dd["state_time"] = Time(dd.index.values)

                tstops = dd.query('state == "OFFLINE"').state_time.values
                if len(tstops) == 0:
                    tintervals = [[t_start, t_end]]
                if len(tstops) > 0:
                    ts = tstops[0]
                    ts_next = ts + TimeDelta(0.1 * u.second)
                    ts_next = Time(ts_next)
                    tintervals = [[t_start, ts]]
                    for ts in tstops[1:]:
                        tintervals.append([ts_next, ts])
                        ts_next = ts + TimeDelta(0.1 * u.second)
                    tintervals.append([ts_next, t_end])
            if len(tstops) == 0:
                logging.info(
                    f"For {queue.name}, found 0 ScriptQueue OFFLINE events in the "
                    f"time period  {t_start} to {t_end}."
                )
            else:
                logging.info(
                    f"For {queue.name}, found {len(tstops)} ScriptQueue restarts in the "
                    f"time period {t_start} to {t_end}, so will query in {len(tstops) + 1} chunks"
                )
                logging.info(f"OFFLINE event at @ {[t.utc.iso for t in tstops]}")

            # Do the script queue queries for each time interval in this queue
            for tinterval in tintervals:
                script_stream_t = get_script_stream(tinterval[0], tinterval[1], efd_client)
                script_status_t = get_script_state(tinterval[0], tinterval[1], queue, efd_client)
                # Merge with script_stream so we get better descriptions
                # and configuration information
                if len(script_status_t) == 0 or len(script_stream_t) == 0:
                    dd = []
                else:
                    dd = pd.merge(
                        script_stream_t,
                        script_status_t,
                        left_on="script_salIndex",
                        right_index=True,
                        suffixes=["", "_s"],
                    )
                    script_status.append(dd)
                logging.info(
                    f"Found {len(dd)} script-status messages during"
                    f" {[e.iso for e in tinterval]} for {queue.name}"
                )
        # Convert to a single dataframe
        script_status = pd.concat(script_status)

    logging.info(f"Found {len(script_status)} script status messages")

    # script_status columns:
    # ['classname', 'description', 'script_salIndex', 'ScriptID', 'blockId',
    # 'config', 'executionId', 'logLevel', 'pauseCheckpoint',
    # 'stopCheckpoint', 'path', 'salIndex', 'finalScriptState', 'scriptState',
    # 'processState', 'timestampProcessStart', 'timestampConfigureStart',
    # 'timestampConfigureEnd', 'timestampRunStart', 'timestampProcessEnd']
    # columns used in final merged dataframe:
    # ['time', 'name', 'description', 'config', 'script_salIndex', 'salIndex',
    # 'finalStatus', 'timestampProcessStart', 'timestampConfigureEnd',
    # 'timestampRunStart', 'timestampProcessEnd']

    def _find_best_script_time(x):
        # Try run start first
        best_time = x.timestampRunStart
        if best_time == TIMESTAMP_ZERO:
            best_time = x.timestampConfigureEnd
        if best_time == TIMESTAMP_ZERO:
            best_time = x.timestampConfigureStart
        if best_time == TIMESTAMP_ZERO:
            best_time = x.timestampProcessStart
        return best_time

    if len(script_status) > 0:
        # Create an index that will slot this into the proper
        # place for runtime / image acquisition, etc
        script_status.index = script_status.apply(_find_best_script_time, axis=1)
        script_status.index = script_status.index.tz_localize("UTC")
        script_status.sort_index(inplace=True)
    return script_status


def get_error_codes(t_start: Time, t_end: Time, efd_client: EfdQueryClient) -> pd.DataFrame:
    """Get all messages from logevent_errorCode topics.

    Parameters
    ----------
    t_start : `astropy.Time`
        The time to start searching for script events.
    t_end : `astropy.Time`
        The time at which to end searching for script events.
    efd_client : `EfdQueryClient`
        EfdClient to query the efd.

    Returns
    -------
    error_messages : `pd.DataFrame`
    """
    # Get error codes
    topics = efd_client.get_topics()
    err_codes = [t for t in topics if "errorCode" in t]

    errs = []
    for topic in err_codes:
        df = efd_client.select_time_series(topic, ["errorCode", "errorReport"], t_start, t_end)
        if len(df) > 0:
            df["topic"] = topic
            errs += [df]
    if len(errs) > 0:
        errs = pd.concat(errs).sort_index()

        def strip_csc(x):
            return (
                x.topic.replace("lsst.sal", "").replace("logevent_errorCode", "").replace(".", "")
                + "CSC error"
            )

        errs["component"] = errs.apply(strip_csc, axis=1)
        # Rename some columns to match narrative log columns
        errs.rename(
            {"errorCode": "error_code", "errorReport": "message_text", "topic": "origin"},
            axis=1,
            inplace=True,
        )
        # Add a salindex so we can color-code based on this as a "source"
        errs["salIndex"] = 4
        errs["finalStatus"] = "ERR"
        errs["timestampProcessStart"] = errs.index.values.copy()
    else:
        # Make an empty dataframe.
        errs = pd.DataFrame(
            [],
            columns=[
                "component",
                "error_code",
                "message_text",
                "origin",
                "salIndex",
                "finalStatus",
                "timestampProcessStart",
            ],
        )

    print(f"Found {len(errs)} error messages")
    return errs


def get_tracebacks(t_start: Time, t_end: Time, efd_client: EfdQueryClient) -> pd.DataFrame:
    """Find tracebacks in lsst.sal.Script.logevent_logMessage.

    Parameters
    ----------
    t_start : `astropy.Time`
        The time to start searching for script events.
    t_end : `astropy.Time`
        The time at which to end searching for script events.
    efd_client : `EfdQueryClient`
        EfdClient to query the efd.

    Returns
    -------
    tracebacks : `pd.DataFrame`
        DataFrame containing tracebacks.
    """
    # Add tracebacks for failed scripts -- these should just slot in
    # right after FAILED scripts, and link with script_salIndex
    topic = "lsst.sal.Script.logevent_logMessage"
    fields = ["message", "traceback", "salIndex"]
    traceback_messages = efd_client.select_time_series(topic, fields, t_start, t_end)
    traceback_messages.rename({"salIndex": "script_salIndex"}, axis=1, inplace=True)
    # First check if there are any messages to query.
    if len(traceback_messages) > 0:
        # Only keep the lines where the traceback wasn't empty.
        traceback_messages.query('traceback != ""', inplace=True)
    # Then check if there are any *traceback* messages to query.
    if len(traceback_messages) > 0:
        # Only keep the lines where the traceback wasn't empty.
        traceback_messages.query('traceback != ""', inplace=True)

        # Add salIndex of queue where the script was run
        def queue_from_script_salindex(x):
            return int(str(x.script_salIndex)[0])

        traceback_messages["salIndex"] = traceback_messages.apply(queue_from_script_salindex, axis=1)

        def make_config_message(x):
            return f"Traceback for {x.script_salIndex}"

        traceback_messages["config"] = traceback_messages.apply(make_config_message, axis=1)
        traceback_messages["finalScriptState"] = "Traceback"
        traceback_messages["timestampProcessStart"] = (
            traceback_messages.index.copy().tz_localize(None).astype("datetime64[ns]")
        )
    # Going to rename some of these columns here to slot into scriptqueue
    traceback_messages.rename({"traceback": "description", "message": "classname"}, axis=1, inplace=True)
    return traceback_messages


def get_narrative_and_errors(
    t_start: Time, t_end: Time, efd_client: EfdQueryClient, narrative_log_client: NarrativeLogClient
) -> pd.DataFrame:
    """Get narrative log and error code messages.

    Parameters
    ----------
    t_start : `astropy.Time`
        The time to start searching for script events.
    t_end : `astropy.Time`
        The time at which to end searching for script events.
    efd_client : `EfdQueryClient`
        EfdClient to query the efd.
    narrative_log_client : `NarrativeLogClient`

    Returns
    -------
    narrative_and_errors : `pd.DataFrame`
    """
    messages = narrative_log_client.query_log(t_start, t_end)
    # Modify narrative log content to match dataframes from errors
    if len(messages) > 0:
        # rename some columns to match error data
        messages.rename({"time_lost_type": "error_code", "user_id": "origin"}, axis=1, inplace=True)
        # Add a salindex so we can color-code based on this as a "source"
        messages["salIndex"] = 0
        messages["error_code"] = 0
        messages["finalStatus"] = "Log"
        messages["timestampProcessStart"] = messages.apply(make_time, args=["date_begin"], axis=1)
        messages["timestampRunStart"] = messages.apply(make_time, args=["date_added"], axis=1)
        messages["timestampProcessEnd"] = messages.apply(make_time, args=["date_end"], axis=1)
    logging.info(f"Found {len(messages)} messages in the narrative log")
    # Get error codes
    errs = get_error_codes(t_start, t_end, efd_client)
    # Merge narrative log messages and error messages
    narrative_and_errors = pd.concat([errs, messages]).sort_index()
    return narrative_and_errors


def get_exposure_info(
    t_start: Time, t_end: Time, efd_client: EfdQueryClient, exposure_log_client: ExposureLogClient
) -> pd.DataFrame:
    """Get exposure information from
    lsst.sal.CCCamera.logevent_endOfImageTelemetry
    and join it with exposure log information.

    Parameters
    ----------
    t_start : `astropy.Time`
        The time to start searching for script events.
    t_end : `astropy.Time`
        The time at which to end searching for script events.
    efd_client : `EfdQueryClient`
        EfdClient to query the efd.
    exposure_log_client : `ExposureLogClient`
        ExposureLogClient to query for exposure logs.

    Returns
    -------
    narrative_and_errors : `pd.DataFrame`
    """
    # Find exposure information - Simonyi Tel
    topic = 'lsst.sal.MTCamera.logevent_endOfImageTelemetry'
    fields = ['imageName', 'imageIndex', 'exposureTime', 'darkTime', 'measuredShutterOpenTime',
              'additionalValues', 'timestampAcquisitionStart', 'timestampDateEnd', 'timestampDateObs']
    image_acquisition_mt = efd_client.select_time_series(topic, fields, t_start, t_end)
    # If there were zero images in this timeperiod, just return now.
    if len(image_acquisition_mt) > 0:
        for col in [c for c in image_acquisition_mt.columns if c.startswith("timestamp")]:
            image_acquisition_mt[col] = Time(image_acquisition_mt[col], format='unix_tai').utc.datetime
        image_acquisition_mt['salIndex'] = 5
        image_acquisition_mt['script_salIndex'] = 0
        image_acquisition_mt['finalStatus'] = "Image Acquired"
        def make_config_col_for_image(x):
            return f"exp {x.exposureTime} // dark {x.darkTime} // open {x.measuredShutterOpenTime} "
        image_acquisition_mt['config'] = image_acquisition_mt.apply(make_config_col_for_image, axis=1)
        image_acquisition_mt.index = image_acquisition_mt['timestampAcquisitionStart'].copy()
        image_acquisition_mt.index = image_acquisition_mt.index.tz_localize("UTC")
        print(f"Found {len(image_acquisition_mt)} image times for MTCamera Simonyi")

    topic = "lsst.sal.CCCamera.logevent_endOfImageTelemetry"
    fields = [
        "imageName",
        "imageIndex",
        "exposureTime",
        "darkTime",
        "measuredShutterOpenTime",
        "additionalValues",
        "timestampAcquisitionStart",
        "timestampDateEnd",
        "timestampDateObs",
    ]
    image_acquisition_cc = efd_client.select_time_series(topic, fields, t_start, t_end)
    # If there were zero images in this timeperiod, just return now.
    if len(image_acquisition_cc) > 0:
        for col in [c for c in image_acquisition_cc.columns if c.startswith("timestamp")]:
            image_acquisition_cc[col] = Time(image_acquisition_cc[col], format="unix_tai").utc.datetime
        image_acquisition_cc["salIndex"] = 5
        image_acquisition_cc["script_salIndex"] = 0
        image_acquisition_cc["finalStatus"] = "Image Acquired"

        def make_config_col_for_image(x):
            return f"exp {x.exposureTime} // dark {x.darkTime} // open {x.measuredShutterOpenTime} "

        image_acquisition_cc["config"] = image_acquisition_cc.apply(make_config_col_for_image, axis=1)
        image_acquisition_cc.index = image_acquisition_cc["timestampAcquisitionStart"].copy()
        image_acquisition_cc.index = image_acquisition_cc.index.tz_localize("UTC")
        logging.info(f"Found {len(image_acquisition_cc)} image times for CCCamera Simonyi")

    # Find exposure information - Aux Tel
    topic = "lsst.sal.ATCamera.logevent_endOfImageTelemetry"
    fields = [
        "imageName",
        "imageIndex",
        "exposureTime",
        "darkTime",
        "measuredShutterOpenTime",
        "additionalValues",
        "timestampAcquisitionStart",
        "timestampDateEnd",
        "timestampDateObs",
    ]
    image_acquisition_at = efd_client.select_time_series(topic, fields, t_start, t_end)
    # If there were zero images in this timeperiod, just return now.
    if len(image_acquisition_at) > 0:
        for col in [c for c in image_acquisition_at.columns if c.startswith("timestamp")]:
            # Is it possible ATCamera is not using tai?
            image_acquisition_at[col] = Time(image_acquisition_at[col], format="unix_tai").utc.datetime
        image_acquisition_at["salIndex"] = 6
        image_acquisition_at["script_salIndex"] = 0
        image_acquisition_at["finalStatus"] = "Image Acquired"

        def make_config_col_for_image(x):
            return f"exp {x.exposureTime} // dark {x.darkTime} // open {x.measuredShutterOpenTime} "

        image_acquisition_at["config"] = image_acquisition_at.apply(make_config_col_for_image, axis=1)
        image_acquisition_at.index = image_acquisition_at["timestampAcquisitionStart"].copy()
        image_acquisition_at.index = image_acquisition_at.index.tz_localize("UTC")
        logging.info(f"Found {len(image_acquisition_at)} image times for ATCamera AuxTel")

    image_acquisition = pd.concat([image_acquisition_cc, image_acquisition_at])

    # Add exposure log information
    exp_logs = exposure_log_client.query_log(t_start, t_end)
    logging.info(f"Found {len(exp_logs)} messages in the exposure log")
    # Modify exposure log and match with exposures to add time tag.
    if len(exp_logs) > 0:
        # Find a time to add the exposure logs into the records
        exp = pd.merge(image_acquisition, exp_logs, how="right", left_on="imageName", right_on="obs_id")
        # Set the time for the exposure log barely after the image start time
        exp_log_image_time = exp["timestampAcquisitionStart"] + EPS_TIME
        exp_logs["img_time"] = exp_log_image_time
        exp_logs.set_index("img_time", inplace=True)
        exp_logs.index = exp_logs.index.tz_localize("UTC")
        exp_logs["salIndex"] = 0
        exp_logs["script_salIndex"] = 0
        # Rename some columns in the exposure log to consolidate here
        exp_logs.rename(
            {
                "obs_id": "imageName",
                "user_id": "config",
                "message_text": "additionalValues",
                "exposure_flag": "finalStatus",
            },
            axis=1,
            inplace=True,
        )
        image_acquisition = pd.concat([image_acquisition, exp_logs]).sort_index()
        logging.info("Joined exposure and exposure log")
    return image_acquisition


def get_consolidated_messages(
    t_start: Time, t_end: Time, tokenfile: str | None = None, site: str | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Get consolidated messages from EFD ScriptQueue, errorCodes,
    CCCamera, exposure and narrative logs.

    Parameters
    ----------
    t_start : `astropy.Time`
        Time of the start of the messages.
    t_end : `astropy.Time`
        Time of the end of the messages.
    tokenfile : `str` or None
        RSP token file. Default None.
    site : `str` or None
        The service site to choose. Default will use usdf-rsp.

    Returns
    -------
    efd_and_messages : `pd.DataFrame`
        A Dataframe of relevant logging and EFD messages.
    cols: `list` [`str`]
        The short-list of columns for display in the table.
    """
    endpoints = get_clients(tokenfile=tokenfile, site=site)
    logging.info(endpoints)

    # Consolidating the information from the various sources requires
    # renaming columns into a more compact set.
    # goal columns :
    cols = [
        "time",
        "name",
        "description",
        "config",
        "script_salIndex",
        "salIndex",
        "finalStatus",
        "timestampProcessStart",
        "timestampConfigureEnd",
        "timestampRunStart",
        "timestampProcessEnd",
    ]

    # columns from scripts
    script_status = get_script_status(t_start, t_end, endpoints["efd"])
    # script_cols = ['classname', 'description', 'config', 'script_salIndex',
    # 'salIndex', 'blockId', 'finalScriptState', 'scriptState',
    # 'timestampProcessStart', 'timestampConfigureEnd',
    # 'timestampRunStart', 'timestampProcessEnd']
    tracebacks = get_tracebacks(t_start, t_end, endpoints["efd"])
    scheduler_configs = get_scheduler_configs(t_start, t_end, endpoints["efd"], endpoints["obsenv"])
    script_status = pd.concat([scheduler_configs, script_status, tracebacks])
    script_status.rename({"classname": "name", "finalScriptState": "finalStatus"}, axis=1, inplace=True)

    # columns from narrative and errors
    narrative_and_errs = get_narrative_and_errors(
        t_start, t_end, endpoints["efd"], endpoints["narrative_log"]
    )
    # narrative_cols = ['component', 'origin', 'message_text',
    # 'error_code', 'salIndex']
    narrative_and_errs.rename(
        {
            "component": "name",
            "origin": "config",
            "message_text": "description",
            "error_code": "script_salIndex",
        },
        axis=1,
        inplace=True,
    )
    # columns from images_and_logs
    image_and_logs = get_exposure_info(
        t_start,
        t_end,
        endpoints["efd"],
        endpoints["exposure_log"],
    )
    # image_cols = ['imageName', 'additionalValues', 'config', 'finalStatus',
    # 'script_salIndex', 'salIndex', 'timestampAcquisitionStart',
    # 'timestampDateObs', 'timestampDateEnd']
    image_and_logs.rename(
        {
            "imageName": "name",
            "additionalValues": "description",
            "timestampAcquisitionStart": "timestampProcessStart",
            "timestampDateObs": "timestampRunStart",
            "timestampDateEnd": "timestampProcessEnd",
        },
        axis=1,
        inplace=True,
    )

    df_list = [script_status, narrative_and_errs, image_and_logs]
    efd_and_messages = pd.concat([df for df in df_list if not df.empty]).sort_index()
    # Wrap description, for on-screen spacing
    efd_and_messages["description"] = efd_and_messages["description"].str.wrap(100)

    # Add some big labels which could be used to indicate foldups
    # The blocks can be complicated - a single BLOCK can actually
    # trigger multiple AddBlock commands (?)
    # So go back and check command_addBlock directly.
    topic = "lsst.sal.Scheduler.command_addBlock"
    block_names = endpoints["efd"].select_time_series(topic, ["id"], t_start, t_end, index=None)
    # Find the FBS setup and starts
    fbs_resume_times = efd_and_messages.query('name == "MTSchedulerResume"')
    scheduler_configs = efd_and_messages.query('name == "Scheduler configuration"')

    def find_fbs_yaml(row, scheduler_configs):
        earlier_configs = scheduler_configs.query("index < @row.name")
        best_config = earlier_configs.iloc[-1].config
        return best_config.split(",")[-1]

    sched_yamls = fbs_resume_times.apply(find_fbs_yaml, args=[scheduler_configs], axis=1)
    sched_yamls = pd.DataFrame(sched_yamls, columns=["id"])
    if len(block_names) > 0 and len(sched_yamls) > 0:
        foldups = pd.concat([block_names, sched_yamls])
    elif len(block_names) == 0:
        foldups = sched_yamls
    else:
        foldups = block_names

    if len(foldups) > 0:
        # If we have some addBlock or resumeScheduler events, add those.
        # Note that we could have images and events -- running from scripts.
        # .. but I don't know how to track these.
        foldups = foldups.sort_index()
        foldups.rename({"id": "name"}, axis=1, inplace=True)
        foldups["salIndex"] = 0
        foldups["script_salIndex"] = -1
        foldups["finalStatus"] = "Job Change"
        foldups["config"] = ""
        foldups["description"] = "New BLOCK or FBS configuration"
        foldups["timestampProcessStart"] = foldups.index.copy()
        foldups["timestampProcessEnd"] = np.concatenate(
            [foldups.index[1:].copy(), np.array([efd_and_messages.index[-1]])]
        )
        efd_and_messages = pd.concat([efd_and_messages, foldups]).sort_index()

    # use an integer index, which makes it easier to pull up values
    # plus avoids occasional failures of time uniqueness
    efd_and_messages.reset_index(drop=False, inplace=True)
    efd_and_messages.rename({"index": "time"}, axis=1, inplace=True)

    print(f"Total combined messages {len(efd_and_messages)}")

    # If there are any missing columns, such as a section of the
    # log was missing, add keys back in
    missing_cols = [c for c in cols if c not in efd_and_messages.keys()]
    for m in missing_cols:
        efd_and_messages[m] = pd.Series()

    return efd_and_messages, cols
