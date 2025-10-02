import logging

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta

from .influx_query import InfluxQueryClient
from .logging_query import ExposureLogClient, NarrativeLogClient
from .ts_xml_enums import CSCState, SalIndex, SalIndexExtended, ScriptState, apply_enum

# To generate a tiny gap in time
EPS_TIME = np.timedelta64(1, "ms")
TIMESTAMP_ZERO = Time(0, format="unix_tai").utc.datetime

logger = logging.getLogger(__name__)

__all__ = [
    "get_scheduler_configs",
    "get_script_stream",
    "get_script_state",
    "get_script_status",
    "get_error_codes",
    "get_scriptqueue_tracebacks",
    "get_all_tracebacks",
    "get_narrative_and_errors",
    "get_exposure_info",
    "get_consolidated_messages",
]


def make_datetime(x: pd.Series, column: str) -> str:
    """Change a timestamp in TAI format to UTC datetime format.

    e.g. convert exposure time into scriptqueue 'time' format
    """
    return Time(x[column], format="isot", scale="tai").utc.datetime


def get_scheduler_configs(
    t_start: Time,
    t_end: Time,
    efd_client: InfluxQueryClient,
    obsenv_client: InfluxQueryClient,
    queue_index: int | None = None,
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

    Defining queue_index will search dependencies and configurations for
    that queue only.

    Parameters
    ----------
    t_start
        The time of the start of the period.
    t_end
        The time at the end of the period.
    efd_client
        An EFD client pointed to the standard EFD database.
    obsenv_client
        A sync EFD client pointed to the obsenv database.
    queue_index
        The salIndex of a specific queue (1=Simonyi, 2=Auxtel, 3=OCS).
        If None, queries all queues, but the initial state may be missed.

    Returns
    -------
    sched_config : `pd.DataFrame`
        A dataframe carrying the configuration information.
        Some columns are compacted into single strings, so
        the entire dataframe can fit into a limited set of columns.
    """
    # The configurationApplied should happen with every scheduler update
    # We have to find this time first, because we need to then use *this*
    # to find the previously enabled obsenv
    topic = "lsst.sal.Scheduler.logevent_configurationApplied"
    fields = ["SchedulerId", "configurations", "salIndex", "schemaVersion", "url", "version"]
    conf_start = efd_client.select_top_n(topic, fields, num=1, time_cut=t_start, index=queue_index)
    conf = efd_client.select_time_series(topic, fields, t_start, t_end, index=queue_index)
    conf = pd.concat([conf_start, conf])
    if len(conf) == 0:
        logger.warning("Could not find scheduler configuration.")
        bad_conf = [t_start.utc.datetime] + ["unknown" for f in fields]
        conf = pd.DataFrame(bad_conf, columns=["time"] + fields)
        conf.set_index("time", inplace=True)
        conf.index = conf.index.tz_localize("UTC")

    # First find the obsenv to find the version of ts_config_ocs
    topic = "lsst.obsenv.summary"
    fields = ["summit_extras", "summit_utils", "ts_standardscripts", "ts_externalscripts", "ts_config_ocs"]
    obsenv_start = obsenv_client.select_top_n(topic, fields, num=1, time_cut=Time(conf.index[0]))
    obsenv = obsenv_client.select_time_series(topic, fields, Time(conf.index[0]), t_end)
    obsenv = pd.concat([obsenv_start, obsenv])
    if len(obsenv) == 0:
        logger.warning("Could not find obsenv values.")
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
    obsenv["salIndex"] = SalIndexExtended.AUTOLOG_OTHER.value
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
    deps_start = efd_client.select_top_n(
        topic, fields, num=1, time_cut=Time(conf.index[0]), index=queue_index
    )
    deps = efd_client.select_time_series(topic, fields, t_start, t_end, index=queue_index)
    deps = pd.concat([deps_start, deps])
    if len(deps) == 0:
        logger.warning("Could not find scheduler dependencies.")
        bad_deps = [t_start.utc.datetime] + ["unknown" for f in fields]
        deps = pd.DataFrame(bad_deps, columns=["time"] + fields)
        deps.set_index("time", inplace=True)
        deps.index = deps.index.tz_localize("UTC")

    # Reconfigure output to fit into script_status fields
    deps["classname"] = "Scheduler dependencies"

    # FBS version information isn't propagated - use seeingModel
    def fbs_version(x: pd.Series) -> str:
        return f"{x.scheduler} {x.seeingModel}"

    deps["description"] = deps.apply(fbs_version, axis=1)
    models = [c for c in deps.columns if "observatory" in c or "Model" in c]

    def build_compact_config_string(x: pd.Series, models: list[str]) -> str:
        dep_string = ""
        for m in models:
            dep_string += f"{m}: {x[m]}, "
        dep_string = dep_string[:-2]
        return dep_string

    deps["config"] = deps.apply(build_compact_config_string, args=[models], axis=1)
    deps["script_salIndex"] = -1

    conf["classname"] = "Scheduler configuration"
    # To get the scheduler relevant info in a single line,
    # pull in ts_config_ocs to the configuration information.
    ts_config_ocs_in_place = []
    for time in conf.index:
        prev_obsenv = obsenv.query("index <= @time")
        if len(prev_obsenv) == 0:
            ts_config_ocs_in_place.append("Unknown")
        else:
            ts_config_ocs_in_place.append(prev_obsenv.iloc[-1]["ts_config_ocs"])
    conf["ts_config_ocs"] = ts_config_ocs_in_place

    def build_link_to_config(x: pd.Series) -> str:
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
    logger.info(f"Found {len(sched_config)} scheduler configuration records")
    return sched_config


def get_script_stream(t_start: Time, t_end: Time, efd_client: InfluxQueryClient) -> pd.DataFrame:
    """Get script description and configuration from
    lsst.sal.Script.logevent_description and lsst.sal.Script.command_configure
    topics.

    Parameters
    ----------
    t_start
        The time to start searching for script events.
    t_end
        The time at which to end searching for script events.
    efd_client
        Sunc EfdClient to query the efd.

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
    t_start: Time, t_end: Time, queue_index: int | None, efd_client: InfluxQueryClient
) -> pd.DataFrame:
    """Get script status from lsst.sal.ScriptQueue.logevent_script topic.

    Parameters
    ----------
    t_start
        The time to start searching for script events.
    t_end
        The time at which to end searching for script events.
    queue_index
        The SalIndex (1/2/3 or None for all queues) to check for script state.
    efd_client
        Sync EfdClient to query the efd.

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
    scripts = efd_client.select_time_series(topic, fields, t_start, t_end, index=queue_index)
    scripts.rename({"scriptSalIndex": "script_salIndex"}, axis=1, inplace=True)
    if len(scripts) == 0:
        logger.info(f"Found 0 script events in {t_start.utc.iso} to {t_end.utc.iso}.")
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


def get_script_status(t_start: Time, t_end: Time, efd_client: InfluxQueryClient) -> pd.DataFrame:
    """Given a start and end time, appropriately query each ScriptQueue to find
    script descriptions, configurations and status.

    This is an appropriate function to call if you just want to retrieve
    a description of the ongoing telescope commands, without additional
    logs or configuration information.

    Parameters
    ----------
    t_start
        The time to start searching for script events.
    t_end
        The time at which to end searching for script events.
    efd_client
        EfdClient to query the efd.

    Returns
    -------
    script_status : `pd.DataFrame`
        DataFrame containing script description, configuration,
        timing information and states.


    Note
    ----
    The (timestamp) index of the returned dataframe is chosen from the
    timestamps recorded for the script.
    In order to best place the script message
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
        restart_events = 0
    else:
        enabled_state = CSCState.ENABLED.value  # noqa: F841
        restart_events = len(dd.query("summaryState == @enabled_state"))

    if restart_events == 0:
        logger.info(f"No queue ENABLED events during time interval {t_start} to {t_end} for any queue.")
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

                tstops = dd.query('state == "ENABLED"').state_time.values
                if len(tstops) == 0:
                    tintervals = [[t_start, t_end]]
                if len(tstops) > 0:
                    ts = tstops[0] - TimeDelta(0.1 * u.second)
                    ts_next = ts
                    tintervals = [[t_start, ts]]
                    for ts in tstops[1:]:
                        tintervals.append([ts_next, ts - TimeDelta(0.1 * u.second)])
                        ts_next = ts
                    tintervals.append([ts_next, t_end])
            if len(tstops) == 0:
                logger.info(
                    f"For {queue.name}, found 0 ScriptQueue ENABLED events in the "
                    f"time period  {t_start} to {t_end}."
                )
            else:
                logger.info(
                    f"For {queue.name}, found {len(tstops)} ScriptQueue restarts in the "
                    f"time period {t_start} to {t_end}, so will query in {len(tstops) + 1} chunks"
                )
                logger.info(f"ENABLED event at @ {[t.utc.iso for t in tstops]}")

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
                logger.info(
                    f"Found {len(dd)} script-status messages during"
                    f" {[e.iso for e in tinterval]} for {queue.name}"
                )
        # Convert to a single dataframe
        script_status = pd.concat(script_status)

    logger.info(f"Found {len(script_status)} script status messages")

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

    def _find_best_script_time(x: pd.Series) -> str:
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


def get_error_codes(t_start: Time, t_end: Time, efd_client: InfluxQueryClient) -> pd.DataFrame:
    """Get all messages from logevent_errorCode topics.

    Parameters
    ----------
    t_start
        The time to start searching for script events.
    t_end
        The time at which to end searching for script events.
    efd_client
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

        def strip_csc(x: pd.Series) -> str:
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
        errs["salIndex"] = SalIndexExtended.ERRORS.value
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

    logger.info(f"Found {len(errs)} error messages")
    return errs


def get_scriptqueue_tracebacks(t_start: Time, t_end: Time, efd_client: InfluxQueryClient) -> pd.DataFrame:
    """Find tracebacks in lsst.sal.Script.logevent_logMessage.

    Parameters
    ----------
    t_start
        The time to start searching for script events.
    t_end
        The time at which to end searching for script events.
    efd_client
        EfdClient to query the efd.

    Returns
    -------
    tracebacks : `pd.DataFrame`
        DataFrame containing tracebacks.
    """
    # Add tracebacks for failed scripts -- these should just slot in
    # right after FAILED scripts, and link with script_salIndex
    query = 'select message, traceback, salIndex from "lsst.sal.Script.logevent_logMessage"'
    query += f"where time >= '{t_start.isot}Z' and time <= '{t_end.isot}Z' and traceback != ''"
    traceback_messages: pd.DataFrame = efd_client.query(query)
    # Then check if there are any *traceback* messages to query.
    if len(traceback_messages) > 0:
        traceback_messages.rename({"salIndex": "script_salIndex"}, axis=1, inplace=True)

        # Add salIndex of queue where the script was run
        def queue_from_script_salindex(x: pd.Series) -> int:
            return int(str(x.script_salIndex)[0])

        traceback_messages["salIndex"] = traceback_messages.apply(queue_from_script_salindex, axis=1)

        def make_config_message(x: pd.Series) -> str:
            return f"Traceback for {x.script_salIndex}"

        traceback_messages["config"] = traceback_messages.apply(make_config_message, axis=1)
        traceback_messages["finalScriptState"] = "Traceback"
        traceback_messages["timestampProcessStart"] = (
            traceback_messages.index.copy().tz_localize(None).astype("datetime64[ns]")
        )
    # Going to rename some of these columns here to slot into scriptqueue
    traceback_messages.rename({"traceback": "description", "message": "classname"}, axis=1, inplace=True)
    return traceback_messages


def get_all_tracebacks(t_start: Time, t_end: Time, efd_client: InfluxQueryClient) -> pd.DataFrame:
    """Find tracebacks all logevent_logMessages.

    This finds all tracebacks, for all CSCs and thus both telescopes.

    Parameters
    ----------
    t_start
        The time to start searching for script events.
    t_end
        The time at which to end searching for script events.
    efd_client
        EfdClient to query the efd.

    Returns
    -------
    tracebacks : `pd.DataFrame`
        DataFrame containing tracebacks.
    """
    topics = efd_client.get_topics()
    log_topics = [t for t in topics if "logMessage" in t]
    tracebacks = []
    for topic in log_topics:
        csc = topic.split(".")[-2]
        query = f'select * from "{topic}"'
        query += f"where time >= '{t_start.isot}Z' and time <= '{t_end.isot}Z' and traceback != ''"
        traceback_messages: pd.DataFrame = efd_client.query(query)
        # Then check if there are any *traceback* messages to query.
        if len(traceback_messages) > 0:

            if "salIndex" not in traceback_messages.columns:
                if csc.startswith("AT"):
                    traceback_messages["salIndex"] = 2
                else:
                    traceback_messages["salIndex"] = 1

            traceback_messages.rename({"salIndex": "script_salIndex"}, axis=1, inplace=True)

            # Add salIndex of queue where the script was run
            def queue_from_script_salindex(x: pd.Series) -> int:
                return int(str(x.script_salIndex)[0])

            traceback_messages["salIndex"] = traceback_messages.apply(queue_from_script_salindex, axis=1)

            def make_config_message(x: pd.Series, csc: str) -> str:
                if (x.script_salIndex) > 3:
                    message = f"{csc} traceback for {x.script_salIndex}"
                else:
                    message = f"{csc} traceback"
                return message

            traceback_messages["config"] = traceback_messages.apply(
                make_config_message,
                args=[
                    csc,
                ],
                axis=1,
            )
            tracebacks.append(traceback_messages)
    # Combine all the tracebacks and add some columns.
    traceback_messages = pd.concat(tracebacks).sort_index()
    traceback_messages["finalScriptState"] = "Traceback"
    traceback_messages["timestampProcessStart"] = (
        traceback_messages.index.copy().tz_localize(None).astype("datetime64[ns]")
    )
    # Going to rename some of these columns here to slot into scriptqueue
    traceback_messages.rename({"traceback": "description", "message": "classname"}, axis=1, inplace=True)

    cols_back = [
        "classname",
        "description",
        "script_salIndex",
        "salIndex",
        "config",
        "finalScriptState",
        "timestampProcessStart",
    ]
    return traceback_messages[cols_back]


def get_narrative_and_errors(
    t_start: Time, t_end: Time, efd_client: InfluxQueryClient, narrative_log_client: NarrativeLogClient
) -> pd.DataFrame:
    """Get narrative log and error code messages.

    Parameters
    ----------
    t_start
        The time to start searching for script events.
    t_end
        The time at which to end searching for script events.
    efd_client
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
        messages["salIndex"] = SalIndexExtended.NARRATIVE_LOG_OTHER.value
        idx = messages.query("component.str.contains('Simonyi') or component.str.contains('simonyi')").index
        messages.loc[idx, "salIndex"] = SalIndexExtended.NARRATIVE_LOG_SIMONYI.value
        idx = messages.query("component.str.contains('Aux') or component.str.contains('aux')").index
        messages.loc[idx, "salIndex"] = SalIndexExtended.NARRATIVE_LOG_AUX.value
        messages["error_code"] = 0
        messages["finalStatus"] = "Log"
        messages["timestampProcessStart"] = messages.apply(make_datetime, args=["date_begin"], axis=1)
        messages["timestampRunStart"] = messages.apply(make_datetime, args=["date_added"], axis=1)
        messages["timestampProcessEnd"] = messages.apply(make_datetime, args=["date_end"], axis=1)
    logger.info(f"Found {len(messages)} messages in the narrative log")
    # Get error codes
    errs = get_error_codes(t_start, t_end, efd_client)
    # Merge narrative log messages and error messages
    if len(errs) == 0:
        narrative_and_errors = messages
    elif len(messages) == 0:
        narrative_and_errors = errs
    else:
        narrative_and_errors = pd.concat([errs, messages]).sort_index()
    return narrative_and_errors


def get_exposure_info(
    t_start: Time, t_end: Time, efd_client: InfluxQueryClient, exposure_log_client: ExposureLogClient
) -> pd.DataFrame:
    """Get exposure information from
    lsst.sal.CCCamera.logevent_endOfImageTelemetry
    and join it with exposure log information.

    Parameters
    ----------
    t_start
        The time to start searching for script events.
    t_end
        The time at which to end searching for script events.
    efd_client
        EfdClient to query the efd.
    exposure_log_client
        ExposureLogClient to query for exposure logs.

    Returns
    -------
    narrative_and_errors : `pd.DataFrame`
    """
    # Find exposure information - Simonyi Tel
    topic = "lsst.sal.MTCamera.logevent_endOfImageTelemetry"
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
    image_acquisition_mt = efd_client.select_time_series(topic, fields, t_start, t_end)
    # If there were zero images in this timeperiod, just return now.
    if len(image_acquisition_mt) > 0:
        for col in [c for c in image_acquisition_mt.columns if c.startswith("timestamp")]:
            image_acquisition_mt[col] = Time(image_acquisition_mt[col], format="unix_tai").utc.datetime
        image_acquisition_mt["salIndex"] = SalIndexExtended.EXP_SIMONYI.value
        image_acquisition_mt["script_salIndex"] = 0
        image_acquisition_mt["finalStatus"] = "Image Acquired"

        def make_config_col_for_image(x: pd.Series) -> str:
            return f"exp {x.exposureTime} // dark {x.darkTime} // open {x.measuredShutterOpenTime} "

        image_acquisition_mt["config"] = image_acquisition_mt.apply(make_config_col_for_image, axis=1)
        image_acquisition_mt.index = image_acquisition_mt["timestampAcquisitionStart"].copy()
        image_acquisition_mt.index = image_acquisition_mt.index.tz_localize("UTC")
        logger.info(f"Found {len(image_acquisition_mt)} image times for MTCamera Simonyi")

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
        image_acquisition_cc["salIndex"] = SalIndexExtended.EXP_SIMONYI.value
        image_acquisition_cc["script_salIndex"] = 0
        image_acquisition_cc["finalStatus"] = "Image Acquired"

        def make_config_col_for_image(x: pd.Series) -> str:
            return f"exp {x.exposureTime} // dark {x.darkTime} // open {x.measuredShutterOpenTime} "

        image_acquisition_cc["config"] = image_acquisition_cc.apply(make_config_col_for_image, axis=1)
        image_acquisition_cc.index = image_acquisition_cc["timestampAcquisitionStart"].copy()
        image_acquisition_cc.index = image_acquisition_cc.index.tz_localize("UTC")
        logger.info(f"Found {len(image_acquisition_cc)} image times for CCCamera Simonyi")

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
        image_acquisition_at["salIndex"] = SalIndexExtended.EXP_AUX.value
        image_acquisition_at["script_salIndex"] = 0
        image_acquisition_at["finalStatus"] = "Image Acquired"

        def make_config_col_for_image(x: pd.Series) -> str:
            return f"exp {x.exposureTime} // dark {x.darkTime} // open {x.measuredShutterOpenTime} "

        image_acquisition_at["config"] = image_acquisition_at.apply(make_config_col_for_image, axis=1)
        image_acquisition_at.index = image_acquisition_at["timestampAcquisitionStart"].copy()
        image_acquisition_at.index = image_acquisition_at.index.tz_localize("UTC")
        logger.info(f"Found {len(image_acquisition_at)} image times for ATCamera AuxTel")

    image_acquisition = pd.concat([image_acquisition_mt, image_acquisition_cc, image_acquisition_at])

    # Add exposure log information
    exp_logs = exposure_log_client.query_log(t_start, t_end)
    logger.info(f"Found {len(exp_logs)} messages in the exposure log")
    # Modify exposure log and match with exposures to add time tag.
    if len(exp_logs) > 0:
        # Find a time to add the exposure logs into the records
        exp = pd.merge(image_acquisition, exp_logs, how="right", left_on="imageName", right_on="obs_id")
        # Set the time for the exposure log barely after the image start time
        exp_log_image_time = exp["timestampAcquisitionStart"] + EPS_TIME
        exp_logs["img_time"] = exp_log_image_time
        exp_logs.set_index("img_time", inplace=True)
        exp_logs.index = exp_logs.index.tz_localize("UTC")
        # Assign the exposure logs to the associated narrative log index
        exp_logs["salIndex"] = SalIndexExtended.NARRATIVE_LOG_OTHER.value
        idx = exp_logs.query("instrument == 'LSSTCam' or instrument == 'LSSTComCam'").index
        exp_logs.loc[idx, "salIndex"] = SalIndexExtended.NARRATIVE_LOG_SIMONYI.value
        idx = exp_logs.query("instrument == 'LATISS'").index
        exp_logs.loc[idx, "salIndex"] = SalIndexExtended.NARRATIVE_LOG_AUX.value
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
        logger.info("Joined exposure and exposure log")
    return image_acquisition


def get_consolidated_messages(t_start: Time, t_end: Time, endpoints: dict) -> tuple[pd.DataFrame, list[str]]:
    """Get consolidated messages from EFD ScriptQueue, errorCodes,
    CCCamera, exposure and narrative logs.

    Parameters
    ----------
    t_start
        Time of the start of the messages.
    t_end
        Time of the end of the messages.
    endpoints
        Endpoints is a dictionary of client connections to the EFD and the
        ConsDb, such as returned by `rubin_nights.connections.get_clients`.
        Must have clients for the `efd`, `obsenv`, `narrative_log` and
        `exposure_log`.

    Returns
    -------
    efd_and_messages : `pd.DataFrame`
        A Dataframe of relevant logging and EFD messages.
    cols: `list` [`str`]
        The short-list of columns for display in the table.
    """

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
    tracebacks = get_scriptqueue_tracebacks(t_start, t_end, endpoints["efd"])
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

    # Add some big labels which could be used to indicate times
    # where activity passes from one task to another.
    # The blocks can be complicated - a single BLOCK can actually
    # trigger multiple AddBlock commands (?)
    # So go back and check command_addBlock directly.
    topic = "lsst.sal.Scheduler.command_addBlock"
    block_names = endpoints["efd"].select_time_series(topic, ["id", "salIndex"], t_start, t_end, index=None)
    block_names.index = block_names.index - EPS_TIME * 30
    idx = block_names.query("salIndex == 1 or salIndex == 3").index
    block_names.loc[idx, "salIndex"] = SalIndexExtended.AUTOLOG_SIMONYI.value
    idx = block_names.query("salIndex == 2").index
    block_names.loc[idx, "salIndex"] = SalIndexExtended.AUTOLOG_AUX.value
    # Find the FBS setup and starts
    mt_fbs_resume_times = efd_and_messages.query("name == 'MTSchedulerResume'")
    at_fbs_resume_times = efd_and_messages.query("name == 'ATSchedulerResume'")
    scheduler_configs = efd_and_messages.query('name == "Scheduler configuration"')

    def find_fbs_yaml(row: pd.Series, scheduler_configs: pd.DataFrame) -> str:
        earlier_configs = scheduler_configs.query("index < @row.name")
        best_config = earlier_configs.iloc[-1].config
        return best_config.split(",")[-1]

    mt_sched_yamls = mt_fbs_resume_times.apply(find_fbs_yaml, args=[scheduler_configs], axis=1)
    mt_sched_yamls = pd.DataFrame(mt_sched_yamls, columns=["id"])
    mt_sched_yamls["salIndex"] = SalIndexExtended.AUTOLOG_SIMONYI.value
    at_sched_yamls = at_fbs_resume_times.apply(find_fbs_yaml, args=[scheduler_configs], axis=1)
    at_sched_yamls = pd.DataFrame(at_sched_yamls, columns=["id"])
    at_sched_yamls["salIndex"] = SalIndexExtended.AUTOLOG_AUX.value
    sched_yamls = pd.concat([mt_sched_yamls, at_sched_yamls])
    if len(block_names) > 0 and len(sched_yamls) > 0:
        task_changes = pd.concat([block_names, sched_yamls])
    elif len(block_names) == 0:
        task_changes = sched_yamls
    else:
        task_changes = block_names

    if len(task_changes) > 0:
        # If we have some addBlock or resumeScheduler events, add those.
        # Note that we could also have images and events running from scripts.
        # .. but I don't know how to track these.
        task_changes = task_changes.sort_index()
        task_changes.rename({"id": "name"}, axis=1, inplace=True)
        task_changes["script_salIndex"] = -1
        task_changes["finalStatus"] = "Task Change"
        task_changes["config"] = ""
        task_changes["description"] = "New BLOCK or FBS configuration"
        task_changes["timestampProcessStart"] = task_changes.index.copy()
        task_changes["timestampProcessEnd"] = np.concatenate(
            [task_changes.index[1:].copy(), np.array([efd_and_messages.index[-1]])]
        )
        # Slide these a fraction of a second earlier to slot before job change
        task_changes.index = task_changes.index - pd.Timedelta(1, "ns")
        efd_and_messages = pd.concat([efd_and_messages, task_changes]).sort_index()

    # use an integer index, which makes it easier to pull up values
    # plus avoids occasional failures of time uniqueness
    efd_and_messages.reset_index(drop=False, inplace=True)
    efd_and_messages.rename({"index": "time"}, axis=1, inplace=True)

    logger.info(f"Total combined messages {len(efd_and_messages)}")

    # If there are any missing columns, such as a section of the
    # log was missing, add keys back in
    missing_cols = [c for c in cols if c not in efd_and_messages.keys()]
    for m in missing_cols:
        efd_and_messages[m] = pd.Series()

    return efd_and_messages, cols
