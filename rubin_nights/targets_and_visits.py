import logging

import numpy as np
import pandas as pd
from astropy.time import Time
from lsst.ts.xml.sal_enums import State as CSCState

logger = logging.getLogger(__name__)

__all__ = [
    "targets_and_visits",
    "flag_potential_bad_visits",
]


def targets_and_visits(
    t_start: Time, t_end: Time, endpoints: dict, queue_index: int = 1
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dataframe showing linked Targets, Observations, NextVisits and Visits.

    Parameters
    ----------
    t_start
        Time of the start of the events.
    t_end
        Time of the end of the events.
    endpoints
        Endpoints is a dictionary of client connections to the EFD and the
        ConsDb, such as returned by `rubin_nights.connections.get_clients`.
    queueIndex
        The SalIndex to query for Targets, corresponding to the Scheduler
        queue. Default of 1 corresponds to the Simonyi queue.
        Using queueIndex = 2 will trigger a request for latiss visits.

    Returns
    -------
    targets_and_visits : `pd.DataFrame`
        A Dataframe of Target, Observation, NextVisit and Visits.
    cols: `list` [`str`]
        The short-list of columns for display in the table.
    target_and_observations : `pd.DataFrame`
        A Dataframe of Targets joined to Observations.
    nextvisit_and_visits : `pd.DataFrame`
        A Dataframe of nextVisits joined to Visits.
    visits : `pd.DataFrame`
        A dataframe of all of the visits during the time period.
    """
    # Fetch the targets
    topic = "lsst.sal.Scheduler.logevent_target"
    targets = endpoints["efd"].select_time_series(topic, "*", t_start, t_end, index=queue_index)
    if len(targets) > 0:
        targets = targets.query("snapshotUri != ''")
    logger.debug(f"{len(targets)} targets events")

    # Fetch the observations
    topic = "lsst.sal.Scheduler.logevent_observation"
    fields = [
        "additionalInformation",
        "blockId",
        "decl",
        "exptime",
        "filter",
        "mjd",
        "nexp",
        "ra",
        "rotSkyPos",
        "salIndex",
        "targetId",
    ]
    observations = endpoints["efd"].select_time_series(topic, fields, t_start, t_end, index=queue_index)
    if len(observations) == 0:
        observations = pd.DataFrame([], columns=fields + ["time"])
    logger.debug(f"{len(observations)} observation events")

    # Fetch and consolidate the nextVisits
    topic = "lsst.sal.ScriptQueue.logevent_nextVisit"
    fields = ["scriptSalIndex", "groupId", "position0", "position1", "cameraAngle"]
    nextvisits = endpoints["efd"].select_time_series(topic, fields, t_start, t_end, index=queue_index)
    logger.debug(f"{len(nextvisits)} next visit events")
    if len(nextvisits) > 0:
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
    logger.debug(f"{len(nextvisits)} next visit events for unique targets")

    # Fetch the visits from the ConsDB
    if queue_index == 2:
        instrument = "latiss"
    else:
        instrument = "lsstcam"
    visits = endpoints["consdb"].get_visits(instrument, t_start, t_end, augment=True)
    logger.debug(f"{len(visits)} visits")

    if len(targets) == 0:
        logger.info("Found 0 targets; returning visits")
        return pd.DataFrame([]), [], pd.DataFrame([]), pd.DataFrame([]), visits

    # In theory, targets and observations could be merged directly on targetId.
    # However, targetId is not unique across Scheduler re-enable times.
    # This can be due to resetting unused targetIds OR it could be due
    # to using different FBS databases for recording observations.
    # Merge only within periods where the scheduler was continuously enabled.

    # Scheduler restarts:
    enabled_state = CSCState.ENABLED.value  # noqa: F841
    topic = "lsst.sal.Scheduler.logevent_summaryState"
    fields = ["summaryState"]
    dd = endpoints["efd"].select_time_series(topic, fields, t_start, t_end, index=queue_index)
    # Identify re-enable times
    restarts = dd.query("summaryState == @enabled_state")

    target_idxs = np.searchsorted(targets.index.values, restarts.index.values)
    target_idx_start = np.concatenate([np.array([0]), target_idxs])
    target_idx_end = np.concatenate([target_idxs, np.array([len(targets)])])
    obs_idxs = np.searchsorted(observations.index.values, restarts.index.values)
    obs_idx_start = np.concatenate([np.array([0]), obs_idxs])
    obs_idx_end = np.concatenate([obs_idxs, np.array([len(targets)])])
    to = []
    for i in range(len(target_idx_start)):
        t_targets = targets.iloc[target_idx_start[i] : target_idx_end[i]]
        t_observations = observations.iloc[obs_idx_start[i] : obs_idx_end[i]]
        if len(t_targets) == 0:
            # Nothing to merge/add from this period.
            continue
        elif len(t_observations) == 0:
            new_df = pd.DataFrame(
                np.zeros((len(t_targets.index.values), len(observations.columns.values))),
                columns=observations.columns.values,
                index=t_targets.index,
            )
            new_df.rename({"time": "time_o"}, axis=1, inplace=True)
            new_df.time_o = np.nan
            t_to = pd.merge(targets, new_df, left_index=True, right_index=True, suffixes=("", "_o"))
            t_to.reset_index("time", inplace=True)
        else:
            t_to = pd.merge_asof(
                t_targets.sort_values("targetId").reset_index("time"),
                t_observations.sort_values("targetId").reset_index("time"),
                on="targetId",
                left_by=["ra", "decl", "skyAngle"],
                right_by=["ra", "decl", "rotSkyPos"],
                suffixes=("", "_o"),
                allow_exact_matches=True,
                direction="forward",
            )
            t_to.sort_values(by="time", inplace=True)
        to.append(t_to)
    if len(to) == 0:
        # No information; make a minimal dataframe to be able to continue.
        to = pd.DataFrame([], columns=["targetId", "blockId", "skyAngle"])
    else:
        to = pd.concat(to)
    to = to.astype({"targetId": int, "blockId": int, "skyAngle": float})
    to.drop([c for c in to.columns if "private" in c], axis=1, inplace=True)
    logger.debug(f"Joined targets and observations for {len(to)} events")

    # If either visit or nextvisit are empty, just quit here.
    if len(visits) == 0:
        logger.warning("Could not retrieve any visits")
        return pd.DataFrame([]), [], to, nextvisits, visits
    elif len(nextvisits) == 0:
        logger.warning("Could not find any nextVisits, can't link to visits")
        return pd.DataFrame([]), [], to, nextvisits, visits

    # nextVisit to visits groupId should be unique
    nv = pd.merge(
        visits,
        nextvisits.reset_index("time"),
        how="outer",
        left_on="group_id",
        right_on="groupId",
        suffixes=["", "_nv"],
    )
    visit_id = np.where(np.isnan(nv["visit_id"].values), 0, nv["visit_id"].values)
    nv["visit_id"] = visit_id
    scriptSalIndex = np.where(np.isnan(nv["scriptSalIndex"].values), 0, nv["scriptSalIndex"].values)
    nv["scriptSalIndex"] = scriptSalIndex
    nv = nv.astype({"visit_id": int, "scriptSalIndex": int, "cameraAngle": float})
    nv.drop([c for c in nv.columns if "private" in c], axis=1, inplace=True)
    logger.debug(f"Joined nextvisit and visits for {len(nv)} records")

    # Join targets and next visit BUT blockId == salScriptId
    # is only unique within times of scriptqueue restarts
    # We can narrow down the links using the angle of the rotator
    # (better would be to fetch times of restarts, but this is cheap)
    # (works for science visits, but other programs may not)

    # Make sure column names in visits take priority
    vt = pd.merge_asof(
        to.sort_values("blockId"),
        nv.sort_values("scriptSalIndex"),
        left_on="blockId",
        right_on="scriptSalIndex",
        suffixes=["_tob", ""],
        left_by=["skyAngle"],
        right_by=["cameraAngle"],
        allow_exact_matches=True,
        direction="forward",
    )
    int_cols = ["visit_id", "day_obs", "seq_num", "scriptSalIndex"]
    for col in int_cols:
        tt = np.where(np.isnan(vt[col].values), 0, vt[col].values)
        vt[col] = tt
    vt = vt.astype(dict([(col, int) for col in int_cols]))
    vt.sort_values("time", inplace=True)
    logger.debug(f"Joined targets+observations with nextvisit+visit for {len(vt)} records")

    # Rename values for clarity
    vt.rename(
        {"time_tob": "time_target", "time_o": "time_observation", "time": "time_nextvisit"},
        axis=1,
        inplace=True,
    )
    # Some handy columns
    cols = [
        "visit_id",
        "day_obs",
        "seq_num",
        "time_target",
        "time_observation",
        "time_nextvisit",
        "obs_start",
        "group_id",
        "s_ra",
        "s_dec",
        "sky_rotation",
        "skyAngle",
        "band",
        "scriptSalIndex",
        "note",
        "observation_reason",
        "target_name",
    ]

    return vt, cols, to, nv, visits


def flag_potential_bad_visits(
    target_visits: pd.DataFrame, extinction: float = 1.5, no_quicklook: bool = True
) -> list[str]:
    """Flag potential bad visits within the target_visits dataframe.

    Parameters
    ----------
    target_visits
        Dataframe containing information on the linked
        target-observation-visit content, such as
        from `targets_and_visits`.
    extinction
        The magnitudes of extinction to allow before considering a visit
        "bad". This can indicate cloud extinction; however mini-donuts
        or other problems with an observation such as a minor tracking glitch
        can also show up as an offset between the measured and predicted
        zeropoint, just as if it were cloud extinction.
    no_quicklook
        Flag a visit as bad if there was no quicklook information.
        Missing quicklook can indicate the visit failed to process, which
        can be an indicator of a bad visit with giant donuts.
        However, missing quicklook can also just indicate that Rapid Analysis
        could not reach the ConsDB, or that it did not have calibration data,
        or that the pointing simply has too many or too few stars.

    Returns
    -------
    flagged_visit_ids : `list` [ `str` ]
        The list of visit_ids corresponding to the flagged visits.

    Notes
    -----
    Visits are always marked "bad" if the target event did not match with
    an observation event. This could happen for rare other reasons, but
    almost always indicates that the script failed due to a fault in the
    observatory, such as a loss of tracking or rotator.
    """
    quicklook_missing = np.where(np.isnan(target_visits.zero_point_median) & (target_visits.visit_id > 0))[0]
    big_zp_offset = np.where(
        (target_visits.zero_point_1s_pred.values - target_visits.zero_point_1s.values) > extinction
    )[0]
    failed_obs = np.where(np.isnan(target_visits.time_observation.values) & (target_visits.visit_id > 0))[0]
    issues = np.concatenate([big_zp_offset, failed_obs])
    if no_quicklook:
        issues = np.concatenate([quicklook_missing, issues])
    issues = np.sort(issues)
    issues = np.unique(issues)
    logger.debug(
        f"Found {len(quicklook_missing)} visits missing quicklook,"
        f" {len(big_zp_offset)} visits with big zeropoint offsets/extinction,"
        f" and {len(failed_obs)} visits with missing observation events,"
        f" out of a total of {len(target_visits)} visits."
    )
    return list(target_visits.iloc[issues]["visit_id"].values)
