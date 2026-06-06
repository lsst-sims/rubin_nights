.. py:currentmodule:: rubin_nights

.. _howto_efd:

Querying the EFD
================

The Engineering and Facilities Database (EFD) is a time-series store fed by the Rubin
Observatory Control System via SAL.  It is backed by InfluxDB and can be queried through
:class:`~influx_query.InfluxQueryClient`.

See also the `lsst-efd-client documentation <https://efd-client.lsst.io>`_ for a
full-featured async client; ``rubin_nights`` provides a simpler client, with both
synchronous and asynchronous capability. The ``rubin_nights`` API is intended
to mimic the ``lsst_efd_client.EfdClient`` API.

The full EFD schema is documented at `ts-xml.lsst.io <https://ts-xml.lsst.io>`_.
Each SAL component (CSC) exposes commands, events, and telemetry as separate topics.
The ``lsst.sal.<CSC>.<type>_<name>`` naming convention is used throughout.
Every index in the EFD is in UTC. Timestamps, when provided for different
topics, are usually UTC, with some notable exceptions in TAI.


Querying a time range
----------------------

:meth:`~influx_query.InfluxQueryClient.select_time_series` returns all records for a
topic between ``t_start`` and ``t_end`` as a DataFrame indexed by UTC timestamp.
Pass ``"*"`` to retrieve all fields, or a list to select specific ones.

.. code-block:: python

    from astropy.time import Time, TimeDelta

    # Specify the start and end of a given day_obs
    day_obs = "2025-06-20"
    t_start = Time(f"{day_obs}T12:00:00", format="isot", scale="utc")
    t_end = t_start + TimeDelta(1, format="jd")

    # Retrieve all Scheduler targets issued during the night
    targets = endpoints["efd"].select_time_series(
        "lsst.sal.Scheduler.logevent_target",
        "*",
        t_start, t_end,
        index=1,   # salIndex of the MTScheduler
    )

    # Retrieve specific fields only
    truss_temp = endpoints["efd"].select_time_series(
        "lsst.sal.ESS.temperature",
        ["temperatureItem6", "temperatureItem7"],
        t_start, t_end,
        index=122,
    )

The ``index`` parameter corresponds to the SAL index (``salIndex``) of the CSC instance.
For topics from CSCs that run as a single instance it can be omitted.

Retrieving the most recent records before a time
-------------------------------------------------

:meth:`~influx_query.InfluxQueryClient.select_top_n` fetches the ``n`` most recent
records that precede ``time_cut``.  This is useful for getting the last known state
before the start of a query window.

.. code-block:: python

    last_targets = endpoints["efd"].select_top_n(
        "lsst.sal.Scheduler.logevent_target",
        "*",
        num=3,
        time_cut=t_start,
    )

Running raw InfluxQL queries
-----------------------------

The ``query`` method accepts a raw InfluxQL query string for cases where the helpers
above are not sufficient:

.. code-block:: python

    query = (
        "SELECT mean(temperatureItem6) AS tma_truss_plus_xy "
        'FROM "efd"."autogen"."lsst.sal.ESS.temperature" '
        f"WHERE salIndex = 122 "
        f"  AND time >= '{t_start.utc.isot}Z' "
        f"  AND time <= '{t_end.utc.isot}Z' "
        "GROUP BY time(5s) FILL(null)"
    )
    result = endpoints["efd"].query(query)

Listing available topics
------------------------

.. code-block:: python

    topics = endpoints["efd"].get_topics()

Accessing LFA data at USDF
---------------------------

URIs written into the EFD at the summit use summit-local bucket paths when
recording the filename in the "largeFileAvailable" topics.
:func:`connections.usdf_lfa` converts such a URI to one that is accessible from the USDF:

.. code-block:: python

    from rubin_nights.connections import usdf_lfa

    summit_uri = "s3://rubinobs-lfa-cp/Scheduler/..."
    usdf_uri = usdf_lfa(summit_uri)


Joining EFD time series with per-visit data
--------------------------------------------

A common pattern is to interpolate a continuous EFD time series onto visit timestamps.
The example below uses a spline to assign truss temperature values to each visit:

.. code-block:: python

    import numpy as np
    from astropy.time import Time
    from scipy.interpolate import UnivariateSpline

    truss_times_mjd = Time(truss_temp.index.values, scale="utc").tai.mjd
    spline = UnivariateSpline(truss_times_mjd, truss_temp["tma_truss_plus_xy"].values, s=0)

    visits["tma_truss_plus_xy"] = spline(visits["exp_midpt_mjd"].values)

Alternatively, take the mean of EFD readings that fall within each exposure window:

.. code-block:: python

    truss_temp["mjd"] = truss_times_mjd

    def mean_during_visit(row, ts):
        window = ts[(ts["mjd"] >= row["obs_start_mjd"]) & (ts["mjd"] <= row["obs_end_mjd"])]
        return window["tma_truss_plus_xy"].mean()

    visits["tma_truss_plus_xy"] = visits.apply(mean_during_visit, args=(truss_temp,), axis=1)



