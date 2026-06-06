.. py:currentmodule:: rubin_nights

.. _howto_consdb:

Querying the ConsDB
===================

The Consolidated Database of Image Metadata (ConsDB) stores per-visit metadata including pointing
information, image quality metrics, and exposure parameters.
The `ConsDb documentation <https://consdb.lsst.io/user-guide/index.html>`_
contains more information about the tables and additional connection
utilities.
See `sdm-schemas.lsst.io <https://sdm-schemas.lsst.io>`_ for the full schema and
`DMTN-227 <https://dmtn-227.lsst.io>`_ for additional background information.

Two query interfaces are available inside and outside the USDF:

* **FastAPI** (``endpoints["consdb"]``) — an interface to the pqserver REST API
  interface of ConsDB. Provides an interface for pure SQL queries.
* **TAP** (``endpoints["consdb_tap"]``) — a TAP interface to the ConsDB.
  Provides a way to send ADQL queries as well as SQL.


One additional interface is available only within the USDF:
* **ConsDbSql** - a direct sqlalchemy connection to the ConsDB is available
within the USDF. This client just runs queries using sqlalchemy + pandas.

All interfaces return pandas DataFrames. The FastAPI and TAP interfaces
are similar in speed in general, but may vary depending on exact queries.

Fetching visits in a given timespan
------------------------------------

The convenience method :meth:`~consdb_query.ConsDbQuery.get_visits` queries the
``visit1`` and ``visit1_quicklook`` tables and returns a combined DataFrame.
It accepts an optional ``visit_constraint`` for additional SQL filtering.

.. code-block:: python

    from astropy.time import Time, TimeDelta

    # Specify the start and end of a given day_obs
    day_obs = "2025-06-20"
    t_start = Time(f"{day_obs}T12:00:00", format="isot", scale="utc")
    t_end = t_start + TimeDelta(1, format="jd")

    visits = endpoints["consdb_tap"].get_visits(
        "lsstcam", t_start, t_end,
        visit_constraint="science_program = 'BLOCK-407'",
        augment=False,
    )

Running custom SQL queries
---------------------------

Both clients expose a ``query`` method that accepts arbitrary SQL:

.. code-block:: python

    query = (
        "SELECT v.*, q.zero_point_median, q.sky_bg_median "
        "FROM cdb_lsstcam.visit1 AS v, cdb_lsstcam.visit1_quicklook AS q "
        "WHERE q.visit_id = v.visit_id "
        "  AND v.science_program = 'BLOCK-407' "
        "  AND v.day_obs = 20251127"
    )
    visits = endpoints["consdb_tap"].query(query)
    # equivalently:
    visits = endpoints["consdb"].query(query)

Common SQL pitfalls
^^^^^^^^^^^^^^^^^^^

* String literals must be enclosed in **single quotes**: ``science_program = 'BLOCK-407'``,
  not double quotes.
* Use ``=`` for comparisons, not ``==`` (Python syntax).
* The ``%`` wildcard in ``LIKE`` clauses must be doubled (``%%``) when using the
  FastAPI client; the TAP client accepts a single ``%``.

Async queries
^^^^^^^^^^^^^

Both clients support asynchronous operation for long-running queries:

.. code-block:: python

    # TAP — submit a job and wait
    job = endpoints["consdb_tap"].tap.submit_job(query)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=120)
    results = job.fetch_result().to_table().to_pandas()

    # FastAPI — async method (use inside an async context)
    results = await endpoints["consdb"].async_query(query)

Adding computed columns
------------------------

:func:`~augment_visits.augment_visits` appends useful derived columns to a visits
DataFrame, including predicted five-sigma limiting depth, cloud extinction estimate,
moon separation, and opsim-compatible columns.

.. code-block:: python

    from rubin_nights.augment_visits import augment_visits

    visits_plus = augment_visits(visits)

The ``augment_visits`` method uses ``rubin_scheduler`` and ``rubin_sim`` to compute
some of this additional data.


Removing bad visits
--------------------

DM maintains a list of known-bad visit IDs at
`lsst-dm/excluded_visits <https://github.com/lsst-dm/excluded_visits>`_.
:func:`~augment_visits.fetch_excluded_visits` downloads that list and
:func:`~augment_visits.exclude_visits` removes the flagged rows:

.. code-block:: python

    from rubin_nights.augment_visits import fetch_excluded_visits, exclude_visits

    bad_ids = fetch_excluded_visits(instrument="lsstcam")
    clean_visits = exclude_visits(visits=visits, bad_visit_ids=bad_ids)

Joining per-detector data
--------------------------

The ``ccdvisit1`` and ``ccdvisit1_quicklook`` tables provide per-detector values.
When you need a per-visit summary, it is simpler to query detectors separately and
aggregate with ``groupby`` before merging into the per-visit table — joining directly
in SQL multiplies every per-visit row by the number of detectors (189 for LSSTCam):

.. code-block:: python

    ccd_query = (
        "SELECT v.visit_id, c.detector, cq.psf_sigma "
        "FROM cdb_lsstcam.visit1 AS v "
        "JOIN cdb_lsstcam.ccdvisit1 AS c ON v.visit_id = c.visit_id "
        "LEFT JOIN cdb_lsstcam.ccdvisit1_quicklook AS cq "
        "  ON c.ccdvisit_id = cq.ccdvisit_id "
        "WHERE c.detector < 189 AND v.day_obs = 20251127"
    )
    ccdvisit = endpoints["consdb"].query(ccd_query)
    per_visit = ccdvisit.groupby("visit_id")["psf_sigma"].median().rename("psf_sigma_median")
    visits = visits.merge(per_visit, left_on="visit_id", right_index=True, how="left")

Joining Scheduler targets with visits
--------------------------------------

:func:`~targets_and_visits.targets_and_visits` links ConsDB visits with the
corresponding ``logevent_target``, ``logevent_observation``, and
``logevent_nextvisit`` EFD topics, adding the script SAL index and block ID to each row.
This requires both ``endpoints["consdb"]`` and ``endpoints["efd"]``.

.. code-block:: python

    from rubin_nights.targets_and_visits import targets_and_visits

    target_visits, cols, targets, next_visits, visits = targets_and_visits(
        t_start, t_end, endpoints, queue_index=1
    )
