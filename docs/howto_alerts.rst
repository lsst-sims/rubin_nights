.. py:currentmodule:: rubin_nights

.. _howto_alerts:

Alert Stream and Pipelines Metrics
====================================

The Prompt Processing pipeline publishes per-detector metrics to InfluxDB under the
``lsst.prompt.prod`` measurement namespace.  These include DiaSource counts, solar
system object associations, and image quality metrics for each processed visit.

The ``endpoints["pp"]`` client returned by :func:`~connections.get_clients` connects
to the relevant ``lsst.prompt`` InfluxDB database.


Querying prompt metrics directly
---------------------------------

To fetch the raw per-detector records from a specific topic:

.. code-block:: python

    dia_det = endpoints["pp"].select_time_series(
        "lsst.prompt.prod.numDiaSourcesGood",
        ["visit", "detector", "numAllDiaSources", "numGoodDiaSources", "run"],
        t_start, t_end,
    )




Using the convenience function
-------------------------------

:func:`~pipelines_metrics.diasource_visit_summaries` queries the three key prompt
processing topics — ``numDiaSourcesGood``, ``numSsObjects``, and
``numDirectSsObjects`` — and returns a per-visit summary DataFrame:

.. code-block:: python

    from astropy.time import Time
    from rubin_nights.pipelines_metrics import diasource_visit_summaries

    t_start = Time("2025-10-24T12:00:00")
    t_end = Time("2025-10-25T12:00:00")

    alert_summary = diasource_visit_summaries(t_start, t_end, endpoints["pp"])

The returned DataFrame is indexed by ``visit_id`` and contains (among other values):

* ``numAllDiaSources_sum`` / ``numAllDiaSources_median`` — sum and median across detectors.
* ``nDiaDetectors_count`` — number of detectors that produced DiaSource results.
* ``numSsObjects_sum`` / ``nSsDetectors_count`` — solar system alert associations.
* ``numDirectSsObjects_sum`` / ``nDSsDetectors_count`` — direct solar system associations.


Joining with visit data
------------------------

Merge the alert summary with ConsDB visit data on ``visit_id`` to combine observing
conditions with prompt processing outputs:

.. code-block:: python

    from rubin_nights.reference_values import SCIENCE_PROGRAMS

    constraint = f"science_program in {SCIENCE_PROGRAMS}"
    visits = endpoints["consdb_tap"].get_visits("lsstcam", t_start, t_end,
                                                visit_constraint=constraint)

    combined = visits.merge(alert_summary, left_on="visit_id", right_index=True, how="outer")

The combined table can be used to correlate image quality metrics (e.g., ``fwhm_geom``,
``cat_m5``, ``clouds``) or visit metadata (``galactic_lat``, ``ecliptic_lat``)
with alert production rates across a night or across a longer
observing campaign.


