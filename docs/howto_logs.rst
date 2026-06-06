.. py:currentmodule:: rubin_nights

.. _howto_logs:

Querying Summit Logs
====================

Three summit logging services can be queried through ``rubin_nights``:

* **Night report** — a structured summary of the observing night, one per telescope.
* **Narrative log** — a running record of observer comments and observatory state changes.
* **Exposure log** — annotations attached to specific exposures.

Clients for all three are included in the dictionary returned by :func:`~connections.get_clients`.

Setting a time window
---------------------

.. code-block:: python

    from astropy.time import Time, TimeDelta

    day_obs = "2025-04-15"
    t_start = Time(f"{day_obs}T12:00:00", format="isot", scale="utc")
    t_end = t_start + TimeDelta(1, format="jd")

Night report
------------

:meth:`~logging_query.NightReportClient.query_night_report` returns the night report
for a given ``day_obs`` and telescope as a dict.  Passing ``return_html=True`` also
returns a rendered HTML string suitable for display in a notebook.

.. code-block:: python

    from IPython.display import display, HTML

    report, html = endpoints["night_report"].query_night_report(
        day_obs=day_obs,
        telescope="Simonyi",
        return_html=True,
    )
    display(HTML(html))

The ``telescope`` argument accepts ``"Simonyi"`` or ``"AuxTel"``.

Narrative log
-------------

:meth:`~logging_query.NarrativeLogClient.query_log` returns log messages between
two times as a DataFrame indexed by timestamp:

.. code-block:: python

    log = endpoints["narrative_log"].query_log(t_start, t_end)
    log[["component", "message_text", "user_id"]].head()

The timestamp index makes it straightforward to join narrative entries with EFD
data or visit tables for contextual analysis.

For more granular control, the underlying ``query`` method accepts the full set of
parameters supported by the `narrativelog REST API
<https://usdf-rsp.slac.stanford.edu/narrativelog/docs#/default/find_messages_messages_get>`_.

.. code-block:: python

    messages = endpoints["narrative_log"].query(
        min_date_begin=t_start.isot,
        max_date_begin=t_end.isot,
        components=["ATMCS", "ATPneumatics"],
    )

Exposure log
------------

:meth:`~logging_query.ExposureLogClient.query_log` retrieves exposure annotations.
This log is typically sparse; many nights will return an empty DataFrame.

.. code-block:: python

    exposure_log = endpoints["exposure_log"].query_log(t_start, t_end)

Exposure log entries reference an exposure ID rather than a wall-clock time, so joining
with visit data requires matching on ``obs_id`` or ``exposure_id``.

The full set of query parameters is documented at the `exposurelog REST API
<https://usdf-rsp.slac.stanford.edu/exposurelog/docs#/default/find_messages_messages_get>`_.

Async queries
-------------

All three log clients expose both synchronous and asynchronous query methods.
The ``query`` method is synchronous; ``async_query`` is its async counterpart for
use in async contexts or event-loop-aware frameworks.

.. code-block:: python

    messages = await endpoints["narrative_log"].async_query(
        min_date_begin=t_start.isot,
        max_date_begin=t_end.isot,
    )
