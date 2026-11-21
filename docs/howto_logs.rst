.. py:currentmodule:: rubin_nights

.. _howto_logs:

Querying Summit Logs
====================

Three summit logging services can be queried through ``rubin_nights``:

* **Night report** — a structured summary of the observing night, covering both telescopes.
* **Narrative log** — a running record of observer comments and observatory state changes.
* **Exposure log** — annotations attached to specific exposures.

Synchronous clients for all three are included in the dictionary returned by
:func:`~connections.get_clients`. Asynchronous clients are available for the
narrative log and exposure log (see `Async queries`_).

Setting a time window
---------------------

.. code-block:: python

    from astropy.time import Time, TimeDelta

    day_obs = "2025-04-15"
    t_start = Time(f"{day_obs}T12:00:00", format="isot", scale="utc")
    t_end = t_start + TimeDelta(1, format="jd")

Night report
------------

:meth:`~logging_query.NightReportClient.query_night_report` returns the night report(s)
for a given ``day_obs`` as a list of dicts, together with an HTML-formatted
version of the first report suitable for display in a notebook.
``day_obs`` can be given as ``"YYYY-MM-DD"`` or as an integer ``YYYYMMDD``.

.. code-block:: python

    from IPython.display import display, HTML

    reports, html = endpoints["night_report"].query_night_report(
        day_obs=day_obs,
        telescope="Simonyi",
        return_html=True,
    )
    display(HTML(html))

The ``telescope`` argument accepts ``"Simonyi"``, ``"AuxTel"``, or ``None`` (the default),
and only affects which telescope summary is included in the HTML; the returned reports
always contain both. With ``return_html=False`` the second return value is an empty string.

The night report client is synchronous only.

Narrative log
-------------

:meth:`~logging_query.NarrativeLogClient.query_log` returns log messages between
two times as a DataFrame indexed by timestamp (UTC):

.. code-block:: python

    log = endpoints["narrative_log"].query_log(t_start, t_end)
    log[["component", "message_text", "user_id"]].head()

The timestamp index makes it straightforward to join narrative entries with EFD
data or visit tables for contextual analysis.

To add or override query parameters, pass a ``user_params`` dict. Any parameter supported
by the `narrativelog REST API
<https://usdf-rsp.slac.stanford.edu/narrativelog/docs#/default/find_messages_messages_get>`_
can be used; ``{"limit": int}`` overrides the default limit of 50000 messages.

.. code-block:: python

    log = endpoints["narrative_log"].query_log(
        t_start, t_end,
        user_params={"components": ["ATMCS", "ATPneumatics"]},
    )

Exposure log
------------

:meth:`~logging_query.ExposureLogClient.query_log` retrieves exposure log messages
(annotations). This log is typically sparse; many nights will return an empty DataFrame.

.. code-block:: python

    exposure_log = endpoints["exposure_log"].query_log(t_start, t_end)

The exposure log messages do not record the time of the exposure itself, so
``t_start`` and ``t_end`` are converted to ``min_day_obs`` and ``max_day_obs`` for the query.
Joining with visit data requires matching on ``obs_id`` (or ``day_obs`` and ``seq_num``).

:meth:`~logging_query.ExposureLogClient.query_exposures` retrieves the exposure records
known to the exposure log over a time range. The default instrument is ``LSSTCam``;
override it with ``user_params``:

.. code-block:: python

    exposures = endpoints["exposure_log"].query_exposures(
        t_start, t_end, user_params={"instrument": "LATISS"}
    )

Both methods accept ``user_params`` in the same way as the narrative log.
The full set of query parameters is documented at the `exposurelog REST API
<https://usdf-rsp.slac.stanford.edu/exposurelog/docs#/default/find_messages_messages_get>`_.

Lower-level queries
-------------------

Every log client also has a lower-level ``query(url_extension, params)`` method that sends
an arbitrary ``params`` dict to an endpoint relative to the client's base url.
The ``query_log`` methods above are generally simpler to use, as they fill in sensible
default parameters and (for the narrative log) reformat the results.

.. code-block:: python

    params = endpoints["narrative_log"].query_params(t_start, t_end)
    raw = endpoints["narrative_log"].query("/narrativelog/messages", params=params)

Async queries
-------------

Asynchronous queries use separate client classes,
:class:`~logging_query.AsyncNarrativeLogClient` and
:class:`~logging_query.AsyncExposureLogClient`. These are not created by
:func:`~connections.get_clients`; instantiate them directly with the same
``api_base`` and ``auth_token``. They have the same method names as the
synchronous clients, but their query methods must be awaited.

.. code-block:: python

    from rubin_nights.connections import get_access_token
    from rubin_nights.logging_query import AsyncExposureLogClient, AsyncNarrativeLogClient
    from rubin_nights.reference_values import API_ENDPOINTS

    api_base = API_ENDPOINTS["usdf"]
    token = get_access_token()

    async with AsyncNarrativeLogClient(api_base, auth_token=token) as narrative_log:
        log = await narrative_log.query_log(t_start, t_end)

    async with AsyncExposureLogClient(api_base, auth_token=token) as exposure_log:
        messages = await exposure_log.query_log(t_start, t_end)
        exposures = await exposure_log.query_exposures(t_start, t_end)

The async clients return the same DataFrames as their synchronous counterparts.
Use ``async with`` (as above) or call ``await client.aclose()`` when finished.
