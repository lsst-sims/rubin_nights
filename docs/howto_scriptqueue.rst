.. py:currentmodule:: rubin_nights

.. _howto_scriptqueue:

ScriptQueue
===========

The ScriptQueue is the observatory's command dispatcher; every script executed at the
summit passes through it.  The :mod:`~rubin_nights.scriptqueue` module provides tools
to mine those records from the EFD and reconstruct a narrative of what ran during a
night.  This is the same data source used by the *SQ Miner* notebook on Times Square
and the *Context Feed* in Nightly Digest.

See `ts-xml.lsst.io <https://ts-xml.lsst.io>`_ for the full SAL schema for
``lsst.sal.ScriptQueue.*`` topics.

Setting up
----------

.. code-block:: python

    import logging
    from astropy.time import Time, TimeDelta
    from rubin_nights import connections, scriptqueue, scriptqueue_formatting
    from rubin_nights.ts_xml_enums import CategoryIndexExtended

    logging.getLogger("rubin_nights").setLevel(logging.INFO)

    endpoints = connections.get_clients()

    day_obs = "2025-12-07"
    t_start = Time(f"{day_obs}T12:00:00", format="isot", scale="utc")
    t_end = t_start + TimeDelta(1, format="jd")

Gathering consolidated messages
--------------------------------

:func:`~scriptqueue.get_consolidated_messages` queries both ScriptQueues (Simonyi
and AuxTel) and merges the results into a single chronologically ordered DataFrame.

.. code-block:: python

    messages, cols = scriptqueue.get_consolidated_messages(
        t_start, t_end, endpoints, all_tracebacks=True
    )

The returned ``cols`` is a subset of the full set of columns in the
returned DataFrame that are most useful when displaying scriptqueue summaries.
Note that ``t_start`` and ``t_end`` can be any astropy Time values.


Filtering the messages
----------------------

The ``scriptqueue.get_consolidated_messages``` function assigns an enum to
every record, following the mapping in
:class:`~ts_xml_enums.CategoryIndexExtended`. These values
can be used to split the consolidated output by telescope or "type" of message.

.. code-block:: python

    # Select all simonyi-related categories
    simonyi_idx = [
        i.value for i in CategoryIndexExtended
        if "SIMONYI" in i.name or "OTHER" in i.name or "MAIN" in i.name or "OCS" in i.name
    ]
    # Select all auxtel-related categories
    auxtel_idx = [
        i.value for i in CategoryIndexExtended
        if "AUX" in i.name or "OTHER" in i.name
    ]

    # Select all log messages, regardless of telescope
    narrative_idx = [
        i.value for i in CategoryIndexExtended
        if "NARRATIVE_LOG" in i.name
    ]
    # Select Simonyi ObservatoryStatus messages
    obs_status_idx = [CategoryIndexExtended.OBSERVATORY_STATUS_SIMONYI.value]

    # Get only the simonyi messages (etc)
    messages = messages[messages["category_index"].isin(simonyi_idx)]


Rendering as HTML
-----------------

:func:`~scriptqueue_formatting.format_html` formats the consolidated messages as an
HTML table for display in a notebook:

.. code-block:: python

    from IPython.display import display, HTML

    # Show all queues in chronological order
    all_html = scriptqueue_formatting.format_html(messages, cols=cols, time_order="oldest first")
    display(HTML(all_html))

A subset of messages such as only the Simonyi-related messages can be
displayed by passing the relevant CategoryIndexes:

.. code-block:: python

    simonyi_idx = [
        i.value for i in CategoryIndexExtended
        if "SIMONYI" in i.name or "OTHER" in i.name or "MAIN" in i.name or "OCS" in i.name
    ]
    simonyi_narrative_html = scriptqueue_formatting.format_html(
        messages, cols=cols,
        time_order="oldest first",
        show_category_index=simonyi_idx,
    )



Lower-level query functions
----------------------------

The individual building blocks used by :func:`~scriptqueue.get_consolidated_messages`
are also available directly:

* :func:`~scriptqueue.get_all_tracebacks` — get tracebacks from all CSCs with logevent_logMessage topics, except the ScriptQueue.
* :func:`~scriptqueue.get_error_codes` — get error messages from all CSCs with logevent_errorCode messages.
* :func:`~scriptqueue.get_narrative_and_errors` — get narrative log entries and error messages.
* :func:`~scriptqueue.get_exposure_info` — get exposure endOfImageTelemetry records and join with exposure_log records.
* :func:`~scriptqueue.get_script_status` - get the summarized ScriptQueue records.
* :func:`~scriptqueue.get_scheduler_configs` - get information on the FBS configuration and software versions.


