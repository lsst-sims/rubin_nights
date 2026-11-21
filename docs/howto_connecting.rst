.. py:currentmodule:: rubin_nights

.. _howto_connecting:

Getting Connected
=================

``rubin_nights`` contains clients to query several internal-project Rubin
metadata services: the EFD, the ConsDB, and the
logging services (narrative log, exposure log, and night report).
:func:`connections.get_clients` is the easiest way to set up all of these at once.


Authentication
--------------

An RSP token is required to query these services.  When running inside an RSP (at the summit,
base, or USDF) the token is available automatically and no extra configuration is needed.

When running outside an RSP, create a token at the relevant site -- if you
go to the RSP login and choose "Access Tokens" from the drop-down, you can
create a new token with the necessary scope. The "read:image", "read:tap"
and "read:sasquatch" scopes should cover what is needed. Note that tokens
are site-specific!
A usdf (prod) token cannot be used to authenticate for usdf-dev services.

All of the query clients take the token itself, as a string, through an
``auth_token`` argument, and send it to the services as a bearer token
(``Authorization: Bearer <token>``).
The older ``auth=(username, password)`` tuple is no longer accepted.

:func:`connections.get_access_token` looks for a token in the following order:

1. An explicit ``tokenfile`` path passed to the function.
2. ``lsst.rsp.get_access_token`` (available when running on an RSP).
3. The environment variable ``ACCESS_TOKEN``.
4. The path given by the environment variable ``ACCESS_TOKEN_FILE``.
5. ``~/.lsst/usdf_rsp`` (the default file location, with a name that
   suggests the usdf-prod RSP services).

Connecting to all services at once
------------------------------------

.. code-block:: python

    import os
    from rubin_nights import connections

    # On an RSP — token and site are detected automatically
    endpoints = connections.get_clients()

    # Outside an RSP — point at your token file
    site = "usdf"
    tokenfile = os.path.join(os.path.expanduser("~"), ".lsst/usdf_rsp")
    endpoints = connections.get_clients(site=site, tokenfile=tokenfile)

    # Or pass the token value directly (overrides tokenfile).
    # Useful for services running behind Gafaelfawr authentication.
    token = connections.get_access_token(tokenfile)
    endpoints = connections.get_clients(site=site, auth_token=token)

The returned dictionary contains the base API url and a client for each available service.
All of the clients returned by :func:`~connections.get_clients` are **synchronous**;
see :ref:`howto_connecting_async` for the asynchronous equivalents.

.. code-block:: text

    {
        "api_base":      <str, e.g. https://usdf-rsp.slac.stanford.edu>,
        "consdb":        <ConsDbFastAPI>,
        "consdb_tap":    <ConsDbTap>,
        "efd":           <InfluxQueryClient for efd>,
        "obsenv":        <InfluxQueryClient for lsst.obsenv>,
        "pp":            <InfluxQueryClient for lsst.prompt>,
        "too":           <InfluxQueryClient for lsst.scimma, or None>,
        "narrative_log": <NarrativeLogClient>,
        "exposure_log":  <ExposureLogClient>,
        "night_report":  <NightReportClient>,
    }

Choosing a site
---------------

The ``site`` argument selects which Rubin deployment to connect to.
Valid values are ``"summit"``, ``"base"``, ``"usdf"``, ``"usdf-dev"``, and ``"usdf-int"``.
When ``site`` is ``None`` (the default), the site is inferred from the
``EXTERNAL_INSTANCE_URL`` environment variable that RSPs set automatically.
Outside any RSP, ``"usdf"`` is used as the default. Remember to use a token
that matches the site you choose.

.. code-block:: python

    # Connect to the USDF development environment instead
    endpoints_dev = connections.get_clients(tokenfile=tokenfile, site="usdf-dev")

.. note::

    On ``usdf-dev`` the EFD clients cannot fetch their credentials from repertoire
    with a usdf-dev token. Instead, they fall back to the ``EFDAUTH`` environment
    variable, which should point to a JSON file of InfluxDB credentials
    (see :ref:`howto_efd_credentials`).

Connecting to individual services
----------------------------------

If you only need a single client, you can instantiate the relevant class directly
rather than using :func:`~connections.get_clients`.
Every client takes ``auth_token`` as a string:

.. code-block:: python

    from rubin_nights.connections import get_access_token
    from rubin_nights.consdb_query import ConsDbTap, ConsDbFastAPI
    from rubin_nights.influx_query import InfluxQueryClient
    from rubin_nights.logging_query import NarrativeLogClient
    from rubin_nights.reference_values import API_ENDPOINTS

    site = "usdf"
    token = get_access_token()
    api_base = API_ENDPOINTS[site]

    consdb_tap = ConsDbTap(api_base=api_base, auth_token=token)
    consdb = ConsDbFastAPI(api_base=api_base, auth_token=token)
    narrative_log = NarrativeLogClient(api_base=api_base, auth_token=token)
    efd = InfluxQueryClient(site=site, db_name="efd", repertoire_site=site, auth_token=token)

.. _howto_connecting_async:

Synchronous and asynchronous clients
------------------------------------

Synchronous and asynchronous queries are provided by separate client classes.
The synchronous clients are the ones returned by :func:`~connections.get_clients`;
the asynchronous clients must be created directly, but take the same
initialization arguments.

.. list-table::
    :header-rows: 1

    * - Service
      - Synchronous client
      - Asynchronous client
    * - ConsDB (FastAPI)
      - :class:`~consdb_query.ConsDbFastAPI`
      - :class:`~consdb_query.AsyncConsDbFastAPI`
    * - ConsDB (TAP)
      - :class:`~consdb_query.ConsDbTap`
      - Use :class:`~consdb_query.ConsDbTap` `.tap` connection directly
    * - EFD and other InfluxDBs
      - :class:`~influx_query.InfluxQueryClient`
      - :class:`~influx_query.AsyncInfluxQueryClient`
    * - Narrative log
      - :class:`~logging_query.NarrativeLogClient`
      - :class:`~logging_query.AsyncNarrativeLogClient`
    * - Exposure log
      - :class:`~logging_query.ExposureLogClient`
      - :class:`~logging_query.AsyncExposureLogClient`
    * - Night report
      - :class:`~logging_query.NightReportClient`
      - (none)

On the asynchronous clients the query methods are coroutines and must be awaited
(for example, inside a Jupyter notebook cell or an ``async def`` function):

.. code-block:: python

    from rubin_nights.consdb_query import AsyncConsDbFastAPI
    from rubin_nights.influx_query import AsyncInfluxQueryClient

    async_consdb = AsyncConsDbFastAPI(api_base=api_base, auth_token=token)
    visits = await async_consdb.query("SELECT * FROM cdb_lsstcam.visit1 WHERE day_obs = 20251127")

    async_efd = AsyncInfluxQueryClient(site=site, db_name="efd", repertoire_site=site, auth_token=token)
    topics = await async_efd.get_topics()

Closing connections
-------------------

The HTTP-based clients hold open a connection pool, which should be closed when
you are done with it. Synchronous clients have a ``close()`` method and can be
used as context managers; asynchronous clients have an ``aclose()`` coroutine
and can be used as async context managers.

.. code-block:: python

    # Close everything returned by get_clients
    connections.close_clients(endpoints)

    # Or let a context manager close a single client
    with NarrativeLogClient(api_base=api_base, auth_token=token) as narrative_log:
        log = narrative_log.query_log(t_start, t_end)

    async with AsyncConsDbFastAPI(api_base=api_base, auth_token=token) as async_consdb:
        visits = await async_consdb.query(query)

