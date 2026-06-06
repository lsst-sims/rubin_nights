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
A usdf (prod) token cannot be used to authenticate for usdf-dev services,


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

The returned dictionary contains clients for each available service:

.. code-block:: text

    {
        "consdb":        <ConsDbFastAPI>,
        "consdb_tap":    <ConsDbTap>,
        "efd":           <InfluxQueryClient for lsst.efd>,
        "obsenv":        <InfluxQueryClient for lsst.obsenv>,
        "pp":            <InfluxQueryClient for lsst.prompt>,
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

Connecting to individual services
----------------------------------

If you only need a single client, you can instantiate the relevant class directly
rather than using :func:`~connections.get_clients`.

.. code-block:: python

    from rubin_nights.connections import get_access_token
    from rubin_nights.consdb_query import ConsDbTap
    from rubin_nights.reference_values import API_ENDPOINTS

    token = get_access_token()
    api_base = API_ENDPOINTS["usdf"]
    consdb_tap = ConsDbTap(api_base=api_base, token=token)


