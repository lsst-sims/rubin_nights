"""Connection utilities."""

import logging
import os
from urllib.parse import urlparse

from .consdb_query import ConsDbFastAPI, ConsDbTap
from .influx_query import InfluxQueryClient
from .logging_query import ExposureLogClient, NarrativeLogClient, NightReportClient

__all__ = ["get_access_token", "get_clients", "localize_lfa"]

logger = logging.getLogger(__name__)


def get_access_token(tokenfile: str | None = None) -> str:
    """Retrieve RSP access token.

    Parameters
    ----------
    tokenfile : `str` or None
        Path to token file.
        Default None will fall back to environment variable,
        ACCESS_TOKEN and then try lsst.rsp.get_access_token().

    Returns
    -------
    token : `str`
        Token value.
    """
    try:
        # Try using lsst-rsp first
        import lsst.rsp.get_access_token as rsp_get_access_token

        token = rsp_get_access_token(tokenfile=tokenfile)
    except ImportError:
        # No lsst-rsp available
        if tokenfile is not None:
            with open(tokenfile, "r") as f:
                token = f.read().strip()
        else:
            token = os.environ.get("ACCESS_TOKEN")

    if token is None:
        logging.warning("No RSP token available.")
        token = ""
    return token


def get_clients(tokenfile: str | None = None, site: str | None = None) -> dict:
    """Return site-specific client connections.

    Parameters
    ----------
    tokenfile : `str` or None
        Passed to `get_access_token`.
    site : `str` or None
        Override site location to a preferred site.
        Most likely to be used to specify `usdf-dev` vs `usdf`.

    Returns
    -------
    endpoints : `dict`
        Dictionary with `efd`, `obsenv`,
        `narrative_log`, `exposure_log`, `night_log`, and `consdb`
        connection information.

    Note
    ----
    The authentication token required to access the log services
    is an RSP token, and is RSP site-specific (including usdf vs usdf-dev).
    For users outside the RSP, a token can be created as described in
    https://nb.lsst.io/environment/tokens.html
    """
    # Set up authentication
    token = get_access_token(tokenfile)
    auth = ("user", token)
    # For more information on rubin tokens see DMTN-234.
    # For information on scopes, see DMTN-235.

    api_endpoints = {
        "usdf": "https://usdf-rsp.slac.stanford.edu",
        "usdf-dev": "https://usdf-rsp-dev.slac.stanford.edu",
        "summit": "https://summit-lsp.lsst.codes",
    }

    if site is None:
        # Guess site from EXTERNAL_INSTANCE_URL (set for RSPs)
        location = os.getenv("EXTERNAL_INSTANCE_URL", "")
        if "summit-lsp" in location:
            site = "summit"
        elif "usdf-rsp-dev" in location:
            site = "usdf-dev"
        elif "usdf-rsp" in location:
            site = "usdf"
        if site is not None and site.startswith("usdf"):
            # Also set up some env variables specific to USDF-RSP
            os.environ["no_proxy"] += ",.consdb"
            os.environ["RUBIN_SIM_DATA_DIR"] = "/sdf/data/rubin/shared/rubin_sim_data"
        # Otherwise, use the USDF resources, outside of the RSP
        if site is None:
            site = "usdf"
    else:
        site = site

    if site == "usdf":
        # And some env variables for S3 through USDF
        os.environ["LSST_DISABLE_BUCKET_VALIDATION"] = "1"
        os.environ["S3_ENDPOINT_URL"] = "https://s3dfrgw.slac.stanford.edu/"

    api_base = api_endpoints[site]
    narrative_log = NarrativeLogClient(api_base, auth)
    exposure_log = ExposureLogClient(api_base, auth)
    night_report = NightReportClient(api_base, auth)
    consdb_query = ConsDbFastAPI(api_base, auth)
    consdb_tap = ConsDbTap(api_base, token=token)
    efd_client = InfluxQueryClient(site, db_name='efd')
    obsenv_client = InfluxQueryClient(site, db_name="lsst.obsenv")
    sasquatch_client = InfluxQueryClient("usdfdev", db_name="lsst.dm")

    endpoints = {
        "api_base": api_base,
        "efd": efd_client,
        "obsenv": obsenv_client,
        "sasquatch": sasquatch_client,
        "consdb": consdb_query,
        "consdb_tap": consdb_tap,
        "narrative_log": narrative_log,
        "exposure_log": exposure_log,
        "night_report": night_report,
    }

    return endpoints


def localize_lfa(uri: str, bucket:str='s3://lfa@') -> str:
    """Convert LFA uri recorded in the EFD to a version accessible at USDF.

    Parameters
    ----------
    uri : `str`
        The URI written into the EFD from the summit.
    bucket : `str`
        The bucket access at the USDF.

    Returns
    -------
    uri : `str`
        The LFA uri at USDF.
    """
    filekey = urlparse(uri).path.lstrip('/')
    return bucket + filekey
