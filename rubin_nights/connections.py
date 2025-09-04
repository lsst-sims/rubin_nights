"""Connection utilities."""

import logging
import os
from urllib.parse import urlparse

from .consdb_query import ConsDbFastAPI, ConsDbTap
from .influx_query import InfluxQueryClient
from .logging_query import ExposureLogClient, NarrativeLogClient, NightReportClient

__all__ = ["get_access_token", "get_clients", "usdf_lfa"]

logger = logging.getLogger(__name__)


def get_access_token(tokenfile: str | None = None) -> str:
    """Retrieve RSP access token.

    Parameters
    ----------
    tokenfile
        Path to the RSP token file. See documentation on RSP tokens at
        https://rsp.lsst.io/v/usdfprod/guides/auth/creating-user-tokens.html
        The token will be read from the tokenfile if available.
        The default value of `None` will attempt to use the
        `lsst.rsp.get_access_token` method if accessible, or then
        look for ACCESS_TOKEN in environment variables.
        If no RSP token is available, access to most services will not
        be available.

    Returns
    -------
    token : `str`
        Token value.
        A zero-length token will not be valid for use.

    Notes
    -----
    RSP access tokens are unique to the different RSP sites, and the
    services which run on a particular site must receive tokens from the
    same site.
    """
    token = None
    # First - tokenfile provided
    if tokenfile is not None:
        with open(tokenfile, "r") as f:
            token = f.read().strip()
    else:
        # Second - are we at an RSP and should use lsst.rsp.get_access_token
        try:
            import lsst.rsp.get_access_token as rsp_get_access_token

            token = rsp_get_access_token(tokenfile=tokenfile)
        except ImportError:
            # Not on an RSP.
            pass
        # Third - try environment variable ACCESS_TOKEN
        if token is None:
            token = os.environ.get("ACCESS_TOKEN", None)
    # Final check to issue warning.
    if token is None:
        token = ""
        logging.warning("No RSP token found.")
    return token


def get_clients(
    tokenfile: str | None = None,
    site: str | None = None,
    auth_token: str | None = None,
) -> dict:
    """Return site-specific client connections.

    Parameters
    ----------
    tokenfile
        Path to the RSP tokenfile. See also `get_access_token`.
    site
        Override site location to a preferred site.
        Most likely to be used to specify `usdf-dev` vs `usdf`.
    auth_token
        The bare authentication token string.
        If not None, this will override any tokenfile argument.
        Useful in services running behind Gafaelfawr authentication.

    Returns
    -------
    endpoints : `dict`
        Dictionary with `efd`, `obsenv`, `sasquatch`,
        `narrative_log`, `exposure_log`, `night_log`, and `consdb`
        connection information.

    Note
    ----
    The authentication token required to access the log services
    is an RSP token, and is RSP site-specific (including usdf vs usdf-dev).
    For users outside the RSP, a token can be created as described in
    https://rsp.lsst.io/v/usdfprod/guides/auth/creating-user-tokens.html
    """
    # For more information on rubin tokens see DMTN-234.
    # For information on scopes, see DMTN-235.
    if auth_token is not None:
        # Override and use provided token as-is.
        token = auth_token
    else:
        # Set up authentication
        token = get_access_token(tokenfile)

    auth = ("user", token)

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
        # Otherwise, use the USDF resources, outside of the RSP
        if site is None:
            site = "usdf"
    else:
        site = site

    api_base = api_endpoints[site]
    narrative_log = NarrativeLogClient(api_base, auth)
    exposure_log = ExposureLogClient(api_base, auth)
    night_report = NightReportClient(api_base, auth)
    consdb_query = ConsDbFastAPI(api_base, auth)
    consdb_tap = ConsDbTap(api_base, token=token)
    efd_client = InfluxQueryClient(site, db_name="efd")
    obsenv_client = InfluxQueryClient(site, db_name="lsst.obsenv")
    sasquatch_client = InfluxQueryClient("usdfdev", db_name="lsst.dm")

    # Be extra helpful with environment variables if using USDF for LFA
    if "usdf" in site:
        # And some env variables for S3 through USDF
        os.environ["LSST_DISABLE_BUCKET_VALIDATION"] = "1"
        os.environ["S3_ENDPOINT_URL"] = "https://s3dfrgw.slac.stanford.edu/"
    # Or if you're actually using one of the USDF RSPs (or kubernetes)
    if "usdf" in os.getenv("EXTERNAL_INSTANCE_URL", ""):
        if os.getenv("RUBIN_SIM_DATA_DIR") is None:
            # Use shared RUBIN_SIM_DATA_DIR
            os.environ["RUBIN_SIM_DATA_DIR"] = "/sdf/data/rubin/shared/rubin_sim_data"

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
    logger.info(f"Endpoint base url: {endpoints['api_base']}")

    return endpoints


def usdf_lfa(uri: str, bucket: str = "s3://lfa@") -> str:
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
    filekey = urlparse(uri).path.lstrip("/")
    return bucket + filekey
