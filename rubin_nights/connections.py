import logging
import os
from urllib.parse import urlparse

from .consdb_query import ConsDbFastAPI, ConsDbTap
from .influx_query import InfluxQueryClient
from .logging_query import ExposureLogClient, NarrativeLogClient, NightReportClient
from .reference_values import API_ENDPOINTS

__all__ = ["get_access_token", "get_clients", "usdf_lfa"]

logger = logging.getLogger(__name__)

DEFAULT_TOKENFILE = "usdf_rsp"


def get_access_token(tokenfile: str | None = None) -> str:
    """Retrieve RSP access token.

    Parameters
    ----------
    tokenfile
        Path to the RSP token file. See documentation on RSP tokens at
        https://rsp.lsst.io/guides/auth/creating-user-tokens.html
        The token will be read from the tokenfile if available.
        If tokenfile is None, then further attempts will be made to
        access the token value from:

        1. `lsst.rsp.get_access_token`
        2. the environment variable "ACCESS_TOKEN"
        3. the environment variable "ACCESS_TOKEN_FILE"
        4. the home directory + '.lsst' + DEFAULT_TOKENFILE

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
    # First - tokenfile explicitly provided.
    if tokenfile is not None:
        with open(tokenfile, "r") as f:
            token = f.read().strip()
    else:
        logger.debug("Tokenfile not specified.")
        # Second - are we at an RSP and should use lsst.rsp.get_access_token
        try:
            import lsst.rsp.get_access_token as rsp_get_access_token

            token = rsp_get_access_token(tokenfile=tokenfile)
        except ImportError:
            # Not on an RSP.
            logger.debug("Attempt to import lsst.rsp.get_access_token failed.")
            pass
        # Third - try environment variable ACCESS_TOKEN (containing token)
        if token is None:
            token = os.environ.get("ACCESS_TOKEN", None)
        # Fourth - try environment variable ACCESS_TOKEN_FILE (file location)
        if token is None:
            logger.debug("$ACCESS_TOKEN not set.")
            tokenfile = os.environ.get("ACCESS_TOKEN_FILE", None)
            if tokenfile is not None:
                logger.debug(f"Checking $ACCESS_TOKEN_FILE {tokenfile}")
                # Try to read this, but an error is not an exception.
                try:
                    with open(tokenfile, "r") as f:
                        token = f.read().strip()
                except FileNotFoundError:
                    logger.debug(f"{tokenfile} does not exist.")
                    pass
        # Fifth - try a default home directory location.
        if token is None:
            logger.debug("$ACCESS_TOKEN_FILE not set.")
            tokenfile = os.path.join(os.path.expanduser("~"), ".lsst", DEFAULT_TOKENFILE)
            logger.debug(f"Checking {tokenfile}")
            # Try to read this, but an error is not an exception.
            try:
                with open(tokenfile, "r") as f:
                    token = f.read().strip()
            except FileNotFoundError:
                logger.debug(f"{tokenfile} does not exist.")
                pass
    # Final check on token value, in order to issue warning.
    if token is None:
        token = ""
        logging.error("No RSP token found.")
    return token


def get_clients(
    tokenfile: str | None = None,
    site: str | None = None,
    auth_token: str | None = None,
) -> dict:
    """Return a wide set of site-specific client connections.

    Parameters
    ----------
    tokenfile
        Path to the RSP tokenfile. See also `get_access_token`.
        Can be None if one of other methods to set token will be successful.
    site
        Override site location to a preferred site.
        Most likely to be used to specify `usdf-dev` vs `usdf`.
        Note that this should be equivalent to the repertoire discovery site.
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

    if site is None:
        # Guess site from EXTERNAL_INSTANCE_URL (set for RSPs)
        location = os.getenv("EXTERNAL_INSTANCE_URL", "")
        if "summit-lsp" in location:
            site = "summit"
        elif "base-lsp" in location:
            site = "base"
        elif "usdf-rsp-int" in location:
            site = "usdf-int"
        elif "usdf-rsp-dev" in location:
            site = "usdf-dev"
        elif "usdf-rsp" in location:
            site = "usdf"
        elif "base" in location:
            site = "base"
        # Otherwise, use the USDF resources, outside of the RSP
        if site is None:
            site = "usdf"
    else:
        site = site

    if site not in API_ENDPOINTS:
        raise ValueError(f"Site {site} must be in {list(API_ENDPOINTS.keys())}")

    api_base = API_ENDPOINTS[site]

    narrative_log = NarrativeLogClient(api_base, auth)
    exposure_log = ExposureLogClient(api_base, auth)
    night_report = NightReportClient(api_base, auth)

    consdb_query = ConsDbFastAPI(api_base, auth)
    consdb_tap = ConsDbTap(api_base, token=token)

    # The EFD is difficult while repertoire and influx deployments
    # are in progress. For now: try usdf-efd first, then local version.
    # This may later apply to all other databases as well.
    try:
        efd_client = InfluxQueryClient("usdf", db_name="efd", api_base=api_base, auth=auth)
        efd_client.get_topics()
    except KeyError:
        logger.warning(f"EFD service not available at USDF. Falling back to {site}.")
        efd_client = InfluxQueryClient(site, db_name="efd", api_base=api_base, auth=auth)

    # Connect to the databases which are not in repertoire yet
    obsenv_client = InfluxQueryClient(site, db_name="lsst.obsenv")
    pp_client = InfluxQueryClient(site, db_name="lsst.prompt")
    too_client = InfluxQueryClient("summit", db_name="lsst.scimma")

    # Be extra helpful with environment variables if using USDF for LFA
    if "usdf" in site:
        # And some env variables for S3 through USDF
        if os.environ.get("LSST_DISABLE_BUCKET_VALIDATION") is None:
            os.environ["LSST_DISABLE_BUCKET_VALIDATION"] = "1"
        if os.environ.get("S3_ENDPOINT_URL") is None:
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
        "pp": pp_client,
        "too": too_client,
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
