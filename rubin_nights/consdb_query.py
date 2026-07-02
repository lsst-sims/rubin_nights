from __future__ import annotations

import datetime
import logging
from collections.abc import Awaitable
from json import JSONDecodeError
from types import TracebackType
from typing import Optional, Type

import httpx
import numpy as np
import pandas as pd
import pyvo
from astropy.time import Time
from pyvo.dal import DALQueryError

try:
    import sqlalchemy
    from psycopg import ProgrammingError

    HAS_SQLALCHEMY = True
except ModuleNotFoundError:
    HAS_SQLALCHEMY = False

from .augment_visits import augment_visits

logger = logging.getLogger(__name__)

__all__ = ["ConsDbTap", "ConsDbFastAPI", "ConsDbSql"]


class ConsDb:
    """Use the child classes, ConsDbTap, ConsDbFastAPI or ConsDbSql,
    instead of this class directly. This class simply implements
    some convenience functions for fetching visits and ccdvisits.
    """

    def query(self, query: str) -> pd.DataFrame | Awaitable[pd.DataFrame]:
        """Execute ConsDB query."""
        raise NotImplementedError("Must be implemented by child class")

    def get_visits(
        self,
        instrument: str,
        t_start: Time | None = None,
        t_end: Time | None = None,
        visit_constraint: str | None = None,
        augment: bool = True,
    ) -> pd.DataFrame:
        """Fetch visit and quicklook values from the ConsDB.

        Parameters
        ----------
        instrument
            The instrument to search for.
            Typical values would include lsstcomcam, latiss, and lsstcam.
            See https://sdm-schemas.lsst.io/ for more details.
        t_start
            The earliest time to match obs_start.
        t_end
            The latest time to match obs_start.
        visit_constraint
            A constraint to apply to the cdb_{instrument}.visit1 table.
            Example: `"visit1.science_program = 'BLOCK-365'"`
        augment
            If True, immediately call `augment_visits.augment_visits`
            after fetching visit1 and visit1_quicklook values from the ConsDB.

        Returns
        -------
        visits : `pd.DataFrame`
            The visit information from cdb_{instrument}.visit1 and
            cdb_{instrument}.visit1_quicklook (if available).
            Additional information may be added, such as `visit_gap`
            if `augment_visits` is True.
        """

        query = (
            f"select visit1.*, visit1_quicklook.* "
            f"from  cdb_{instrument}.visit1 as visit1 "
            f"left join cdb_{instrument}.visit1_quicklook as visit1_quicklook "
            f"on visit1.visit_id = visit1_quicklook.visit_id "
        )
        constraint = []
        if t_start is not None:
            constraint.append(f" visit1.obs_start_mjd >= {t_start.mjd} ")
        if t_end is not None:
            constraint.append(f" visit1.obs_start_mjd <= {t_end.mjd} ")
        if visit_constraint is not None:
            constraint.append(f" ({visit_constraint}) ")
        constraint_str = "and".join(constraint)
        if len(constraint_str) > 0:
            query = query + f" where {constraint_str}"
        query += " order by visit1.visit_id"
        logger.debug(f"Query executed: {query}")
        visits = self.query(query)
        assert isinstance(visits, pd.DataFrame)

        if visits.empty:
            logger.info(f"No visits for {instrument} retrieved from consdb")
            return pd.DataFrame([])

        if augment:
            visits = augment_visits(visits, instrument=instrument)
        return visits

    def query_ccdvisits(
        self,
        instrument: str,
        visit_id: int,
        detector_min: int | None = None,
        detector_max: int | None = None,
    ) -> pd.DataFrame:
        """Fetch ccdvisit data.

        Parameters
        ----------
        instrument
            The instrument to search for.
            Typical values would include lsstcomcam, latiss, and lsstcam.
            See https://sdm-schemas.lsst.io/ for more details.
        visit_id
            The visit for which to fetch the detector values.
        detector_min, detector_max
            The minimum and maximum detector number to fetch.
            Values of None will fetch all detectors.
            Values of `detector_min=90, detector_max=98` will fetch
            the center raft.

        Returns
        -------
        ccdvisits : `pd.DataFrame`
            The visit information from cdb_{instrument}.visit1 and the
            per-detector ccdvisit information.
        """
        query = (
            f"select v.*, c.detector, cq.* "
            f"from cdb_{instrument}.visit1 as v join cdb_{instrument}.ccdvisit1 as c "
            f"on v.visit_id = c.visit_id  "
            f"left join cdb_{instrument}.ccdvisit1_quicklook as cq "
            f"on c.ccdvisit_id = cq.ccdvisit_id "
            f"where v.visit_id = {visit_id}"
        )
        if detector_min is not None:
            query += f" and c.detector >= {detector_min}"
        if detector_max is not None:
            query += f" and c.detector <= {detector_max}"
        ccdvisits = self.query(query)
        assert isinstance(ccdvisits, pd.DataFrame)
        return ccdvisits


class ConsDbTap(ConsDb):
    """Query the ConsDB through the TAP service.

    Parameters
    ----------
    api_base
        Base API for services.
        e.g. https://usdf-rsp.slac.stanford.edu
    token
        The token for authentication.
    query_timeout
        Seconds to wait for a query to complete, in remote service.

    Notes
    -----
    This class provides a `pyvo.dal.TAPService` connection to the Consdb,
    accessible at `ConsDbTap.tap`.
    """

    def __init__(self, api_base: str, token: str, query_timeout: float = 10 * 60):
        url = api_base + "/api/consdbtap"
        self.query_timeout = query_timeout
        cred = pyvo.auth.CredentialStore()
        cred.set_password("x-oauth-basic", token)
        self.credential = cred.get("ivo://ivoa.net/sso#BasicAA")
        self.tap = pyvo.dal.TAPService(url, session=self.credential)

    def __repr__(self) -> str:
        return self.tap.baseurl

    def query(self, query: str) -> pd.DataFrame:
        """Execute TAP ConsDB query.

        Parameters
        ----------
        query
            SQL query.

        Returns
        -------
        results : `pd.DataFrame`
        """
        job = self.tap.submit_job(query)
        job.run()
        job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=self.query_timeout)
        if job.phase == "COMPLETED":
            results = job.fetch_result().to_table().to_pandas()
            job.delete()
        else:
            try:
                job.raise_if_error()
            except DALQueryError as e:
                logger.error(e)
            results = pd.DataFrame([])
        return results


def _fastapi_to_pandas(messages: dict) -> pd.DataFrame:
    # Turn json dictionary returned from consdb into pandas dataframe
    # De-duplicate columns (assumes they are identical).
    # Non-duplication breaks later groupby.
    results = pd.DataFrame(messages["data"], columns=messages["columns"])
    # Check for duplicate columns.
    indices = np.where(pd.Series(results.columns.duplicated()))[0]
    newcols = results.columns.to_list()
    for i in indices:
        newcols[i] = newcols[i] + "_duplicate"
    # Have to change only some instances of the duplicates
    results.columns = newcols
    results.drop(results.columns[indices], axis=1, inplace=True)
    return results


class ConsDbFastAPI(ConsDb):
    """Query the ConsDB through the REST API / FastAPI interface.

    Synchronous queries.

    Parameters
    ----------
    api_base
        Base API for services.
        e.g. https://usdf-rsp.slac.stanford.edu
    auth
        The username and password for authentication.
    query_timeout
        Seconds to wait for the query to return data.

    """

    # From within the USDF RSP, you could also use
    # http://consdb-pq.consdb:8080/ for the ConsDB api_base.
    # This may be slightly faster without F5 load balancer packet checking.
    def __init__(self, api_base: str, auth: tuple, query_timeout: float = 10 * 60) -> None:
        self.base_url = api_base + "/consdb"
        timeout = httpx.Timeout(timeout=query_timeout, connect=60.0)
        transport = httpx.HTTPTransport(retries=2)
        self.httpx_client = httpx.Client(
            base_url=self.base_url, timeout=timeout, transport=transport, auth=auth
        )

    def __repr__(self) -> str:
        return self.base_url

    def close(self) -> None:
        self.httpx_client.close()

    def __enter__(self) -> ConsDbFastAPI:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def query(self, query: str) -> pd.DataFrame:
        """Execute synchronous FastAPI ConsDB query.

        Parameters
        ----------
        query
            SQL query.

        Returns
        -------
        results : `pd.DataFrame`
        """
        params = {"query": query}
        try:
            response = self.httpx_client.post("/query", json=params)
            # We add this little test here because sometimes the consdb
            # FastAPI connections to the consdb itself fall asleep.
            if response.status_code == 500:
                try:
                    sql_problems = response.json()["message"].replace("\n\n", "\n")
                    if "OperationalError" in sql_problems:
                        # Just try again - consdb to FastAPI fell asleep?
                        logger.info(
                            f"Consdb Operational error at "
                            f"{datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')} "
                            f"- trying again."
                        )
                        response = self.httpx_client.post("/query", json=params)
                except JSONDecodeError:
                    error_message = "SQL query error, failing in sqlalchemy not postgres."
                    error_message += " A common issue might be using a single % for wildcards, instead of %%."
                    logger.error(error_message)
            response.raise_for_status()
        except httpx.RequestError as exc:
            error_message = f"An error occurred while requesting {exc.request.url!r}.\n"
            error_message += (
                f"Error at UTC time " f"{datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')}"
            )
            logger.error(error_message)
        except httpx.HTTPStatusError as exc:
            # This might be a problem with the server closing the connection
            # Or it might be a problem with the sql query.
            # All messages from the database are in the response.
            error_message = (
                f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.\n"
            )
            try:
                sql_problems = response.json()["message"].replace("\n\n", "\n")
                error_message += f"{sql_problems}\n"
            except Exception:
                pass
            error_message += (
                f"Error at UTC time " f"{datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')}"
            )
            logger.error(error_message)
        if response.status_code != 200:
            messages = dict()
        else:
            messages = response.json()
        if len(messages) > 0:
            results = _fastapi_to_pandas(messages)
        else:
            results = pd.DataFrame([])
        return results


class AsyncConsDbFastAPI(ConsDb):
    """Query the ConsDB through the REST API / FastAPI interface.

    Asynchronous queries.

    Parameters
    ----------
    api_base
        Base API for services.
        e.g. https://usdf-rsp.slac.stanford.edu
    auth
        The username and password for authentication.
    query_timeout
        Seconds to wait for the query to return data.

    """

    # From within the USDF RSP, you could also use
    # http://consdb-pq.consdb:8080/ for the ConsDB api_base.
    # This may be slightly faster without F5 load balancer packet checking.
    def __init__(self, api_base: str, auth: tuple, query_timeout: float = 10 * 60) -> None:
        self.base_url = api_base + "/consdb"
        timeout = httpx.Timeout(timeout=query_timeout, connect=60.0)
        atransport = httpx.AsyncHTTPTransport(retries=2)
        self.async_client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, transport=atransport, auth=auth
        )

    def __repr__(self) -> str:
        return self.base_url

    async def aclose(self) -> None:
        await self.async_client.aclose()

    async def __aenter__(self) -> AsyncConsDbFastAPI:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.aclose()

    async def query(self, query: str) -> pd.DataFrame:
        """Execute asynchronous FastAPI ConsDB query.

        Parameters
        ----------
        query
            SQL query.

        Returns
        -------
        results : `pd.DataFrame`
        """
        params = {"query": query}
        try:
            response = await self.async_client.post("/query", json=params)
            if response.status_code == 500:
                sql_problems = response.json()["message"].replace("\n\n", "\n")
                if "OperationalError" in sql_problems:
                    # Just try again - consdb to FastAPI fell asleep?
                    logger.info(
                        f"Consdb Operational error at "
                        f"{datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')} "
                        f"- trying again."
                    )
                    response = await self.async_client.post("/query", json=params)
            response.raise_for_status()
        except httpx.RequestError as exc:
            error_message = f"An error occurred while requesting {exc.request.url!r}.\n"
            error_message += (
                f"Error at UTC time " f"{datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')}"
            )
            logger.error(error_message)
        except httpx.HTTPStatusError as exc:
            # This might be a problem with the server closing the connection
            # Or it might be a problem with the sql query.
            # All messages from the database are in the response.
            error_message = (
                f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.\n"
            )
            try:
                sql_problems = response.json()["message"].replace("\n\n", "\n")
                error_message += f"{sql_problems}\n"
            except Exception:
                pass
            error_message += (
                f"Error at UTC time " f"{datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')}"
            )
            logger.error(error_message)
        if response.status_code != 200:
            messages = dict()
        else:
            messages = response.json()
        if len(messages) > 0:
            results = _fastapi_to_pandas(messages)
        else:
            results = pd.DataFrame([])
        return results


class ConsDbSql(ConsDb):
    """Query the ConsDB through pandas with a SQLAlchemy Postgres connection.

    Parameters
    ----------
    site
        Two options for site, to connect directly to the postgres servers,
        either 'usdf' or 'summit'. Note that these postgres servers are
        not exposed outside of the USDF or Summit; you must use
        one of the other ConsDb query services in that case.
    connection_string
        Optional kwarg to explicitly set the connection string instead
        of using the defaults for 'usdf' or 'summit'.

    Notes
    -----
    Credentials must be available in ~/.lsst/postgres-credentials.txt

    For access external to the USDF, base or summit, a different method must
    be used.
    """

    def __init__(self, site: str = "usdf", connection_string: str | None = None) -> None:
        # Authentication for the native postgres connection is via
        # credentials in ~/.lsst/postgres-credentials.txt
        if not HAS_SQLALCHEMY:
            logging.warning(
                "Cannot use ConsDbSql class without installing "
                "SQLAlchemy. Please install sqlalchemy or use a different method."
            )
            return

        if connection_string is None:
            if site.lower() == "summit":
                self.conn_str = "postgresql+psycopg://usdf@postgresdb01.cp.lsst.org/exposurelog"
            else:
                self.conn_str = "postgresql+psycopg://usdf@usdf-summitdb-logical-replica-svc.sdf.slac.stanford.edu/exposurelog"
        else:
            self.conn_str = connection_string

        self.engine = sqlalchemy.create_engine(self.conn_str)

    def __del__(self) -> None:
        self.engine.dispose()

    def __repr__(self) -> str:
        return self.conn_str

    def query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query

        Parameters
        ----------
        query : `str`
            SQL query.

        Returns
        -------
        results : `pd.DataFrame`
        """
        # engine.begin() opens a transaction scoped to this query and commits
        # on success or rolls back on error, so no connection is left idle in
        # transaction once the query returns.
        try:
            with self.engine.begin() as conn:
                result = pd.read_sql(query, conn)
        except ProgrammingError as e:
            logger.error(e)
            result = pd.DataFrame([])
        return result

    async def async_query(self, query: str) -> pd.DataFrame:
        """The ConsDbSql client uses sqlalchemy + pandas.read_sql.
        Async queries are unavailable.
        """
        logger.error("Using sqlalchemy + pandas; async unavailable.")
        raise NotImplementedError
