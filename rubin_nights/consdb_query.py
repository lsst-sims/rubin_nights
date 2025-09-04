"""Execute queries for the ConsDB."""

import datetime
import logging

import httpx
import numpy as np
import pandas as pd
import pyvo
from astropy.time import Time

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

    def query(self, query: str) -> pd.DataFrame:
        """The simple query method is implemented in the child classes,
        according to the specific service/interface used to access the ConsDB.
        """
        raise NotImplementedError

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
            f"select *, q.* from  cdb_{instrument}.visit1 "
            f"left join cdb_{instrument}.visit1_quicklook as q "
            f"on visit1.visit_id = q.visit_id "
        )
        constraint = []
        if t_start is not None:
            constraint.append(f" obs_start_mjd >= {t_start.mjd} ")
        if t_end is not None:
            constraint.append(f" obs_start_mjd <= {t_end.mjd} ")
        if visit_constraint is not None:
            constraint.append(f" ({visit_constraint}) ")
        constraint_str = "and".join(constraint)
        if len(constraint_str) > 0:
            query = query + f" where {constraint_str}"
        logger.debug(f"Query executed: {query}")
        visits = self.query(query)

        if len(visits) == 0:
            logger.info(f"No visits for {instrument} retrieved from consdb")
            return pd.DataFrame([])

        if augment:
            visits = augment_visits(visits, instrument)
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
    """

    def __init__(self, api_base: str, token: str):
        url = api_base + "/api/consdbtap"
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
        try:
            results = self.tap.search(query).to_table().to_pandas()
        except Exception as e:
            logger.warning(e)
            results = pd.DataFrame([])
        return results


class ConsDbFastAPI(ConsDb):
    """Query the ConsDB through the REST API / FastAPI interface.

    Parameters
    ----------
    api_base
        Base API for services.
        e.g. https://usdf-rsp.slac.stanford.edu
    auth
        The username and password for authentication.
    query_timeout

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

    def __del__(self) -> None:
        self.httpx_client.close()

    def __repr__(self) -> str:
        return self.base_url

    def query(self, query: str) -> pd.DataFrame:
        """Execute FastAPI ConsDB query.

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
            if response.status_code == 500:
                sql_problems = response.json()["message"].replace("\n\n", "\n")
                if "OperationalError" in sql_problems:
                    # Just try again - consdb to FastAPI fell asleep?
                    logger.info(
                        f"Consdb Operational error at "
                        f"{datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')} "
                        f"- trying again."
                    )
                    response = self.httpx_client.post("/query", json=params)
            response.raise_for_status()
        except httpx.RequestError as exc:
            logger.error(f"An error occurred while requesting {exc.request.url!r}.")
            logger.error(
                f"Error at UTC time {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')}"
            )
        except httpx.HTTPStatusError as exc:
            # This might be a problem with the server closing the connection
            # Or it might be a problem with the sql query.
            # All messages from the database are in the response.
            logger.error(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
            try:
                sql_problems = response.json()["message"].replace("\n\n", "\n")
                logger.error(f"{sql_problems}")
            except Exception:
                pass
            logger.error(
                f"Error at UTC time {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')}"
            )
        if response.status_code != 200:
            messages = dict()
        else:
            messages = response.json()
        if len(messages) > 0:
            results = pd.DataFrame(messages["data"], columns=messages["columns"])
            # Check for duplicate columns.
            indices = np.where(pd.Series(results.columns.duplicated()))[0]
            newcols = results.columns.to_list()
            for i in indices:
                newcols[i] = newcols[i] + "_duplicate"
            # Have to change only some instances of the duplicates
            results.columns = newcols
            results.drop(results.columns[indices], axis=1, inplace=True)
        else:
            results = pd.DataFrame([])
        return results


class ConsDbSql(ConsDb):
    """Query the ConsDB through pandas with a SQLAlchemy Postgres connection.

    Parameters
    ----------
    site
        Two options for site, to connect directly to the postgres servers,
        either "usdf" or "summit". Note that these postgres servers are
        not exposed outside of the USDF or Summit; you must use
        one of the other ConsDb query services in that case.

    Notes
    -----
    Credentials must be available in ~/.lsst/postgres-credentials.txt

    For access external to the USDF or summit, a different access method must
    be used.
    """

    def __init__(self, site: str = "usdf") -> None:
        # Internal to USDF the sql connection string is
        # postgresql://usdf@usdf-summitdb-replica.slac.stanford.edu/exposurelog
        # At summit the sql connection string is
        # postgresql://usdf@postgresdb01.cp.lsst.org/exposurelog

        # Authentication for the native postgres connection is via
        # credentials in ~/.lsst/postgres-credentials.txt
        if not HAS_SQLALCHEMY:
            logging.warning(
                "Cannot use ConsDbSql class without installing "
                "SQLAlchemy. Please install sqlalchemy or use a different method."
            )
            return

        if site.lower() == "summit":
            self.conn_str = "postgresql+psycopg://usdf@postgresdb01.cp.lsst.org/exposurelog"
        else:
            self.conn_str = "postgresql+psycopg://usdf@usdf-summitdb-replica.slac.stanford.edu/exposurelog"

        self.engine = sqlalchemy.create_engine(self.conn_str)
        self.conn = self.engine.connect()

    def __del__(self) -> None:
        self.conn.close()
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
        try:
            result = pd.read_sql(query, self.conn)
        except ProgrammingError as e:
            self.conn.rollback()
            logger.error(e)
        return result
