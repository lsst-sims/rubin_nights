import argparse
import logging
import sqlite3

import pandas as pd

import rubin_nights.rubin_sim_addons as rsim
from rubin_nights import connections

logging.getLogger(__name__).setLevel(logging.INFO)


def query_consdb_to_opsim(tokenfile: str | None = None, site: str | None = None) -> pd.DataFrame:
    endpoints = connections.get_clients(tokenfile, site)

    instrument = "lsstcam"
    query = (
        f"select v.*, q.* from  cdb_{instrument}.visit1 as v "
        f"left join cdb_{instrument}.visit1_quicklook as q "
        f"on v.visit_id = q.visit_id "
        f"where v.img_type != 'bias' and v.img_type != 'flat' "
        f"and v.s_ra is not NULL "
        f"and v.day_obs >= 20250415 "
    )

    consdb_visits = endpoints["consdb"].query(query)
    consdb_visits = endpoints["consdb"].augment_visits(consdb_visits)
    opsim = rsim.consdb_to_opsim(consdb_visits)
    return opsim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenfile", type=str, default=None)
    parser.add_argument("--site", type=str, default=None)
    parser.add_argument("--outfile", type=str, default=None)
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    opsim = query_consdb_to_opsim(args.tokenfile, args.site)

    if args.outfile is None:
        args.outfile = "consdb_opsim.db"
    connection = sqlite3.connect(args.outfile)
    opsim.to_sql("observations", connection, index=False)
    connection.close()
