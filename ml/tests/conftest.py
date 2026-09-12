"""Synthetic snapshot builders. No test in ml/tests opens ucc.duckdb or anything under ml/data/.

The column lists are the live schema of ucc.duckdb (context pack §3, verified 2026-09-12); a
snapshot written here has exactly the columns export_snapshot.py would write, so dataset.py is
tested against the real shape with tiny, hand-built rows.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from ucc_ml.provenance import sha256_file, write_json

SNAPSHOT_COLUMNS: dict[str, list[str]] = {
    "co_filings": ["transactionid", "masterdocumentid", "transactiontype", "filingtype", "documenttype",
                   "filingdate", "continuation", "terminationflag", "fileid"],
    "co_debtors": ["debtorid", "organizationname", "address1", "city", "state", "zipcode", "fileid",
                   "actiontype", "recordstatus", "efsuniqueid"],
    "co_secured_parties": ["spid", "organizationname", "address1", "city", "state", "zipcode", "fileid",
                           "actiontype", "recordstatus", "assignor"],
    "ct_filings": ["id_lien_flng_nbr", "lien_status", "cd_flng_type", "debtor_nm_bus", "debtor_ad_str1",
                   "debtor_ad_city", "debtor_ad_state", "debtor_ad_zip", "sec_party_nm_bus",
                   "sec_party_ad_str1", "sec_party_ad_city", "sec_party_ad_state", "sec_party_ad_zip",
                   "dt_lapse", "dt_accept"],
    "scope_all": ["region", "fileid", "loan_year", "borrower", "borrower_address", "borrower_city",
                  "borrower_state", "borrower_zip", "lender", "route_a", "route_b"],
}
SOURCE_TABLE_NAMES: dict[str, str] = {
    "co_filings": "filings", "co_debtors": "debtors", "co_secured_parties": "secured_parties",
    "ct_filings": "ct_filings", "scope_all": "scope_all",
}
_BOOLEAN = {("scope_all", "route_a"), ("scope_all", "route_b")}


def _frame(name: str, rows: list[dict]) -> pd.DataFrame:
    cols = SNAPSHOT_COLUMNS[name]
    unknown = {k for r in rows for k in r} - set(cols)
    if unknown:
        raise KeyError(f"{name}: unknown columns {sorted(unknown)}")
    return pd.DataFrame({c: [r.get(c) for r in rows] for c in cols}, columns=cols, dtype=object)


def _typed_select(name: str) -> str:
    parts = []
    for c in SNAPSHOT_COLUMNS[name]:
        typ = "BOOLEAN" if (name, c) in _BOOLEAN else "VARCHAR"
        parts.append(f'"{c}"::{typ} AS "{c}"')
    return ", ".join(parts)


def write_snapshot(snapshot_dir: Path, tables: dict[str, list[dict]]) -> Path:
    """Write every snapshot table (missing ones empty) as Parquet plus a manifest.json."""
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    manifest_tables = {}
    try:
        for name in SNAPSHOT_COLUMNS:
            df = _frame(name, tables.get(name, []))
            con.register("_t", df)
            target = snapshot_dir / f"{name}.parquet"
            con.execute(f"COPY (SELECT {_typed_select(name)} FROM _t) TO '{target.as_posix()}' (FORMAT PARQUET)")
            con.unregister("_t")
            manifest_tables[name] = {"source_table": SOURCE_TABLE_NAMES[name], "columns": SNAPSHOT_COLUMNS[name],
                                     "row_count": len(df), "sha256": sha256_file(target),
                                     "bytes": target.stat().st_size}
    finally:
        con.close()
    write_json(snapshot_dir / "manifest.json", {
        "schema_version": 1, "source_path": "synthetic", "source_bytes": 0, "source_sha256": "synthetic",
        "duckdb_version": duckdb.__version__, "python_version": "test", "exported_at": "2026-01-01T00:00:00Z",
        "tables": manifest_tables,
    })
    return snapshot_dir


def build_source_duckdb(path: Path, tables: dict[str, list[dict]]) -> Path:
    """A tiny ucc.duckdb look-alike with the SOURCE table names, for testing export_snapshot.py."""
    path = Path(path)
    con = duckdb.connect(str(path))
    try:
        for name, source_name in SOURCE_TABLE_NAMES.items():
            df = _frame(name, tables.get(name, []))
            con.register("_t", df)
            con.execute(f"CREATE TABLE {source_name} AS SELECT {_typed_select(name)} FROM _t")
            con.unregister("_t")
    finally:
        con.close()
    return path


# --- row builders: every field defaults to a valid, eligible value -------------------------

def co_filing(fileid: str, filingdate: str | None = "2001-02-03T00:00:00.000", master: str | None = None,
              filingtype: str = "ucc", terminationflag: str = "false") -> dict:
    return {"transactionid": f"T-{fileid}", "masterdocumentid": master or f"M-{fileid}",
            "transactiontype": "UCC1", "filingtype": filingtype, "documenttype": "UCC1",
            "filingdate": filingdate, "continuation": "false", "terminationflag": terminationflag,
            "fileid": fileid}


def co_debtor(fileid: str, name: str | None, address: str | None = "1 MAIN ST", city: str | None = "DENVER",
              state: str | None = "CO", zipcode: str | None = "80202", debtorid: str | None = None) -> dict:
    return {"debtorid": debtorid or f"D-{fileid}-{name}", "organizationname": name, "address1": address,
            "city": city, "state": state, "zipcode": zipcode, "fileid": fileid, "actiontype": "ADD",
            "recordstatus": "A", "efsuniqueid": None}


def co_sp(fileid: str, name: str | None) -> dict:
    return {"spid": f"S-{fileid}-{name}", "organizationname": name, "address1": "9 BANK ST", "city": "DENVER",
            "state": "CO", "zipcode": "80202", "fileid": fileid, "actiontype": "ADD", "recordstatus": "A",
            "assignor": None}


def ct_row(lien_id: str | None, name: str | None, address: str | None = "1 MAIN ST", city: str | None = "HARTFORD",
           state: str | None = "CT", zipcode: str | None = "06103", lender: str | None = "FIRST BANK",
           dt_accept: str | None = "2001-02-03T00:00:00.000", status: str = "Active",
           ftype: str = "ORIG FIN STMT") -> dict:
    return {"id_lien_flng_nbr": lien_id, "lien_status": status, "cd_flng_type": ftype, "debtor_nm_bus": name,
            "debtor_ad_str1": address, "debtor_ad_city": city, "debtor_ad_state": state, "debtor_ad_zip": zipcode,
            "sec_party_nm_bus": lender, "sec_party_ad_str1": "9 BANK ST", "sec_party_ad_city": "HARTFORD",
            "sec_party_ad_state": "CT", "sec_party_ad_zip": "06103", "dt_lapse": None, "dt_accept": dt_accept}


def scope_row(region: str, fileid: str, borrower: str, lender: str | None, address: str = "1 MAIN ST",
              city: str = "DENVER", state: str = "CO", zipcode: str = "80202", loan_year: str = "2001",
              route_a: bool = False, route_b: bool = True) -> dict:
    return {"region": region, "fileid": fileid, "loan_year": loan_year, "borrower": borrower,
            "borrower_address": address, "borrower_city": city, "borrower_state": state, "borrower_zip": zipcode,
            "lender": lender, "route_a": route_a, "route_b": route_b}


@pytest.fixture
def snapshot_factory(tmp_path):
    def _make(tables: dict[str, list[dict]], name: str = "snap") -> Path:
        return write_snapshot(tmp_path / name, tables)
    return _make
