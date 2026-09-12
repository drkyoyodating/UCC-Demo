#!/usr/bin/env python
"""Export the five source tables from ucc.duckdb to Parquet, read-only, with a manifest.

Runs under ./.venv/bin/python (duckdb 1.4.5 -- the version that wrote ucc.duckdb). Nothing in
.venv-ml opens the DuckDB file; it reads these Parquet files. The manifest records row counts
(source and Parquet, asserted equal), the sha256 of every Parquet file, the source file's sha256
and size, and the DuckDB version, so a candidate dataset can prove which snapshot it came from.

    ./.venv/bin/python ml/tools/export_snapshot.py --source ucc.duckdb --out ml/data/snapshots/v1

Deliberately stdlib + duckdb only: ucc_ml is not installed in .venv.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import duckdb

#: parquet file stem -> source table
TABLES: dict[str, str] = {
    "co_filings": "filings",
    "co_debtors": "debtors",
    "co_secured_parties": "secured_parties",
    "ct_filings": "ct_filings",
    "scope_all": "scope_all",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(source: Path, out: Path) -> dict:
    source, out = Path(source), Path(out)
    if not source.exists():
        raise FileNotFoundError(source)
    out.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(source)
    con = duckdb.connect(str(source), read_only=True)
    con.execute(f"SET temp_directory='{(out / '.duckdb_tmp').as_posix()}'")   # never spill next to ucc.duckdb
    tables: dict[str, dict] = {}
    try:
        for name, table in TABLES.items():
            described = con.execute(f"DESCRIBE {table}").fetchall()
            columns = [r[0] for r in described]
            column_types = [r[1] for r in described]
            (source_rows,) = con.execute(f"SELECT count(*) FROM {table}").fetchone()
            target = out / f"{name}.parquet"
            # ORDER BY ALL makes a repeat export byte-identical; duplicates are identical rows.
            con.execute(f"COPY (SELECT * FROM {table} ORDER BY ALL) TO '{target.as_posix()}' (FORMAT PARQUET)")
            (parquet_rows,) = con.execute(f"SELECT count(*) FROM read_parquet('{target.as_posix()}')").fetchone()
            if parquet_rows != source_rows:
                raise RuntimeError(f"{name}: wrote {parquet_rows} rows but source has {source_rows}")
            tables[name] = {
                "source_table": table, "columns": columns, "column_types": column_types,
                "row_count": int(source_rows), "sha256": sha256_file(target), "bytes": target.stat().st_size,
            }
            print(f"{name:20s} {source_rows:>12,} rows  -> {target}", flush=True)
    finally:
        con.close()
    manifest = {
        "schema_version": 1,
        "source_path": str(source),
        "source_bytes": source.stat().st_size,
        "source_sha256": source_sha,
        "duckdb_version": duckdb.__version__,
        "python_version": sys.version.split()[0],
        "exported_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tables": tables,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export ucc.duckdb tables to Parquet (read-only) with a manifest.")
    parser.add_argument("--source", required=True, type=Path, help="path to ucc.duckdb")
    parser.add_argument("--out", required=True, type=Path, help="snapshot directory, e.g. ml/data/snapshots/v1")
    ns = parser.parse_args(argv)
    export(ns.source, ns.out)
    print(f"wrote {ns.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
