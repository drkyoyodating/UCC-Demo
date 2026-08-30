#!/usr/bin/env python
"""P9-early -- ingest the other free jurisdictions. Connecticut + Philadelphia.

Pulled forward from P9 because the founder wants a CROSS-JURISDICTION labelling
set, and cross-source linkage is the single biggest gap between this demo and
what Tex actually does (310+ sources).

CONNECTICUT data.ct.gov xfev-8smz -- one flat row carrying BOTH parties with full
addresses, plus dt_lapse. Same Socrata pager as Colorado, unchanged.

PHILADELPHIA phl.carto.com rtt_summary -- Carto SQL API, free and anonymous.
document_type='ORIGINAL FINANCING STATEMENT' only: 30,369 rows. grantors/grantees
are SEMICOLON-DELIMITED multi-party free text and must be split before
canonicalisation. No per-party address -- street_address is the COLLATERAL
PROPERTY and is the wrong join key for entity disambiguation, so it is not used
as one. Post-2001 county-recorded UCCs are fixture and real-property-related
filings: a real-estate-collateral population, NOT a general business-lien one.
"""
from __future__ import annotations
import sys, time, gzip, os
from pathlib import Path
import requests, duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_pages"; RAW.mkdir(exist_ok=True)
UA = {"User-Agent": "ucc-demo/1.0 (+https://github.com/drkyoyodating/UCC-Demo)"}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ct():
    d = RAW / "ct"; d.mkdir(exist_ok=True)
    cols = ("id_lien_flng_nbr,lien_status,cd_flng_type,debtor_nm_bus,debtor_ad_str1,"
            "debtor_ad_city,debtor_ad_state,debtor_ad_zip,sec_party_nm_bus,sec_party_ad_str1,"
            "sec_party_ad_city,sec_party_ad_state,sec_party_ad_zip,dt_lapse,dt_accept")
    off, pages, rows = 0, 0, 0
    while True:
        f = d / f"{off:09d}.csv.gz"
        if not f.exists():
            r = requests.get("https://data.ct.gov/resource/xfev-8smz.csv", headers=UA, timeout=300,
                             params={"$select": cols, "$order": "id_lien_flng_nbr",
                                     "$limit": "50000", "$offset": str(off)})
            r.raise_for_status()
            tmp = f.with_suffix(".gz.tmp")
            gzip.open(tmp, "wb").write(r.content); os.replace(tmp, f)
        n = sum(1 for _ in gzip.open(f, "rb")) - 1
        rows += n; pages += 1
        log(f"  CT offset={off:>8,} rows={n:,}")
        if n < 50000: break
        off += 50000
    log(f"CT: {pages} pages, {rows:,} rows")
    return rows


def philly():
    d = RAW / "philly"; d.mkdir(exist_ok=True)
    off, rows = 0, 0
    while True:
        f = d / f"{off:09d}.csv.gz"
        if not f.exists():
            q = ("SELECT document_id,display_date,grantors,grantees,zip_code,street_address "
                 "FROM rtt_summary WHERE document_type='ORIGINAL FINANCING STATEMENT' "
                 f"ORDER BY document_id LIMIT 10000 OFFSET {off}")
            r = requests.get("https://phl.carto.com/api/v2/sql", headers=UA, timeout=300,
                             params={"q": q, "format": "csv"})
            r.raise_for_status()
            tmp = f.with_suffix(".gz.tmp")
            gzip.open(tmp, "wb").write(r.content); os.replace(tmp, f)
        n = sum(1 for _ in gzip.open(f, "rb")) - 1
        rows += n
        log(f"  PHL offset={off:>8,} rows={n:,}")
        if n < 10000: break
        off += 10000
    log(f"PHL: {rows:,} rows")
    return rows


def main():
    ct(); philly()
    con = duckdb.connect(str(ROOT / "ucc.duckdb"))
    con.execute("SET memory_limit='2GB'")
    con.execute(f"""CREATE OR REPLACE TABLE ct_filings AS
        SELECT * FROM read_csv('{RAW/'ct'}/*.csv.gz', header=true, all_varchar=true)""")
    con.execute(f"""CREATE OR REPLACE TABLE phl_filings AS
        SELECT * FROM read_csv('{RAW/'philly'}/*.csv.gz', header=true, all_varchar=true)""")
    for t in ("ct_filings", "phl_filings"):
        log(f"{t}: {con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]:,} rows loaded")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
