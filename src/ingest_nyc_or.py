#!/usr/bin/env python
"""Ingest NYC ACRIS + Oregon. Completes the four-jurisdiction set (Philadelphia
excluded by the founder: only 8 machinery-relevant records, its register is
real-property fixture filings not equipment finance).

NYC ACRIS -- nbbg-wtuz personal-property PARTIES (11,035,386 rows) and sv7x-dduq
MASTER (4,547,264). Roles resolve against 7isb-wh4c: party1_type=DEBTOR,
party2_type=SECURED PARTY. Joined on document_id.

OREGON -- 2kf7-i54h is SECURED PARTIES ONLY, no debtor field (a correction to the
project's earlier settled record). It carries lapse_date, which is the one thing
Colorado lacks.
"""
from __future__ import annotations
import gzip, os, sys, time
from pathlib import Path
import requests, duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_pages"
UA = {"User-Agent": "ucc-demo/1.0 (+https://github.com/drkyoyodating/UCC-Demo)"}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def walk(name, url, cols, order, page=50000):
    d = RAW / name; d.mkdir(parents=True, exist_ok=True)
    off, rows, t0 = 0, 0, time.time()
    while True:
        f = d / f"{off:09d}.csv.gz"
        if not f.exists():
            for attempt in range(6):
                try:
                    r = requests.get(url, headers=UA, timeout=300,
                                     params={"$select": cols, "$order": order,
                                             "$limit": str(page), "$offset": str(off)})
                    if r.status_code == 200:
                        break
                    time.sleep(min(2 ** attempt, 30))
                except requests.RequestException:
                    time.sleep(min(2 ** attempt, 30))
            else:
                raise RuntimeError(f"{name} offset {off} failed")
            tmp = f.with_suffix(".gz.tmp")
            gzip.open(tmp, "wb").write(r.content); os.replace(tmp, f)
        n = sum(1 for _ in gzip.open(f, "rb")) - 1
        rows += n
        if (off // page) % 20 == 0:
            log(f"  {name} offset={off:>10,} rows={rows:,} ({time.time()-t0:.0f}s)")
        if n < page:
            break
        off += page
    log(f"{name}: {rows:,} rows in {time.time()-t0:.0f}s")
    return rows


def main():
    walk("nyc_parties", "https://data.cityofnewyork.us/resource/nbbg-wtuz.csv",
         "document_id,party_type,name,address_1,city,state,zip", "document_id")
    walk("nyc_master", "https://data.cityofnewyork.us/resource/sv7x-dduq.csv",
         "document_id,doc_type,recorded_datetime,ucc_collateral,document_amt,recorded_borough", "document_id")
    walk("or_sp", "https://data.oregon.gov/resource/2kf7-i54h.csv",
         "lt_cd,filenumber,filing_date,secured_party,addr1,city,state,postalcode", ":id")

    con = duckdb.connect(str(ROOT / "ucc.duckdb")); con.execute("SET memory_limit='3GB'")
    for t, p in [("nyc_parties", "nyc_parties"), ("nyc_master", "nyc_master"), ("or_sp", "or_sp")]:
        con.execute(f"""CREATE OR REPLACE TABLE {t} AS
            SELECT * FROM read_csv('{RAW/p}/*.csv.gz', header=true, all_varchar=true, ignore_errors=true)""")
        log(f"{t}: {con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]:,} loaded")
    log("NYC doc_type mix:")
    for dt, k in con.execute("""SELECT doc_type, count(*) k FROM nyc_master
                                GROUP BY 1 ORDER BY k DESC LIMIT 12""").fetchall():
        log(f"    {str(dt):14s} {k:,}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
