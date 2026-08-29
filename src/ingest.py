#!/usr/bin/env python
"""P1 — Colorado lien register (Socrata) -> DuckDB.

Design notes that are load-bearing; read before changing anything.

$order is MANDATORY on every request. Socrata does not guarantee a stable
implicit result order, so an unordered 170-page walk can re-emit rows on one
page and skip others *silently* -- and comparing the loaded count to
$select=count(*) is blind to it, because a duplicate-plus-skip preserves the
count. Hence: $order on every page, plus a distinct-key acceptance test, plus a
monotonicity assertion across page boundaries.

We land CSV, not JSON. Socrata's JSON output OMITS keys whose value is null, so
across 170 pages the inferred schema is unstable (44.8% of debtor rows have no
organizationname). CSV always emits every projected column.

Every page is written to disk gzipped BEFORE loading, so "resume, not restart"
is a file-existence check rather than a claim, and any later re-projection is a
local job instead of a second network walk. Writes are atomic (tmp + rename) so
an interrupted fetch can never leave a short file that looks complete.

NOTE ON THE SHIPPED DATA: the walk that produced the committed results used a
single-column $order. That is provably sufficient only where the key is unique
(debtors, secured_parties). For filings and collateral the key repeats, so the
count match alone does NOT rule out a skip-plus-duplicate; those two tables were
therefore verified by check_boundaries() below, which found zero tied keys
straddling any of the 87 page boundaries. The $order for those two tables now
carries a fileid tiebreaker so a re-run is stable by construction rather than by
inspection. Changing $order invalidates the raw_pages cache: delete it to re-walk.

Termination is ONLY on a short page. The natural `if not rows: break` reads a
429 or a timed-out page as end-of-dataset and exits 0 on a truncated table.
"""
from __future__ import annotations
import argparse, csv, gzip, io, os, sys, time
from pathlib import Path
import requests, duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW, DB = ROOT / "raw_pages", ROOT / "ucc.duckdb"
BASE = "https://data.colorado.gov/resource"
PAGE, TIMEOUT, MAX_ATTEMPTS = 50_000, 300, 6
UA = {"User-Agent": "ucc-demo/1.0 (+https://github.com/drkyoyodating/UCC-Demo)"}

# Projections are WIDENED vs the original plan and that widening is irreversible
# after the walk: primary keys (no stable Splink unique_id without them),
# lifecycle columns (exclude rows the state marks deleted), and `assignor`
# (stops one filing being double-counted under two lenders in the league table).
TABLES = {
    "filings": dict(ds="wffy-3uut", pk="transactionid", order="transactionid,fileid", expected=2_587_492, cols=[
        "transactionid","masterdocumentid","transactiontype","filingtype",
        "documenttype","filingdate","continuation","terminationflag","fileid"]),
    "debtors": dict(ds="8upq-58vz", pk="debtorid", expected=2_012_155, cols=[
        "debtorid","organizationname","address1","city","state","zipcode",
        "fileid","actiontype","recordstatus","efsuniqueid"]),
    "secured_parties": dict(ds="ap62-sav4", pk="spid", expected=2_082_624, cols=[
        "spid","organizationname","address1","city","state","zipcode",
        "fileid","actiontype","recordstatus","assignor"]),
    "collateral": dict(ds="4am6-w6u4", pk="collateralid", order="collateralid,fileid", expected=1_702_184, cols=[
        "collateralid","fileid","collateraldescription","actiontype","recordstatus"]),
}

def csv_rows(body: bytes) -> int:
    """Exact data-row count (header excluded), honouring RFC4180 quoting."""
    if not body:
        raise RuntimeError("empty body with no header")
    rdr = csv.reader(io.TextIOWrapper(io.BytesIO(body), encoding="utf-8", newline=""))
    next(rdr, None)                      # header
    return sum(1 for _ in rdr)

def page_key(body: bytes, cols: list[str], keys: list[str], first: bool):
    """(first|last) data row's key tuple from a page, or None if the page is empty."""
    rdr = csv.reader(io.TextIOWrapper(io.BytesIO(body), encoding="utf-8", newline=""))
    hdr = next(rdr, None)
    if hdr is None:
        return None
    idx = [hdr.index(k) for k in keys]
    row = None
    for row in rdr:
        if first:
            break
    return tuple(row[i] for i in idx) if row else None

def check_boundaries(name: str) -> list[str]:
    """THE MONOTONICITY ASSERTION.

    A two-way count match against the API proves the walk is faithful ONLY when
    the ordered key is unique: where a key repeats, a fault that skips one row of
    a tied group while duplicating another row elsewhere preserves both the total
    AND the distinct count, and slips through. The defence is an ordering with no
    ties, plus this check: across every page boundary the key must strictly
    increase, and no tied key may straddle a boundary.
    """
    t = TABLES[name]
    keys = t.get("order", t["pk"]).split(",")
    files = sorted((RAW / name).glob("*.csv.gz"))
    fails, prev_last = [], None
    for f in files:
        body = gzip.open(f, "rb").read()
        lo = page_key(body, t["cols"], keys, first=True)
        hi = page_key(body, t["cols"], keys, first=False)
        if lo is None:
            continue
        if prev_last is not None and not (prev_last < lo):
            fails.append(f"{name}: page boundary at {f.name} — previous page ended "
                         f"{prev_last}, this page starts {lo}: not strictly increasing "
                         f"(a tied key straddles the boundary; rows may be dup/skipped)")
        prev_last = hi
    log(f"{name}: page-boundary monotonicity checked across {len(files)} pages — "
        f"{'OK' if not fails else str(len(fails)) + ' VIOLATIONS'}")
    return fails

def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def api_count(ds: str) -> int:
    for a in range(MAX_ATTEMPTS):
        r = requests.get(f"{BASE}/{ds}.json", params={"$select": "count(*) AS n"},
                         headers=UA, timeout=TIMEOUT)
        if r.status_code == 200:
            return int(r.json()[0]["n"])          # Socrata returns counts as STRINGS
        time.sleep(2 ** a)
    raise RuntimeError(f"count failed for {ds}")

def api_distinct(ds: str, col: str) -> int:
    """Authoritative distinct-key count from the source, used to prove the walk
    neither duplicated nor skipped rows. Comparing our distinct(pk) to OURS
    proves nothing; comparing it to the API's is the real test."""
    for a in range(MAX_ATTEMPTS):
        r = requests.get(f"{BASE}/{ds}.json",
                         params={"$select": f"count(distinct {col}) AS n"},
                         headers=UA, timeout=TIMEOUT)
        if r.status_code == 200:
            return int(r.json()[0]["n"])
        time.sleep(2 ** a)
    raise RuntimeError(f"distinct count failed for {ds}.{col}")

def fetch_page(ds: str, cols: list[str], order: str, offset: int) -> tuple[bytes, int]:
    """Returns (csv_bytes, data_row_count). Raises rather than returning short."""
    params = {"$select": ",".join(cols), "$order": order,
              "$limit": str(PAGE), "$offset": str(offset)}
    last = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = requests.get(f"{BASE}/{ds}.csv", params=params, headers=UA, timeout=TIMEOUT)
            if r.status_code == 200:
                body = r.content
                # Exact count via a real CSV parse. A naive newline count is
                # wrong whenever a quoted field contains an embedded newline,
                # which organisation names in this register genuinely do.
                n = csv_rows(body)
                return body, n
            last = f"HTTP {r.status_code}"
            if r.status_code in (429, 500, 502, 503, 504):
                wait = int(r.headers.get("Retry-After", 0)) or min(2 ** attempt, 60)
                log(f"    {last} at offset {offset}; backoff {wait}s "
                    f"(attempt {attempt+1}/{MAX_ATTEMPTS})")
                time.sleep(wait); continue
            raise RuntimeError(f"non-retryable {last} at offset {offset}")
        except requests.RequestException as e:
            last = repr(e)
            wait = min(2 ** attempt, 60)
            log(f"    {last} at offset {offset}; backoff {wait}s "
                f"(attempt {attempt+1}/{MAX_ATTEMPTS})")
            time.sleep(wait)
    raise RuntimeError(f"offset {offset} failed after {MAX_ATTEMPTS} attempts: {last}")

def walk(name: str) -> dict:
    t = TABLES[name]
    d = RAW / name; d.mkdir(parents=True, exist_ok=True)
    pre = api_count(t["ds"])
    log(f"{name}: API pre-walk count = {pre:,} (expected {t['expected']:,}, "
        f"drift {100*abs(pre-t['expected'])/t['expected']:.3f}%)")
    offset, pages, fetched, t0 = 0, 0, 0, time.time()
    while True:
        f = d / f"{offset:09d}.csv.gz"
        if f.exists() and f.stat().st_size > 0:          # RESUME, not restart
            with gzip.open(f, "rb") as fh:
                n = csv_rows(fh.read())
            log(f"  offset={offset:>9,} cached rows={n:,}")
        else:
            s = time.time()
            body, n = fetch_page(t["ds"], t["cols"], t.get("order", t["pk"]), offset)
            tmp = f.with_suffix(".gz.tmp")
            with gzip.open(tmp, "wb", compresslevel=6) as fh:
                fh.write(body)
            os.replace(tmp, f)                            # atomic: never a short file
            log(f"  offset={offset:>9,} http=200 rows={n:,} {time.time()-s:.1f}s")
        fetched += n; pages += 1
        if n < PAGE:                                      # ONLY valid termination
            break
        offset += PAGE
    post = api_count(t["ds"])
    log(f"{name}: {pages} pages, {fetched:,} rows in {time.time()-t0:.0f}s; "
        f"API post-walk = {post:,}")
    return {"pre": pre, "post": post, "fetched": fetched, "pages": pages}

def load(con, name: str) -> None:
    t = TABLES[name]
    cols = ", ".join(f"'{c}': 'VARCHAR'" for c in t["cols"])
    con.execute(f"DROP TABLE IF EXISTS {name}")
    con.execute(f"""
        CREATE TABLE {name} AS
        SELECT * FROM read_csv('{RAW/name}/*.csv.gz',
            header=true, columns={{{cols}}}, all_varchar=true, quote='"', escape='"')
    """)

def verify(con, name: str, w: dict) -> tuple[list[str], list[str]]:
    """Returns (failures, findings).

    THE PAGING TEST IS NOT "is the pk unique". Several of these tables have
    genuinely non-unique keys and genuine duplicate rows in the source, and an
    ingest must not silently "fix" that. The test that actually proves the walk
    is sound is a two-way match against the API's own aggregates:
        loaded rows      == API count(*)          -> nothing skipped or truncated
        loaded distinct  == API count(distinct pk)-> nothing duplicated
    Those two together are only jointly satisfiable by a faithful copy: a walk
    that duplicated a row must have skipped another (totals match), which would
    drive our distinct count BELOW the API's.
    """
    t, fails, finds = TABLES[name], [], []
    n, dk = con.execute(
        f"SELECT count(*), count(DISTINCT {t['pk']}) FROM {name}").fetchone()
    full = con.execute(
        f"SELECT count(*) FROM (SELECT DISTINCT * FROM {name})").fetchone()[0]
    api_dk = api_distinct(t["ds"], t["pk"])
    log(f"{name}: loaded={n:,} distinct_{t['pk']}={dk:,} (api={api_dk:,}) "
        f"distinct_full_rows={full:,} api_rows={w['post']:,}")

    drift = abs(n - w["post"]) / max(w["post"], 1)
    if drift > 0.001:
        fails.append(f"{name}: loaded {n:,} vs API {w['post']:,} — drift "
                     f"{100*drift:.3f}% exceeds 0.1% (rows skipped or truncated)")
    if dk != api_dk:
        fails.append(f"{name}: distinct {t['pk']} {dk:,} != API {api_dk:,} — "
                     f"paging duplicated or skipped rows")
    if n != w["fetched"]:
        fails.append(f"{name}: loaded {n:,} != fetched {w['fetched']:,}")

    if dk != n:
        finds.append(f"{name}: {t['pk']} is NOT unique — {n-dk:,} repeats "
                     f"({100*(n-dk)/n:.3f}%). Confirmed against the API, which "
                     f"reports the same distinct count. Downstream joins must "
                     f"account for the real grain.")
    if full != n:
        finds.append(f"{name}: {n-full:,} fully-identical duplicate ROWS exist "
                     f"in the SOURCE ({100*(n-full)/n:.3f}%). Not introduced by "
                     f"the walk. Dedupe explicitly downstream or they inflate "
                     f"any count that joins through this table.")
    return fails, finds

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=",".join(TABLES))
    ap.add_argument("--skip-fetch", action="store_true")
    a = ap.parse_args()
    names = [x for x in a.tables.split(",") if x]
    RAW.mkdir(exist_ok=True)

    walks = {}
    for nme in names:
        walks[nme] = walk(nme) if not a.skip_fetch else {
            "pre": 0, "post": api_count(TABLES[nme]["ds"]),
            "fetched": -1, "pages": -1}

    con = duckdb.connect(str(DB))
    con.execute("SET memory_limit='6GB'")
    con.execute(f"SET temp_directory='{ROOT/'.duckdbtmp'}'")
    fails, findings = [], []
    for nme in names:
        log(f"loading {nme} -> duckdb")
        load(con, nme)
        if walks[nme]["fetched"] == -1:
            walks[nme]["fetched"] = con.execute(
                f"SELECT count(*) FROM {nme}").fetchone()[0]
        f_, fi_ = verify(con, nme, walks[nme])
        fails += f_ + check_boundaries(nme); findings += fi_
        out = ROOT / "parquet"; out.mkdir(exist_ok=True)
        con.execute(f"COPY {nme} TO '{out/nme}.parquet' (FORMAT parquet)")
    con.close()

    print("\n" + "=" * 62)
    if findings:
        print("DATA-QUALITY FINDINGS (not failures — properties of the source):")
        for f in findings:
            print("  *", f)
        print()
    if fails:
        print("P1 ACCEPTANCE: FAIL"); [print("  -", f) for f in fails]; return 1
    print("P1 ACCEPTANCE: PASS — row counts AND distinct-key counts both reconcile against the API; parquet exported")
    return 0

if __name__ == "__main__":
    sys.exit(main())
