#!/usr/bin/env python
"""P3 -- working corpora: EQUIPMENT-filing debtors (P5) and secured parties (P5b).

Corpus decision (founder, 2026-08-30): **A -- party rows on EQUIPMENT filings only.**
The alternative ("all rows for any debtor appearing on >=1 EQUIPMENT filing") was
measured at 85,112 rows against A's 37,587 -- 2.3x the comparison space for the
IDENTICAL 24,488 distinct keys, because a debtor is identified by its name. It buys
extra address observations per entity and nothing else, against a 4h P5 budget.

Three things here are not obvious and all three are load-bearing:

1. **Collateral MUST be deduplicated before the join.** P1 established that
   `collateral` carries 3,808 fully-identical duplicate rows in the SOURCE
   (verified against the API's own count(distinct collateralid)). Joining parties
   to it naively multiplies party rows by the duplicate factor and inflates every
   downstream count. We join to `SELECT DISTINCT fileid`, which is immune to both
   the exact duplicates and the legitimate many-collateral-rows-per-file grain.

2. **`is_active` is computed, not filtered.** Rows the state marks deleted or
   inactive stay in the table behind a flag, so the filter is a downstream
   decision that can be changed without rebuilding, and so the counts stay
   reconcilable with the numbers reported at the P3 gate.

3. **The 10% stability holdout is assigned by a SEEDED HASH of the party id, not
   by sampling.** P6 runs resolution twice -- run 1 on the base 90%, run 2 on all
   100% -- to measure canonical-ID churn under a refresh. That number is
   meaningless if the split moves between runs, and `random.sample` or
   `ORDER BY random()` would move it. md5 of "SEED:party_id" is stable across
   processes, machines and reruns; Python's builtin hash() is NOT (it is salted
   per process) and must never be used here.
"""
from __future__ import annotations
import hashlib, sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from normalize import normalize_name          # noqa: E402
from splink_contract import SEED               # noqa: E402

HOLDOUT_PCT = 10


def in_holdout(party_id: str) -> bool:
    """Deterministic 10% assignment. Stable across processes and machines."""
    h = hashlib.md5(f"{SEED}:{party_id}".encode()).hexdigest()
    return int(h[:8], 16) % 100 < HOLDOUT_PCT


def build(con: duckdb.DuckDBPyConnection, source: str, pk: str, extra: str, out: str) -> dict:
    df = con.execute(f"""
        SELECT p.{pk} AS unique_id, p.organizationname AS name_raw,
               p.address1, p.city, p.state, p.zipcode, p.fileid,
               p.actiontype, p.recordstatus, p.{extra} AS extra_col
        FROM {source} p
        WHERE p.fileid IN (SELECT DISTINCT fileid FROM collateral
                           WHERE collateraldescription = 'EQUIPMENT')
          AND p.organizationname IS NOT NULL
          AND trim(p.organizationname) <> ''
    """).df()
    norm = [normalize_name(n) for n in df["name_raw"]]
    df["name_clean"] = [a for a, _ in norm]
    df["suffix"] = [b for _, b in norm]
    df["is_active"] = (
        df["recordstatus"].eq("active")
        & ~df["actiontype"].isin(["delete only", "change and delete"])
    )
    df["is_holdout"] = [in_holdout(str(u)) for u in df["unique_id"]]
    df = df[df["name_clean"].notna()]          # a key-less row cannot be blocked

    con.register("_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {out} AS SELECT * FROM _df")
    con.unregister("_df")
    con.execute(f"COPY {out} TO '{ROOT/'parquet'/out}.parquet' (FORMAT parquet)")

    act = df[df.is_active]
    return {
        "rows": len(df),
        "active_rows": len(act),
        "distinct_raw": df.name_raw.nunique(),
        "distinct_key": df.name_clean.nunique(),
        "active_distinct_key": act.name_clean.nunique(),
        "holdout_rows": int(df.is_holdout.sum()),
        "base_rows": int((~df.is_holdout).sum()),
        "with_suffix": int(df.suffix.notna().sum()),
        "zip_present": float(df.zipcode.notna().mean()),
        "addr_present": float(df.address1.notna().mean()),
    }


def main() -> int:
    con = duckdb.connect(str(ROOT / "ucc.duckdb"))
    con.execute("SET memory_limit='2GB'")
    eq_files = con.execute(
        "SELECT count(DISTINCT fileid) FROM collateral WHERE collateraldescription='EQUIPMENT'"
    ).fetchone()[0]
    print(f"EQUIPMENT filings (distinct fileid, dedup-safe): {eq_files:,}\n")

    specs = [("debtors", "debtorid", "efsuniqueid", "corpus_debtors_eq"),
             ("secured_parties", "spid", "assignor", "corpus_lenders_eq")]
    stats = {}
    for src, pk, extra, out in specs:
        s = build(con, src, pk, extra, out)
        stats[out] = s
        print(f"{out}")
        print(f"   rows                {s['rows']:>8,}   (active {s['active_rows']:,})")
        print(f"   distinct raw name   {s['distinct_raw']:>8,}")
        print(f"   distinct name_clean {s['distinct_key']:>8,}   (active {s['active_distinct_key']:,})")
        print(f"   base / holdout      {s['base_rows']:>8,} / {s['holdout_rows']:,} "
              f"({100*s['holdout_rows']/max(s['rows'],1):.2f}%)")
        print(f"   suffix present      {s['with_suffix']:>8,} ({100*s['with_suffix']/max(s['rows'],1):.1f}%)")
        print(f"   address1 / zipcode  {100*s['addr_present']:>7.1f}% / {100*s['zip_present']:.1f}%\n")

    fails = []
    for out, s in stats.items():
        if not 8.0 <= 100 * s["holdout_rows"] / max(s["rows"], 1) <= 12.0:
            fails.append(f"{out}: holdout {100*s['holdout_rows']/s['rows']:.2f}% outside 8-12%")
        if s["rows"] == 0:
            fails.append(f"{out}: empty")
        n_null = con.execute(f"SELECT count(*) FROM {out} WHERE name_clean IS NULL").fetchone()[0]
        if n_null:
            fails.append(f"{out}: {n_null} rows with NULL name_clean survived")
        n_dup = con.execute(
            f"SELECT count(*)-count(DISTINCT unique_id) FROM {out}").fetchone()[0]
        if n_dup:
            fails.append(f"{out}: unique_id is not unique ({n_dup} repeats) -- Splink requires it")
    con.close()
    print("=" * 62)
    if fails:
        print("P3 ACCEPTANCE: FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("P3 ACCEPTANCE: PASS -- both corpora built, unique_id unique, "
          "holdout seeded and in range, parquet exported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
