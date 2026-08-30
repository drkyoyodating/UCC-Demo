#!/usr/bin/env python
"""Generate labelling pairs for each new AREA, kept strictly WITHIN jurisdiction.

FOUNDER DIRECTIVE, and it is a correct one: never generate cross-jurisdiction
pairs. "There are hundreds of businesses called YoYo in the US in different
states" -- a shared name across two registers is NOT evidence of a shared firm,
and asking a human to judge it from a name alone would manufacture noise and
call it ground truth. Cross-source linkage needs corroborating evidence
(a shared address, a shared officer, a filing that references the other), which
this data does not carry. Each jurisdiction is sampled and labelled on its own.

Pairs are stratified on NAME SIMILARITY rather than model score: most of these
areas have no trained model, and ground truth must not be sampled by the thing
it will later judge.
"""
from __future__ import annotations
import sys
from pathlib import Path
import duckdb, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from splink_contract import SEED  # noqa: E402

AREAS = [
    ("area_eqlender",  "CO - ACTIVE EQUIPMENT LENDERS", 60, True),
    ("area_inventory", "CO - INVENTORY (DEALER FLOORPLAN)", 60, True),
    ("area_agri",      "CO - AGRICULTURAL (CURRENT)", 50, True),
    ("area_ct",        "CONNECTICUT - SECURED PARTIES", 60, True),
    ("area_phl",       "PHILADELPHIA - SECURED PARTIES", 40, False),
]
BANDS = [("exact", 1.0, 1.01), ("very-high", 0.95, 1.0), ("high", 0.90, 0.95),
         ("medium", 0.85, 0.90), ("low", 0.75, 0.85)]

d = duckdb.connect(); d.execute("SET memory_limit='2GB'")
out = []
for tbl, label, n_target, has_addr in AREAS:
    d.execute(f"CREATE OR REPLACE TABLE a AS SELECT * FROM '{ROOT}/parquet/{tbl}.parquet'")
    # candidate pairs: share a 4-char name prefix (and, where available, a ZIP)
    join = "x.p4 = y.p4"
    if has_addr:
        join += " OR (x.zipcode IS NOT NULL AND x.zipcode = y.zipcode)"
    d.execute(f"""CREATE OR REPLACE TABLE cand AS
        WITH k AS (SELECT *, substr(name_clean,1,4) p4 FROM a)
        SELECT x.name_raw an, x.address1 aa, x.city ac, x.state ast, x.zipcode az, x.name_clean anc,
               y.name_raw bn, y.address1 ba, y.city bc, y.state bst, y.zipcode bz, y.name_clean bnc,
               jaro_winkler_similarity(x.name_clean, y.name_clean) sim
        FROM k x JOIN k y ON ({join})
        WHERE (x.name_clean < y.name_clean)
           OR (x.name_clean = y.name_clean AND coalesce(CAST(x.address1 AS VARCHAR),'') < coalesce(CAST(y.address1 AS VARCHAR),''))""")
    tot = d.execute("SELECT count(*) FROM cand").fetchone()[0]
    per = max(1, n_target // len(BANDS))
    got = []
    for bname, lo, hi in BANDS:
        # USING SAMPLE binds to the table scan, not the filtered result, so the
        # filter must be pushed into a subquery or it samples then filters and
        # returns almost nothing. This cost one run to discover.
        rows = d.execute(f"""SELECT * FROM (
                               SELECT * FROM cand WHERE sim >= {lo} AND sim < {hi}
                             ) USING SAMPLE {per} ROWS (reservoir, {SEED})""").df()
        rows["band"] = bname
        got.append(rows)
    df = pd.concat(got, ignore_index=True) if got else pd.DataFrame()
    df["area"] = label
    out.append(df)
    print(f"  {label:36s} candidates {tot:>10,}   sampled {len(df):>3}")
d.close()

full = pd.concat(out, ignore_index=True)
full = full.sample(frac=1.0, random_state=SEED + 7).reset_index(drop=True)
full.insert(0, "pair_id", [f"X{i:03d}" for i in range(1, len(full) + 1)])
full.to_csv(ROOT / "area_pairs_key.csv", index=False)
print(f"\ntotal new pairs: {len(full)}")
print(full.groupby(["area", "band"]).size().to_string())
