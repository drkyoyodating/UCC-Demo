#!/usr/bin/env python
"""Build the six labelling AREAS for the expanded set.

Areas 1-3 are Colorado, re-scoped to the machinery-relevant universe P7 now
targets. Areas 4-5 are the two new jurisdictions. Area 6 is the cross-source
linkage test -- the same firm appearing in two different states' registers -- and
is the skill no other portfolio piece will demonstrate.

Pairs are stratified on NAME SIMILARITY, not model score, because most of these
areas have no trained model. That is the right call: the labels establish ground
truth, and ground truth must not be sampled by the thing it will judge.
"""
from __future__ import annotations
import sys
from pathlib import Path
import duckdb, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from normalize import normalize_name          # noqa: E402

con = duckdb.connect(str(ROOT / "ucc.duckdb"))
con.execute("SET memory_limit='2GB'")


def norm_df(df, namecol):
    n = [normalize_name(x) for x in df[namecol]]
    df = df.copy()
    df["name_clean"] = [a for a, _ in n]
    df["suffix"] = [b for _, b in n]
    return df[df.name_clean.notna()]


# ---- CO area 1: the 747 inferred active equipment lenders, 2013+ ----------
con.execute("""CREATE OR REPLACE TABLE area_eqlender AS
WITH eq AS (SELECT DISTINCT organizationname nm FROM secured_parties
            WHERE fileid IN (SELECT DISTINCT fileid FROM collateral
                             WHERE collateraldescription='EQUIPMENT')
              AND organizationname IS NOT NULL)
SELECT DISTINCT sp.organizationname name_raw, sp.address1, sp.city, sp.state, sp.zipcode
FROM secured_parties sp JOIN filings f USING(fileid)
WHERE sp.organizationname IN (SELECT nm FROM eq)
  AND substr(f.filingdate,1,4)>='2013'""")

# ---- CO area 2: INVENTORY (dealer floorplan) ------------------------------
con.execute("""CREATE OR REPLACE TABLE area_inventory AS
SELECT DISTINCT d.organizationname name_raw, d.address1, d.city, d.state, d.zipcode
FROM debtors d WHERE d.organizationname IS NOT NULL AND trim(d.organizationname)<>''
  AND d.fileid IN (SELECT DISTINCT fileid FROM collateral
                   WHERE collateraldescription='INVENTORY')""")

# ---- CO area 3: agricultural, still filed today ---------------------------
con.execute("""CREATE OR REPLACE TABLE area_agri AS
SELECT DISTINCT d.organizationname name_raw, d.address1, d.city, d.state, d.zipcode
FROM debtors d JOIN filings f USING(fileid)
WHERE d.organizationname IS NOT NULL AND trim(d.organizationname)<>''
  AND substr(f.filingdate,1,4)>='2013'
  AND d.fileid IN (SELECT DISTINCT fileid FROM collateral WHERE collateraldescription IN
      ('CATTLE AND CALVES','ALL LIVESTOCK','FARM PRODUCTS','ALL FIELD CROPS','WHEAT','CORN','HAY'))""")

# ---- area 4: Connecticut (both parties on one row) ------------------------
con.execute("""CREATE OR REPLACE TABLE area_ct AS
SELECT DISTINCT sec_party_nm_bus name_raw, sec_party_ad_str1 address1,
       sec_party_ad_city city, sec_party_ad_state state, sec_party_ad_zip zipcode
FROM ct_filings WHERE sec_party_nm_bus IS NOT NULL AND trim(sec_party_nm_bus)<>''""")

# ---- area 5: Philadelphia (semicolon-delimited, NO party address) ---------
con.execute("""CREATE OR REPLACE TABLE area_phl AS
SELECT DISTINCT trim(g) name_raw, NULL address1, 'PHILADELPHIA' city, 'PA' state, NULL zipcode
FROM (SELECT unnest(str_split(grantees,';')) g FROM phl_filings WHERE grantees IS NOT NULL)
WHERE trim(g)<>''""")

for t in ("area_eqlender", "area_inventory", "area_agri", "area_ct", "area_phl"):
    print(f"  {t:18s} {con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]:>9,} distinct records")

frames = {}
for t in ("area_eqlender", "area_inventory", "area_agri", "area_ct", "area_phl"):
    df = norm_df(con.execute(f"SELECT * FROM {t}").df(), "name_raw")
    frames[t] = df
    print(f"  {t:18s} -> {len(df):>9,} normalised, {df.name_clean.nunique():>8,} distinct keys")
con.close()

out = ROOT / "parquet"
d = duckdb.connect()
for k, v in frames.items():
    d.register("_f", v)
    d.execute(f"COPY (SELECT * FROM _f) TO '{out}/{k}.parquet' (FORMAT parquet)")
    d.unregister("_f")
d.close()
print("\nwrote parquet/area_*.parquet")
