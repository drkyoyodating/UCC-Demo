#!/usr/bin/env python
"""Build the MACHINERY-ONLY labelling areas, per the founder's narrowed scope.

The filter is INFERENCE-DRIVEN and this is the point: Colorado stopped coding
collateral after 2012, so we cannot read what a lien is for. But WHO LENT is the
strongest available proxy -- a filing by Caterpillar Financial, John Deere
Construction & Forestry, Kubota Credit or Wagner Equipment (Caterpillar's
Colorado dealer) is construction/heavy-machinery finance whatever the collateral
field says, or does not say.

Explicitly EXCLUDED as other industries: IRS liens, hospital liens, consumer
goods, general accounts-receivable lending, and the agricultural-only crop
categories. Farm CREDIT lenders are kept only where the lender is a machinery
captive (Deere, AGCO, CNH, Kubota all sell heavy equipment); crop and livestock
liens are not machinery.

JURISDICTIONS ARE NEVER CONFLATED. Pairs are generated strictly within a single
register. "There are hundreds of businesses called YoYo in different states" --
a shared name across two registers is not evidence of a shared firm, and asking
a human to judge it from a name alone would manufacture noise and label it truth.
"""
from __future__ import annotations
import sys
from pathlib import Path
import duckdb, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from normalize import normalize_name          # noqa: E402
from splink_contract import SEED              # noqa: E402

PAT = ("DEERE|CATERPILLAR|KOMATSU|BOBCAT|CASE CREDIT|CNH|NEW HOLLAND|AGCO|KUBOTA|VOLVO|TEREX|"
       "JLG|GENIE|VERMEER|DITCH WITCH|WAGNER EQUIPMENT|MACHINERY|EQUIPMENT FINANCE|"
       "EQUIPMENT CO|EQUIPMENT LEASING|CONSTRUCTION EQUIP|HEAVY EQUIP|CRANE|EXCAVAT|TRACTOR|"
       "DE LAGE LANDEN|SNAP ON|SNAP-ON|HITACHI|LIEBHERR|DOOSAN|TAKEUCHI|MANITOWOC|SANY|JCB")
BANDS = [("exact", 1.0, 1.01), ("very-high", 0.95, 1.0), ("high", 0.90, 0.95),
         ("medium", 0.85, 0.90), ("low", 0.75, 0.85)]

con = duckdb.connect(str(ROOT / "ucc.duckdb")); con.execute("SET memory_limit='2GB'")

SPECS = [
  ("CO", "MACHINERY LENDERS", 300, f"""
     SELECT DISTINCT sp.organizationname name_raw, sp.address1, sp.city, sp.state, sp.zipcode
     FROM secured_parties sp JOIN filings f USING(fileid)
     WHERE regexp_matches(upper(sp.organizationname),'{PAT}') AND substr(f.filingdate,1,4)>='1990'"""),
  ("CO", "MACHINERY BORROWERS", 300, f"""
     SELECT DISTINCT d.organizationname name_raw, d.address1, d.city, d.state, d.zipcode
     FROM debtors d JOIN filings f USING(fileid)
     WHERE d.organizationname IS NOT NULL AND trim(d.organizationname)<>''
       AND substr(f.filingdate,1,4)>='1990'
       AND d.fileid IN (SELECT fileid FROM secured_parties
                        WHERE regexp_matches(upper(organizationname),'{PAT}'))"""),
  ("CT", "MACHINERY LENDERS", 150, f"""
     SELECT DISTINCT sec_party_nm_bus name_raw, sec_party_ad_str1 address1,
            sec_party_ad_city city, sec_party_ad_state state, sec_party_ad_zip zipcode
     FROM ct_filings WHERE regexp_matches(upper(sec_party_nm_bus),'{PAT}')"""),
  ("CT", "MACHINERY BORROWERS", 150, f"""
     SELECT DISTINCT debtor_nm_bus name_raw, debtor_ad_str1 address1,
            debtor_ad_city city, debtor_ad_state state, debtor_ad_zip zipcode
     FROM ct_filings WHERE debtor_nm_bus IS NOT NULL
       AND regexp_matches(upper(sec_party_nm_bus),'{PAT}')"""),
  ("PHL", "EQUIPMENT FINANCE", 40, f"""
     SELECT DISTINCT trim(g) name_raw, NULL::VARCHAR address1, 'PHILADELPHIA' city,
            'PA' state, NULL::VARCHAR zipcode
     FROM (SELECT unnest(str_split(grantors,';')) g, grantees FROM phl_filings
           WHERE grantors IS NOT NULL AND regexp_matches(upper(grantees),'{PAT}'))
     WHERE trim(g)<>''"""),
]

d = duckdb.connect(); d.execute("SET memory_limit='2GB'")
out = []
for juris, area, target, sql in SPECS:
    df = con.execute(sql).df()
    n = [normalize_name(x) for x in df.name_raw]
    df["name_clean"] = [a for a, _ in n]
    df = df[df.name_clean.notna()]
    d.register("a", df)
    d.execute("""CREATE OR REPLACE TABLE cand AS
      WITH k AS (SELECT *, substr(name_clean,1,4) p4 FROM a)
      SELECT x.name_raw an,x.address1 aa,x.city ac,x.state ast,x.zipcode az,
             y.name_raw bn,y.address1 ba,y.city bc,y.state bst,y.zipcode bz,
             jaro_winkler_similarity(x.name_clean,y.name_clean) sim
      FROM k x JOIN k y ON x.p4=y.p4
      WHERE x.name_clean < y.name_clean
         OR (x.name_clean = y.name_clean
             AND coalesce(CAST(x.address1 AS VARCHAR),'') < coalesce(CAST(y.address1 AS VARCHAR),''))""")
    d.unregister("a")
    per, got = max(1, target // len(BANDS)), []
    for bname, lo, hi in BANDS:
        r = d.execute(f"""SELECT * FROM (SELECT * FROM cand WHERE sim>={lo} AND sim<{hi})
                          USING SAMPLE {per} ROWS (reservoir, {SEED})""").df()
        r["band"] = bname; got.append(r)
    g = pd.concat(got, ignore_index=True)
    g["jurisdiction"] = juris; g["area"] = area
    out.append(g)
    print(f"  {juris:4s} {area:22s} records {len(df):>7,}  candidates "
          f"{d.execute('SELECT count(*) FROM cand').fetchone()[0]:>10,}  sampled {len(g):>4}")
con.close(); d.close()

full = pd.concat(out, ignore_index=True)
full = full.sort_values(["jurisdiction", "area"]).reset_index(drop=True)
full.insert(0, "pair_id", [f"M{i:04d}" for i in range(1, len(full) + 1)])
full.to_csv(ROOT / "machinery_pairs_key.csv", index=False)
print(f"\nTOTAL NEW PAIRS: {len(full)}")
print(full.groupby(["jurisdiction", "area"]).size().to_string())
