#!/usr/bin/env python
"""Build the v4 labelling sheet from the approved in-scope corpus.

Fields carry everything the founder specified: name, address, LOAN YEAR, lender
(may be blank), region. Two question types as before.

IDENTITY MATCHING, per the founder's rule: the name field is one string split on
spaces; middle initials are ignored; a match forwards OR backwards is the same
party. HOWARD JOHN F and JOHN HOWARD are one man. Duplicates collapse; loans do not.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import duckdb, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from normalize import normalize_name  # noqa: E402
from splink_contract import SEED      # noqa: E402

_ABBR = [(" STREET", " ST"), (" AVENUE", " AVE"), (" ROAD", " RD"), (" DRIVE", " DR"),
         (" BOULEVARD", " BLVD"), (" HIGHWAY", " HWY"), (" STATE HWY", " HWY"),
         (" INTERSTATE", " I"), (" COUNTY ROAD", " CO RD"), (" COUNTY RD", " CO RD"),
         (" SUITE", " STE"), (" NORTH ", " N "), (" SOUTH ", " S "), (" EAST ", " E "),
         (" WEST ", " W "), (" LANE", " LN"), (" COURT", " CT"), (" PLACE", " PL"),
         (" CIRCLE", " CIR"), (" PARKWAY", " PKWY"), (" TERRACE", " TER"),
         (" POST OFFICE BOX", " PO BOX"), (" P O BOX", " PO BOX")]

_BIZ_HINT = re.compile(r"-|\b(INC|LLC|LLP|LP|CORP|CO|COMPANY|LTD|PLLC|PC|TRUST|"
                       r"PARTNERSHIP|ENTERPRISES|INDUSTRIES|GROUP|SERVICES)\b")


def akey(a: str) -> str:
    if not isinstance(a, str):
        return ""
    s = " " + "".join(c if c.isalnum() else " " for c in a.upper()) + " "
    for lo, sh in _ABBR:
        s = s.replace(lo, sh)
    return " ".join(s.split())


def ikey(name_raw: str) -> str:
    """Identity key. For a person: tokens SORTED with middle initials dropped, so
    forwards and backwards collapse to one key (HOWARD JOHN F == JOHN HOWARD).
    For anything company-like: the normalised name, order preserved."""
    nc, suf = normalize_name(name_raw)
    if not nc:
        return ""
    if suf or _BIZ_HINT.search(str(name_raw).upper()):
        return nc                                   # company: order matters
    t = [x for x in nc.split() if len(x) > 1]       # drop middle initials
    if 2 <= len(t) <= 3:
        return " ".join(sorted(t))                  # person: order does NOT matter
    return nc


con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
df = con.execute("""SELECT region, borrower, borrower_address, borrower_city,
                    borrower_state, borrower_zip, lender, loan_year, route_a, route_b
                    FROM scope_all""").df()
con.close()
df["ikey"] = [ikey(n) for n in df.borrower]
df["akey"] = [akey(a) for a in df.borrower_address]
df = df[df.ikey != ""]

# collapse identity+address; keep the LOAN HISTORY rather than dropping duplicates
ent = (df.groupby(["region", "ikey", "akey"], as_index=False)
         .agg(borrower=("borrower", "first"), address=("borrower_address", "first"),
              city=("borrower_city", "first"), state=("borrower_state", "first"),
              zipc=("borrower_zip", "first"), lender=("lender", "first"),
              first_loan=("loan_year", "min"), last_loan=("loan_year", "max"),
              loans=("loan_year", "size")))
print(f"in-scope rows {len(df):,} -> entities {len(ent):,}  "
      f"({len(df)/max(len(ent),1):.2f} loans each)")
for r in ("CO", "CT"):
    print(f"   {r}: {int((ent.region==r).sum()):,} entities")

d = duckdb.connect(); rows = []
for region in ("CO", "CT"):
    e = ent[ent.region == region]
    d.register("e", e)
    d.execute("""CREATE OR REPLACE TABLE cand AS
      WITH k AS (SELECT *, substr(ikey,1,4) p4 FROM e)
      SELECT x.borrower an,x.address aa,x.city ac,x.state ast,x.zipc az,x.lender al,
             x.first_loan afl,x.last_loan all_,x.loans aln,x.akey aak,x.ikey aik,
             y.borrower bn,y.address ba,y.city bc,y.state bst,y.zipc bz,y.lender bl,
             y.first_loan bfl,y.last_loan bll,y.loans bln,y.akey bak,y.ikey bik,
             jaro_winkler_similarity(x.ikey,y.ikey) sim
      FROM k x JOIN k y ON x.p4=y.p4 OR (x.akey<>'' AND x.akey=y.akey)
      WHERE x.ikey < y.ikey OR (x.ikey=y.ikey AND x.akey < y.akey)""")
    d.unregister("e")
    bands = [("same-identity/diff-address", "sim>=1.0 AND aak<>bak"),
             ("very-high 0.95-1.0", "sim>=0.95 AND sim<1.0"),
             ("high 0.90-0.95", "sim>=0.90 AND sim<0.95"),
             ("medium 0.85-0.90", "sim>=0.85 AND sim<0.90"),
             ("low 0.75-0.85", "sim>=0.75 AND sim<0.85")]
    for bn_, w in bands:
        r = d.execute(f"SELECT * FROM (SELECT * FROM cand WHERE {w}) USING SAMPLE 110 ROWS (reservoir, {SEED})").df()
        r["band"] = bn_; r["question"] = "ENTITY - same firm?"; r["region"] = region
        rows.append(r)
    d.execute("""CREATE OR REPLACE TABLE ap AS SELECT aak a, count(DISTINCT aik) n
                 FROM (SELECT aak,aik FROM cand UNION ALL SELECT bak,bik FROM cand)
                 WHERE aak<>'' GROUP BY 1""")
    r = d.execute(f"""SELECT * FROM (SELECT c.* FROM cand c JOIN ap p ON p.a=c.aak
          WHERE c.aak<>'' AND c.aak=c.bak AND c.sim<0.85) USING SAMPLE 100 ROWS (reservoir, {SEED})""").df()
    r["band"] = "same-address/different-name"; r["question"] = "PREMISES - one operation?"
    r["region"] = region
    rows.append(r)
d.close()

full = pd.concat(rows, ignore_index=True).sort_values(["region", "question", "band"]).reset_index(drop=True)
full.insert(0, "pair_id", [f"{r.region}{i:04d}" for i, r in enumerate(full.itertuples(), 1)])
full.to_csv(ROOT / "sheet_v4_key.csv", index=False)
print(f"\nsheet rows: {len(full)}")
print(full.groupby(["region", "question"]).size().to_string())
