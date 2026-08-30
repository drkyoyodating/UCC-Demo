#!/usr/bin/env python
"""Build the per-jurisdiction labelling workbook. 300 pairs per jurisdiction.

TWO CORRECTIONS THE FOUNDER CALLED OUT, both about not wasting his time:

1. FORMATTING TWINS ARE NOT A LABELLING PROBLEM. "Kubota Credit Corporation,
   U.S.A." vs "KUBOTA CREDIT CORPORATION, U.S.A." differ only in case; the
   normaliser already collapses them. Records are therefore deduplicated on
   (name_clean, address_clean, city, zip) with the ADDRESS normalised too, so
   pairs that differ only in punctuation or spacing never reach the sheet.
   Only genuine judgement calls are asked for.

2. JURISDICTIONS ARE NEVER MIXED. One sheet per register. A shared name across
   two registers is not evidence of a shared firm.

TWO QUESTION TYPES, and they are different questions:
  ENTITY   -- "same firm?"        -> SAME / DIFFERENT / UNSURE
  PREMISES -- "one operation?"    -> ONE-OP / SEPARATE / UNSURE
The premises rows exist because WOOD DONNA L and WOOD DONALD J at one PO box are
two legal borrowers and one operation. Merging them is wrong; discarding them
loses the relationship. Asking a different question captures it.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import duckdb, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from normalize import normalize_name          # noqa: E402
from splink_contract import SEED              # noqa: E402

LEND = ("CATERPILLAR|KOMATSU|BOBCAT|TEREX|JLG|GENIE INDUST|VERMEER|DITCH WITCH|HITACHI|LIEBHERR|"
        "DOOSAN|TAKEUCHI|MANITOWOC|SANY|JCB |KUBOTA|CNH |CASE CREDIT|NEW HOLLAND|VOLVO|"
        "JOHN DEERE CONSTRUCTION|DEERE CONSTRUCTION|WAGNER EQUIPMENT|MACHINERY|"
        "CONSTRUCTION EQUIP|HEAVY EQUIP|CRANE|FARIS |POWER EQUIPMENT|4 RIVERS")
BORR = ("EXCAVAT|PAVING|GRADING|CONCRETE|CONSTRUCTION|DRILLING|TRENCH|DEMOLITION|ASPHALT|"
        "EARTHWORK|SITEWORK|CRANE|AGGREGATE|QUARRY|PIPELINE|UNDERGROUND UTIL|BACKHOE|DOZER|"
        "GRAVEL|READY MIX|MASONRY|LANDSCAP|SEPTIC|WELL DRILL|BUILDERS|CONTRACT")

_ABBR = {r"\bSTREET\b": "ST", r"\bAVENUE\b": "AVE", r"\bROAD\b": "RD", r"\bDRIVE\b": "DR",
         r"\bBOULEVARD\b": "BLVD", r"\bNORTH\b": "N", r"\bSOUTH\b": "S", r"\bEAST\b": "E",
         r"\bWEST\b": "W", r"\bSUITE\b": "STE", r"\bPOST OFFICE BOX\b": "PO BOX",
         r"\bP\s*O\s*BOX\b": "PO BOX", r"\bCOUNTY ROAD\b": "CO RD", r"\bHIGHWAY\b": "HWY"}


#: Address values that are not addresses. Verified present in the register.
_JUNK_ADDR = {"", "NONE", "NONE PROVIDED", "NA", "N A", "UNKNOWN", "SAME", "COMPANY",
              "NOT PROVIDED", "NOT AVAILABLE", "ADDRESS UNKNOWN", "NO ADDRESS", "X", "XX",
              "TBD", "PO BOX", "BOX", "SEE ATTACHED", "VARIOUS"}


def addr_clean(a):
    if not isinstance(a, str) or not a.strip():
        return ""
    s = re.sub(r"[^A-Z0-9 ]", " ", a.upper())
    for pat, rep in _ABBR.items():
        s = re.sub(pat, rep, s)
    s = " ".join(s.split())
    return "" if s in _JUNK_ADDR else s


_MIDDLE = re.compile(r"^([A-Z]+) ([A-Z]+)(?: ([A-Z])| ([A-Z]{1,2}))?$")


def person_key(name_clean: str) -> str | None:
    """If the name looks like a person, return FIRST+LAST with any middle initial
    dropped. Used ONLY for duplicate removal, never to exclude anyone.

    Founder's rule: two records with the SAME first and last name -- with or
    without a middle initial -- at the SAME address in the SAME state are one
    person filed twice, and the second copy dilutes the results. WOOD DONNA L and
    WOOD DONNA are one record. WOOD DONNA L and WOOD DONALD J are two people and
    both are kept.
    """
    if not isinstance(name_clean, str):
        return None
    t = name_clean.split()
    if not (2 <= len(t) <= 3):
        return None
    if len(t) == 3 and len(t[2]) > 2:          # third token is a word, not an initial
        return None
    return f"{t[0]}|{t[1]}"


def records(con, sql):
    df = con.execute(sql).df()
    n = [normalize_name(x) for x in df.name_raw]
    df["name_clean"] = [a for a, _ in n]
    df["suffix"] = [b or "" for _, b in n]
    df["addr_clean"] = [addr_clean(a) for a in df.address1]
    df["city_c"] = [str(c).upper().strip() if isinstance(c, str) else "" for c in df.city]
    df = df[df.name_clean.notna()]
    # DEDUP KEY, per the founder's criteria:
    #   case            -> IGNORED (normalize_name uppercases)
    #   direct spelling -> MATTERS (nothing is fuzzy-collapsed; AMEERICAN stays AMEERICAN)
    #   address         -> MATTERS (normalised for formatting only: STREET->ST, P O BOX->PO BOX)
    #   ENTITY TYPE     -> MATTERS  <- `suffix` was MISSING from this key, so ACME LLC and
    #                      ACME INC at one address were collapsing into one record and the
    #                      pair never reached the sheet. That is rule R3's whole subject.
    #   last name       -> MATTERS (carried inside name_clean; nothing strips it)
    df = df.drop_duplicates(
        subset=["name_clean", "suffix", "addr_clean", "city_c", "zipcode"])
    # PERSON DUPLICATE REMOVAL. Same first+last (middle initial ignored), same
    # address, same state -> one person filed twice. Drop the copy. Individuals are
    # never excluded from the corpus; only exact re-filings of the same individual
    # at the same place are collapsed, so results are not diluted by doubles.
    pk = [person_key(n) for n in df.name_clean]
    df = df.assign(_pk=pk)
    ppl = df[df._pk.notna()].drop_duplicates(subset=["_pk", "addr_clean", "city_c", "zipcode"])
    orgs = df[df._pk.isna()]
    return pd.concat([ppl, orgs], ignore_index=True).drop(columns=["_pk"])


def pairs(d, df, n_entity, n_premises):
    d.register("a", df)
    d.execute("""CREATE OR REPLACE TABLE cand AS
      WITH k AS (SELECT *, substr(name_clean,1,4) p4 FROM a)
      SELECT x.name_raw an,x.address1 aa,x.city_c ac,x.zipcode az,x.name_clean anc,
             x.addr_clean aac,x.suffix asuf,
             y.name_raw bn,y.address1 ba,y.city_c bc,y.zipcode bz,y.name_clean bnc,
             y.addr_clean bac,y.suffix bsuf,
             jaro_winkler_similarity(x.name_clean,y.name_clean) sim
      FROM k x JOIN k y ON x.p4=y.p4 OR (x.addr_clean<>'' AND x.addr_clean=y.addr_clean)
      WHERE x.name_clean < y.name_clean
         OR (x.name_clean = y.name_clean AND x.suffix < y.suffix)
         OR (x.name_clean = y.name_clean AND x.suffix = y.suffix AND x.addr_clean < y.addr_clean)""")
    d.unregister("a")
    got = []
    # ENTITY question: stratified across name-similarity, formatting twins excluded
    bands = [("same-name/different-ENTITY-TYPE", "sim>=1.0 AND asuf<>bsuf"),
             ("exact-name/diff-address", "sim>=1.0 AND asuf=bsuf AND aac<>bac"),
             ("very-high 0.95-1.0", "sim>=0.95 AND sim<1.0"),
             ("high 0.90-0.95", "sim>=0.90 AND sim<0.95"),
             ("medium 0.85-0.90", "sim>=0.85 AND sim<0.90"),
             ("low 0.75-0.85", "sim>=0.75 AND sim<0.85")]
    per = max(1, n_entity // len(bands))
    for bn_, w in bands:
        r = d.execute(f"""SELECT * FROM (SELECT * FROM cand WHERE {w})
                          USING SAMPLE {per} ROWS (reservoir, {SEED})""").df()
        r["band"] = bn_; r["question"] = "ENTITY - same firm?"; got.append(r)
    # PREMISES question: identical normalised address, clearly different names.
    #
    # CRITICAL FILTER. The most common addresses in this register are NOT premises --
    # they are registered agents and filing services. "950 TECHNOLOGY WAY, SUITE 301"
    # appears 92,610 times in Colorado AND 13,162 times in Connecticut; an address
    # spanning two state registers is an agent, not a yard. Likewise "330 N BRAND
    # BLVD, SUITE 700; ATTN: SPRS" and "PO BOX 2046". Including them would fill the
    # premises sample with pairs whose answer is trivially SEPARATE and teach nothing.
    #
    # A genuine shared operation carries a handful of names. An office building or a
    # filing agent carries hundreds. Cap at 6 distinct firms per address.
    d.execute("""CREATE OR REPLACE TABLE addr_pop AS
        SELECT aac a, count(DISTINCT anc) n FROM (
            SELECT aac, anc FROM cand UNION ALL SELECT bac, bnc FROM cand)
        WHERE aac <> '' GROUP BY 1""")
    # USING SAMPLE binds to the table scan, not the filtered/joined result, so the
    # whole selection must sit inside a subquery. This is the second time this has
    # bitten; it silently returned zero rows rather than erroring.
    n_avail = d.execute("""SELECT count(*) FROM cand c JOIN addr_pop p ON p.a=c.aac
          WHERE c.aac<>'' AND c.aac=c.bac AND c.sim<0.85 AND p.n<=6""").fetchone()[0]
    r = d.execute(f"""SELECT * FROM (
            SELECT c.* FROM cand c JOIN addr_pop p ON p.a = c.aac
            WHERE c.aac<>'' AND c.aac=c.bac AND c.sim<0.85 AND p.n <= 6
          ) USING SAMPLE {n_premises} ROWS (reservoir, {SEED})""").df()
    print(f"      premises candidates after agent filter: {n_avail:,} -> sampled {len(r)}")
    r["band"] = "same-address/different-name"; r["question"] = "PREMISES - one operation?"
    got.append(r)
    return pd.concat(got, ignore_index=True)


con = duckdb.connect(str(ROOT / "ucc.duckdb")); con.execute("SET memory_limit='2GB'")
d = duckdb.connect(); d.execute("SET memory_limit='2GB'")

JURIS = {
 "COLORADO": f"""
   SELECT DISTINCT p.organizationname name_raw, p.address1, p.city, p.zipcode FROM (
     SELECT sp.organizationname, sp.address1, sp.city, sp.zipcode, sp.fileid FROM secured_parties sp
       WHERE regexp_matches(upper(sp.organizationname),'{LEND}')
     UNION ALL
     SELECT dd.organizationname, dd.address1, dd.city, dd.zipcode, dd.fileid FROM debtors dd
       WHERE dd.organizationname IS NOT NULL AND (
         regexp_matches(upper(dd.organizationname),'{BORR}')
         OR dd.fileid IN (SELECT fileid FROM secured_parties
                          WHERE regexp_matches(upper(organizationname),'{LEND}')))
   ) p JOIN filings f ON f.fileid=p.fileid WHERE substr(f.filingdate,1,4)>='1990'""",
 "CONNECTICUT": f"""
   SELECT DISTINCT name_raw, address1, city, zipcode FROM (
     SELECT sec_party_nm_bus name_raw, sec_party_ad_str1 address1, sec_party_ad_city city,
            sec_party_ad_zip zipcode FROM ct_filings
       WHERE regexp_matches(upper(sec_party_nm_bus),'{LEND}')
         AND substr(dt_accept,1,4) >= '1990'
     UNION ALL
     SELECT debtor_nm_bus, debtor_ad_str1, debtor_ad_city, debtor_ad_zip FROM ct_filings
       WHERE debtor_nm_bus IS NOT NULL AND substr(dt_accept,1,4) >= '1990'
         AND (regexp_matches(upper(debtor_nm_bus),'{BORR}')
         OR regexp_matches(upper(sec_party_nm_bus),'{LEND}')))""",
}

out = []
for juris, sql in JURIS.items():
    df = records(con, sql)
    p = pairs(d, df, 250, 50)
    p["jurisdiction"] = juris
    out.append(p)
    print(f"  {juris:14s} records {len(df):>8,}  pairs {len(p):>4}  "
          f"(entity {int((p.question.str.startswith('ENTITY')).sum())}, "
          f"premises {int((p.question.str.startswith('PREMISES')).sum())})")
con.close(); d.close()

full = pd.concat(out, ignore_index=True)
full = full.sort_values(["jurisdiction", "question", "band"]).reset_index(drop=True)
full.insert(0, "pair_id", [f"{r.jurisdiction[:2]}{i:03d}" for i, r in enumerate(full.itertuples(), 1)])
full.to_csv(ROOT / "workbook_key.csv", index=False)
print(f"\nTOTAL {len(full)} pairs")
print(full.groupby(["jurisdiction", "question"]).size().to_string())
