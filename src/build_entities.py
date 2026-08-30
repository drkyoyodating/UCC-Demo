#!/usr/bin/env python
"""Build the ENTITY -> LOANS table. One row per (party, address); loans aggregated.

Founder's rule: identical first+last name (middle initial ignored) at the same
address in the same state is ONE PERSON. But if those records carry different
filing numbers or different lien dates they are not duplicate rows -- they are
the same person with SEVERAL machine loans. Collapse the identity, keep the loans.

That is the difference between a de-duplicated list and a loan history, and the
loan history is the thing worth publishing: it is the entity timeline, the
refinancing signal, and the lender-relationship edge all at once.

Scope: construction and heavy-machinery finance, 1990 onward, usable address.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from normalize import normalize_name  # noqa: E402

LEND = ("CATERPILLAR|KOMATSU|BOBCAT|TEREX|JLG|GENIE INDUST|VERMEER|DITCH WITCH|HITACHI|LIEBHERR|"
        "DOOSAN|TAKEUCHI|MANITOWOC|SANY|JCB |KUBOTA|CNH |CASE CREDIT|NEW HOLLAND|VOLVO|"
        "JOHN DEERE CONSTRUCTION|DEERE CONSTRUCTION|DEERE & COMPANY|WAGNER EQUIPMENT|MACHINERY|"
        "CONSTRUCTION EQUIP|HEAVY EQUIP|CRANE|FARIS |POWER EQUIPMENT|4 RIVERS|EQUIPMENT FINANCE|"
        "EQUIPMENT CO|EQUIPMENT LEASING")
JUNK = ("'', 'NONE', 'NONE PROVIDED', 'NA', 'N/A', 'UNKNOWN', 'SAME', 'COMPANY', "
        "'NOT PROVIDED', 'TBD', 'X', 'XX', 'NO ADDRESS'")

con = duckdb.connect(str(ROOT / "ucc.duckdb"))
con.execute("SET memory_limit='3GB'")

con.create_function("nc", lambda s: normalize_name(s)[0], ["VARCHAR"], "VARCHAR", null_handling="special")
con.create_function("nsuf", lambda s: normalize_name(s)[1], ["VARCHAR"], "VARCHAR", null_handling="special")

# identity key for a person: FIRST+LAST, middle initial dropped. NULL for orgs.
#: Words that make a name a BUSINESS however few tokens it has. Without this,
#: "SEMA CONSTRUCTION INC" loses its suffix, leaves two tokens, and is classified
#: as a person -- which would apply the person-duplicate rule to a company.
_BIZ = set("""CONSTRUCTION TRUCKING BROTHERS BROS EQUIPMENT SERVICES SERVICE SUPPLY RENTAL
RENTALS LEASING EXCAVATING EXCAVATION PAVING GRADING CONCRETE DRILLING CONTRACTORS CONTRACTING
BUILDERS ENTERPRISES INDUSTRIES MACHINE MACHINERY MOTORS AUTO FARMS RANCH RANCHES DAIRY
UNIVERSITY COLLEGE SCHOOL DISTRICT COUNTY CITY STATE BANK CREDIT FINANCIAL CAPITAL TRUST
HOLDINGS PROPERTIES REALTY DEVELOPMENT PARTNERS GROUP ASSOCIATES COMPANIES SYSTEMS SOLUTIONS
TRANSPORT TRANSPORTATION LOGISTICS ENERGY OIL GAS MINING AGGREGATE READY MIX ASPHALT ROOFING
PLUMBING ELECTRIC ELECTRICAL MECHANICAL WELDING FABRICATION STEEL LUMBER FEED GRAIN SEED
LANDSCAPING LANDSCAPE NURSERY GREENHOUSE SEPTIC UTILITIES UTILITY PIPELINE SPECIALTIES
CONSTRUCTORS TRUCKS TRACTOR IMPLEMENT SALES CENTER CENTRE""".split())


def pkey(n):
    """FIRST|LAST for a person-looking name, middle initial dropped. None for orgs.

    Used ONLY to collapse the same individual filed twice at one address. It never
    excludes anyone, so a false negative merely leaves two rows; a false POSITIVE
    would merge a company, which is why the business-word veto runs first.
    """
    if not isinstance(n, str):
        return None
    t = n.split()
    if not (2 <= len(t) <= 3):
        return None
    if any(w in _BIZ for w in t):
        return None
    if len(t) == 3 and len(t[2]) > 2:
        return None
    if any(any(ch.isdigit() for ch in w) for w in t):
        return None
    return f"{t[0]}|{t[1]}"
con.create_function("pkey", pkey, ["VARCHAR"], "VARCHAR", null_handling="special")


def pkey2(name_raw):
    """Person key, with ENTITY TYPE as the decisive veto.

    A name that carried a corporate suffix (INC / LLC / CORP / CO ...) is a
    COMPANY no matter how much the stem reads like a person. Without this,
    "STUTSMAN-GERBAZ, INC." loses its suffix, leaves two plausible-looking tokens
    and is classified as an individual. The suffix is the strongest signal
    available and it is free.
    """
    nm, suf = normalize_name(name_raw)
    if suf:
        return None
    return pkey(nm)


con.create_function("pkey2", pkey2, ["VARCHAR"], "VARCHAR", null_handling="special")

_ABBR = [(" STREET", " ST"), (" AVENUE", " AVE"), (" ROAD", " RD"), (" DRIVE", " DR"),
         (" BOULEVARD", " BLVD"), (" HIGHWAY", " HWY"), (" STATE HWY", " HWY"),
         (" COUNTY ROAD", " CO RD"), (" COUNTY RD", " CO RD"), (" SUITE", " STE"),
         (" NORTH ", " N "), (" SOUTH ", " S "), (" EAST ", " E "), (" WEST ", " W "),
         (" POST OFFICE BOX", " PO BOX"), (" P O BOX", " PO BOX")]


def akey(a):
    """Address key with FORMATTING normalised. Without it the same entity splits:
    STUTSMAN-GERBAZ appeared twice on 'HIGHWAY 82' vs 'STATE HIGHWAY 82' (91 and
    75 loans), and LIGHTNING VENTURES on 'TOMAH ROAD' vs 'TOMAH RD' (79 and 68).
    """
    if not isinstance(a, str):
        return ""
    s = " " + "".join(c if c.isalnum() else " " for c in a.upper()) + " "
    for long, short in _ABBR:
        s = s.replace(long, short)
    return " ".join(s.split())


con.create_function("akey", akey, ["VARCHAR"], "VARCHAR", null_handling="special")

con.execute(f"""
CREATE OR REPLACE TABLE co_machinery_loans AS
SELECT p.party_role, p.name_raw, p.address1, p.city, p.state, p.zipcode,
       f.fileid, f.filingdate, f.filingtype, f.continuation, f.terminationflag,
       sp.organizationname AS lender_name
FROM (
    SELECT 'DEBTOR' AS party_role, organizationname name_raw, address1, city, state, zipcode, fileid
      FROM debtors WHERE organizationname IS NOT NULL AND trim(organizationname) <> ''
    UNION ALL
    SELECT 'SECURED_PARTY', organizationname, address1, city, state, zipcode, fileid
      FROM secured_parties WHERE organizationname IS NOT NULL AND trim(organizationname) <> ''
) p
JOIN filings f ON f.fileid = p.fileid
JOIN (SELECT DISTINCT fileid, first(organizationname) OVER (PARTITION BY fileid) organizationname
      FROM secured_parties WHERE regexp_matches(upper(organizationname), '{LEND}')) sp
  ON sp.fileid = p.fileid
WHERE substr(f.filingdate,1,4) >= '1990'
  AND p.address1 IS NOT NULL AND upper(trim(p.address1)) NOT IN ({JUNK})
""")

con.execute("""
CREATE OR REPLACE TABLE co_entities AS
SELECT
  coalesce(pkey2(name_raw), nc(name_raw))                          AS identity_key,
  any_value(name_raw)                                              AS name,
  nc(name_raw)                                                     AS name_clean,
  akey(address1)                                                   AS address_key,
  any_value(address1) AS address, any_value(city) AS city,
  any_value(state) AS state, any_value(zipcode) AS zipcode,
  any_value(party_role) AS party_role,
  -- HEURISTIC, and deliberately labelled as one. Telling a two-token person name
  -- (SMITH JOHN) from a two-token company name (WILDERNESS EARTHWORKS) is not
  -- reliably solvable, and a business-word list is whack-a-mole. It is kept
  -- because it is HARMLESS WHERE IT IS USED: its only job is to drop a middle
  -- initial when collapsing the same individual filed twice at one address, and
  -- companies do not have middle initials. A false positive changes nothing.
  -- Do NOT use this column to count or display "people" -- it would overstate.
  any_value(pkey2(name_raw)) IS NOT NULL                            AS person_like,
  count(DISTINCT fileid)                                           AS loan_count,
  min(substr(filingdate,1,10))                                     AS first_loan,
  max(substr(filingdate,1,10))                                     AS last_loan,
  count(DISTINCT lender_name)                                      AS distinct_lenders,
  list(DISTINCT lender_name)[1:5]                                  AS lenders,
  count(*) FILTER (WHERE terminationflag='true')                   AS terminated,
  count(*) FILTER (WHERE continuation='true')                      AS continued
FROM co_machinery_loans
GROUP BY identity_key, name_clean, address_key
""")

n_loans, n_ent = con.execute(
    "SELECT (SELECT count(*) FROM co_machinery_loans),(SELECT count(*) FROM co_entities)").fetchone()
print(f"machinery loan rows (1990+, usable address): {n_loans:,}")
print(f"distinct ENTITIES (identity + address)     : {n_ent:,}")
print(f"  collapse ratio: {n_loans/max(n_ent,1):.2f} loan rows per entity\n")
print("entities with the most machine loans:")
for r in con.execute("""SELECT name, city, loan_count, distinct_lenders, first_loan, last_loan, party_role
    FROM co_entities WHERE party_role='DEBTOR' ORDER BY loan_count DESC LIMIT 8""").fetchall():
    print(f"   {str(r[0])[:34]:34s} {str(r[1])[:14]:14s} {r[2]:>4} loans  "
          f"{r[3]:>2} lenders  {r[4]}..{r[5]}")
print("\nperson_like entities with several machine loans (flag is heuristic -- see comment):")
for r in con.execute("""SELECT name, address, city, loan_count, first_loan, last_loan
    FROM co_entities WHERE person_like AND party_role='DEBTOR' AND loan_count>1
    ORDER BY loan_count DESC LIMIT 6""").fetchall():
    print(f"   {str(r[0])[:26]:26s} {str(r[1])[:24]:24s} {str(r[2])[:12]:12s} "
          f"{r[3]:>3} loans  {r[4]}..{r[5]}")
con.close()
