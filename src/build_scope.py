#!/usr/bin/env python
"""Build the approved in-scope corpus: Colorado + Connecticut, both routes.

CONNECTICUT IS BACK IN. It was cut for having no collateral field -- but Route B
does not need one: it reads the BORROWER'S OWN NAME, and Connecticut publishes
both party names. Route A works there too. The collateral field was never the
only evidence available; it was just the one Colorado happened to have.

ROUTE A  lender is a heavy-construction manufacturer / captive / dealer.
ROUTE B  a heavy-construction equipment or trade word appears in the borrower's
         own name -- now including the CONCRETE family, because a concrete outfit
         runs mixers, pumpers and boom trucks by definition.

COMPLETENESS, enforced hard: (person OR business) + address + year + route A or B.
No placeholder addresses. Lender may be blank.
"""
from __future__ import annotations
import sys
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from heavy_filter import (BORROWER_SQL, LENDER_SQL,           # noqa: E402
                          heavy_row, is_heavy_borrower, is_heavy_lender)

JUNK = ("'','NONE','NONE PROVIDED','NA','N/A','UNKNOWN','SAME','COMPANY',"
        "'NOT PROVIDED','TBD','X','XX','NO ADDRESS','ADDRESS UNKNOWN','VARIOUS'")

con = duckdb.connect(str(ROOT / "ucc.duckdb")); con.execute("SET memory_limit='3GB'")

con.execute(f"""CREATE OR REPLACE TABLE scope_co AS
SELECT DISTINCT 'CO' AS region, f.fileid, substr(f.filingdate,1,4) AS loan_year,
       d.organizationname AS borrower, d.address1 AS borrower_address,
       d.city AS borrower_city, d.state AS borrower_state, d.zipcode AS borrower_zip,
       sp.organizationname AS lender,
       regexp_matches(upper(coalesce(sp.organizationname,'')), '{LENDER_SQL}') AS route_a,
       regexp_matches(upper(d.organizationname), '{BORROWER_SQL}')            AS route_b
FROM filings f
JOIN debtors d ON d.fileid = f.fileid
LEFT JOIN secured_parties sp ON sp.fileid = f.fileid
WHERE substr(f.filingdate,1,4) >= '1990'
  AND d.organizationname IS NOT NULL AND trim(d.organizationname) <> ''
  AND d.address1 IS NOT NULL AND upper(trim(d.address1)) NOT IN ({JUNK})
  AND (regexp_matches(upper(coalesce(sp.organizationname,'')), '{LENDER_SQL}')
    OR regexp_matches(upper(d.organizationname), '{BORROWER_SQL}'))""")

con.execute(f"""CREATE OR REPLACE TABLE scope_ct AS
SELECT DISTINCT 'CT' AS region, id_lien_flng_nbr AS fileid,
       substr(dt_accept,1,4) AS loan_year,
       debtor_nm_bus AS borrower, debtor_ad_str1 AS borrower_address,
       debtor_ad_city AS borrower_city, debtor_ad_state AS borrower_state,
       debtor_ad_zip AS borrower_zip, sec_party_nm_bus AS lender,
       regexp_matches(upper(coalesce(sec_party_nm_bus,'')), '{LENDER_SQL}') AS route_a,
       regexp_matches(upper(debtor_nm_bus), '{BORROWER_SQL}')               AS route_b
FROM ct_filings
WHERE substr(dt_accept,1,4) >= '1990'
  AND debtor_nm_bus IS NOT NULL AND trim(debtor_nm_bus) <> ''
  AND debtor_ad_str1 IS NOT NULL AND upper(trim(debtor_ad_str1)) NOT IN ({JUNK})
  AND (regexp_matches(upper(coalesce(sec_party_nm_bus,'')), '{LENDER_SQL}')
    OR regexp_matches(upper(debtor_nm_bus), '{BORROWER_SQL}'))""")

con.execute("CREATE OR REPLACE TABLE scope_all AS SELECT * FROM scope_co UNION ALL SELECT * FROM scope_ct")

# --- EXACT-PREDICATE PASS -------------------------------------------------
# The SQL above uses the raw LENDER_SQL / BORROWER_SQL regexes, which are a
# fast SUPERSET. They do NOT carry the two guards that live in Python:
#   * LENDER_DENY  -- banks, machine tools and pure agriculture, checked BEFORE
#                     the manufacturer whitelist ("1ST SOURCE BANK, CONSTRUCTION
#                     EQUIPMENT DIVISION" matches the whitelist and is a bank).
#   * the personal-name guard -- "CRANE, ROBERT GALE" is not a crane firm.
# Without this pass the guards are decorative: they would filter is_heavy_*()
# callers while the actual pull kept every row. Applying them in SQL would mean
# maintaining the same logic twice, so instead the regex pre-filters 8.4M source
# rows down to ~110k and the authoritative Python predicates decide those.
_cand = con.execute("SELECT * FROM scope_all").df()
_keep = [bool(heavy_row(b, l)) for b, l in zip(_cand.borrower, _cand.lender)]
_cand = _cand[_keep]
# route flags must agree with the predicates that actually decided the row
_cand["route_a"] = [bool(is_heavy_lender(l)) for l in _cand.lender]
_cand["route_b"] = [bool(is_heavy_borrower(b)) for b in _cand.borrower]
con.register("_scope_exact", _cand)
con.execute("CREATE OR REPLACE TABLE scope_all AS SELECT * FROM _scope_exact")
con.unregister("_scope_exact")
con.execute("CREATE OR REPLACE TABLE scope_co AS SELECT * FROM scope_all WHERE region='CO'")
con.execute("CREATE OR REPLACE TABLE scope_ct AS SELECT * FROM scope_all WHERE region='CT'")

print("=== IN-SCOPE CORPUS: heavy construction equipment finance ===")
print(f"{'region':8s} {'rows':>9s} {'filings':>9s} {'borrowers':>10s} {'lenders':>8s} "
      f"{'routeA':>8s} {'routeB':>8s} {'2013+':>8s}  years")
for r in ('CO', 'CT'):
    row = con.execute(f"""SELECT count(*), count(DISTINCT fileid), count(DISTINCT borrower),
        count(DISTINCT lender), count(*) FILTER (WHERE route_a), count(*) FILTER (WHERE route_b),
        count(*) FILTER (WHERE loan_year >= '2013'), min(loan_year), max(loan_year)
        FROM scope_all WHERE region = '{r}'""").fetchone()
    print(f"{r:8s} {row[0]:>9,} {row[1]:>9,} {row[2]:>10,} {row[3]:>8,} {row[4]:>8,} "
          f"{row[5]:>8,} {row[6]:>8,}  {row[7]}-{row[8]}")
t = con.execute("""SELECT count(*), count(DISTINCT fileid), count(DISTINCT borrower)
                   FROM scope_all""").fetchone()
print(f"{'TOTAL':8s} {t[0]:>9,} {t[1]:>9,} {t[2]:>10,}")
print("\nCONCRETE-family borrowers now in scope via Route B (founder's ruling):")
for n, k in con.execute("""SELECT borrower, count(DISTINCT fileid) k FROM scope_all
    WHERE route_b AND NOT route_a AND regexp_matches(upper(borrower),
      '\\b(CONCRETE|CEMENT|MIXER|SHOTCRETE|PRECAST|READY\\s+MIX|FLATWORK|REBAR)S?\\b')
    GROUP BY 1 ORDER BY k DESC LIMIT 8""").fetchall():
    print(f"   {str(n)[:46]:46s} {k:,}")
con.close()
