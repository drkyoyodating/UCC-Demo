#!/usr/bin/env python
"""Relational views over the resolved corpus, plus the cross-register evidence.

Everything here is a SQL query against ucc.duckdb and is emitted to
docs/data/stats.json so the published page can display the analytics rather than
merely have performed them. The queries are shipped alongside their results so a
reader can see what was actually asked.

Three of the eight capability-map skills are closed by this file and its
consumer in docs/index.html:
  * SQL / relational analytics -- the views below.
  * Graph over resolved entities -- lender -> borrower adjacency, which the page
    renders by highlighting a lender's borrowers across the map.
  * Cross-source linkage -- firms whose name appears in BOTH registers, surfaced
    deliberately as NOT-merged. See the note on that query.
"""
from __future__ import annotations
import json
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data" / "stats.json"

Q = {
 "headline": """
    SELECT count(*) AS filings,
           count(DISTINCT borrower) AS borrowers,
           count(DISTINCT lender) AS lenders,
           min(loan_year) AS first_year, max(loan_year) AS last_year
    FROM scope_all""",
 "by_region": """
    SELECT region, count(*) AS filings, count(DISTINCT borrower) AS borrowers
    FROM scope_all GROUP BY region ORDER BY region""",
 "by_criterion": """
    SELECT CASE WHEN route_a AND route_b THEN 'both'
                WHEN route_a THEN 'lender' ELSE 'borrower' END AS criterion,
           count(*) AS filings
    FROM scope_all GROUP BY 1 ORDER BY 2 DESC""",
 "top_lenders": """
    SELECT lender, count(*) AS filings, count(DISTINCT borrower) AS borrowers
    FROM scope_all WHERE lender <> '' GROUP BY 1 ORDER BY filings DESC LIMIT 12""",
 "top_borrowers": """
    SELECT borrower, region, count(*) AS filings,
           min(loan_year) AS first_year, max(loan_year) AS last_year
    FROM scope_all GROUP BY 1,2 ORDER BY filings DESC LIMIT 12""",
 "by_decade": """
    SELECT (CAST(loan_year AS INT)/10)*10 AS decade, count(*) AS filings
    FROM scope_all WHERE loan_year ~ '^[0-9]{4}$' GROUP BY 1 ORDER BY 1""",
 # CROSS-SOURCE LINKAGE. The same normalised name filing in both registers.
 # These are deliberately NOT merged into one entity: a shared name across two
 # states is not evidence of a shared firm -- there are many businesses called
 # the same thing in different states -- so the pipeline never generates a
 # cross-jurisdiction pair. Surfacing them here shows the linkage was FOUND and
 # declined, which is the honest result and a stronger claim than a merge.
 "cross_register": """
    SELECT co.borrower AS name,
           co.n AS co_filings, ct.n AS ct_filings,
           co.city AS co_city, ct.city AS ct_city
    FROM (SELECT borrower, count(*) n, min(borrower_city) city
          FROM scope_all WHERE region='CO' GROUP BY 1) co
    JOIN (SELECT borrower, count(*) n, min(borrower_city) city
          FROM scope_all WHERE region='CT' GROUP BY 1) ct
      ON upper(trim(co.borrower)) = upper(trim(ct.borrower))
    ORDER BY (co.n + ct.n) DESC LIMIT 15""",
}


def main() -> int:
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    out = {}
    for k, q in Q.items():
        cur = con.execute(q)
        cols = [d[0] for d in cur.description]
        out[k] = {"sql": " ".join(q.split()),
                  "cols": cols,
                  "rows": [list(r) for r in cur.fetchall()]}
        print(f"{k:<16} {len(out[k]['rows']):>4} rows")
    con.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":"), default=str))
    print(f"\n{OUT.relative_to(ROOT)}  {OUT.stat().st_size/1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
