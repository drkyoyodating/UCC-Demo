#!/usr/bin/env python
"""P4 -- EDA. Half a page of numbers, each one naming the P5 decision it changes.

The single highest-value output here is the BLOCKING CARDINALITY. Splink compares
only pairs a blocking rule proposes, so those counts decide (a) whether P5 finishes
inside its budget and (b) the recall ceiling no amount of model tuning can lift.
Measuring them before training is cheaper than discovering a quadratic blow-up at
hour 12.
"""
from __future__ import annotations
import sys
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "eda.md"
L: list[str] = []


def w(s: str = "") -> None:
    L.append(s)
    print(s)


def main() -> int:
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    q = lambda s: con.execute(s).fetchall()
    q1 = lambda s: con.execute(s).fetchone()

    w("# P4 — Exploratory data analysis")
    w()
    w("Every number below is followed by the P5 design decision it changes. "
      "Nothing here is reported for its own sake.")
    w()

    for corpus, label in [("corpus_debtors_eq", "DEBTORS (P5)"),
                          ("corpus_lenders_eq", "LENDERS (P5b)")]:
        n, dk, dr = q1(f"SELECT count(*), count(DISTINCT name_clean), "
                       f"count(DISTINCT name_raw) FROM {corpus}")
        w(f"## {label} — `{corpus}`")
        w()
        w(f"- **{n:,} rows → {dr:,} distinct raw names → {dk:,} distinct `name_clean`.** "
          f"Normalisation alone collapsed {dr-dk:,} variants ({100*(dr-dk)/dr:.1f}%) "
          f"before any model runs.")

        # --- suffix distribution ---
        rows = q(f"SELECT coalesce(suffix,'(none)') s, count(*) c FROM {corpus} "
                 f"GROUP BY 1 ORDER BY c DESC LIMIT 8")
        tot = sum(c for _, c in rows)
        w(f"- Suffix distribution (top 8 of "
          f"{q1(f'SELECT count(DISTINCT coalesce(suffix,chr(48))) FROM {corpus}')[0]}): "
          + ", ".join(f"`{s}` {c:,}" for s, c in rows))
        w("  → *P5: suffix is a comparison feature, not part of the blocking key. "
          "A suffix-only difference gets a mild discount, never a penalty.*")

        # --- naive exact-match clusters ---
        big = q(f"SELECT name_clean, count(*) c, count(DISTINCT zipcode) z FROM {corpus} "
                f"WHERE name_clean IS NOT NULL GROUP BY 1 HAVING c>1 ORDER BY c DESC LIMIT 5")
        multi = q1(f"SELECT count(*) FROM (SELECT name_clean FROM {corpus} "
                   f"GROUP BY 1 HAVING count(*)>1)")[0]
        w(f"- **{multi:,} keys already have ≥2 rows by exact `name_clean` match** — "
          f"that is the floor Splink must beat, not its result. Largest: "
          + ", ".join(f"`{k}` ×{c} ({z} ZIPs)" for k, c, z in big))

        # --- completeness ---
        a, c_, z = q1(f"SELECT avg((address1 IS NOT NULL)::INT), avg((city IS NOT NULL)::INT), "
                      f"avg((zipcode IS NOT NULL)::INT) FROM {corpus}")
        w(f"- Completeness — address1 {100*a:.1f}%, city {100*c_:.1f}%, zipcode {100*z:.1f}%.")
        w("  → *P5: high enough that address belongs in the comparison set and ZIP is a "
          "viable blocking key. If ZIP had been sparse, blocking on it would silently "
          "drop the missing rows from every comparison.*")

        # --- BLOCKING CARDINALITY: the number that decides P5's runtime ---
        w()
        w(f"### Blocking cardinality — `{corpus}`")
        w()
        w("| rule | blocks | comparisons | largest block | pairs in largest |")
        w("|---|---:|---:|---:|---:|")
        for rule, expr in [("`zipcode`", "zipcode"),
                           ("`substr(name_clean,1,4)`", "substr(name_clean,1,4)")]:
            blocks, pairs, mx = q1(f"""
                SELECT count(*), sum(c*(c-1)/2), max(c) FROM (
                  SELECT count(*) c FROM {corpus} WHERE {expr} IS NOT NULL
                  GROUP BY {expr})""")
            w(f"| {rule} | {blocks:,} | {int(pairs or 0):,} | {mx:,} | "
              f"{int(mx*(mx-1)/2):,} |")
        # Record-level: the same two rules applied to DISTINCT party records.
        for rule, expr in [("`zipcode` (records)", "zipcode"),
                           ("`substr(name_clean,1,4)` (records)", "substr(name_clean,1,4)")]:
            blocks, pairs, mx = q1(f"""
                SELECT count(*), sum(c*(c-1)/2), max(c) FROM (
                  SELECT count(*) c FROM (SELECT DISTINCT name_clean, address1, city, zipcode
                                          FROM {corpus}) WHERE {expr} IS NOT NULL
                  GROUP BY {expr})""")
            w(f"| {rule} | {blocks:,} | {int(pairs or 0):,} | {mx:,} | "
              f"{int(mx*(mx-1)/2):,} |")
        naive = q1(f"SELECT count(*) FROM {corpus}")[0]
        w(f"| *(no blocking — for scale)* | 1 | {naive*(naive-1)//2:,} | {naive:,} | "
          f"{naive*(naive-1)//2:,} |")
        w()

        # --- does blocking actually lose true matches? ---
        # Three iterations, all recorded because the process is the point.
        #  v1: "identical keys agreeing on neither rule" -> 0. True BY CONSTRUCTION
        #      (identical keys share their own prefix). A tautology dressed as a finding.
        #  v2: "keys differing in the first 4 chars but agreeing once spaces collapse"
        #      -> 4,712, published as a recall ceiling. The P4 audit hand-judged 30 of
        #      them: 0 were genuine variants. 70% involve an "X AND Y" name whose
        #      space-collapse coincidentally hits SANDWICH / RANDOLPH / BAND- / LAND-.
        #      So v2 measured coincidence and overstated loss by ~2 orders of magnitude.
        #  v3 (this): pairs whose keys are IDENTICAL once spaces are removed. Decidable,
        #      no sampling and no judgement call: these ARE the same normalised name
        #      written with different spacing, and the prefix rule provably cannot
        #      propose them.
        n_var, n_resc = q1(f"""
            WITH k AS (SELECT DISTINCT name_clean nc, replace(name_clean,' ','') sq,
                              substr(name_clean,1,4) p4
                       FROM {corpus} WHERE name_clean IS NOT NULL),
                 z AS (SELECT DISTINCT name_clean nc, zipcode zp
                       FROM {corpus} WHERE zipcode IS NOT NULL)
            SELECT count(*),
                   count(*) FILTER (WHERE EXISTS (SELECT 1 FROM z za JOIN z zb ON za.zp=zb.zp
                                                  WHERE za.nc=a.nc AND zb.nc=b.nc))
            FROM k a JOIN k b ON a.sq=b.sq AND a.nc<b.nc AND a.p4<>b.p4""")
        ex = q(f"""
            WITH k AS (SELECT DISTINCT name_clean nc, replace(name_clean,' ','') sq,
                              substr(name_clean,1,4) p4
                       FROM {corpus} WHERE name_clean IS NOT NULL)
            SELECT a.nc, b.nc FROM k a JOIN k b ON a.sq=b.sq AND a.nc<b.nc AND a.p4<>b.p4
            ORDER BY length(a.nc) DESC LIMIT 3""")
        w(f"- **Measured blocking loss: {n_var} spacing-variant key pairs, of which "
          f"{n_resc} are rescued by the ZIP rule, leaving {n_var-n_resc} unreachable by "
          f"either rule.** These are pairs whose keys are *identical once spaces are "
          f"removed* — decidably the same name, differently spaced, and provably outside "
          f"the prefix rule. Examples: "
          + "; ".join(f"`{x}` / `{y}`" for x, y in ex) + ".")
        w(f"  → *P5: a third blocking rule would be worth at most {n_var-n_resc} pairs on "
          f"this corpus, so none is added. Recall is reported as conditional on the union "
          f"of the two rules, with this figure stated as the known loss.*")
        w()

        if corpus == "corpus_lenders_eq":
            rows_, recs = q1(f"""SELECT count(*), count(DISTINCT (name_clean, address1,
                                 city, zipcode)) FROM {corpus}""")
            w(f"- ⚠ **Comparison space warning.** This corpus has {rows_:,} rows but only "
              f"{recs:,} distinct (name, address, city, zip) RECORDS — a "
              f"{rows_/recs:.0f}× row-level redundancy, because one lender files "
              f"thousands of times (`WAGNER EQUIPMENT` ×6,334). Blocking on raw rows "
              f"costs ~74M comparisons against the debtor corpus's ~4.4M.")
            w(f"  → *P5b: resolve DISTINCT PARTY RECORDS, not rows, then map canonical ids "
              f"back to rows for the league table. Same answer, ~{(rows_/recs)**2:.0f}× less "
              f"work, and it removes the duplicate-row mass that would otherwise dominate "
              f"the match-weight histogram and flatter the high-weight labelling stratum.*")
            w()

    # --- corpus-independent context ---
    w("## Context that shapes the write-up")
    w()
    ft = q("""SELECT f.filingtype, count(DISTINCT f.fileid) c FROM filings f
             WHERE f.fileid IN (SELECT DISTINCT fileid FROM collateral
                                WHERE collateraldescription='EQUIPMENT')
             GROUP BY 1 ORDER BY c DESC LIMIT 6""")
    w("- EQUIPMENT filings by `filingtype`: "
      + ", ".join(f"`{t}` {c:,}" for t, c in ft)
      + ". → *The register is not UCC-only; the write-up must say 'lien register', "
        "and the five-year lapse premise governs only the `ucc` rows.*")
    na = q1("""SELECT count(*) FROM (
                 SELECT name_raw FROM corpus_debtors_eq UNION ALL
                 SELECT name_raw FROM corpus_lenders_eq)
               WHERE regexp_matches(name_raw, '[^\\x00-\\x7F]')""")[0]
    w(f"- Non-ASCII names in both corpora combined: **{na}**. → *Confirms the decision not "
      f"to build a Unicode confusable map: a stage that provably fires ~never is padding. "
      f"This is the measured number, not an assumption.*")
    ho = q1("SELECT count(*) FROM corpus_debtors_eq WHERE is_holdout")[0]
    w(f"- Stability holdout reserved: {ho:,} debtor rows (seeded, reproducible). "
      f"→ *P6 run 1 uses base only; run 2 uses all rows.*")
    con.close()

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
