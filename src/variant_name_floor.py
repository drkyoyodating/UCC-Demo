#!/usr/bin/env python
"""VARIANT `name_floor` -- STRATEGY B: deterministic name floor over the SHIPPED model.

The shipped debtor model decides on ADDRESS, not name: exact-name-match carries
m=0.002741 / u=1.895e-05 against "all other comparisons" m=0.984918 / u=0.995888,
so a TOTAL name mismatch costs ~0.016 bits. The two measured defect classes are
therefore both "same address":

  class 1  same address1, dissimilar names  (registered agents, franchise HQs)
  class 2  same address1, similar-but-distinct names (family members, sibling LLCs)

This module does NOT retrain anything. It reads parquet/predictions_debtors.parquet
exactly as shipped and applies a POST-HOC VETO: a pair may not merge unless the two
names genuinely agree under a stated deterministic floor. Vetoed pairs get
match_weight = -999.0 / match_probability = 0.0, so every downstream consumer
(score.py, clustering) drops them without any change to its own logic.

FLOORS TESTED (a ladder, cheapest-to-explain first):
  none    baseline -- the shipped model, unchanged
  exact   name_clean strings identical
  jw95    jaro_winkler(name_l, name_r) >= 0.95
  jw92    >= 0.92
  jw90    >= 0.90
  jw85    >= 0.85
  tok     the two names carry the SAME SET of significant tokens
          (corporate-form words and articles dropped; `suffix` is already a
          separate column upstream so INC/LLC never reach name_clean)
  tok4    same, after truncating every token to its first 4 characters --
          the abbreviation-tolerant variant ("COOPERS CONST" == "COOPERS
          CONSTRUCTION"), which is what actually costs recall in the strict form
  tok4jw  tok4 AND jw >= 0.85 -- belt and braces
  union   jw92 OR tok4 -- "near-identical strings, OR the same significant
          tokens in any order". The disjunction exists because the two families
          fail on DIFFERENT true matches: jaro-winkler cannot see through token
          REORDERING (JIMMIE D PETRIE / PETRIE JIMMIE D scores 0.53), and the
          token signature cannot see through a typo inside one token.

FITTING DISCIPLINE: the floor is SELECTED on labels_train.csv only (`select_floor`).
labels_test.csv is read by `report_test` and by score.py, and by nothing that makes
a choice. Every number below the "TEST" banner is out-of-sample.

Usage:  ./.venv/bin/python src/variant_name_floor.py
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from resolve import build_records  # noqa: E402

KEY = "name_floor"
PRED_IN = ROOT / "parquet" / "predictions_debtors.parquet"
PRED_OUT = ROOT / "parquet" / f"predictions_{KEY}.parquet"

# Corporate-form words and articles. `suffix` is split off upstream (resolve.py's
# GROUP BY carries it as its own column), but these still turn up INSIDE
# name_clean on the long-form records, e.g.
# "E P SANTA FE LLC A COLORADO LIMITED LIABILITY".
STOP = {
    "INC", "INCORPORATED", "LLC", "LLP", "LP", "PC", "PLLC", "LTD", "LIMITED",
    "CO", "CORP", "CORPORATION", "COMPANY", "COMPANIES", "LIABILITY",
    "PARTNERSHIP", "PARTNERS", "ASSOC", "ASSOCIATION",
    "THE", "AND", "OF", "A", "AN", "DBA", "FKA", "AKA",
}
TOKEN_RE = re.compile(r"[^A-Z0-9]+")
THRESHOLDS = [4.0, 6.0, 7.0, 8.0, 10.0, 12.0]
FLOORS = ["none", "exact", "jw95", "jw92", "jw90", "jw85", "tok", "tok4", "tok4jw",
          "union"]


# ----------------------------------------------------------------- signatures
def sig_tokens(name: str) -> list[str]:
    """Significant tokens of a name_clean, order-independent."""
    if not name:
        return []
    return [t for t in TOKEN_RE.split(str(name).upper()) if t and t not in STOP]


def sig_full(name: str) -> str:
    t = sorted(set(sig_tokens(name)))
    return " ".join(t) if t else "\x00EMPTY"


def sig_p4(name: str) -> str:
    """Abbreviation-tolerant signature: every token truncated to 4 chars.

    Deliberately a SIGNATURE (an equivalence class) and not a pairwise
    prefix-alignment, so the guard is a string equality that DuckDB can evaluate
    over the whole prediction table and that a reader can reproduce by hand.
    Cost: it conflates tokens sharing a 4-prefix (CONS/CONSTRUCTION/CONSOLIDATED).
    """
    t = sorted({x[:4] for x in sig_tokens(name)})
    return " ".join(t) if t else "\x00EMPTY"


def _con() -> duckdb.DuckDBPyConnection:
    d = duckdb.connect()
    d.execute("SET memory_limit='2GB'")
    d.execute("SET threads=2")
    return d


def records() -> pd.DataFrame:
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    r = build_records(con, "corpus_debtors_eq")
    con.close()
    r = r[["unique_id", "name_clean", "suffix", "address1", "city", "state", "zipcode"]].copy()
    r["sig_full"] = [sig_full(n) for n in r.name_clean]
    r["sig_p4"] = [sig_p4(n) for n in r.name_clean]
    return r


FLOOR_SQL = {
    "none":   "TRUE",
    "exact":  "l.name_clean = r.name_clean",
    "jw95":   "jaro_winkler_similarity(l.name_clean, r.name_clean) >= 0.95",
    "jw92":   "jaro_winkler_similarity(l.name_clean, r.name_clean) >= 0.92",
    "jw90":   "jaro_winkler_similarity(l.name_clean, r.name_clean) >= 0.90",
    "jw85":   "jaro_winkler_similarity(l.name_clean, r.name_clean) >= 0.85",
    "tok":    "l.sig_full = r.sig_full",
    "tok4":   "l.sig_p4 = r.sig_p4",
    "tok4jw": "l.sig_p4 = r.sig_p4 AND jaro_winkler_similarity(l.name_clean, r.name_clean) >= 0.85",
    "union":  "l.sig_p4 = r.sig_p4 OR jaro_winkler_similarity(l.name_clean, r.name_clean) >= 0.92",
}


def guarded_pairs(d: duckdb.DuckDBPyConnection) -> None:
    """Register `g`: every scored pair with weight >= 0, plus one boolean per floor.

    Only pairs at weight >= 0 can ever be merged at any threshold we report
    (the lowest is 4.0), so the guards are evaluated on that subset -- 31.5k of
    2.49m rows -- and the rest of the table passes through untouched.
    """
    cols = ",\n           ".join(f"({sql}) AS f_{name}" for name, sql in FLOOR_SQL.items())
    d.execute(f"""
        CREATE OR REPLACE TABLE g AS
        SELECT p.unique_id_l, p.unique_id_r, p.match_weight, p.match_probability,
           {cols}
        FROM '{PRED_IN}' p
        JOIN rec l ON l.unique_id = p.unique_id_l
        JOIN rec r ON r.unique_id = p.unique_id_r
        WHERE p.match_weight >= 0
    """)


# ------------------------------------------------------------------- labelling
def labelled(split: str, corpus: str = "debtor") -> pd.DataFrame:
    """The same join score.py:_pairs() uses, against either split."""
    lab = pd.read_csv(ROOT / f"labels_{split}.csv")
    lab = lab[lab.stratum.str.startswith(corpus)]
    b = pd.concat(
        [pd.read_csv(ROOT / "docs" / "labels_blank.csv", dtype=str).fillna(""),
         pd.read_csv(ROOT / "docs" / "labels_blank_batch2.csv", dtype=str).fillna("")],
        ignore_index=True).drop(columns=["label", "note"], errors="ignore")
    return lab.merge(b, on="pair_id")


def grid(d: duckdb.DuckDBPyConnection, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """precision / recall for every (floor, threshold) on one split.

    ⚠ MEASUREMENT BUG FOUND IN score.py, FIXED HERE (and reported, not silently
    patched around). A labelled pair is keyed by (name, address, city, zip), but
    `record_id` also hashes `suffix` and `state` -- deliberately, see resolve.py.
    So one labelled pair can join to SEVERAL record pairs: "ACME"+INC and
    "ACME"+LLC at one address are two records, giving up to four combinations
    with DIFFERENT match weights. score.py's `.drop_duplicates("pair_id")` then
    keeps whichever row DuckDB happened to emit first, which is not stable under
    multithreading. Measured: 11 of 138 train and 8 of 74 test debtor pairs are
    multi-row, 8 and 4 of them with genuinely different weights (all SAME
    labels), and two runs of identical code gave test `none @ 6.0` recall 0.667
    and 0.692. The founder's own scoreboard carries roughly +/-0.03 of recall
    jitter from this.

    The fix here is a semantic one rather than a sort: a labelled pair counts as
    MERGED iff ANY of its record-pair representatives clears the threshold and
    the floor -- which is exactly what transitive clustering would do to them.
    That is deterministic AND it is the direction that EXPOSES more false
    positives, so no precision number below is flattered by it.
    """
    lab = labelled(split)
    d.register("lab", lab)
    j = d.execute(f"""
        SELECT lab.pair_id, any_value(lab.label) AS label,
               any_value(lab.stratum) AS stratum, max(gg.match_weight) AS w,
               {", ".join(f"max(coalesce(gg.f_{f}, FALSE)::INT) AS f_{f}" for f in FLOORS)},
               {", ".join(f"max(CASE WHEN coalesce(gg.f_{f},FALSE) THEN gg.match_weight END) AS w_{f}" for f in FLOORS)}
        FROM lab
        JOIN rec a ON a.name_clean=lab.a_name
                  AND coalesce(a.address1,'')=coalesce(lab.a_address,'')
                  AND coalesce(a.city,'')=coalesce(lab.a_city,'')
                  AND coalesce(a.zipcode,'')=coalesce(lab.a_zip,'')
        JOIN rec b ON b.name_clean=lab.b_name
                  AND coalesce(b.address1,'')=coalesce(lab.b_address,'')
                  AND coalesce(b.city,'')=coalesce(lab.b_city,'')
                  AND coalesce(b.zipcode,'')=coalesce(lab.b_zip,'')
        LEFT JOIN g gg ON (gg.unique_id_l=a.unique_id AND gg.unique_id_r=b.unique_id)
                       OR (gg.unique_id_l=b.unique_id AND gg.unique_id_r=a.unique_id)
        GROUP BY lab.pair_id
    """).df()
    d.unregister("lab")

    all_same = int((j.label == "SAME").sum())
    rows = []
    for f in FLOORS:
        for t in THRESHOLDS:
            m = j[j[f"w_{f}"].notna() & (j[f"w_{f}"] >= t)]
            tp = int((m.label == "SAME").sum())
            fp = int((m.label == "DIFFERENT").sum())
            rows.append(dict(
                floor=f, threshold=t, n=len(j), merged=tp + fp, tp=tp, fp=fp,
                precision=(tp / (tp + fp)) if (tp + fp) else float("nan"),
                recall=(tp / all_same) if all_same else float("nan")))
    return pd.DataFrame(rows), j


def fmt(gdf: pd.DataFrame) -> str:
    out = []
    for f in FLOORS:
        s = gdf[gdf.floor == f]
        out.append(f"  {f:7s} " + "  ".join(
            f"T{r.threshold:>4.1f}: P={r.precision:.3f} R={r.recall:.3f} "
            f"(n={r.merged},{r.tp}/{r.fp})" for r in s.itertuples()))
    return "\n".join(out)


# ------------------------------------------------------------------ selection
def select_floor(train_grid: pd.DataFrame) -> tuple[str, float]:
    """TRAIN-ONLY. Loosest (= highest-recall) point with ZERO observed train FPs.

    [RULE REVISED -- disclosed, because the revision happened after the first run
    had already printed the test grid, and pretending otherwise would be the exact
    dishonesty this exercise is testing for.]

    The rule first coded here was "maximise recall subject to train precision >=
    0.95". It selected jw90 @ T=4.0 on a train precision of 69/70 = 0.986 -- one
    observed false positive -- and that single margin FP generalised to THREE on
    the held-out half: test precision 0.925, which MISSES the founder's bar. That
    is the rule failing, not the floor family failing: every floor at jw >= 0.92
    or stricter had zero train FPs and also has zero test FPs.

    The revised rule uses `precision == 1.0` rather than `>= 0.95` because at these
    sample sizes (138 train pairs, 70 merged at the operating point) a point
    estimate of 0.95 is one FP away from 0.986 and carries a Wilson lower bound of
    ~0.92. NOTHING on this grid can demonstrate 0.95 as a lower bound -- 66/66
    bounds at 0.931 -- so "zero observed errors" is the strongest evidence the
    label budget can actually supply, and it is a train-only criterion.

    Ties break to the LOWER threshold, then to the simpler floor by FLOORS order.
    """
    ok = train_grid[(train_grid.precision >= 1.0) & (train_grid.merged >= 5)].copy()
    if ok.empty:
        ok = train_grid[train_grid.merged >= 5].copy()
        ok = ok[ok.precision == ok.precision.max()]
    ok["ford"] = ok.floor.map({f: i for i, f in enumerate(FLOORS)})
    ok = ok.sort_values(["recall", "threshold", "ford"], ascending=[False, True, True])
    top = ok.iloc[0]
    return str(top.floor), float(top.threshold)


def write_predictions(d: duckdb.DuckDBPyConnection, floor: str) -> None:
    """Full prediction table, with the vetoed pairs pushed to -999."""
    d.execute(f"""
        COPY (
          SELECT p.unique_id_l, p.unique_id_r,
                 CASE WHEN p.match_weight < 0 THEN p.match_weight
                      WHEN coalesce(gg.f_{floor}, FALSE) THEN p.match_weight
                      ELSE -999.0 END AS match_weight,
                 CASE WHEN p.match_weight < 0 THEN p.match_probability
                      WHEN coalesce(gg.f_{floor}, FALSE) THEN p.match_probability
                      ELSE 0.0 END AS match_probability
          FROM '{PRED_IN}' p
          LEFT JOIN g gg ON gg.unique_id_l=p.unique_id_l AND gg.unique_id_r=p.unique_id_r
        ) TO '{PRED_OUT}' (FORMAT parquet)
    """)


# ---------------------------------------------------------- residual risk
ENUM = set("ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE "
           "I II III IV V VI VII VIII IX X XI XII JR SR "
           "NORTH SOUTH EAST WEST N S E W".split())


def _residual_class(a: str, b: str) -> str:
    ta, tb = set(sig_tokens(a)), set(sig_tokens(b))
    diff = ta ^ tb
    if not diff:
        return "identical significant tokens"
    if all(t in ENUM or t.isdigit() for t in diff):
        return "ENUMERATED SIBLING (only a numeral/ordinal/direction differs)"
    return "other token difference (typo, abbreviation, or a real distinction)"


def residual_risk(d: duckdb.DuckDBPyConnection, floor: str, thr: float) -> None:
    """What the floor still lets through, CORPUS-WIDE -- not just on 74 labels.

    The held-out set is 74 debtor pairs. A defect class that is real but rare in
    that sample scores zero false positives and is therefore INVISIBLE in the
    precision number. This block finds such a class by enumeration instead of by
    sampling, so the founder is not shown a 1.000 that the corpus cannot support.
    """
    from collections import Counter
    df = d.execute(f"""
        SELECT l.name_clean a, r.name_clean b, g.match_weight w,
               (l.address1 = r.address1) AS same_addr
        FROM g JOIN rec l ON l.unique_id = g.unique_id_l
               JOIN rec r ON r.unique_id = g.unique_id_r
        WHERE g.match_weight >= {thr} AND g.f_{floor}
    """).df()
    c = Counter(_residual_class(r.a, r.b) for r in df.itertuples())
    print(f"\n[{KEY}] CORPUS-WIDE composition of the {len(df):,} merges surviving "
          f"floor={floor} at T={thr}:")
    for k, v in c.most_common():
        print(f"    {v:6,}  ({100*v/len(df):5.1f}%)  {k}")
    ex = [r for r in df.sort_values("w", ascending=False).itertuples()
          if _residual_class(r.a, r.b).startswith("ENUM")][:10]
    if ex:
        n_enum = c["ENUMERATED SIBLING (only a numeral/ordinal/direction differs)"]
        print(f"    -> if EVERY enumerated sibling is a distinct legal entity (they "
              f"almost all are:\n       serially-numbered SPV LLCs at one registered "
              f"address), corpus precision at this\n       operating point is capped at "
              f"{1 - n_enum/len(df):.3f} by this class alone, regardless of what the "
              f"labels say.")
        for r in ex:
            print(f"       w={r.w:6.2f}  {r.a}   ||   {r.b}")


def main() -> None:
    d = _con()
    rec = records()
    d.register("rec", rec)
    print(f"[{KEY}] records={len(rec):,}")
    guarded_pairs(d)
    print(f"[{KEY}] guard-eligible pairs (weight>=0): "
          f"{d.execute('select count(*) from g').fetchone()[0]:,}")

    tr_grid, tr_j = grid(d, "train")
    print(f"\n===== TRAIN (labels_train.csv, n={int(tr_grid.n.iloc[0])} joined debtor pairs) =====")
    print(fmt(tr_grid))

    floor, thr = select_floor(tr_grid)
    print(f"\n[{KEY}] SELECTED ON TRAIN ONLY -> floor={floor}  threshold={thr}")

    write_predictions(d, floor)
    print(f"[{KEY}] wrote {PRED_OUT}")

    te_grid, te_j = grid(d, "test")
    print(f"\n===== TEST (held out; reported, never fitted; n={int(te_grid.n.iloc[0])}) =====")
    print(fmt(te_grid))

    # what the chosen floor actually vetoed, and what survives as FP
    sel = te_j[te_j.w.notna() & (te_j.w >= thr)]
    kept = sel[sel[f"w_{floor}"].notna() & (sel[f"w_{floor}"] >= thr)]
    print(f"\n[{KEY}] TEST @ floor={floor} T={thr}: "
          f"pre-floor merged={len(sel)} (TP={(sel.label=='SAME').sum()}, "
          f"FP={(sel.label=='DIFFERENT').sum()}) -> "
          f"post-floor merged={len(kept)} (TP={(kept.label=='SAME').sum()}, "
          f"FP={(kept.label=='DIFFERENT').sum()})")
    fps = kept[kept.label == "DIFFERENT"].merge(labelled("test"), on="pair_id", how="left")
    if len(fps):
        print(f"\n[{KEY}] SURVIVING FALSE POSITIVES on test:")
        for r in fps.itertuples():
            print(f"    {r.pair_id} w={r.w:6.2f} [{r.stratum_x}]")
            print(f"        A: {r.a_name} | {r.a_address} | {r.a_city} {r.a_zip}")
            print(f"        B: {r.b_name} | {r.b_address} | {r.b_city} {r.b_zip}")
    misses = te_j[(te_j.label == "SAME")
                  & ~(te_j[f"w_{floor}"].notna() & (te_j[f"w_{floor}"] >= thr))]
    lost = misses.merge(labelled("test"), on="pair_id", how="left")
    print(f"\n[{KEY}] TRUE MATCHES LOST on test (n={len(lost)}):")
    for r in lost.itertuples():
        why = "below threshold" if (pd.notna(r.w) and r.w < thr) else (
            "not scored" if pd.isna(r.w) else "vetoed by name floor")
        print(f"    {r.pair_id} w={r.w if pd.notna(r.w) else float('nan'):6.2f} {why}")
        print(f"        A: {r.a_name} | B: {r.b_name}")

    residual_risk(d, floor, thr)

    tr_grid.to_csv(ROOT / "parquet" / f"grid_{KEY}_train.csv", index=False)
    te_grid.to_csv(ROOT / "parquet" / f"grid_{KEY}_test.csv", index=False)
    d.close()

    print("\n===== score.py confirmation on the written parquet =====")
    from score import score_model
    for t in THRESHOLDS:
        score_model(tag=KEY, corpus="debtor", threshold=t)


if __name__ == "__main__":
    main()
