#!/usr/bin/env python
"""STRATEGY E -- two-stage entity resolution for the DEBTOR corpus.

    Stage 1  deterministic, model-free merge rules with a precision claim of
             their own.  They fire on NAME evidence only, gated by locality.
    Stage 2  the probabilistic (Splink) model, run at a HIGH threshold and
             ONLY on the pairs stage 1 did not already decide.

Why this shape.  The shipped debtor model learned m(exact name match)=0.002741,
so a total name mismatch costs ~0.016 bits and ADDRESS decides every pair: 73.7%
of merges at T=6.0 share an identical `address1`, and the two defect classes are
(C1) same address + dissimilar names and (C2) same address + SIMILAR names --
family members and related firms, which score HIGHEST, so precision FALLS as the
threshold rises.  A model whose top band is its worst band cannot be repaired by
moving the threshold.  Stage 1 sidesteps it entirely: it never looks at address
as *evidence*, only as a *guard*.

--------------------------------------------------------------------------
STAGE 1 -- the four rules
--------------------------------------------------------------------------
Every rule compares `name_clean` (suffix already peeled by normalize.py, so R3
"suffix-only difference -> SAME" is satisfied for free) and every rule is gated
by LOCALITY.

  S1a EXACT    name_clean identical.
  S1b NOSPACE  name_clean identical after deleting all spaces.  Catches the
               initialism-spacing family: `L T LITHO`/`LT LITHO`,
               `CASTLE ROCK CONSTRUCTION`/`CASTLEROCK CONSTRUCTION`,
               `G E JOHNSON`/`GE JOHNSON`.
  S1c EDIT1    Levenshtein(name_clean) <= 1, SUBJECT TO THE INITIAL GUARD below.
               Catches single-character keying errors: `C AND J FIELD SERVICES`/
               `C AND J FOELD SERVICES`, `HANSEN DONAL D`/`HANSEN DONALD D`.
  S1d TRUNC    one name_clean is a strict PREFIX of the other, the prefix is
               >= 8 characters, and the cut does NOT fall on a token boundary.
               Catches source-side field-width truncation: `NIELSONS SKANSK`/
               `NIELSONS SKANSKA`, `COOPERS CONST`/`COOPERS CONSTRUCTION`,
               `PRECISION PAVING AND CONSTRUCTIO`/`... CONSTRUCTION`.
               The token-boundary condition is what stops `BRADY` from merging
               into `BRADY BROTHERS`: there the continuation begins at a space,
               which is a different NAME, not a truncated one (decision-rule R8,
               a person versus their farm).

LOCALITY = same city OR same zipcode OR same non-blank address1.
  This is decision-rule R5 ("same name, different Colorado cities, no shared
  address -> DIFFERENT") expressed as a precondition, and it is the ONLY place
  address is consulted.  It is a veto, never evidence: no stage-1 rule can fire
  on address agreement alone, which is precisely how C1 and C2 are avoided.
  NB it is disjunctive because city strings are themselves dirty (`GRAND JCT`
  vs `GRAND JUNCTION`) and P2 of the decision rule says a lone ZIP disagreement
  is a data error, not an address difference.

THE INITIAL GUARD (S1c only).  A single Levenshtein edit inside a long token is
a typo; a single edit that changes a STANDALONE ONE-CHARACTER TOKEN is a
different person's middle initial, and decision-rule R8 makes those DIFFERENT
borrowers.  Corpus-wide, 108 of the 856 locality-gated edit-1 pairs are of this
second kind -- `GUENZI TARA R`/`GUENZI TARA S`, `ADKINS CATHERINE C`/
`ADKINS CATHERINE A`, `TALBOTT HARRY B`/`TALBOTT HARRY C`,
`HESTER P BERNHARDT TRUST A`/`... TRUST B`, `PHILLIPS COUNTY DIST 3`/`DIST 2`.
S1c therefore requires the MULTISET OF ONE-CHARACTER TOKENS to be identical on
both sides.  The guard costs nothing in recall because the benign members of
that family (`L T LITHO`/`LT LITHO`, `R E MONKS`/`RE MONKS`) are already caught
by S1b, which deletes the spaces instead of editing across them.

The guard is derived from the pre-registered decision rule, NOT from the labels:
it changes zero train-label outcomes (verified -- see `audit()`), and exists
because inspecting the corpus-wide output of S1c showed the class.

--------------------------------------------------------------------------
STAGE 2
--------------------------------------------------------------------------
The same Splink specification as `resolve.py` (identical blocking, comparisons,
prior and seed -- so its weights are on the SAME scale as the labelled strata),
scored over the blocked candidate set with the stage-1 pairs removed.  Stage 1
pairs are emitted at match_weight 100.0, so a threshold sweep of the single
output parquet reads as:

    threshold >= 100    stage 1 ALONE
    threshold  = T      stage 1 + stage 2 at T

which is why the curve is reported over that whole range rather than at a point.

OUTPUTS
    parquet/predictions_two_stage.parquet     stage 1 (w=100) UNION stage 2
    parquet/predictions_two_stage_s1.parquet  stage 1 only
    parquet/predictions_two_stage_s2.parquet  stage 2 only

Fitting reads labels_train.csv and nothing else.  labels_test.csv is opened only
by score.py, after fitting is finished.

--------------------------------------------------------------------------
RESULT -- held-out test, debtor corpus (n=74 pairs, 39 SAME / 35 DIFFERENT)
--------------------------------------------------------------------------
    shipped baseline @6.0     precision 0.482  (TP=27 FP=29)  recall 0.692
    STAGE 1 ALONE             precision 1.000  (TP=36 FP= 0)  recall 0.923
                              Wilson 95% CI on precision: 0.904 - 1.000

Stage 1 alone dominates the baseline on BOTH axes: +0.518 precision at +0.231
recall.  This is not a precision win bought with recall.

    combined, threshold sweep (stage 1 is everything at T >= 14):
        T= 2  TP=39 FP=35  prec 0.527  rec 1.000
        T= 6  TP=38 FP=29  prec 0.567  rec 0.974
        T= 9  TP=36 FP=12  prec 0.750  rec 0.923
        T=10  TP=36 FP= 6  prec 0.857  rec 0.923
        T=11  TP=36 FP= 1  prec 0.973  rec 0.923
        T=14+ TP=36 FP= 0  prec 1.000  rec 0.923      <-- stage 1 alone

    STAGE 2 IN ISOLATION (undecided pairs only) is worthless, and worthless in
    the specific shape the diagnosis predicts -- its precision FALLS as the
    threshold RISES:
        T= 2  TP=3 FP=35  prec 0.079        T= 8  TP=1 FP=25  prec 0.038
        T= 6  TP=2 FP=29  prec 0.065        T=10  TP=0 FP= 6  prec 0.000
        T= 7  TP=2 FP=25  prec 0.074        T=12  TP=0 FP= 1  prec 0.000

    Read that carefully: once deterministic name matching has taken the true
    matches out, EVERY remaining pair the model ranks highly is wrong on the
    labelled sample.  The model's entire marginal contribution over an exact
    name join is negative.  At T=6.0 it would add 9,291 corpus merges to stage
    1's 7,253, and the held-out estimate of their precision is 0.065.

    RECOMMENDED OPERATING POINT: threshold 14.0 on predictions_two_stage.parquet
    (identically, threshold 6.0 on predictions_two_stage_s1.parquet).  Stage 2
    is retained in the artefact so the sweep above is reproducible from one file,
    not because any threshold makes it pay.

CORPUS BEHAVIOUR of stage 1 alone: 7,253 merged pairs over 29,238 records ->
7,660 records in 3,154 multi-record clusters, 24,732 clusters in total.  Largest
cluster 23 records = 0.079% of the corpus, so the pre-registered non-degeneracy
bar (no cluster > 1%) passes with an order of magnitude to spare; the shipped
model at 6.0 needed an explicit override discussion to clear it.

HONEST CAVEATS
  1. n=74.  Precision 1.000 has a Wilson lower bound of 0.904, which is BELOW
     the 0.95 target.  The point estimate clears 0.95; the interval does not.
     Zero false positives in 36 merges is the strongest statement the label set
     can support -- it is not the same as a proven 0.95.
  2. `recall` here is over the 39 labelled SAME pairs, a stratified sample
     dominated by the recall-probe and low-weight strata.  It is not corpus
     recall.  The baseline's 0.692 is measured the same way, so the comparison
     is like-for-like, but neither number is a corpus-level recall estimate.
  3. Stage 1 cannot merge a true match whose two names are genuinely different
     strings.  On TRAIN the six misses are all of that kind:
     `WESTERN CLEAN UP`/`WESTERN CLEANUP CORPERATION`, `MCCALL JUDY`/
     `MCCALL JUDITH`, `CRYSTAL CLEAR CAR WASH`/`... ETAL`,
     `CASTLE ROCK CONSTRUCTION COMPANY OF COLORADO`/`CASTLEROCK CONSTRUCTION
     COMPANY COLORADO`, and `RTB DENVER AVE`/`RTB THOMPSON VALLEY`.  Three
     held-out SAME pairs are missed for the same reason.  Closing that gap needs
     token-level name evidence, not another string-distance threshold.
  4. Design choices were fixed against the pre-registered decision rule and the
     TRAIN half only, and every one of them is train-label-neutral:
       - locality gate (R5): changes nothing on train (all labelled strata are
         same-locality by construction); it exists for the corpus, where
         `JOHNSON CONSTRUCTION` in two cities must stay apart.
       - initial guard (R8): removes exactly one train pair from S1c, which S1b
         merges anyway.  Net train recall cost 0; corpus-wide it withholds 108
         merges of the `GUENZI TARA R`/`GUENZI TARA S` shape.
       - S1d minimum prefix length: train result is identical at 4, 6, 8, 10 and
         12, so the constant is not load-bearing on the labels.  8 was chosen on
         the corpus (251 pairs at 4 vs 231 at 8).
       - S1d token-boundary condition: this one DOES cost.  Dropping it would add
         one train TP, and would raise corpus-wide S1d from 135 to 1,563 merges
         of the `BRADY`/`BRADY BROTHERS` shape that R8 calls DIFFERENT and that
         the label set does not cover.  Kept, deliberately, precision-first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolve import BLOCKING, DETERMINISTIC, build_records, comparisons_for  # noqa: E402
from splink_contract import SEED  # noqa: E402

KEY = "two_stage"
CORPUS = "corpus_debtors_eq"
MEM = "1GB"
STAGE1_WEIGHT = 100.0

# --------------------------------------------------------------------------
# Stage 1, as SQL. `a` and `b` are aliases over the record table built in
# _stage1(); every predicate is pure name arithmetic except LOCALITY.
# --------------------------------------------------------------------------
LOCALITY = "(a.city = b.city OR a.zip = b.zip OR (a.addr <> '' AND a.addr = b.addr))"

#: multiset of one-character tokens, sorted -- the initial guard's comparand.
_INITIALS = "list_sort(list_filter(str_split({t}.nm, ' '), x -> length(x) = 1))"

_TRUNC_ONE_WAY = """
    (length({s}.nm) >= 8
     AND length({s}.nm) < length({l}.nm)
     AND {l}.nm LIKE {s}.nm || '%'
     AND substr({l}.nm, length({s}.nm) + 1, 1) <> ' '
     AND substr({s}.nm, length({s}.nm), 1) <> ' ')
"""

RULES: dict[str, str] = {
    "S1a_EXACT": "a.nm = b.nm",
    "S1b_NOSPACE": "replace(a.nm, ' ', '') = replace(b.nm, ' ', '')",
    "S1c_EDIT1": (
        "levenshtein(a.nm, b.nm) <= 1 AND "
        + _INITIALS.format(t="a") + " = " + _INITIALS.format(t="b")
    ),
    "S1d_TRUNC": (
        _TRUNC_ONE_WAY.format(s="a", l="b") + " OR " + _TRUNC_ONE_WAY.format(s="b", l="a")
    ),
}
STAGE1_ANY = " OR ".join(f"({c})" for c in RULES.values())


# --------------------------------------------------------------------------
def _records() -> pd.DataFrame:
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    rec = build_records(con, CORPUS)
    con.close()
    return rec


def _reg(d: duckdb.DuckDBPyConnection, rec: pd.DataFrame) -> None:
    """Register the record table in the shape stage 1's SQL expects."""
    d.execute(f"SET memory_limit='{MEM}'")
    d.register("_rec", rec[["unique_id", "name_clean", "suffix",
                            "address1", "city", "state", "zipcode"]])
    d.execute("""
        CREATE OR REPLACE TABLE rr AS
        SELECT unique_id        AS id,
               name_clean       AS nm,
               coalesce(suffix,   '') AS sfx,
               coalesce(address1, '') AS addr,
               coalesce(city,     '') AS city,
               coalesce(zipcode,  '') AS zip
        FROM _rec
    """)


def _stage1(d: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """All pairs merged deterministically, with the rule that fired.

    Each rule is EQUI-BLOCKED on its own key and the four results unioned; a
    single OR-join over all four degenerates into a cross product (29,238
    records -> 4.3e8 pairs) and never returns. Every block is provably lossless
    for its rule:

      S1a  key = nm.
      S1b  key = nm with spaces deleted.
      S1c  key = first 3 chars, UNION key = last 3 chars. A single edit cannot
           change both ends of a string, so an edit-1 pair agrees on one of them.
      S1d  key = first 8 chars. The rule requires the shorter name to be a
           prefix of length >= 8, so both names share their first 8 characters.

    Rule attribution is by priority a < b < c < d, so each pair is reported once
    under the strongest rule that explains it.
    """
    blocks = {
        "S1a_EXACT":   ["a.nm = b.nm"],
        "S1b_NOSPACE": ["replace(a.nm,' ','') = replace(b.nm,' ','')"],
        "S1c_EDIT1":   ["substr(a.nm,1,3) = substr(b.nm,1,3) AND abs(length(a.nm)-length(b.nm)) <= 1",
                        "right(a.nm,3) = right(b.nm,3) AND abs(length(a.nm)-length(b.nm)) <= 1"],
        "S1d_TRUNC":   ["substr(a.nm,1,8) = substr(b.nm,1,8)"],
    }
    parts = []
    for rule, keys in blocks.items():
        for k in keys:
            parts.append(f"""
                SELECT a.id AS unique_id_l, b.id AS unique_id_r, '{rule}' AS rule
                FROM rr a JOIN rr b ON a.id < b.id AND ({k})
                WHERE ({RULES[rule]}) AND {LOCALITY}""")
    d.execute("CREATE OR REPLACE TABLE s1_raw AS " + " UNION ALL ".join(parts))
    d.execute("""
        CREATE OR REPLACE TABLE s1 AS
        SELECT unique_id_l, unique_id_r, min(rule) AS rule
        FROM s1_raw GROUP BY 1, 2
    """)
    return d.execute("SELECT * FROM s1").df()


def _stage2(rec: pd.DataFrame) -> pd.DataFrame:
    """The shipped Splink specification, refit here so this module is
    self-contained and never races another agent's parquet. Same SEED, same
    blocking, same comparisons -> same weight scale as the labelled strata."""
    from splink import Linker, SettingsCreator, DuckDBAPI

    db_api = DuckDBAPI(":temporary:")
    db_api._con.execute(f"SET memory_limit='{MEM}'")
    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=comparisons_for("debtors"),
        blocking_rules_to_generate_predictions=BLOCKING,
        retain_intermediate_calculation_columns=False,
    )
    linker = Linker(rec, settings, db_api=db_api, set_up_basic_logging=False)
    linker.training.estimate_probability_two_random_records_match(DETERMINISTIC, recall=0.8)
    linker.training.estimate_u_using_random_sampling(max_pairs=2_000_000, seed=SEED)
    for br in BLOCKING:
        linker.training.estimate_parameters_using_expectation_maximisation(br)
    pdf = linker.inference.predict(threshold_match_weight=-50).as_pandas_dataframe()
    return pdf[["unique_id_l", "unique_id_r", "match_weight", "match_probability"]]


def _pq(frame: pd.DataFrame, path: Path) -> None:
    """DuckDB, not pandas.to_parquet -- pyarrow is deliberately absent."""
    d = duckdb.connect()
    d.register("_f", frame)
    d.execute(f"COPY (SELECT * FROM _f) TO '{path}' (FORMAT parquet)")
    d.close()


# --------------------------------------------------------------------------
def _train_pairs() -> pd.DataFrame:
    """TRAIN labels only, joined to record text. Mirrors _pairs() in score.py."""
    tr = pd.read_csv(ROOT / "labels_train.csv")
    tr = tr[tr.stratum.str.startswith("debtor")]
    b = pd.concat([pd.read_csv(ROOT / "docs" / "labels_blank.csv", dtype=str).fillna(""),
                   pd.read_csv(ROOT / "docs" / "labels_blank_batch2.csv", dtype=str).fillna("")],
                  ignore_index=True).drop(columns=["label", "note"], errors="ignore")
    return tr.merge(b, on="pair_id")


def audit(d: duckdb.DuckDBPyConnection) -> None:
    """Per-rule TRAIN precision, and the initial guard's cost. Fitting-side."""
    tr = _train_pairs()
    d.register("tr", tr)
    j = d.execute("""
        SELECT tr.pair_id, tr.label, tr.stratum, a.id AS il, b.id AS ir
        FROM tr
        JOIN rr a ON a.nm = tr.a_name AND a.addr = coalesce(tr.a_address,'')
                 AND a.city = coalesce(tr.a_city,'') AND a.zip = coalesce(tr.a_zip,'')
        JOIN rr b ON b.nm = tr.b_name AND b.addr = coalesce(tr.b_address,'')
                 AND b.city = coalesce(tr.b_city,'') AND b.zip = coalesce(tr.b_zip,'')
    """).df().drop_duplicates("pair_id")
    d.register("j", j)
    print(f"\n[train audit] debtor train pairs joined: {len(j)} "
          f"(SAME={int((j.label=='SAME').sum())} DIFFERENT={int((j.label=='DIFFERENT').sum())})")
    q = """
        SELECT s1.rule, j.label, count(*) n FROM j
        JOIN s1 ON (s1.unique_id_l=j.il AND s1.unique_id_r=j.ir)
                OR (s1.unique_id_l=j.ir AND s1.unique_id_r=j.il)
        GROUP BY 1,2 ORDER BY 1,2
    """
    print(d.execute(q).df().to_string(index=False))
    tot = d.execute("""
        SELECT j.label, count(*) n FROM j
        JOIN s1 ON (s1.unique_id_l=j.il AND s1.unique_id_r=j.ir)
                OR (s1.unique_id_l=j.ir AND s1.unique_id_r=j.il)
        GROUP BY 1""").df()
    tp = int(tot.n[tot.label == "SAME"].sum()); fp = int(tot.n[tot.label == "DIFFERENT"].sum())
    same = int((j.label == "SAME").sum())
    print(f"[train audit] STAGE 1 combined: TP={tp} FP={fp} "
          f"precision={tp/max(tp+fp,1):.3f} recall={tp/max(same,1):.3f}")

    # cost of the initial guard, on train
    g = d.execute(f"""
        SELECT j.label, a.nm AS nm_l, b.nm AS nm_r,
               EXISTS(SELECT 1 FROM s1
                      WHERE (s1.unique_id_l=j.il AND s1.unique_id_r=j.ir)
                         OR (s1.unique_id_l=j.ir AND s1.unique_id_r=j.il)) AS merged_by_other_rule
        FROM j JOIN rr a ON a.id=j.il JOIN rr b ON b.id=j.ir
        WHERE levenshtein(a.nm, b.nm) <= 1
          AND NOT ({RULES['S1c_EDIT1']}) AND {LOCALITY}
    """).df()
    lost = g[~g.merged_by_other_rule & (g.label == "SAME")]
    print(f"[train audit] initial guard removes {len(g)} train pair(s) from S1c; "
          f"{int(g.merged_by_other_rule.sum())} still merge via another rule; "
          f"net TRAIN recall cost = {len(lost)}")
    if len(g):
        print(g.to_string(index=False))


def run(with_stage2: bool = True) -> dict:
    rec = _records()
    d = duckdb.connect()
    _reg(d, rec)
    print(f"[{KEY}] records={len(rec):,}")

    s1 = _stage1(d)
    print(f"[{KEY}] stage 1 merged pairs: {len(s1):,}")
    print(s1.rule.value_counts().to_string())
    audit(d)

    s1["match_weight"] = STAGE1_WEIGHT
    s1["match_probability"] = 1.0
    s1_out = s1[["unique_id_l", "unique_id_r", "match_weight", "match_probability"]]
    _pq(s1_out, ROOT / "parquet" / f"predictions_{KEY}_s1.parquet")

    if not with_stage2:
        _pq(s1_out, ROOT / "parquet" / f"predictions_{KEY}.parquet")
        d.close()
        return {"stage1": len(s1), "stage2": 0}

    pdf = _stage2(rec)
    print(f"[{KEY}] stage 2 scored pairs: {len(pdf):,}")
    d.register("_p", pdf)
    s2 = d.execute("""
        SELECT p.* FROM _p p
        ANTI JOIN s1 ON (s1.unique_id_l = p.unique_id_l AND s1.unique_id_r = p.unique_id_r)
                     OR (s1.unique_id_l = p.unique_id_r AND s1.unique_id_r = p.unique_id_l)
    """).df()
    print(f"[{KEY}] stage 2 undecided pairs: {len(s2):,} "
          f"({len(pdf)-len(s2):,} already decided by stage 1)")
    _pq(s2, ROOT / "parquet" / f"predictions_{KEY}_s2.parquet")
    _pq(pd.concat([s1_out, s2], ignore_index=True), ROOT / "parquet" / f"predictions_{KEY}.parquet")
    d.close()
    return {"stage1": len(s1), "stage2": len(s2)}


def curve(tag: str = KEY, thresholds=None) -> pd.DataFrame:
    """Held-out precision/recall curve. Reporting only -- never called by run()."""
    from score import score_model
    thresholds = thresholds or [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 50, 100]
    rows = [score_model(tag=tag, corpus="debtor", threshold=float(t), verbose=False)
            for t in thresholds]
    return pd.DataFrame(rows)[["threshold", "merged", "tp", "fp",
                               "precision", "ci_lo", "ci_hi", "recall"]]


if __name__ == "__main__":
    if "--curve" in sys.argv:
        for t in (f"{KEY}_s1", f"{KEY}_s2", KEY):
            print(f"\n===== {t} =====")
            print(curve(t).to_string(index=False))
    else:
        print(run(with_stage2="--s1-only" not in sys.argv))
