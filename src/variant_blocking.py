#!/usr/bin/env python
"""VARIANT `blocking` -- STRATEGY G: fix the debtor model at the BLOCKING stage.

WHAT IS WRONG WITH THE SHIPPED MODEL, STATED AS A BLOCKING PROBLEM
------------------------------------------------------------------
The shipped debtor model blocks on

    block_on("zipcode")  UNION  block_on("substr(name_clean,1,4)")

which proposes 2,489,916 candidate pairs out of 427,415,703 possible. The
`zipcode` half contributes 2,110,302 of them, and it is a NAME-FREE rule: it
proposes every pair of debtors that happen to share a five-digit ZIP, no matter
how unrelated their names are. Because the learned model then values names at
almost nothing (exact-name-match m=0.002741, so a total name mismatch costs only
~0.016 bits) the address comparison decides those pairs, and a shared address
inside a shared ZIP merges them.

So the two documented defect classes are really one blocking defect and one
comparison defect:

  C1  same address, DISSIMILAR names (registered agents, farm co-ops, shared
      PO boxes). These pairs exist ONLY because a name-free blocking rule
      proposed them. No comparison-level fix is needed: stop proposing them.
  C2  same address, SIMILAR names -- SEGELKE SHIRLEY / SEGELKE BURLIE E,
      LEY HENRY JR / LEY BARBARA. These survive a name-prefix rule, because the
      shared token IS the surname the prefix keys on. Blocking can only reach
      them if the key extends past the surname.

MEASURED AT BLOCKING, ON labels_train.csv ONLY (138 debtor pairs, 71 SAME / 67
DIFFERENT). "SAME%" is candidate-set recall -- the ceiling on model recall.
"DIFF%" is how many labelled negatives are still proposed; 1 - DIFF% is free
precision that costs no threshold and no comparison tuning. `ceilP` is the
precision you would get if the model merged every candidate it was handed.

    rule                       SAME%    DIFF%   ceilP
    zipcode                   100.0%   100.0%   0.514   <- the culprit
    substr(name_clean,1,4)     97.2%    26.9%   0.793   <- kills all of C1
    substr(name_clean,1,8)     94.4%     9.0%   0.918
    substr(name_clean,1,10)    93.0%     1.5%   0.985
    alast  (last sorted token) 94.4%     9.0%   0.918
    sig4   (sorted 4-char tok) 87.3%     0.0%   1.000
    first2_4 (see below)       93.0%     0.0%   1.000   <- chosen
    desp12 (see below)         84.5%     0.0%   1.000
    first2_4 | sig4            93.0%     0.0%   1.000
    first2_4 | sig4 | desp12   95.8%     0.0%   1.000   <- SHIPPED HERE

THE CHOSEN KEY
--------------
`first2_4` = the FIRST TWO whitespace/punctuation tokens of name_clean, each
truncated to four characters, joined by a space.

  "SEGELKE SHIRLEY"      -> "SEGE SHIR"
  "SEGELKE BURLIE E"     -> "SEGE BURL"     (C2 pair separated at blocking)
  "COOPERS CONST"        -> "COOP CONS"
  "COOPERS CONSTRUCTION" -> "COOP CONS"     (abbreviation still blocks together)

Truncating each token to 4 rather than taking a flat 8-character prefix is what
keeps the abbreviation family together; requiring the SECOND token is what
splits the family/related-entity family apart. Both properties are needed and a
flat prefix gives only one of them: substr(...,1,10) reaches ceilP 0.985 but at
93.0% SAME it loses the same positives AND is a worse rule on the corpus
(45,840 pairs but a 186-record block, versus first2_4's identical block ceiling
with a semantically meaningful key).

`sig4` = the SORTED multiset of 4-char token prefixes. It exists only to survive
token REORDERING ("PETRIE JIMMIE D" / "JIMMIE D PETRIE"), which no prefix rule
can see through. It is unioned in because it costs 200 extra comparisons.

`desp12` = the first 12 characters of name_clean with ALL separators removed.
It exists only to survive TOKENISATION drift -- "L T LITHO"/"LT LITHO",
"CASTLE ROCK CONSTRUCTION..."/"CASTLEROCK CONSTRUCTION..." -- which is a
normalisation defect that this module is not allowed to fix at its source
(normalize.py belongs to another agent). Twelve characters, not eight: the
ladder desp6/7/8/9/10/12 was measured on train and every length below 12 starts
readmitting labelled negatives (desp8 admits 3, desp6 admits 13), while desp12
admits none and still recovers two of the five positives that first2_4 | sig4
loses. It is the LAST rule added and it is added only because it is strictly
dominant on the train labels: +2 candidate positives, +0 candidate negatives,
+995 comparisons.

COMPARISON COUNTS (29,238 debtor records; measured, not estimated)
    zipcode                      2,110,302   largest block   404
    substr(name_clean,1,4)         401,196   largest block   365
    first2_4                        51,418   largest block   186
    sig4                             9,481   largest block    25
    UNION zipcode | pfx4         2,489,916   <- SHIPPED (matches its scored_pairs exactly)
    desp12                          28,394   largest block   184
    UNION first2_4 | sig4           51,618
    UNION first2_4 | sig4 | desp12  52,613   <- THIS VARIANT, 47.3x cheaper

WHAT RECALL IS LOST AT BLOCKING, AND WHERE
------------------------------------------
Candidate-set recall is 95.8% (68 of 71 train SAME pairs). The missing 4.2% is a
HARD CEILING on this variant: no threshold, no comparison and no post-hoc rule
can recover a pair that was never scored. Before desp12 was added the ceiling was
93.0% and the five lost pairs were, in full:

    TEAM PANELS INTERNATIONAL   / T E A M PANELS INTERNATIONAL  (letter-spaced)
    L T LITHO                   / LT LITHO                      (letter-spaced)
    RTB DENVER AVE              / RTB THOMPSON VALLEY           (branch sites)
    CASTLE ROCK CONSTRUCTION... / CASTLEROCK CONSTRUCTION...     (space inserted)
    MCCALL JUDY                 / MCCALL JUDITH                 (given-name form)

Three of the five are TOKENISATION artefacts -- the same string with spaces
added or removed. They are a normalisation defect, not a blocking defect, and
the honest fix lives in normalize.py (which this module is forbidden to touch);
a de-spaced key would recover them, and it is named in the write-up as the next
move rather than smuggled in here. The fourth (RTB DENVER AVE / RTB THOMPSON
VALLEY) is two branch sites of one debtor sharing only a 3-letter stem, and NO
name-based blocking rule can propose it without also proposing the C2 family
pairs -- it is the real price of this strategy. The fifth (JUDY/JUDITH) is a
nickname, reachable only by a name-variant dictionary.

`desp12` recovers the L T LITHO and CASTLE ROCK cases. The three that remain
lost are T E A M PANELS (letter-spacing pushes the informative characters past
position 12), RTB DENVER AVE / RTB THOMPSON VALLEY, and MCCALL JUDY / JUDITH.

WHY EM IS STILL TRAINED ON THE OLD RULES
----------------------------------------
`blocking_rules_to_generate_predictions` and the rules passed to EM are
deliberately DIFFERENT. Splink cannot learn the parameters of a column it has
blocked on -- inside a `first2_4` block the name is nearly constant, so training
EM there would leave the name comparison untrained and silently degenerate (the
exact failure documented at the top of `comparisons_for` in resolve.py). EM is
therefore trained on the SHIPPED pair of rules (zipcode, substr(name,1,4)) with
the SHIPPED seed, so the learned m/u -- and hence the match weight of any given
pair -- are the shipped ones. The ONLY thing this variant changes is WHICH PAIRS
ARE PROPOSED. That makes it a clean single-factor ablation: every difference in
the score curve is attributable to blocking and to nothing else.

FITTING DISCIPLINE
------------------
The blocking rules were selected on labels_train.csv alone (`select_rules()`
below re-derives the table above from train and prints it). labels_test.csv is
read by `report_test()` and by score.py, and by nothing that makes a choice.

Usage:  ./.venv/bin/python src/variant_blocking.py            # select + fit + score
        ./.venv/bin/python src/variant_blocking.py --select   # train-only analysis
        ./.venv/bin/python src/variant_blocking.py --score    # re-score existing parquet
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd
from splink import Linker, SettingsCreator, DuckDBAPI, block_on
from splink.blocking_analysis import count_comparisons_from_blocking_rule

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from splink_contract import SEED                       # noqa: E402
from resolve import build_records, comparisons_for, DETERMINISTIC  # noqa: E402

KEY = "blocking"
CORPUS = "corpus_debtors_eq"
PRED_OUT = ROOT / "parquet" / f"predictions_{KEY}.parquet"
MEM = "1GB"

#: Rules used ONLY to generate candidate pairs. This is the intervention.
BLOCK_PREDICT = [block_on("blk_first2_4"), block_on("blk_sig4"), block_on("blk_desp12")]

#: Rules used ONLY to train EM -- the shipped pair, unchanged, so that the
#: learned parameters are the shipped parameters (see module docstring).
BLOCK_TRAIN = [block_on("zipcode"), block_on("substr(name_clean,1,4)")]

_TOKEN = re.compile(r"[^A-Z0-9]+")


def _tokens(s) -> list[str]:
    return [t for t in _TOKEN.split(str(s or "").upper()) if t]


def first2_4(s) -> str | None:
    """First two tokens, each truncated to 4 chars. None for an empty name."""
    t = _tokens(s)
    return " ".join(x[:4] for x in t[:2]) or None


def desp12(s) -> str | None:
    """First 12 chars with every separator removed; None if the name is shorter,
    so that short names do not all collide into one hub block."""
    t = re.sub(r"[^A-Z0-9]", "", str(s or "").upper())
    return t[:12] if len(t) >= 12 else None


def sig4(s) -> str | None:
    """Sorted multiset of 4-char token prefixes -- order-insensitive."""
    t = sorted(x[:4] for x in _tokens(s))
    return " ".join(t) or None


def add_blocking_columns(df: pd.DataFrame) -> pd.DataFrame:
    """The derived keys. Computed in pandas, not SQL, so the SAME function that
    selected the rule on the train labels is the one Splink blocks on -- a
    reimplementation in SQL is exactly where a blocking rule silently drifts
    away from the analysis that justified it."""
    df = df.copy()
    df["blk_first2_4"] = [first2_4(x) for x in df.name_clean]
    df["blk_sig4"] = [sig4(x) for x in df.name_clean]
    df["blk_desp12"] = [desp12(x) for x in df.name_clean]
    return df


# ----------------------------------------------------------------- train-only
def _train_pairs() -> pd.DataFrame:
    """Debtor TRAIN labels joined to records. Mirrors _pairs() in src/score.py
    (the same blank-file join on pair_id) but reads labels_TRAIN.csv."""
    tr = pd.read_csv(ROOT / "labels_train.csv")
    tr = tr[tr.stratum.str.startswith("debtor")]
    b = pd.concat([pd.read_csv(ROOT / "docs" / "labels_blank.csv", dtype=str).fillna(""),
                   pd.read_csv(ROOT / "docs" / "labels_blank_batch2.csv", dtype=str).fillna("")],
                  ignore_index=True).drop(columns=["label", "note"], errors="ignore")
    return tr.merge(b, on="pair_id")


def select_rules(verbose=True) -> pd.DataFrame:
    """Re-derive the selection table in the docstring from labels_train.csv.

    Prints, per candidate key: candidate-set recall on the labelled SAME pairs,
    the share of labelled DIFFERENT pairs still proposed, and the ceiling
    precision those two imply. Reads no test label.
    """
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    rec = add_blocking_columns(build_records(con, CORPUS))
    con.close()
    tr = _train_pairs()
    d = duckdb.connect(); d.execute(f"SET memory_limit='{MEM}'")
    d.register("r", rec); d.register("te", tr)
    j = d.execute("""
        SELECT te.pair_id, te.label,
               a.name_clean na, b.name_clean nb, a.zipcode za, b.zipcode zb,
               a.blk_first2_4 fa, b.blk_first2_4 fb, a.blk_sig4 sa, b.blk_sig4 sb,
               a.blk_desp12 da, b.blk_desp12 db
        FROM te
        JOIN r a ON a.name_clean=te.a_name AND coalesce(a.address1,'')=coalesce(te.a_address,'')
                AND coalesce(a.city,'')=coalesce(te.a_city,'') AND coalesce(a.zipcode,'')=coalesce(te.a_zip,'')
        JOIN r b ON b.name_clean=te.b_name AND coalesce(b.address1,'')=coalesce(te.b_address,'')
                AND coalesce(b.city,'')=coalesce(te.b_city,'') AND coalesce(b.zipcode,'')=coalesce(te.b_zip,'')
    """).df().drop_duplicates("pair_id")
    d.close()
    S, D = j.label == "SAME", j.label == "DIFFERENT"
    cands = {
        "zipcode":                 j.za.astype(str) == j.zb.astype(str),
        "substr(name,1,4)":        j.na.str[:4] == j.nb.str[:4],
        "substr(name,1,8)":        j.na.str[:8] == j.nb.str[:8],
        "substr(name,1,10)":       j.na.str[:10] == j.nb.str[:10],
        "first2_4":                j.fa == j.fb,
        "sig4":                    j.sa == j.sb,
        "desp12":                  j.da == j.db,
        "first2_4 | sig4":         (j.fa == j.fb) | (j.sa == j.sb),
        "first2_4 | sig4 | desp12": (j.fa == j.fb) | (j.sa == j.sb) | (j.da == j.db),
    }
    rows = []
    for k, m in cands.items():
        s, dd = int((m & S).sum()), int((m & D).sum())
        rows.append(dict(rule=k, same=s, same_pct=100 * s / S.sum(),
                         diff=dd, diff_pct=100 * dd / D.sum(),
                         ceil_precision=s / (s + dd) if s + dd else float("nan")))
    tab = pd.DataFrame(rows)
    if verbose:
        print(f"\n=== BLOCKING SELECTION on labels_train.csv "
              f"(debtor n={len(j)}: {int(S.sum())} SAME / {int(D.sum())} DIFFERENT) ===")
        print(tab.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
        keep = (j.fa == j.fb) | (j.sa == j.sb) | (j.da == j.db)
        print("\nSAME pairs LOST at blocking by 'first2_4 | sig4 | desp12' "
              "(hard ceiling on recall -- unrecoverable downstream):")
        for _, r in j[S & ~keep].iterrows():
            print(f"    {r.na!r:52s} || {r.nb!r}")
        print("\nDIFFERENT pairs STILL PROPOSED by 'first2_4 | sig4 | desp12':")
        lost = j[D & keep]
        for _, r in lost.iterrows():
            print(f"    {r.na!r:52s} || {r.nb!r}")
        if not len(lost):
            print("    (none)")
    return tab


# ------------------------------------------------------------------- fit
def fit(seed: int = SEED) -> dict:
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    df = add_blocking_columns(build_records(con, CORPUS))
    con.close()
    print(f"[{KEY}] records={len(df):,} (from {int(df.n_rows.sum()):,} rows)")

    db_api = DuckDBAPI(":temporary:")
    db_api._con.execute(f"SET memory_limit='{MEM}'")

    print(f"[{KEY}] comparison counts (the cost of each rule, made visible):")
    for label, brs in (("TRAIN-only (EM)", BLOCK_TRAIN), ("PREDICT", BLOCK_PREDICT)):
        for br in brs:
            c = count_comparisons_from_blocking_rule(
                table_or_tables=[df], blocking_rule=br, link_type="dedupe_only",
                db_api=db_api, unique_id_column_name="unique_id")
            n = c["number_of_comparisons_to_be_scored_post_filter_conditions"]
            sql = br.get_blocking_rule("duckdb").blocking_rule_sql
            print(f"    {label:16s} {sql:<64s} {n:>12,}")

    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=comparisons_for("debtors"),
        blocking_rules_to_generate_predictions=BLOCK_PREDICT,
        retain_intermediate_calculation_columns=True,
    )
    linker = Linker(df, settings, db_api=db_api, set_up_basic_logging=False)
    linker.training.estimate_probability_two_random_records_match(DETERMINISTIC, recall=0.8)
    linker.training.estimate_u_using_random_sampling(max_pairs=2_000_000, seed=seed)
    for br in BLOCK_TRAIN:                       # <- NOT the prediction rules
        linker.training.estimate_parameters_using_expectation_maximisation(br)

    preds = linker.inference.predict(threshold_match_weight=-50)
    pdf = preds.as_pandas_dataframe()
    print(f"[{KEY}] scored pairs: {len(pdf):,}  (shipped model: 2,489,916)")
    if not len(pdf):
        raise RuntimeError("ZERO predictions -- check the prior and EM before blocking rules")

    out = pdf[["unique_id_l", "unique_id_r", "match_weight", "match_probability"]]
    d = duckdb.connect(); d.execute(f"SET memory_limit='{MEM}'")
    d.register("_f", out)
    d.execute(f"COPY (SELECT * FROM _f) TO '{PRED_OUT}' (FORMAT parquet)")
    d.close()
    print(f"[{KEY}] wrote {PRED_OUT}")

    # Degeneracy guard, reported not gated: a tight blocking rule plus a low
    # threshold can still blob if one key is a hub.
    stats = {}
    for T in (2.0, 4.0, 6.0):
        cdf = linker.clustering.cluster_pairwise_predictions_at_threshold(
            preds, threshold_match_weight=T).as_pandas_dataframe()
        sizes = cdf.groupby("cluster_id").size()
        stats[T] = dict(clusters=int(cdf.cluster_id.nunique()),
                        singletons=int((sizes == 1).sum()),
                        largest=int(sizes.max()),
                        largest_pct=round(100 * int(sizes.max()) / len(df), 3))
    print(f"[{KEY}] cluster shape: {json.dumps(stats)}")
    return dict(records=len(df), scored_pairs=len(pdf), clusters=stats)


# ------------------------------------------------------------------ report
THRESHOLDS = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]


def report_test():
    """OUT-OF-SAMPLE. Full curve, plus the shipped baseline for comparison."""
    from score import score_model
    print(f"\n=== HELD-OUT TEST CURVE -- variant '{KEY}' (debtor) ===")
    rows = []
    for T in THRESHOLDS:
        r = score_model(tag=KEY, corpus="debtor", threshold=T, verbose=False)
        rows.append(r)
        print(f"  T={T:5.1f}  TP={r['tp']:3d} FP={r['fp']:3d}  "
              f"precision={r['precision']:.3f} [{r['ci_lo']:.3f}-{r['ci_hi']:.3f}]  "
              f"recall={r['recall']:.3f}")
    print("\n=== SHIPPED BASELINE (tag 'debtors') for comparison ===")
    for T in THRESHOLDS:
        r = score_model(tag="debtors", corpus="debtor", threshold=T, verbose=False)
        print(f"  T={T:5.1f}  TP={r['tp']:3d} FP={r['fp']:3d}  "
              f"precision={r['precision']:.3f}  recall={r['recall']:.3f}")
    return rows


if __name__ == "__main__":
    args = set(sys.argv[1:])
    if "--score" not in args:
        select_rules()
    if not (args & {"--select", "--score"}):
        fit()
    if "--select" not in args:
        report_test()
