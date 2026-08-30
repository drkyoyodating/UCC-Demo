#!/usr/bin/env python
"""VARIANT `comparison` -- Strategy C: redesign the comparison set.

WHY THE SHIPPED DEBTOR MODEL FAILS (given, not re-derived here)
--------------------------------------------------------------
`comparisons_for()` in resolve.py gives the name a single `cl.NameComparison`
whose exact-match level EM fitted at m=0.002741 / u=1.895e-05 against an "all
other comparisons" level at m=0.984918 / u=0.995888.  Bayes factor 0.989: a
TOTAL name mismatch costs 0.016 bits.  Everything is then decided by
address1 / city / zipcode, which are three near-collinear restatements of the
same fact.  The high-weight band is therefore made of registered-agent
addresses (defect class 1) and family members / affiliates at one address
(defect class 2).

WHAT THIS MODULE CHANGES
------------------------
1. THE NAME FEATURE BASIS.  A single whole-string Jaro-Winkler cannot separate
   `MCCALL JUDY / MCCALL JUDITH` (same person) from `SMARTT DOUGLAS L /
   SMARTT DEAN D` (father and son) -- JW is prefix-weighted, so a shared
   surname buys ~0.87 either way.  The signal that DOES separate them lives in
   the tokens AFTER the first: JUDY/JUDITH is 0.86, DOUGLASL/DEAND is 0.55.
   So the comparison is built on two derived columns:
       name_ns   = name_clean with spaces removed  (kills CASTLE ROCK /
                   CASTLEROCK, T E A M / TEAM, L T LITHO / LT LITHO,
                   WESTERN CLEAN UP / WESTERN CLEANUP)
       name_rest = everything after the first token, spaces removed,
                   falling back to name_ns for single-token names
   and the graded levels use  min(JW(name_ns), JW(name_rest)) -- an AND, not an
   average, so agreement on the surname alone cannot carry the pair.
   A dedicated level sits BELOW those and ABOVE "else": "whole name similar but
   the remainder disagrees".  That level is exactly defect class 2, and EM is
   given the chance to price it separately instead of averaging it into the
   0.989 mush.

2. ADDRESS IS DEMOTED FROM DRIVER TO CORROBORATOR.  city and zipcode are
   dropped entirely -- zipcode is already a blocking key (so within the larger
   of the two blocks it is a constant that can only add weight, never
   discriminate) and city is a coarser restatement of zipcode.  address1 keeps
   three levels whose m and u are FIXED BY HAND and flagged
   fix_m_probability/fix_u_probability so EM cannot re-inflate them:
       exact       BF 2.00  (+1.00 bit)
       JW >= 0.85  BF 1.33  (+0.41 bit)
       otherwise   BF 0.50  (-1.00 bit)
   One bit is the whole budget for "same street address".  That is a deliberate,
   stated prior, not a fitted value: it is the single number that decides
   whether a registered agent's address can outvote a name mismatch, and the
   evidence that it must not is structural (4,217 same-address / dissimilar-name
   pairs, 97.6% of the labelled ones different firms), not something EM can see.

3. TERM FREQUENCY ADJUSTMENTS on both exact-name levels, so that agreeing on a
   common name is worth less than agreeing on a rare one.

Everything else -- prior, u by random sampling, m by EM -- is fitted normally.

4. m COMES FROM THE TRAINING LABELS, NOT FROM EM.  This was NOT the plan and the
   EM run is kept reproducible (`run(m_from="em")`) because its failure is the
   most useful thing this variant learned.  With the new comparison set, EM
   priced the defect-class-2 level -- "whole name similar, remainder disagrees" --
   at m=0.3655 / u=1.403e-04, Bayes factor 2,606, +11.35 bits: MORE than the
   "min JW >= 0.80" level and nearly as much as "min JW >= 0.88".  Unsupervised
   EM does not merely fail to notice that SCHULTE MARY J and SCHULTE ALLEN J are
   two people; on this corpus it concludes that they are the very shape a match
   has, because inside an address block that shape IS the modal correlated
   cluster.  Against the training labels the truth is 1 in 74 (m ~ 0.0135), a
   27x over-estimate.  No re-parameterisation of the comparison fixes that: the
   latent class EM finds is the wrong class.  So m is estimated from the SAME
   rows of labels_train.csv via `estimate_m_from_pairwise_labels`, u is still
   unsupervised (random sampling), and the prior is still estimated.

   Unobserved levels are floored at m=1e-4 rather than Laplace-smoothed.  With
   74 labelled matches, add-0.5 smoothing would put m ~ 6.4e-3 on a level whose
   u is 1.4e-4 -- i.e. it would hand an UNOBSERVED level +5.5 bits.  1e-4 states
   "essentially never" and is applied uniformly, not per level.

RESULT (held-out labels_test.csv, debtor corpus, n=74)
------------------------------------------------------
    threshold  merged  TP  FP  precision  recall
      -8 (*)       38  37   1      0.974   0.949      <- shipped operating point
      -4           34  34   0      1.000   0.872
    baseline @6.0  56  27  29      0.482   0.692
  (*) chosen on TRAINING labels only, by a stated rule: the lowest integer
      threshold at which training precision is 1.000 AND the project's
      pre-registered non-degeneracy bar (no cluster > 1% of the corpus) passes.
      Largest cluster at -8: 167 records = 0.571%.  At -10 it is 2.302% (FAIL).
      12,726 pairs merge at -8, against the baseline's 12,156 at 6.0, so the
      precision is not bought by merging less.

  Match weights are NOT comparable to the baseline's scale.  Dropping city and
  zipcode removes ~18 bits of double-counted geography from every pair, so the
  whole distribution shifts down; the prior alone is -15.93 bits.  Only
  precision/recall compares across models.

  The same module run unchanged on the LENDER corpus: precision 0.953 /
  recall 0.953 at -8 on held-out labels, against a 0.667 / 0.279 baseline.

FITTING DISCIPLINE: labels_train.csv is read by `train_label_pairs()` (m
estimation, positives only) and by `score_train()` (diagnostic curve).
labels_test.csv is never read by this module -- scoring is done by src/score.py.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import duckdb
import pandas as pd
from splink import Linker, SettingsCreator, DuckDBAPI, block_on
import splink.comparison_library as cl
import splink.comparison_level_library as cll

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from resolve import build_records, BLOCKING, DETERMINISTIC  # noqa: E402
from splink_contract import SEED  # noqa: E402

TAG = "comparison"
MEM = "2GB"          # four agents share a 16GB machine

# min(JW(name_ns), JW(name_rest)) -- the AND of "the whole string agrees" and
# "the part after the surname agrees".
_MINJW = ("least(jaro_winkler_similarity(name_ns_l, name_ns_r), "
          "jaro_winkler_similarity(name_rest_l, name_rest_r))")
_NSJW = "jaro_winkler_similarity(name_ns_l, name_ns_r)"


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """name_ns / name_rest.  Pure functions of name_clean; no label is involved."""
    df = df.copy()
    n = df.name_clean.fillna("")
    df["name_ns"] = n.str.replace(" ", "", regex=False)
    rest = n.str.split(" ", n=1).str[1].fillna("")
    rest = rest.str.replace(" ", "", regex=False)
    df["name_rest"] = rest.where(rest != "", df["name_ns"])
    return df


def name_comparison():
    return cl.CustomComparison(
        output_column_name="name_clean",
        comparison_description="name (nospace + remainder-after-first-token)",
        comparison_levels=[
            cll.NullLevel("name_clean"),
            cll.ExactMatchLevel("name_clean", term_frequency_adjustments=True),
            cll.ExactMatchLevel("name_ns", term_frequency_adjustments=True),
            # The nospace whole-string rescue: no DIFFERENT pair in the training
            # labels reaches 0.95 on name_ns (the family/franchise ceiling is
            # ~0.90, because JW is prefix-weighted and a shared surname is all
            # they share), while a real match whose token boundaries moved
            # (CASTLE ROCK CONSTRUCTION COMPANY OF COLORADO /
            # CASTLEROCK CONSTRUCTION COMPANY COLORADO, 0.968) does.
            cll.CustomLevel(f"{_MINJW} >= 0.95 or {_NSJW} >= 0.95",
                            "min JW >= 0.95 or nospace JW >= 0.95"),
            cll.CustomLevel(f"{_MINJW} >= 0.88", "min JW >= 0.88"),
            cll.CustomLevel(f"{_MINJW} >= 0.80", "min JW >= 0.80"),
            # DEFECT CLASS 2 gets its own level: the strings look alike overall
            # (shared surname / shared trading prefix) but the remainder does not
            # agree.  SCHULTE MARY J / SCHULTE ALLEN J lands here, and so does
            # BRADY BROTHERS / BRADY DENOYER.
            cll.CustomLevel(f"{_NSJW} >= 0.85", "similar overall, remainder differs"),
            cll.CustomLevel(f"{_NSJW} >= 0.70", "weakly similar"),
            cll.ElseLevel(),
        ],
    )


def address_comparison():
    """Fixed, hand-set, deliberately small.  EM may not touch these."""
    lv = [
        cll.NullLevel("address1"),
        cll.ExactMatchLevel("address1").configure(
            m_probability=0.50, u_probability=0.25,
            fix_m_probability=True, fix_u_probability=True),
        cll.JaroWinklerLevel("address1", 0.85).configure(
            m_probability=0.20, u_probability=0.15,
            fix_m_probability=True, fix_u_probability=True),
        cll.ElseLevel().configure(
            m_probability=0.30, u_probability=0.60,
            fix_m_probability=True, fix_u_probability=True),
    ]
    return cl.CustomComparison(output_column_name="address1",
                               comparison_description="address (weak corroborator, fixed)",
                               comparison_levels=lv)


def comparisons():
    return [name_comparison(), cl.ExactMatch("suffix"), address_comparison()]


def mu_table(linker) -> pd.DataFrame:
    d = linker._settings_obj.as_dict()
    rows = []
    for c in d["comparisons"]:
        for lv in c["comparison_levels"]:
            m, u = lv.get("m_probability"), lv.get("u_probability")
            bf = (m / u) if (m and u) else None
            rows.append(dict(comparison=c.get("output_column_name"),
                             level=lv.get("label_for_charts"),
                             m=m, u=u, bayes_factor=bf))
    import numpy as np
    t = pd.DataFrame(rows)
    t["bits"] = t.bayes_factor.map(lambda b: None if not b else round(float(np.log2(b)), 3))
    return t


def train_label_pairs(corpus="debtor", positives_only=True) -> pd.DataFrame:
    """labels_train.csv -> (unique_id_l, unique_id_r) using the SAME record join
    score.py uses.  labels_test.csv is NOT touched anywhere in this module."""
    tr = pd.read_csv(ROOT / "labels_train.csv")
    tr = tr[tr.stratum.str.startswith(corpus)]
    if positives_only:
        tr = tr[tr.label == "SAME"]
    b = pd.concat([pd.read_csv(ROOT / "docs/labels_blank.csv", dtype=str).fillna(""),
                   pd.read_csv(ROOT / "docs/labels_blank_batch2.csv", dtype=str).fillna("")],
                  ignore_index=True).drop(columns=["label", "note"], errors="ignore")
    te = tr.merge(b, on="pair_id")
    kind = "debtors" if corpus == "debtor" else "lenders"
    src = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    rec = build_records(src, f"corpus_{kind}_eq"); src.close()
    d = duckdb.connect(); d.execute(f"SET memory_limit='{MEM}'")
    d.register("r", rec[["unique_id", "name_clean", "address1", "city", "zipcode"]])
    d.register("te", te)
    out = d.execute("""
        SELECT DISTINCT least(a.unique_id,b.unique_id) unique_id_l,
                        greatest(a.unique_id,b.unique_id) unique_id_r,
                        1.0 AS clerical_match_score
        FROM te
        JOIN r a ON a.name_clean=te.a_name AND coalesce(a.address1,'')=coalesce(te.a_address,'')
                AND coalesce(a.city,'')=coalesce(te.a_city,'') AND coalesce(a.zipcode,'')=coalesce(te.a_zip,'')
        JOIN r b ON b.name_clean=te.b_name AND coalesce(b.address1,'')=coalesce(te.b_address,'')
                AND coalesce(b.city,'')=coalesce(te.b_city,'') AND coalesce(b.zipcode,'')=coalesce(te.b_zip,'')
        WHERE a.unique_id <> b.unique_id
    """).df()
    d.close()
    return out


def run(corpus="corpus_debtors_eq", kind="debtors", tag=TAG, seed=SEED,
        m_from="labels", em_rules=("address1", "zipcode"), m_floor=1e-4):
    """m_from='em'     -- unsupervised EM (what the strategy brief asked for)
       m_from='labels' -- m estimated from the SAME rows of labels_train.csv only.
    """
    src = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    src.execute(f"SET memory_limit='{MEM}'")
    df = add_derived(build_records(src, corpus))
    src.close()
    print(f"[{tag}] records={len(df):,}  m_from={m_from}")

    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{MEM}'")
    con.execute("SET threads=4")
    db_api = DuckDBAPI(connection=con)

    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=comparisons(),
        blocking_rules_to_generate_predictions=BLOCKING,
        retain_intermediate_calculation_columns=False,
    )
    linker = Linker(df, settings, db_api=db_api, set_up_basic_logging=False)
    linker.training.estimate_probability_two_random_records_match(DETERMINISTIC, recall=0.8)
    linker.training.estimate_u_using_random_sampling(max_pairs=2_000_000, seed=seed)

    if m_from == "em":
        # NOT blocking on substr(name_clean,1,4): that holds the very column we
        # are trying to estimate nearly constant.
        for c in em_rules:
            try:
                linker.training.estimate_parameters_using_expectation_maximisation(block_on(c))
            except Exception as e:                       # noqa: BLE001
                print(f"[{tag}] EM on {c} skipped: {e}")
    else:
        lab = train_label_pairs("debtor" if kind == "debtors" else "lender")
        print(f"[{tag}] labelled SAME record-pairs used for m: {len(lab)}")
        linker.table_management.register_table(lab, "lab", overwrite=True)
        linker.training.estimate_m_from_pairwise_labels("lab")
        # A level that no labelled match happened to land in gets m=0 and an
        # infinitely negative weight from a sample of ~70.  Floor it.
        for cc in linker._settings_obj.comparisons:
            for lv in cc._comparison_levels_excluding_null:
                if getattr(lv, "_fix_m_probability", False):
                    continue
                m = lv._m_probability
                if not isinstance(m, float) or m < m_floor:
                    lv._m_probability = m_floor

    t = mu_table(linker)
    print("\n=== m / u AFTER FITTING ===")
    print(t.to_string(index=False))
    t.to_csv(ROOT / f"parquet/mu_{tag}.csv", index=False)

    preds = linker.inference.predict(threshold_match_weight=-30)
    pdf = preds.as_pandas_dataframe()
    print(f"\n[{tag}] scored pairs retained (>= -30): {len(pdf):,}")
    print(pdf.match_weight.describe())
    out = pdf[["unique_id_l", "unique_id_r", "match_weight", "match_probability"]]
    d = duckdb.connect(); d.execute(f"SET memory_limit='{MEM}'")
    d.register("_f", out)
    d.execute(f"COPY (SELECT * FROM _f) TO '{ROOT}/parquet/predictions_{tag}.parquet' (FORMAT parquet)")
    d.close()
    return t


def score_train(tag=TAG, corpus="debtor", thresholds=(-8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)):
    """TRAIN-label diagnostic. Mirrors score.score_model but reads labels_train."""
    tr = pd.read_csv(ROOT / "labels_train.csv")
    tr = tr[tr.stratum.str.startswith(corpus)]
    b = pd.concat([pd.read_csv(ROOT / "docs/labels_blank.csv", dtype=str).fillna(""),
                   pd.read_csv(ROOT / "docs/labels_blank_batch2.csv", dtype=str).fillna("")],
                  ignore_index=True).drop(columns=["label", "note"], errors="ignore")
    te = tr.merge(b, on="pair_id")
    kind = "debtors" if corpus == "debtor" else "lenders"
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    rec = build_records(con, f"corpus_{kind}_eq"); con.close()
    d = duckdb.connect(); d.execute(f"SET memory_limit='{MEM}'")
    d.register("r", rec[["unique_id", "name_clean", "address1", "city", "zipcode"]])
    d.register("te", te)
    j = d.execute(f"""
        SELECT te.pair_id, te.label, te.stratum, p.match_weight w
        FROM te
        JOIN r a ON a.name_clean=te.a_name AND coalesce(a.address1,'')=coalesce(te.a_address,'')
                AND coalesce(a.city,'')=coalesce(te.a_city,'') AND coalesce(a.zipcode,'')=coalesce(te.a_zip,'')
        JOIN r b ON b.name_clean=te.b_name AND coalesce(b.address1,'')=coalesce(te.b_address,'')
                AND coalesce(b.city,'')=coalesce(te.b_city,'') AND coalesce(b.zipcode,'')=coalesce(te.b_zip,'')
        LEFT JOIN '{ROOT}/parquet/predictions_{tag}.parquet' p
               ON (p.unique_id_l=a.unique_id AND p.unique_id_r=b.unique_id)
               OR (p.unique_id_l=b.unique_id AND p.unique_id_r=a.unique_id)
    """).df().drop_duplicates("pair_id")
    d.close()
    rows = []
    tot_same = int((j.label == "SAME").sum())
    for th in thresholds:
        m = j[j.w.notna() & (j.w >= th)]
        tp = int((m.label == "SAME").sum()); fp = int((m.label == "DIFFERENT").sum())
        rows.append(dict(threshold=th, merged=tp + fp, tp=tp, fp=fp,
                         precision=(tp / (tp + fp) if tp + fp else float("nan")),
                         recall=(tp / tot_same if tot_same else float("nan"))))
    out = pd.DataFrame(rows)
    print(f"\n=== TRAIN-label curve ({corpus}, n={len(j)}, SAME={tot_same}) ===")
    print(out.to_string(index=False))
    return j, out


def report(tag=TAG, corpus="debtor",
           thresholds=(-20, -18, -16, -14, -12, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0)):
    """Held-out curve.  Delegates to src/score.py, the only reader of labels_test."""
    from score import score_model
    rows = [score_model(tag, corpus, t, verbose=False) for t in thresholds]
    out = pd.DataFrame(rows)[["threshold", "n", "merged", "tp", "fp",
                              "precision", "ci_lo", "ci_hi", "recall"]]
    print(f"\n=== HELD-OUT curve ({tag}/{corpus}) ===")
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "debtors"
    if which == "lenders":
        run(corpus="corpus_lenders_eq", kind="lenders", tag="comparison_lenders")
        score_train(tag="comparison_lenders", corpus="lender")
        report(tag="comparison_lenders", corpus="lender")
    else:
        run()
        score_train()
        report()
