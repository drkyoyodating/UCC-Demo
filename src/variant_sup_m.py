#!/usr/bin/env python
"""VARIANT `sup_m` -- SUPERVISED m FROM HAND LABELS (Strategy A).

Self-contained. Touches nothing the shipped pipeline owns: it IMPORTS
`build_records`, `BLOCKING`, `DETERMINISTIC` from src/resolve.py, writes only
parquet/predictions_sup_m.parquet and variants/*, and is scored by src/score.py
against the held-out labels.

    ./.venv/bin/python src/variant_sup_m.py

==========================================================================
WHY
==========================================================================
The shipped debtor model's EM fit is degenerate on the name field:

    name_clean  Exact match   m = 0.002741  u = 1.8953e-05  BF =   144.6  (+7.18 bits)
    name_clean  ELSE          m = 0.984918  u = 0.995888    BF =     0.989 (-0.016 bits)

EM concluded that only 0.27% of TRUE matches share a name, so A TOTAL NAME
MISMATCH COSTS 0.016 BITS -- nothing. Meanwhile city (BF 42.6) and zipcode
(BF 181.9) each carry hundreds of times more evidence than the name. The
shipped model is not a name matcher; it is an address matcher with a name
column attached, and every documented defect follows from that one number.

EM is unsupervised: it found *a* two-component mixture of the corpus, just not
the one we mean by "the same firm". On this corpus the dominant latent split is
"same postal area / different postal area", and that is what it locked onto.
We have 246 hand labels. The textbook fix is to stop asking EM what a match
looks like and tell it, via `estimate_m_from_pairwise_labels`.

==========================================================================
WHAT CHANGES (exactly one thing)
==========================================================================
  prior : UNCHANGED -- estimate_probability_two_random_records_match on the same
          two deterministic rules at recall=0.8  (-> 1.6015e-05, identical).
  u     : UNCHANGED -- random sampling, max_pairs=2e6, seed=SEED (20260830). u
          is a property of the corpus, not of the match definition; re-using it
          keeps the comparison against the baseline clean.
  m     : REPLACED. No EM is run at all. m comes from the labelled SAME pairs
          in labels_train.csv.
  features: UNCHANGED -- byte-identical to resolve.comparisons_for(). Otherwise
          the comparison confounds "supervised m" with "different features".

==========================================================================
TWO API TRAPS, both verified in the installed 4.0.16 source
==========================================================================
(1) `estimate_m_from_pairwise_labels` IGNORES `clerical_match_score`.
    splink/internals/m_from_labels.py hardcodes `cast(1.0 as float8) as
    match_probability` for every row of the labels table. So the table must
    contain the SAME pairs ONLY. Feeding it all 138 debtor training labels
    would define "true match" as "any pair a human bothered to look at", which
    is the one change that would make this worse than EM. The 67 DIFFERENT
    labels are still load-bearing -- they select the threshold and they measure
    the fit -- but they are not fed to the estimator.

(2) A level never observed among the labelled SAME pairs is NOT floored at
    1e-6, which is what `ComparisonLevel.m_probability` returns for
    LEVEL_NOT_OBSERVED_TEXT and what you would guess from reading that property
    alone. `estimate_m_from_pairwise_labels` leaves `_m_probability = None`,
    and the property then returns the comparison library's
    `default_m_probability` -- an arbitrary constant baked into the template:

        name_clean  JW >= 0.7   unobserved -> m = 0.01250  ->  +1.65 bits
        address1    ELSE        unobserved -> m = 0.01667  ->  -5.87 bits
        zipcode     ELSE        unobserved -> m = 0.05000  ->  -4.31 bits

    Splink prints only "m values not fully trained" at predict() time and then
    scores 2.5M pairs with those constants. With 71 SAME pairs three levels land
    here, and one of them (name JW>=0.7, +1.65 bits) is POSITIVE evidence
    invented by a library default -- it is what keeps pairs like
    COOKSEY DAMARIS E / COOKSEY VERNON L alive at weight 10.09. This module
    PRINTS every such level explicitly. They are deliberately LEFT at the
    default: hand-picking a floor after seeing the labels is exactly the tuning
    the pre-registration forbids, and (measured) smoothing them changes the
    train confusion matrix by zero pairs.

==========================================================================
THRESHOLD -- chosen on labels_train.csv, by 5-fold cross-validation
==========================================================================
The in-sample train curve is optimistic because m was fitted on it. So the
operating point is chosen by 5-fold CV WITHIN THE TRAINING LABELS
(`cv_threshold_curve` below): for each fold, m is re-fitted from the SAME pairs
of the other four folds and the held-out fold is scored with it.

RULE, stated before the test file was opened: the lowest 0.5-step match weight
whose CROSS-VALIDATED training precision is >= 0.95 while merging >= 10 pairs.
It returns 14.0 (CV precision 1.000 at 14.0; 0.928 at 13.5).

`labels_test.csv` is read by src/score.py and by nothing in this file.
"""
from __future__ import annotations

import json, math, sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from splink import Linker, SettingsCreator, DuckDBAPI
import splink.comparison_library as cl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from splink_contract import SEED                             # noqa: E402
from resolve import build_records, BLOCKING, DETERMINISTIC   # noqa: E402

TAG = "sup_m"
MEM = "2GB"                      # four variants run concurrently on a 16GB box
CORPUS = {"debtors": "corpus_debtors_eq", "lenders": "corpus_lenders_eq"}
VDIR = ROOT / "variants"

#: `default_m_probability` baked into the Splink 4.0.16 comparison-library
#: templates, read off the constructed Comparison objects. Reproduced here ONLY
#: so the CV can replicate Splink's own fallback exactly; nothing sets these.
LIB_DEFAULT_M = {
    "name_clean": {4: 0.95, 3: 0.0125, 2: 0.0125, 1: 0.0125, 0: 0.0125},
    "suffix":     {1: 0.95, 0: 0.05},
    "address1":   {3: 0.95, 2: 1 / 60, 1: 1 / 60, 0: 1 / 60},
    "city":       {1: 0.95, 0: 0.05},
    "zipcode":    {1: 0.95, 0: 0.05},
}
#: (comparison, gamma column, number of non-null levels). gamma counts DOWN from
#: the most similar level, matching Splink's comparison_vector_value.
GAMMA = [("name_clean", "g_name", 5), ("suffix", "g_suffix", 2),
         ("address1", "g_addr", 4), ("city", "g_city", 2), ("zipcode", "g_zip", 2)]


def comparisons():
    """IDENTICAL to resolve.comparisons_for(). Do not 'improve' it here."""
    return [
        cl.NameComparison("name_clean"),
        cl.ExactMatch("suffix"),
        cl.JaroWinklerAtThresholds("address1", [0.9, 0.7]),
        cl.ExactMatch("city"),
        cl.ExactMatch("zipcode"),
    ]


def _pq(frame, path):
    """DuckDB writes parquet; pandas.to_parquet needs pyarrow, deliberately absent."""
    d = duckdb.connect(); d.execute(f"SET memory_limit='{MEM}'")
    d.register("_f", frame)
    d.execute(f"COPY (SELECT * FROM _f) TO '{path}' (FORMAT parquet)")
    d.close()


# --------------------------------------------------------------- label wiring
def _label_fields() -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(ROOT / "docs" / "labels_blank.csv", dtype=str).fillna(""),
         pd.read_csv(ROOT / "docs" / "labels_blank_batch2.csv", dtype=str).fillna("")],
        ignore_index=True,
    ).drop(columns=["label", "note"], errors="ignore")


def train_pairs(rec: pd.DataFrame, corpus: str = "debtor") -> pd.DataFrame:
    """labels_train.csv joined to record ids. NEVER touches labels_test.csv.

    The join key is the record TEXT (name_clean, address1, city, zipcode),
    lifted verbatim from score._pairs() so the pairs this fits on and the pairs
    score.py measures are matched by the same rule.
    """
    tr = pd.read_csv(ROOT / "labels_train.csv")
    tr = tr[tr.stratum.str.startswith(corpus)].merge(_label_fields(), on="pair_id")
    d = duckdb.connect(); d.execute(f"SET memory_limit='{MEM}'")
    d.register("r", rec[["unique_id", "name_clean", "address1", "city", "zipcode"]])
    d.register("tr", tr)
    out = d.execute("""
        SELECT tr.pair_id, tr.label, tr.stratum,
               a.unique_id AS unique_id_l, b.unique_id AS unique_id_r
        FROM tr
        JOIN r a ON a.name_clean=tr.a_name
                AND coalesce(a.address1,'')=coalesce(tr.a_address,'')
                AND coalesce(a.city,'')   =coalesce(tr.a_city,'')
                AND coalesce(a.zipcode,'')=coalesce(tr.a_zip,'')
        JOIN r b ON b.name_clean=tr.b_name
                AND coalesce(b.address1,'')=coalesce(tr.b_address,'')
                AND coalesce(b.city,'')   =coalesce(tr.b_city,'')
                AND coalesce(b.zipcode,'')=coalesce(tr.b_zip,'')
    """).df().drop_duplicates("pair_id")
    d.close()
    return out


def gamma_frame(rec: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """Comparison vector + name term-frequency for each labelled training pair.

    The CASE ladders are transcriptions of the `sql_condition` strings in the
    fitted model JSON, so `cv_threshold_curve` reproduces Splink's own
    match_weight to 4e-15 (asserted in __main__).
    """
    d = duckdb.connect(); d.execute(f"SET memory_limit='{MEM}'")
    d.register("r", rec[["unique_id", "name_clean", "suffix", "address1", "city", "zipcode"]])
    d.register("lab", lab)
    g = d.execute("""
        SELECT lab.pair_id, lab.label, lab.stratum,
          CASE WHEN a.name_clean IS NULL OR b.name_clean IS NULL THEN -1
               WHEN a.name_clean = b.name_clean THEN 4
               WHEN jaro_winkler_similarity(a.name_clean,b.name_clean) >= 0.92 THEN 3
               WHEN jaro_winkler_similarity(a.name_clean,b.name_clean) >= 0.88 THEN 2
               WHEN jaro_winkler_similarity(a.name_clean,b.name_clean) >= 0.7  THEN 1
               ELSE 0 END AS g_name,
          CASE WHEN a.suffix IS NULL OR b.suffix IS NULL THEN -1
               WHEN a.suffix = b.suffix THEN 1 ELSE 0 END AS g_suffix,
          CASE WHEN a.address1 IS NULL OR b.address1 IS NULL THEN -1
               WHEN a.address1 = b.address1 THEN 3
               WHEN jaro_winkler_similarity(a.address1,b.address1) >= 0.9 THEN 2
               WHEN jaro_winkler_similarity(a.address1,b.address1) >= 0.7 THEN 1
               ELSE 0 END AS g_addr,
          CASE WHEN a.city IS NULL OR b.city IS NULL THEN -1
               WHEN a.city = b.city THEN 1 ELSE 0 END AS g_city,
          CASE WHEN a.zipcode IS NULL OR b.zipcode IS NULL THEN -1
               WHEN a.zipcode = b.zipcode THEN 1 ELSE 0 END AS g_zip,
          a.name_clean AS shared_name
        FROM lab
        JOIN r a ON a.unique_id = lab.unique_id_l
        JOIN r b ON b.unique_id = lab.unique_id_r
    """).df().drop_duplicates("pair_id")
    d.close()
    tf = rec.name_clean.value_counts() / len(rec)     # Splink's tf table, exactly
    g["tf_name"] = g.shared_name.map(tf)
    return g


# ---------------------------------------------------- parameter table / report
def m_table(linker) -> pd.DataFrame:
    rows = []
    for c in linker._settings_obj.comparisons:
        for lv in c._comparison_levels_excluding_null:
            m, u = lv.m_probability, lv.u_probability
            bf = (m / u) if (m and u) else float("nan")
            rows.append(dict(
                comparison=c.output_column_name,
                level=lv.label_for_charts,
                gamma=lv.comparison_vector_value,
                m=m, u=u, bayes_factor=bf,
                bits=math.log2(bf) if (bf == bf and bf > 0) else float("nan"),
                # True when the level was never seen among the labelled SAME
                # pairs, i.e. `m` above is a comparison-library constant, not data.
                m_from_library_default=(lv._m_probability is None),
            ))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- the fit
def fit(kind: str = "debtors", corpus_label: str = "debtor", tag: str = TAG) -> dict:
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    rec = build_records(con, CORPUS[kind]); con.close()
    print(f"[{tag}] records={len(rec):,}")

    lab = train_pairs(rec, corpus_label)
    same = lab[lab.label == "SAME"][["unique_id_l", "unique_id_r"]].reset_index(drop=True)
    n_diff = int((lab.label == "DIFFERENT").sum())
    print(f"[{tag}] training labels joined to records: {len(lab)} "
          f"(SAME={len(same)}, DIFFERENT={n_diff})")
    print(f"[{tag}] m estimated from the {len(same)} SAME pairs ONLY "
          f"(estimate_m_from_pairwise_labels ignores clerical_match_score)")
    if len(same) < 30:
        raise RuntimeError("too few SAME labels to estimate m -- check the record join")

    db_api = DuckDBAPI(":temporary:")
    db_api._con.execute(f"SET memory_limit='{MEM}'")

    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=comparisons(),
        blocking_rules_to_generate_predictions=BLOCKING,
        retain_intermediate_calculation_columns=True,
    )
    linker = Linker(rec, settings, db_api=db_api, set_up_basic_logging=False)
    linker.training.estimate_probability_two_random_records_match(DETERMINISTIC, recall=0.8)
    linker.training.estimate_u_using_random_sampling(max_pairs=2_000_000, seed=SEED)
    linker.table_management.register_table(same, "__labels_sup_m", overwrite=True)
    linker.training.estimate_m_from_pairwise_labels("__labels_sup_m")

    mt = m_table(linker)
    print(f"\n[{tag}] SUPERVISED PARAMETERS  (baseline exact-name m was 0.002741)")
    with pd.option_context("display.width", 200, "display.max_rows", 60):
        print(mt.to_string(index=False))
    dflt = mt[mt.m_from_library_default]
    if len(dflt):
        print(f"\n[{tag}] !! {len(dflt)} level(s) NEVER OBSERVED among the {len(same)} labelled "
              f"SAME pairs. Splink silently substitutes the comparison library's "
              f"default_m_probability (NOT 1e-6):")
        for _, r in dflt.iterrows():
            print(f"      {r.comparison:<11s} / {r.level:<44s} m={r.m:.5f} -> {r.bits:+.2f} bits")

    preds = linker.inference.predict(threshold_match_weight=-50)
    pdf = preds.as_pandas_dataframe()
    print(f"\n[{tag}] scored pairs: {len(pdf):,}")
    if not len(pdf):
        raise RuntimeError("ZERO predictions -- check prior/u before touching blocking")

    VDIR.mkdir(exist_ok=True)
    linker.misc.save_model_to_json(str(VDIR / f"model_{tag}.json"), overwrite=True)
    mt.to_csv(VDIR / f"params_{tag}.csv", index=False)
    _pq(pdf[["unique_id_l", "unique_id_r", "match_weight", "match_probability"]],
        ROOT / "parquet" / f"predictions_{tag}.parquet")
    return dict(tag=tag, kind=kind, records=len(rec), scored_pairs=len(pdf),
                n_same_labels=len(same), n_diff_labels=n_diff, params=mt)


# ------------------------------------- TRAIN-ONLY cross-validated threshold
def _fit_m_counts(same_g: pd.DataFrame) -> dict:
    """Replicate estimate_m_from_pairwise_labels arithmetically: per comparison,
    the proportion of labelled SAME pairs at each level, over the pairs where
    that comparison is non-null. Unobserved levels fall back to the library
    default, exactly as Splink does."""
    M = {}
    for col, g, nlev in GAMMA:
        sub = same_g[same_g[g] >= 0]
        M[col] = {}
        for k in range(nlev):
            c = int((sub[g] == k).sum())
            M[col][k] = (c / len(sub)) if (len(sub) and c) else LIB_DEFAULT_M[col][k]
    return M


def _weights(g: pd.DataFrame, M: dict, U: dict, prior: float) -> np.ndarray:
    w = np.full(len(g), math.log2(prior / (1 - prior)))
    for col, gcol, _ in GAMMA:
        gv = g[gcol].values
        for k in np.unique(gv):
            if k < 0:
                continue
            w[gv == k] += math.log2(M[col][int(k)] / U[col][int(k)])
    ex = (g.g_name.values == 4)                 # tf adjustment, weight 1.0
    w[ex] += np.log2(U["name_clean"][4] / g.tf_name.values[ex])
    return w


def _u_and_prior_from_model(tag: str = TAG):
    mj = json.load(open(VDIR / f"model_{tag}.json"))
    U = {}
    for c in mj["comparisons"]:
        nm = c["output_column_name"]
        lv = [l for l in c["comparison_levels"] if not l.get("is_null_level")]
        for i, l in enumerate(lv):
            U.setdefault(nm, {})[len(lv) - 1 - i] = l["u_probability"]
    return U, mj["probability_two_random_records_match"]


def cv_threshold_curve(g: pd.DataFrame, tag: str = TAG, folds: int = 5, seed: int = SEED):
    """5-fold CV *within labels_train.csv*. m is refitted per fold from the SAME
    pairs of the other folds; the held-out fold is scored with it. This is the
    honest in-training precision curve and the only instrument used to choose
    the operating point."""
    U, prior = _u_and_prior_from_model(tag)
    rng = np.random.default_rng(seed)
    fold = np.empty(len(g), dtype=int)
    for lb in ("SAME", "DIFFERENT"):                       # stratified folds
        idx = np.where(g.label.values == lb)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % folds
    g = g.assign(fold=fold)
    w = np.empty(len(g))
    for f in range(folds):
        M = _fit_m_counts(g[(g.fold != f) & (g.label == "SAME")])
        sel = g.fold.values == f
        w[sel] = _weights(g[sel], M, U, prior)
    g = g.assign(cv_w=w, insample_w=_weights(g, _fit_m_counts(g[g.label == "SAME"]), U, prior))
    n_same = int((g.label == "SAME").sum())
    rows = []
    for t in np.arange(0.0, 22.5, 0.5):
        for col, nm in (("cv_w", "cv"), ("insample_w", "insample")):
            m = g[g[col] >= t]
            tp = int((m.label == "SAME").sum()); fp = int((m.label == "DIFFERENT").sum())
            rows.append(dict(threshold=round(float(t), 1), curve=nm, merged=tp + fp, tp=tp, fp=fp,
                             precision=(tp / (tp + fp)) if (tp + fp) else float("nan"),
                             recall=tp / n_same))
    return pd.DataFrame(rows), g


def pick_threshold_on_train(curve: pd.DataFrame, target: float = 0.95, min_merged: int = 10) -> float:
    """STATED RULE, train only: the lowest 0.5-step weight whose CROSS-VALIDATED
    training precision is >= target while merging >= min_merged pairs. A
    precision computed on three pairs is not a precision."""
    c = curve[curve.curve == "cv"]
    ok = c[(c.precision >= target) & (c.merged >= min_merged)]
    if len(ok):
        return float(ok.threshold.min())
    fall = c[c.merged >= min_merged]
    return float(fall.loc[fall.precision.idxmax()].threshold)


# ------------------------------------------------------------------- driver
if __name__ == "__main__":
    r = fit()

    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    rec = build_records(con, CORPUS["debtors"]); con.close()
    g = gamma_frame(rec, train_pairs(rec, "debtor"))
    curve, gw = cv_threshold_curve(g)

    # sanity: the arithmetic weight model must reproduce Splink's own output
    d = duckdb.connect(); d.execute(f"SET memory_limit='{MEM}'")
    d.register("gw", gw[["pair_id", "insample_w"]])
    d.register("lab", train_pairs(rec, "debtor"))
    chk = d.execute(f"""
        SELECT max(abs(gw.insample_w - p.match_weight)) AS max_abs_diff, count(*) n
        FROM gw JOIN lab USING (pair_id)
        JOIN '{ROOT}/parquet/predictions_{TAG}.parquet' p
          ON (p.unique_id_l=lab.unique_id_l AND p.unique_id_r=lab.unique_id_r)
          OR (p.unique_id_l=lab.unique_id_r AND p.unique_id_r=lab.unique_id_l)
    """).df(); d.close()
    print(f"\n[check] CV weight model reproduces Splink on {int(chk.n[0])} train pairs, "
          f"max |diff| = {chk.max_abs_diff[0]:.2e}")
    assert chk.max_abs_diff[0] < 1e-9, "CV weight model diverged from Splink"

    print("\n" + "=" * 84)
    print("TRAIN threshold selection -- 5-fold CV inside labels_train.csv (test never read)")
    piv = curve.pivot_table(index="threshold", columns="curve",
                            values=["precision", "recall", "merged", "fp"])
    with pd.option_context("display.width", 200, "display.max_rows", 60):
        print(piv.round(3).to_string())
    thr = pick_threshold_on_train(curve)
    print(f"\nTHRESHOLD SELECTED ON TRAIN (CV precision >= 0.95, merged >= 10): {thr}")
    VDIR.mkdir(exist_ok=True)
    curve.to_csv(VDIR / f"train_cv_curve_{TAG}.csv", index=False)
    json.dump({"threshold": thr, "rule": "lowest 0.5-step weight with 5-fold CV train "
                                         "precision >= 0.95 and >= 10 merged pairs",
               "n_same_labels": r["n_same_labels"], "n_diff_labels": r["n_diff_labels"],
               "scored_pairs": r["scored_pairs"]},
              open(VDIR / f"choice_{TAG}.json", "w"), indent=2)

    print("\n" + "=" * 84)
    print("HELD-OUT TEST (labels_test.csv -- first and only read, after the point was fixed)")
    from score import score_model
    for t in [2.0, 4.0, 6.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 20.0]:
        mark = "  <-- SELECTED ON TRAIN" if t == thr else ""
        score_model(tag=TAG, corpus="debtor", threshold=t)
        if mark:
            print(mark)
