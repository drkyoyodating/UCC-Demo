#!/usr/bin/env python
"""VARIANT `ensemble` -- STRATEGY I: EVIDENCE STACKING, weights FIT on labels_train.

The shipped debtor model is a Fellegi-Sunter fit whose EM run learned
m(exact name match) = 0.002741. A total name mismatch therefore costs ~0.016
bits and ADDRESS decides every pair: 73.7% of merges at T=6.0 share an identical
address1, and the two defect classes are both "same address" --

  C1  same address, DISSIMILAR names  (4,217 pairs, 97.6% wrong)
  C2  same address, SIMILAR names     (4,625 pairs; family members and sibling
      entities: SCHULTE MARY J / SCHULTE ALLEN J, LEY HENRY JR / LEY BARBARA)

C2 is the one that matters, because those pairs score HIGHEST -- precision FALLS
as the threshold rises.

This module does not repair the EM fit and does not post-hoc veto it. It replaces
the score outright with an EXPLICIT, HAND-FITTED linear scorer:

    logit(SAME) = b0 + b1*name_sim + b2*suffix_agree + b3*address_agree + b4*city_agree

with b FIT BY LOGISTIC REGRESSION ON labels_train.csv, implemented by hand in
numpy (IRLS with a ridge penalty; scikit-learn is deliberately absent). The
POINT of the exercise is the coefficient vector itself: if labelled data says the
address term should be near zero -- or negative -- that is a measurement, not a
hand-set floor, and it is a far stronger claim than "we added a name gate".

TWO MODELS ARE FIT AND BOTH ARE REPORTED:

  lr4   the four terms above, exactly as specified. The headline finding.
  ens   the same estimator over a richer, still fully explicit feature vector.
        Three additions carry the weight:

        * name_tokmin -- the MINIMUM, over every token of either name, of that
          token's best Jaro-Winkler partner in the other name. This is the C2
          detector. Whole-string JW cannot see C2 (SCHULTE MARY J / SCHULTE ALLEN
          J scores 0.904, higher than the true match COOPERS CONST / COOPERS
          CONSTRUCTION at 0.930's near neighbours) because the shared surname and
          shared middle initial dominate the string. Per-token minimum inverts
          that: MARY has no partner in {SCHULTE, ALLEN, J}, so tokmin = 0.483,
          while CONST -> CONSTRUCTION keeps tokmin = 0.883. One number separates
          "every token of each name has a counterpart" from "one name carries a
          token the other has never heard of", which is precisely what
          distinguishes an abbreviation from a sibling.

        * address FREQUENCY split. `addr_eq` is not one event. Two records
          sharing a 2-name address is evidence; two records sharing a 40-name
          address (a registered agent, a strip mall, a PO-box bank) is nearly
          none. The address term is split into addr_eq_rare (<=2 distinct names
          ever seen at that address1) and addr_eq_shared (>=3), so the fit can
          price them separately instead of averaging a real signal against noise.

        * suffix DISAGREEMENT as its own term, separate from agreement, with
          "one or both suffixes absent" as the reference level. ACME LLC vs
          ACME INC is evidence AGAINST; ACME LLC vs ACME (no suffix) is not.

TRAIN RESIDUALS -> the two extra terms in `ens2`
  Read off the TRAIN residuals of `ens`, and off nothing else:

  * Its two surviving train false merges are both people who share a surname AND
    a first initial -- STEDMAN JEFF / STEDMAN JANELLE, SEGELKE SHIRLEY / SEGELKE
    BURLIE E. Jaro-Winkler gives a shared initial a prefix bonus, so tokmin still
    reads 0.60 / 0.66. `name_tokmin_fc` re-scores every token pair as 0 unless the
    two tokens share their FIRST CHARACTER, which is the cheapest statement of
    "JUDY may abbreviate JUDITH; BURLIE does not abbreviate SHIRLEY".
  * Its worst train recall losses are token-SEGMENTATION differences, where
    tokmin is exactly 0 on a genuine match: L T LITHO / LT LITHO, TEAM PANELS
    INTERNATIONAL / T E A M PANELS INTERNATIONAL. `nospace_eq` (the two names are
    identical once every space is removed) is 1 on both and 0 on every family
    pair, so it buys recall without touching the C2 failure mode.

  These are 2 more parameters on 138 rows. The ridge is what keeps them finite,
  the bootstrap CI is printed so the reader can see how little they are pinned
  down, and the model that ships is chosen by a stated train-only rule below.

RESULT (held-out, labels_test.csv, debtor corpus, 70 labelled pairs / 37 SAME)
  shipped model = `ens2` at its train-selected operating point T = 0.0:
      TP=35  FP=0   precision 1.000 (Wilson 95%: 0.901-1.000)   recall 0.946
  baseline (shipped Splink debtor model) at T=6.0 on the same 70 pairs:
      TP=26  FP=26  precision 0.500   recall 0.703
  The baseline's precision FALLS as its threshold rises (0.587 @3.0 -> 0.500 @6.0
  -> 0.107 @8.0 -> 0.000 @10.0), which is the C2 signature; this model's does not.
  The result is invariant to score.py's duplicate-record ambiguity: taking the
  MIN and the MAX weight over every record combination a labelled pair maps to
  gives TP=35 FP=0 either way.

  PROVENANCE WARNING, recorded because it nearly produced a false number: an
  earlier fit of this module was invalidated when labels_train/labels_test were
  re-drawn by another agent while the scoring pass was still running, putting 22
  of the 70 held-out pairs into the fitting set. `parquet/_ensemble_fitted_on.txt`
  is written at fit time and lists exactly which pair_ids were fitted on; check it
  against labels_test.csv before believing any number this module produces.

FITTING DISCIPLINE
  * Only labels_train.csv is read here. labels_test.csv is never opened by this
    file; scoring goes through score.score_model, which reads it and nothing else.
  * The label sample is STRATIFIED BY THE SHIPPED MODEL'S OWN SCORE BAND. That
    sampling is covariate-dependent, not outcome-dependent, so an unweighted
    logistic fit is consistent for the slopes under correct specification --
    unlike case-control sampling, which biases only the intercept. The intercept
    here is nonetheless NOT interpretable as a population log-odds, and no
    threshold is justified from it; thresholds are reported as a curve.
  * A design-weighted fit (weight = N_h / n_h, the column the label file already
    carries) is ALSO run and printed, as a specification check on the above.

CANDIDATE SET: the same two blocking rules the shipped model uses
(block_on zipcode; block_on substr(name_clean,1,4)), so recall is comparable and
a pair absent from the output is absent for the same reason it would be there.

Usage:  ./.venv/bin/python src/variant_ensemble.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from resolve import build_records  # noqa: E402

KEY = "ensemble"
PRED_OUT = ROOT / "parquet" / f"predictions_{KEY}.parquet"
#: The token-alignment features are nested-lambda SQL over 2.5M pairs and cost
#: ~40 min. They depend on the CORPUS ONLY -- never on a label -- so they are
#: cached. A re-fit (new label split, new coefficients) is then a projection over
#: this table and costs seconds. This mattered: the first fit of this module was
#: silently invalidated when another agent re-drew labels_train/labels_test
#: mid-run, and without the cache the re-fit would have cost a second 40 minutes.
FEATS_CACHE = ROOT / "parquet" / f"_{KEY}_features.parquet"
MEM = "1GB"          # up to nine Splink runs may share a 16GB machine
LN2 = float(np.log(2.0))

# --------------------------------------------------------------------------
# Feature algebra. ONE SQL text, used for BOTH the training pairs and the
# 2.5M candidate pairs, so a feature can never be computed two different ways.
# --------------------------------------------------------------------------
_TOKMIN = """least(
      list_min(list_transform(string_split({A}.name_clean,' '),
               x -> list_max(list_transform(string_split({B}.name_clean,' '),
                    y -> jaro_winkler_similarity(x,y))))),
      list_min(list_transform(string_split({B}.name_clean,' '),
               x -> list_max(list_transform(string_split({A}.name_clean,' '),
                    y -> jaro_winkler_similarity(x,y)))))
    )"""

_TOKMIN_FC = """least(
      list_min(list_transform(string_split({A}.name_clean,' '),
               x -> list_max(list_transform(string_split({B}.name_clean,' '),
                    y -> CASE WHEN substr(x,1,1)=substr(y,1,1)
                              THEN jaro_winkler_similarity(x,y) ELSE 0.0 END)))),
      list_min(list_transform(string_split({B}.name_clean,' '),
               x -> list_max(list_transform(string_split({A}.name_clean,' '),
                    y -> CASE WHEN substr(x,1,1)=substr(y,1,1)
                              THEN jaro_winkler_similarity(x,y) ELSE 0.0 END))))
    )"""

FEATURE_SQL = """
SELECT p.*,
  jaro_winkler_similarity(a.name_clean, b.name_clean)                     AS name_jw,
  {TOKMIN}                                                                AS name_tokmin,
  CASE WHEN a.suffix IS NOT NULL AND b.suffix IS NOT NULL
            AND a.suffix =  b.suffix THEN 1.0 ELSE 0.0 END                AS sfx_eq,
  CASE WHEN a.suffix IS NOT NULL AND b.suffix IS NOT NULL
            AND a.suffix <> b.suffix THEN 1.0 ELSE 0.0 END                AS sfx_neq,
  CASE WHEN nullif(trim(a.address1),'') IS NOT NULL
            AND nullif(trim(a.address1),'') = nullif(trim(b.address1),'')
       THEN 1.0 ELSE 0.0 END                                              AS addr_eq,
  CASE WHEN nullif(trim(a.address1),'') IS NOT NULL
            AND nullif(trim(a.address1),'') = nullif(trim(b.address1),'')
            AND coalesce(fa.n_names, 1) <= 2 THEN 1.0 ELSE 0.0 END        AS addr_eq_rare,
  CASE WHEN nullif(trim(a.address1),'') IS NOT NULL
            AND nullif(trim(a.address1),'') = nullif(trim(b.address1),'')
            AND coalesce(fa.n_names, 1) >= 3 THEN 1.0 ELSE 0.0 END        AS addr_eq_shared,
  CASE WHEN nullif(trim(a.address1),'') IS NOT NULL
            AND nullif(trim(b.address1),'') IS NOT NULL
            AND nullif(trim(a.address1),'') <> nullif(trim(b.address1),'')
            AND jaro_winkler_similarity(a.address1, b.address1) >= 0.90
       THEN 1.0 ELSE 0.0 END                                              AS addr_near,
  CASE WHEN nullif(trim(a.city),'') IS NOT NULL
            AND nullif(trim(a.city),'') = nullif(trim(b.city),'')
       THEN 1.0 ELSE 0.0 END                                              AS city_eq,
  CASE WHEN nullif(trim(a.zipcode),'') IS NOT NULL
            AND nullif(trim(a.zipcode),'') = nullif(trim(b.zipcode),'')
       THEN 1.0 ELSE 0.0 END                                              AS zip_eq,
  CASE WHEN replace(a.name_clean,' ','') = replace(b.name_clean,' ','')
       THEN 1.0 ELSE 0.0 END                                              AS nospace_eq,
  {TOKMINFC}                                                              AS name_tokmin_fc
FROM {PAIRS} p
JOIN rec a ON a.unique_id = p.unique_id_l
JOIN rec b ON b.unique_id = p.unique_id_r
LEFT JOIN addrfreq fa ON fa.address1 = nullif(trim(a.address1),'')
                     AND nullif(trim(a.address1),'') = nullif(trim(b.address1),'')
"""

#: The four terms the strategy specifies, in order. b3 (address) is the finding.
COLS_LR4 = ["name_jw", "sfx_eq", "addr_eq", "city_eq"]
#: The richer explicit vector. addr_eq is REPLACED by its two frequency levels
#: (their sum is addr_eq, so keeping all three would be exactly collinear).
COLS_ENS = ["name_jw", "name_tokmin", "sfx_eq", "sfx_neq",
            "addr_eq_rare", "addr_eq_shared", "addr_near", "city_eq", "zip_eq"]
#: ens + two terms chosen from the TRAIN residuals of `ens` and from nothing else
#: (see the "TRAIN RESIDUALS" block in the docstring). No test label was consulted.
COLS_ENS2 = COLS_ENS + ["nospace_eq", "name_tokmin_fc"]


def feature_sql(pairs_tbl: str) -> str:
    return FEATURE_SQL.format(PAIRS=pairs_tbl,
                              TOKMIN=_TOKMIN.format(A="a", B="b"),
                              TOKMINFC=_TOKMIN_FC.format(A="a", B="b"))


# --------------------------------------------------------------------------
# Logistic regression, by hand. IRLS + ridge (scikit-learn is not installed).
# --------------------------------------------------------------------------
def fit_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
                 l2: float = 1.0, iters: int = 200, tol: float = 1e-9):
    """Newton / IRLS on the penalised log-likelihood. Intercept is column 0 and
    is NOT penalised. The ridge exists because several features are near-
    separating on 138 rows: without it a coefficient runs to infinity and the
    fitted 'score' becomes a hard rule with a fake confidence attached."""
    n, k = X.shape
    if w is None:
        w = np.ones(n)
    w = w / w.mean()
    P = np.eye(k) * l2
    P[0, 0] = 0.0
    b = np.zeros(k)
    for _ in range(iters):
        eta = X @ b
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        W = np.clip(p * (1 - p), 1e-6, None) * w
        g = X.T @ (w * (y - p)) - P @ b
        H = (X * W[:, None]).T @ X + P
        step = np.linalg.solve(H, g)
        b_new = b + step
        if np.max(np.abs(b_new - b)) < tol:
            b = b_new
            break
        b = b_new
    return b


def _bootstrap_se(X, y, w, l2, cols, reps=400, seed=20260830):
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            out.append(fit_logistic(X[idx], y[idx], None if w is None else w[idx], l2))
        except np.linalg.LinAlgError:
            continue
    B = np.array(out)
    return B.std(axis=0), np.percentile(B, 2.5, axis=0), np.percentile(B, 97.5, axis=0)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def open_db():
    d = duckdb.connect()
    d.execute(f"SET memory_limit='{MEM}'")
    d.execute("SET threads=4")
    d.execute(f"SET temp_directory='{tempfile.gettempdir()}'")
    return d


def load_records() -> pd.DataFrame:
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    rec = build_records(con, "corpus_debtors_eq")
    con.close()
    return rec[["unique_id", "name_clean", "suffix", "address1",
                "city", "state", "zipcode"]]


def train_pairs(d) -> pd.DataFrame:
    """labels_train.csv joined to records. The join is copied from _pairs() in
    src/score.py verbatim (blank + batch2 on pair_id, then record text on
    name/address/city/zip) so train and test are keyed identically."""
    tr = pd.read_csv(ROOT / "labels_train.csv")
    tr = tr[tr.stratum.str.startswith("debtor")]
    (ROOT / "parquet" / f"_{KEY}_fitted_on.txt").write_text(
        "\n".join(sorted(tr.pair_id)) + "\n")
    b = pd.concat([pd.read_csv(ROOT / "docs" / "labels_blank.csv", dtype=str).fillna(""),
                   pd.read_csv(ROOT / "docs" / "labels_blank_batch2.csv", dtype=str).fillna("")],
                  ignore_index=True).drop(columns=["label", "note"], errors="ignore")
    tr = tr.merge(b, on="pair_id")
    d.register("tr", tr)
    j = d.execute("""
        SELECT tr.pair_id, tr.label, tr.stratum, tr.weight,
               a.unique_id AS unique_id_l, b.unique_id AS unique_id_r
        FROM tr
        JOIN rec a ON a.name_clean=tr.a_name AND coalesce(a.address1,'')=coalesce(tr.a_address,'')
                  AND coalesce(a.city,'')=coalesce(tr.a_city,'') AND coalesce(a.zipcode,'')=coalesce(tr.a_zip,'')
        JOIN rec b ON b.name_clean=tr.b_name AND coalesce(b.address1,'')=coalesce(tr.b_address,'')
                  AND coalesce(b.city,'')=coalesce(tr.b_city,'') AND coalesce(b.zipcode,'')=coalesce(tr.b_zip,'')
    """).df().drop_duplicates("pair_id")
    d.unregister("tr")
    return j


def build_candidates(d):
    """The shipped blocking rules, unchanged: block_on(zipcode) UNION
    block_on(substr(name_clean,1,4))."""
    d.execute("""
        CREATE OR REPLACE TABLE cand AS
        SELECT DISTINCT least(a.unique_id,b.unique_id) AS unique_id_l,
                        greatest(a.unique_id,b.unique_id) AS unique_id_r
        FROM rec a JOIN rec b
          ON a.zipcode = b.zipcode AND a.unique_id < b.unique_id
        UNION
        SELECT DISTINCT least(a.unique_id,b.unique_id),
                        greatest(a.unique_id,b.unique_id)
        FROM rec a JOIN rec b
          ON substr(a.name_clean,1,4) = substr(b.name_clean,1,4) AND a.unique_id < b.unique_id
    """)
    return d.execute("SELECT count(*) FROM cand").fetchone()[0]


# --------------------------------------------------------------------------
def curve_train(F: pd.DataFrame, y: np.ndarray, wt: np.ndarray, cols, beta,
                step: float = 1.0):
    """Precision/recall ON TRAIN, design-weighted AND raw. Used only to choose a
    shipping threshold; every headline number is the held-out curve."""
    eta = np.column_stack([np.ones(len(F))] + [F[c].values for c in cols]) @ beta
    mw = eta / LN2
    rows = []
    for t in np.arange(-6, 26, step):
        m = mw >= t
        tp, fp = int(((y == 1) & m).sum()), int(((y == 0) & m).sum())
        wtp = float(wt[(y == 1) & m].sum()); wfp = float(wt[(y == 0) & m].sum())
        rows.append(dict(t=t, tp=tp, fp=fp,
                         prec=tp / (tp + fp) if tp + fp else np.nan,
                         rec=tp / max((y == 1).sum(), 1),
                         wprec=wtp / (wtp + wfp) if wtp + wfp else np.nan))
    return pd.DataFrame(rows)


def main():
    d = open_db()
    rec = load_records()
    d.register("rec_df", rec)
    d.execute("CREATE OR REPLACE TABLE rec AS SELECT * FROM rec_df")
    d.execute("""CREATE OR REPLACE TABLE addrfreq AS
                 SELECT nullif(trim(address1),'') AS address1,
                        count(DISTINCT name_clean) AS n_names
                 FROM rec WHERE nullif(trim(address1),'') IS NOT NULL GROUP BY 1""")
    print(f"[{KEY}] records={len(rec):,}  "
          f"addresses={d.execute('SELECT count(*) FROM addrfreq').fetchone()[0]:,}")

    # ---- training matrix -------------------------------------------------
    tp = train_pairs(d)
    d.register("trp", tp)
    d.execute("CREATE OR REPLACE TABLE trpairs AS SELECT * FROM trp")
    F = d.execute(feature_sql("trpairs")).df()
    print(f"[{KEY}] train pairs matched to records: {len(F)} of {len(tp)} "
          f"({(F.label=='SAME').sum()} SAME / {(F.label=='DIFFERENT').sum()} DIFFERENT)")
    y = (F.label == "SAME").values.astype(float)
    wt = F.weight.values.astype(float)

    fits = {}
    for name, cols in (("lr4", COLS_LR4), ("ens", COLS_ENS), ("ens2", COLS_ENS2)):
        X = np.column_stack([np.ones(len(F))] + [F[c].values.astype(float) for c in cols])
        b_u = fit_logistic(X, y, None, l2=1.0)
        b_w = fit_logistic(X, y, wt, l2=1.0)
        se, lo, hi = _bootstrap_se(X, y, None, 1.0, cols)
        fits[name] = dict(cols=cols, beta=b_u, beta_w=b_w, se=se, lo=lo, hi=hi)
        print(f"\n=== FITTED COEFFICIENTS [{name}]  (log-odds; ridge l2=1.0, "
              f"intercept unpenalised) ===")
        print(f"{'term':16s} {'beta':>8s} {'boot_se':>8s} {'boot 95% CI':>20s} "
              f"{'design-wtd':>11s}")
        for i, c in enumerate(["(intercept)"] + cols):
            print(f"{c:16s} {b_u[i]:8.3f} {se[i]:8.3f} "
                  f"  [{lo[i]:7.3f},{hi[i]:7.3f}] {b_w[i]:11.3f}")

    # ---- train curves ----------------------------------------------------
    for name in ("lr4", "ens", "ens2"):
        f = fits[name]
        c = curve_train(F, y, wt, f["cols"], f["beta"])
        print(f"\n--- TRAIN curve [{name}] (in-sample, for threshold choice only) ---")
        print(c[(c.t >= 0) & (c.t <= 20)].to_string(index=False,
              float_format=lambda v: f"{v:.3f}"))

    # ---- MODEL SELECTION, on TRAIN only ---------------------------------
    # Stated rule, applied mechanically: for each model take t* = the lowest
    # 0.5-step threshold at which it makes ZERO false merges on train; ship the
    # model with the highest train recall at its own t*; ties break to the model
    # with FEWER features. No held-out label is consulted anywhere in this file.
    sel = []
    for name in ("lr4", "ens", "ens2"):
        f = fits[name]
        c = curve_train(F, y, wt, f["cols"], f["beta"], step=0.5)
        z = c[(c.fp == 0) & (c.tp > 0)]
        if len(z):
            r = z.iloc[0]
            sel.append((name, float(r.t), float(r.rec), len(f["cols"])))
        else:
            sel.append((name, float("nan"), 0.0, len(f["cols"])))
    print("\n--- TRAIN model selection (zero-FP operating point) ---")
    for n, t, r, k in sel:
        print(f"  {n:5s} t*={t:5.2f}  train recall at t*={r:.3f}  ({k} features)")
    best = sorted(sel, key=lambda x: (-x[2], x[3]))[0][0]
    print(f"  -> SHIPPING model '{best}' as predictions_{KEY}.parquet")

    # ---- score every candidate pair --------------------------------------
    n_cand = build_candidates(d)
    print(f"\n[{KEY}] candidate pairs under the shipped blocking: {n_cand:,}")
    # Materialise the feature table ONCE. The token-alignment features are
    # nested-lambda SQL and cost ~10 min over the candidate set; computing them
    # once and projecting four scores off the result is the difference between
    # one run and four.
    ok = False
    if FEATS_CACHE.exists():
        n = d.execute(f"SELECT count(*) FROM '{FEATS_CACHE}'").fetchone()[0]
        have = set(d.execute(f"SELECT * FROM '{FEATS_CACHE}' LIMIT 0").df().columns)
        need = set(["unique_id_l", "unique_id_r"] + COLS_ENS2)
        ok = (n == n_cand) and need.issubset(have)
        print(f"[{KEY}] feature cache: rows={n:,} usable={ok}")
    if ok:
        d.execute(f"CREATE OR REPLACE VIEW feats AS SELECT * FROM '{FEATS_CACHE}'")
    else:
        d.execute("CREATE OR REPLACE TABLE feats AS " + feature_sql("cand"))
        d.execute(f"COPY (SELECT * FROM feats) TO '{FEATS_CACHE}' (FORMAT parquet)")
        print(f"[{KEY}] features materialised and cached -> {FEATS_CACHE}")
    print(f"[{KEY}] feature rows: "
          f"{d.execute('SELECT count(*) FROM feats').fetchone()[0]:,}")

    def write(name, path):
        f = fits[name]
        e = " + ".join([f"({f['beta'][0]})"] +
                       [f"({f['beta'][i+1]})*{c}" for i, c in enumerate(f["cols"])])
        d.execute(f"""
            COPY (SELECT unique_id_l, unique_id_r,
                         ({e}) / {LN2} AS match_weight,
                         1.0/(1.0+exp(-least(greatest(({e}),-30),30))) AS match_probability
                  FROM feats) TO '{path}' (FORMAT parquet)""")
        print(f"[{KEY}] wrote {path}")

    for name in ("lr4", "ens", "ens2"):
        write(name, ROOT / "parquet" / f"predictions_{KEY}_{name}.parquet")
    write(best, PRED_OUT)          # the shipped artefact, chosen on train
    d.close()


if __name__ == "__main__":
    main()
