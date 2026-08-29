#!/usr/bin/env python
"""P5 / P5b -- Splink resolution. `resolve(...)` is a CALLABLE, not a script.

Written as a parameterised function from line one so that P5b (lenders) is a
second CALL rather than a refactor at hour 15, which is the worst possible moment
to be restructuring the model code.

Everything decided in advance is in DECISIONS.md under "P5 -- PRE-REGISTRATION",
committed at cfe134b BEFORE this file existed. In particular `pick_threshold()`
below implements that written rule literally; it takes no arguments that could
be tuned after seeing a result.

THE TRAP (documented, and defused here explicitly): Splink's
`probability_two_random_records_match` defaults to 0.0001 and untrained `m`
values produce ZERO matches with NO error. We set the prior by estimation and
run EM per blocking rule. If predictions ever come back empty, check this and
`pip freeze | grep pandas` before touching the blocking rules.

DETERMINISM: EM in 4.0.16 has no RNG -- it is a fixed-point iteration. The only
stochastic step is `estimate_u_using_random_sampling`, which takes a seed. Seeded
plus fixed input means the pipeline is reproducible, which is the precondition
for P6's stability metric measuring clustering churn rather than our own noise.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

import duckdb
import pandas as pd
from splink import Linker, SettingsCreator, DuckDBAPI, block_on
import splink.comparison_library as cl
from splink.blocking_analysis import count_comparisons_from_blocking_rule

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from splink_contract import SEED  # noqa: E402

BLOCKING = [block_on("zipcode"), block_on("substr(name_clean,1,4)")]

#: FOUNDER OVERRIDE, 2026-08-30, taken BEFORE any label existed.
#: `pick_threshold()` below is left EXACTLY as pre-registered (commit cfe134b) --
#: it is the historical record and must not be edited to match the outcome. What
#: it returned on the debtor corpus was 11.5, from a "valley" whose smoothed count
#: was 27 out of 2,489,916 scored pairs: a sparse spot in tail noise, not a
#: boundary between modes. The rule's premise (a bimodal weight distribution) does
#: not hold on this data.
#:
#: 6.0 is the value the PRE-REGISTRATION ITSELF names as the fallback for "no
#: interior minimum", which is the pathology actually present -- the implementation
#: only tested for a minimum on the interval BOUNDARY, so it could not detect a
#: spurious interior one. The override therefore invokes the rule's own escape
#: hatch rather than substituting a hand-picked number.
#:
#: This does not compromise Ship Gate 1. The evaluation is blind because the
#: labeller never sees weights, predictions or cluster ids -- and no labels existed
#: when this was chosen. Selecting on cluster-count grounds before labels exist is
#: legitimate; selecting after they exist would not be. Frozen from here.
THRESHOLD_REASON = {
    "debtors": "FOUNDER DECISION (2026-08-30, before any label existed): rule returned "
               "11.5 from a valley of smoothed-count 27 out of 2,489,916 pairs -- tail "
               "noise, not a mode boundary. 6.0 is the pre-registration's own named "
               "fallback for 'no meaningful interior minimum', which is the pathology "
               "actually present.",
    "lenders": "LEAD DECISION under the founder's delegation of threshold choice "
               "(locked row 15), by the stated rule 'lowest threshold at which the "
               "pre-registered non-degeneracy bar passes'. Rule's fallback gave 6.0; "
               "measured CC largest-cluster 6.0->2.31% FAIL, 7.0->1.04% FAIL, "
               "8.0->0.85% PASS.",
}

THRESHOLD_OVERRIDE = {
    "debtors": 6.0,
    # LENDERS = 8.0, set by a stated rule under the founder's delegation of the
    # threshold choice: THE LOWEST THRESHOLD AT WHICH THE PRE-REGISTERED
    # NON-DEGENERACY BAR PASSES. The rule's own fallback fired at 6.0 (the lender
    # weight distribution is monotone over [0,15]), but 6.0 produced a 210-record
    # cluster = 2.31% of the corpus, failing the <=1% bar that was pre-registered
    # in the same commit. Measured: CC at 6.0 -> 2.31% FAIL, 7.0 -> 1.04% FAIL,
    # 8.0 -> 0.85% PASS. So 8.0, chosen by the bar rather than by eye, and still
    # before any label exists.
    #
    # HONESTY: 8.0 passes the bar but does NOT eliminate over-merging. Hand
    # inspection of the largest clusters at 8.0 still finds COLORADO NATIONAL BANK
    # + COLORADO BUSINESS BANK + COLORADO BUSINESS PARK in one cluster, and
    # AMERICAN ENERGY FINANCE inside the AMERICAN NATIONAL BANK cluster. The 1%
    # bar is a crude degeneracy guard, not a correctness proof. Tuning further to
    # make clusters LOOK clean is exactly the shopping the pre-registration forbids;
    # P6's labelled precision measures this properly and the write-up states it.
    "lenders": 8.0,
}
DETERMINISTIC = [
    "l.name_clean = r.name_clean and l.zipcode = r.zipcode",
    "l.name_clean = r.name_clean and l.address1 = r.address1",
]


def record_id(*parts) -> str:
    """MD5 over the FULL record key. Must cover every column the GROUP BY uses.

    [corrected] The pre-registration wrote the record as
    (name_clean, address1, city, zipcode) but the query groups by six columns,
    so records differing only in `suffix` or `state` collided on the same id.
    That is a coherence bug, not a tuning choice -- but the fix is not merely to
    hash more columns, because the alternative (dropping suffix from the grouping)
    would have been WRONG in a way that matters: merging `ACME`+`LLC` with
    `ACME`+`INC` at one address pre-decides locked ground-truth rule 8(a) inside
    the data preparation, before the model ever scores the pair, and destroys the
    exact signal P2 split the suffix out to preserve. Suffix stays in the key and
    stays a comparison feature; the MODEL decides whether the two are one firm.
    """
    return hashlib.md5("|".join(str(x or "") for x in parts).encode()).hexdigest()[:16]


def build_records(con, corpus: str, base_only: bool = False) -> pd.DataFrame:
    """Distinct party RECORDS, not rows.

    P4 measured 5.04x row-level redundancy in the lender corpus (one lender files
    thousands of times) and 1.30x in the debtor corpus. Resolving rows would cost
    130x the comparisons on the lender side for no extra information, and -- the
    part that actually matters -- would let one bank's repeat filings dominate
    term-frequency and u-probability estimation and flatter P6's high-weight
    labelling stratum.
    """
    where = " AND NOT is_holdout" if base_only else ""
    df = con.execute(f"""
        SELECT name_clean, suffix, address1, city, state, zipcode,
               count(*) AS n_rows, min(unique_id) AS example_party_id
        FROM {corpus}
        WHERE name_clean IS NOT NULL {where}
        GROUP BY ALL
    """).df()
    df["unique_id"] = [record_id(*t) for t in zip(
        df.name_clean, df.suffix, df.address1, df.city, df.state, df.zipcode)]
    if df.unique_id.duplicated().any():          # md5 collision or grouping bug
        raise RuntimeError("record_id is not unique -- Splink requires it")
    return df


def comparisons_for(kind: str) -> list:
    """Both corpora get the SAME feature set. The weights are learned, not assumed.

    [corrected] The first version hand-implemented P4's finding that "address
    disagreement must not veto a lender match" by DELETING address1 and zipcode
    from the lender comparison and substituting `state`. That was wrong twice over
    and produced a model that merged NOTHING:

      * It starved EM. With only name/suffix/city/state, and `state` almost
        constant (Colorado), EM converged on a degenerate solution where the match
        signal WAS `state` (m=1.0, u=0.54) and an exact name match carried
        m=0.0616 -- i.e. it concluded only 6% of true matches share a name. Peak
        match weight across 1,290,471 scored pairs was 4.71, so nothing could clear
        any sensible threshold and every one of the 9,096 records stayed a singleton.
      * It confused "this evidence is weak" with "remove this evidence". Deleting
        address discards the POSITIVE signal when addresses do agree, which is
        exactly when a lender match is most certain.

    The correct expression of P4's finding is to keep the feature and let EM learn
    its weight from the data: because true lender matches frequently disagree on
    address (VECTRA BANK has 199 distinct addresses across 60 ZIPs), EM assigns
    address-disagreement a small negative weight on its own. The difference between
    the corpora then shows up as MEASURED parameters rather than as a hand-coded
    assumption -- which is also a far better thing to publish.

    `state` is dropped from both: it is ~100% Colorado, carries no information, and
    is what the degenerate lender fit latched onto.
    """
    return [
        cl.NameComparison("name_clean"),
        cl.ExactMatch("suffix"),
        cl.JaroWinklerAtThresholds("address1", [0.9, 0.7]),
        cl.ExactMatch("city"),
        cl.ExactMatch("zipcode"),
    ]


def pick_threshold(weights: pd.Series) -> tuple[float, str]:
    """THE PRE-REGISTERED RULE, implemented literally. No tunable arguments.

    DECISIONS.md, committed cfe134b before this file existed:
      bins 0.5 wide over [-10,25]; 3-bin centred moving average; minimum smoothed
      bin in [0,15]; threshold = its left edge; ties break to the HIGHER weight
      (precision is the gated number, recall is reported regardless); fall back to
      match weight 6.0 if that interval has no interior minimum.
    """
    import numpy as np
    edges = np.arange(-10.0, 25.0 + 0.5, 0.5)
    counts, _ = np.histogram(weights.clip(-10, 25), bins=edges)
    k = np.ones(3) / 3.0
    sm = np.convolve(counts.astype(float), k, mode="same")
    lo = int(np.searchsorted(edges, 0.0))
    hi = int(np.searchsorted(edges, 15.0))
    window = sm[lo:hi]
    if window.size == 0:
        return 6.0, "fallback: empty decision interval"
    mn = window.min()
    idxs = [i for i, v in enumerate(window) if v == mn]
    best = idxs[-1]                                  # ties -> HIGHEST weight
    if best in (0, window.size - 1):                 # minimum on the boundary
        return 6.0, "fallback: no interior minimum in [0,15] (monotone)"
    return float(edges[lo + best]), f"valley at smoothed-count {mn:.1f}"


def resolve(corpus: str, kind: str, *, seed: int = SEED, base_only: bool = False,
            tag: str = "", threshold: float | None = None) -> dict:
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    df = build_records(con, corpus, base_only)
    con.close()
    print(f"[{tag or kind}] records={len(df):,} (from {int(df.n_rows.sum()):,} rows, "
          f"{df.n_rows.sum()/len(df):.2f}x redundancy)")

    db_api = DuckDBAPI(":temporary:")
    for br in BLOCKING:
        c = count_comparisons_from_blocking_rule(
            table_or_tables=[df], blocking_rule=br, link_type="dedupe_only",
            db_api=db_api, unique_id_column_name="unique_id")
        print(f"[{tag or kind}] blocking {br.__class__.__name__}: "
              f"{c['number_of_comparisons_to_be_scored_post_filter_conditions']:,} comparisons")

    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=comparisons_for(kind),
        blocking_rules_to_generate_predictions=BLOCKING,
        retain_intermediate_calculation_columns=True,
    )
    linker = Linker(df, settings, db_api=db_api, set_up_basic_logging=False)
    linker.training.estimate_probability_two_random_records_match(DETERMINISTIC, recall=0.8)
    linker.training.estimate_u_using_random_sampling(max_pairs=2_000_000, seed=seed)
    for br in BLOCKING:
        linker.training.estimate_parameters_using_expectation_maximisation(br)

    # -50 is effectively "no floor". An earlier version passed -10 intending to
    # retain the full distribution; predict() FILTERS at that weight, so ~620k
    # scored pairs -- including 2,390 identical-name pairs -- were dropped before
    # they reached the histogram, and the histogram the threshold rule reads was
    # truncated at its left edge. The rule searches [0,15] so the selection was
    # unaffected, but the diagnostics built on it were wrong.
    preds = linker.inference.predict(threshold_match_weight=-50)
    pdf = preds.as_pandas_dataframe()
    print(f"[{tag or kind}] scored pairs: {len(pdf):,}")
    if not len(pdf):
        raise RuntimeError("ZERO predictions -- check the prior and EM before blocking rules")

    rule_thr, rule_branch = pick_threshold(pdf.match_weight)   # always computed and reported
    if threshold is None:
        thr, branch = rule_thr, rule_branch
    else:
        thr = threshold
        branch = (f"OVERRIDE to {threshold} -- rule returned {rule_thr} via "
                  f"'{rule_branch}'. {THRESHOLD_REASON.get(kind, '')}")
    clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
        preds, threshold_match_weight=thr)
    cdf = clusters.as_pandas_dataframe()
    sizes = cdf.groupby("cluster_id").size()
    biggest = int(sizes.max())
    out = {
        "kind": kind, "tag": tag or kind, "records": len(df),
        "rows": int(df.n_rows.sum()), "scored_pairs": len(pdf),
        "threshold": thr, "threshold_branch": branch,
        "rule_threshold": rule_thr, "rule_branch": rule_branch,
        "clusters": int(cdf.cluster_id.nunique()),
        "singletons": int((sizes == 1).sum()),
        "largest_cluster": biggest,
        "largest_pct": 100 * biggest / len(df),
        "p50": int(sizes.median()), "p95": int(sizes.quantile(0.95)),
    }
    # Write parquet via DuckDB, not pandas: pandas.to_parquet needs pyarrow, which
    # is deliberately not installed (the venv carries only load-bearing packages).
    # DuckDB writes parquet natively and is already a pinned dependency.
    def _pq(frame, path):
        d = duckdb.connect()
        d.register("_f", frame)
        d.execute(f"COPY (SELECT * FROM _f) TO '{path}' (FORMAT parquet)")
        d.close()

    mdir = ROOT / "models"; mdir.mkdir(exist_ok=True)
    linker.misc.save_model_to_json(str(mdir / f"model_{tag or kind}.json"), overwrite=True)
    _pq(cdf, ROOT / "parquet" / f"clusters_{tag or kind}.parquet")
    _pq(pdf[["unique_id_l", "unique_id_r", "match_weight", "match_probability"]],
        ROOT / "parquet" / f"predictions_{tag or kind}.parquet")
    return out


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "debtors"
    corpus = {"debtors": "corpus_debtors_eq", "lenders": "corpus_lenders_eq"}[kind]
    r = resolve(corpus, kind, tag=kind, threshold=THRESHOLD_OVERRIDE.get(kind))
    print("\n" + json.dumps(r, indent=2))
    ok = r["largest_pct"] <= 1.0
    print(f"\nNON-DEGENERACY BAR (no cluster >1% of corpus): "
          f"largest={r['largest_pct']:.3f}% -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
