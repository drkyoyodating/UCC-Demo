#!/usr/bin/env python
"""Score any candidate model against the HELD-OUT test labels.

Usage:  from score import score_model
        score_model(tag='v2', corpus='debtor', threshold=6.0)

Reads parquet/predictions_<tag>.parquet, joins to labels_test.csv on the record
pair, and reports precision / recall on the held-out half only. Never reads
labels_train.csv. Stratum-weighted, same estimator as src/evaluate.py.
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import duckdb, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
EDGES = [2, 4, 6, 7, 8, 10, 999]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def _pairs(corpus):
    """Held-out labelled pairs, keyed by the RECORD TEXT so they survive a
    model change (record ids are stable, but this is robust either way)."""
    te = pd.read_csv(ROOT / "labels_test.csv")
    te = te[te.stratum.str.startswith(corpus)]
    b = pd.concat([pd.read_csv(ROOT / "docs" / "labels_blank.csv", dtype=str).fillna(""),
                   pd.read_csv(ROOT / "docs" / "labels_blank_batch2.csv", dtype=str).fillna("")],
                  ignore_index=True).drop(columns=["label", "note"], errors="ignore")
    return te.merge(b, on="pair_id")


def score_model(tag: str, corpus: str = "debtor", threshold: float = 6.0, verbose=True,
                corpus_table: str | None = None):
    """Score a prediction set against the held-out labels.

    `corpus_table` overrides the physical table the records are read from. It
    defaults to the historical `corpus_{kind}_eq`, which is the STALE
    Colorado-only EQUIPMENT-collateral corpus; pass "corpus_scope_all" to score
    a model resolved on the current CO+CT heavy-construction scope.
    """
    kind = "debtors" if corpus == "debtor" else "lenders"
    te = _pairs(corpus)
    d = duckdb.connect()
    sys.path.insert(0, str(ROOT / "src"))
    from resolve import build_records
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    rec = build_records(con, corpus_table or f"corpus_{kind}_eq"); con.close()
    d.register("r", rec[["unique_id", "name_clean", "suffix", "address1", "city", "state", "zipcode"]])
    d.register("te", te)
    pq = f"{ROOT}/parquet/predictions_{tag}.parquet"
    # DETERMINISM FIX (2026-08-30). The joins below match a labelled pair to
    # records by TEXT, so when several records share one (name, address, city,
    # zip) tuple a single pair_id yields several rows. The old code ended with
    # .drop_duplicates("pair_id"), which kept whichever row the PARALLEL join
    # emitted first -- baseline recall flipped between 0.703 and 0.730 across
    # identical runs, and every downstream number inherited that.
    #
    # Aggregating with MAX over the group is order-independent, so the result is
    # now byte-identical run to run. MAX is also the right semantics: a labelled
    # pair counts as merged if ANY record combination of it scored at or above
    # the threshold. `label` and `stratum` are functionally dependent on
    # pair_id, so any_value on them is exact, not a sample.
    j = d.execute(f"""
        SELECT te.pair_id,
               any_value(te.label)   AS label,
               any_value(te.stratum) AS stratum,
               max(p.match_weight)   AS w
        FROM te
        JOIN r a ON a.name_clean=te.a_name AND coalesce(a.address1,'')=coalesce(te.a_address,'')
                AND coalesce(a.city,'')=coalesce(te.a_city,'') AND coalesce(a.zipcode,'')=coalesce(te.a_zip,'')
        JOIN r b ON b.name_clean=te.b_name AND coalesce(b.address1,'')=coalesce(te.b_address,'')
                AND coalesce(b.city,'')=coalesce(te.b_city,'') AND coalesce(b.zipcode,'')=coalesce(te.b_zip,'')
        LEFT JOIN '{pq}' p ON (p.unique_id_l=a.unique_id AND p.unique_id_r=b.unique_id)
                           OR (p.unique_id_l=b.unique_id AND p.unique_id_r=a.unique_id)
        GROUP BY te.pair_id
    """).df()
    d.close()
    merged = j[j.w.notna() & (j.w >= threshold)]
    tp = int((merged.label == "SAME").sum()); fp = int((merged.label == "DIFFERENT").sum())
    all_same = int((j.label == "SAME").sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec_ = tp / all_same if all_same else float("nan")
    lo, hi = wilson(tp, tp + fp)
    if verbose:
        print(f"[{tag}/{corpus}@{threshold}] held-out n={len(j)}  merged={tp+fp}  "
              f"TP={tp} FP={fp}   precision={prec:.3f} ({lo:.3f}-{hi:.3f})   recall={rec_:.3f}")
    return dict(tag=tag, corpus=corpus, threshold=threshold, n=len(j), merged=tp + fp,
                tp=tp, fp=fp, precision=prec, ci_lo=lo, ci_hi=hi, recall=rec_)


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "debtors"
    c = sys.argv[2] if len(sys.argv) > 2 else "debtor"
    th = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
    score_model(t, c, th)
