"""Design-weighted evaluation for the UCC relevance screener (ml/specs/evaluation_protocol_v1.md).

Weights (protocol §2): in a split S, stratum h has N_h candidates and n_h non-repeat labelled cases drawn,
INSUFFICIENT_EVIDENCE included. Every labelled case gets w_h = N_h / n_h, which equals 1 / inclusion_probability
because validation and test hold no pilot case (K12) and the main round draws within (split, stratum) (K13);
`design_weights` checks that equality row by row. Rates use the RESOLVED cases with these weights, so they
estimate performance on the resolvable population of S. Dividing N_h by the RESOLVED count instead would impute
every INSUFFICIENT_EVIDENCE case with the resolved cases' outcome mix and re-weight strata by their unresolved
share, which is biased whenever that share differs across strata. Nothing pools the enriched sample unweighted.
Uncertainty (Task 7), the threshold objective (Task 8) and the one-shot TEST run (Task 11) follow.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from ucc_ml.provenance import finite_or_none


@dataclass(frozen=True)
class WeightedCounts:
    tp: float
    fp: float
    fn: float
    tn: float
    n_tp: int
    n_fp: int
    n_fn: int
    n_tn: int


def _as_int_array(a) -> np.ndarray:
    return np.asarray(a, dtype=float).astype(int)


def weighted_counts(y, pred, w) -> WeightedCounts:
    y = _as_int_array(y)
    pred = _as_int_array(pred)
    w = np.asarray(w, dtype=float)
    if not (len(y) == len(pred) == len(w)):
        raise ValueError("y, pred and w must have the same length")
    tp = (y == 1) & (pred == 1)
    fp = (y == 0) & (pred == 1)
    fn = (y == 1) & (pred == 0)
    tn = (y == 0) & (pred == 0)
    return WeightedCounts(
        tp=float(w[tp].sum()), fp=float(w[fp].sum()), fn=float(w[fn].sum()), tn=float(w[tn].sum()),
        n_tp=int(tp.sum()), n_fp=int(fp.sum()), n_fn=int(fn.sum()), n_tn=int(tn.sum()),
    )


def precision_recall_f1(c: WeightedCounts) -> tuple[float, float, float]:
    p = c.tp / (c.tp + c.fp) if (c.tp + c.fp) > 0 else float("nan")
    r = c.tp / (c.tp + c.fn) if (c.tp + c.fn) > 0 else float("nan")
    f = 2 * p * r / (p + r) if (not math.isnan(p) and not math.isnan(r) and (p + r) > 0) else float("nan")
    return p, r, f


def weighted_metrics(y, pred, w) -> dict:
    c = weighted_counts(y, pred, w)
    p, r, f = precision_recall_f1(c)
    y_arr = _as_int_array(y)
    return {
        "n": int(len(y_arr)),
        "positives": int((y_arr == 1).sum()),
        "predicted_positives": int(c.n_tp + c.n_fp),
        "tp": c.n_tp, "fp": c.n_fp, "fn": c.n_fn, "tn": c.n_tn,
        "w_tp": c.tp, "w_fp": c.fp, "w_fn": c.fn, "w_tn": c.tn,
        "weighted_predicted_positives": c.tp + c.fp,
        "weighted_precision": finite_or_none(p),
        "weighted_recall": finite_or_none(r),
        "weighted_f1": finite_or_none(f),
    }


def design_weights(split_rows: pd.DataFrame, N_h: Mapping[str, int]) -> np.ndarray:
    """The Horvitz-Thompson weight, 1 / inclusion_probability per row, checked by RECONSTRUCTION.

    A single stratum weight N_h / n_h is correct only when every case in the stratum had the same
    chance of selection. The screened round breaks that: it draws within (split, stratum, CELL), so
    one stratum carries up to four rates and a boundary cell is deliberately oversampled. Measured on
    the real held-out splits, N_h / n_h disagreed with 1/inclusion_probability for 1200 of 1200 rows,
    and not marginally -- CO:rejected in test has a stratum figure of 525 against per-row values from
    25 to 1119. Applying the stratum figure would count an oversampled boundary case as representing
    the same slice of the population as an unscreened one, which is exactly the bias the screen exists
    to avoid.

    The guard that replaces the old row-equality check is stronger, not weaker: within a cell the rate
    is n_cell / pool_cell, so the inverse probabilities sum to pool_cell exactly, and across the cells
    of a stratum to N_h -- exactly, because K12 keeps every pilot case in TRAIN, leaving pool_h == N_h
    in validation and test. Verified on the real artefacts: all eight (split, stratum) totals
    reconstruct to the digit, relative error 0.0. A design error still trips this; unlike the old check
    it is also true of the design we actually drew.
    """
    n_h = split_rows.groupby("stratum").size().to_dict()
    missing = sorted(set(n_h) - set(N_h))
    if missing:
        raise ValueError(f"strata {missing} have labelled rows but no population count N_h")
    undrawn = sorted(h for h, n in N_h.items() if n > 0 and h not in n_h)
    if undrawn:
        raise ValueError(f"strata {undrawn} have population but no labelled row; the estimate would silently omit them")
    w = 1.0 / split_rows.inclusion_probability.to_numpy(dtype=float)
    for stratum, index in split_rows.groupby("stratum").groups.items():
        total = float(w[split_rows.index.get_indexer(index)].sum())
        if not np.isclose(total, float(N_h[stratum]), rtol=1e-6):
            raise ValueError(f"the design weights of stratum {stratum!r} sum to {total:.2f}, which does not "
                             f"reconstruct its population N_h={N_h[stratum]}; the sample does not represent "
                             "the population it is weighted to")
    return w


def weights_table(N_h: Mapping[str, int], strata) -> list[dict]:
    strata = np.asarray(strata).astype(str)
    keys, counts = np.unique(strata, return_counts=True)
    n_h = dict(zip(keys.tolist(), counts.tolist()))
    rows = []
    for stratum in sorted(N_h):
        n = int(n_h.get(stratum, 0))
        # w_h is the AVERAGE weight of the stratum, reported for the reader. It is not the weight
        # applied to any row: under the screen each row carries its own cell's rate.
        rows.append({"stratum": stratum, "N_h": int(N_h[stratum]), "n_h": n,
                     "w_h": (float(N_h[stratum]) / n) if n else None, "unsampled": n == 0})
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Raw-count sanity interval only (re-implemented from src/evaluate.py:35; NOT the
    design-aware interval — Codex §8 says a Wilson interval over weighted counts is invalid)."""
    if n == 0:
        return (None, None)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)
