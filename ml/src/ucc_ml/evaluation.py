"""Design-weighted evaluation for the UCC relevance screener (ml/specs/evaluation_protocol_v1.md).

Weights (protocol §2): in a split S, stratum h has N_h candidates and n_h non-repeat labelled cases drawn,
INSUFFICIENT_EVIDENCE included. Every labelled case gets `1 / inclusion_probability`, the rate of the CELL it was
drawn from. `w_h = N_h / n_h` is the stratum's AVERAGE weight, reported for the reader and applied to nothing: the
screened round draws within (split, stratum, cell), so one stratum carries up to four rates and no single value is
the weight of every row in it. `design_weights` checks that the weights RECONSTRUCT N_h. Rates use the RESOLVED
cases with these weights, so they
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


# ---------------------------------------------------------------------------
# Uncertainty (protocol §5): stratified cluster Bayesian bootstrap with Jeffreys pseudo-counts
# ---------------------------------------------------------------------------
BOOTSTRAP_METHOD = "stratified cluster Bayesian bootstrap with Jeffreys pseudo-counts (prior 0.5)"
REPORT_KEYS: tuple[str, ...] = ("n", "positives", "negatives", "weights", "model", "rules", "delta_model_minus_rules",
                                "unresolved_share", "unresolved_share_among_model_positive", "review_queue", "per_region",
                                "per_stratum", "bootstrap")


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float | None
    lower: float | None
    upper: float | None
    n_resamples: int
    n_failed: int

    def to_dict(self) -> dict:
        return {"estimate": self.estimate, "lower": self.lower, "upper": self.upper,
                "n_resamples": self.n_resamples, "n_failed": self.n_failed}


def _percentile_result(estimate, replicates, level: float) -> BootstrapResult:
    replicates = np.asarray(replicates, dtype=float)
    n_failed = int(np.isnan(replicates).sum())
    if n_failed == len(replicates):
        return BootstrapResult(finite_or_none(estimate), None, None, len(replicates), n_failed)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.nanpercentile(replicates, [100 * alpha, 100 * (1 - alpha)])
    return BootstrapResult(finite_or_none(estimate), float(lo), float(hi), len(replicates), n_failed)


def jeffreys_cluster_draws(y, pred_model, pred_rules, strata, groups, N_h, weights, *, n_resamples: int = 2000,
                           seed: int = 0, prior: float = 0.5) -> dict:
    """Stratified cluster Bayesian bootstrap with Jeffreys pseudo-counts. y: 1 / 0 / -1 (INSUFFICIENT_EVIDENCE) for EVERY
    non-repeat labelled case of the split. Returns {stratum: (n_resamples, 9) estimated population cell totals};
    cell = (1-y)*4 + (1-model)*2 + (1-rules) for resolved cases, 8 = unresolved.

    A row contributes its DESIGN WEIGHT, not 1.0. Counting rows and rescaling the resulting shares to N_h makes the
    expected cell share the unweighted sample share, which is algebraically w_h = N_h / n_h -- the retired estimator,
    rediscovered. Under the screen that is not a near-miss: the boundary cells are ~19% of the sample and ~6%, ~2%
    and ~1% of the population, and they are enriched in RELEVANT cases, so both the centre and the interval move
    upward. Measured on the real test split before this was fixed, the review-queue precision interval
    [0.0763, 0.3517] did not contain the design-correct value 0.0358.

    The pseudo-mass scales with the stratum's mean weight PER CLUSTER. Jeffreys smoothing was calibrated against
    totals on the SAMPLE scale (n_h ~ 200); weighted totals sit on the POPULATION scale (N_h ~ 10^5), where a fixed
    0.5 is negligible and the property that keeps a stratum with no sampled false positive uncertain silently
    vanishes -- measured, that stratum's precision lower bound goes from 0.477 to 0.999, i.e. no uncertainty at all.
    The scale is N_h / (number of clusters), not N_h / (number of rows), because the cluster is the unit of evidence
    being resampled: repeating one borrower's rows five times must not change the smoothing, and dividing by rows
    would cut it fivefold. Measured width ratio under that duplication: 1.000 by clusters, 0.969-1.022 by rows."""
    y = np.asarray(y, dtype=int); pm = _as_int_array(pred_model); pr = _as_int_array(pred_rules)
    w = np.asarray(weights, dtype=float)
    strata = np.asarray(strata).astype(str); groups = np.asarray(groups).astype(str)
    missing = sorted(set(strata.tolist()) - set(N_h))
    if missing:
        raise ValueError(f"strata {missing} have labelled rows but no population count N_h")
    rng = np.random.default_rng(seed)
    cell = np.where(y < 0, 8, (1 - np.clip(y, 0, 1)) * 4 + (1 - pm) * 2 + (1 - pr))
    out = {}
    for h in sorted(set(strata.tolist())):
        m = strata == h
        allowed = np.zeros(9, dtype=bool); allowed[8] = True
        for yy in (1, 0):
            for mm in (1, 0):
                for rr in set(pr[m].tolist()):
                    allowed[(1 - yy) * 4 + (1 - mm) * 2 + (1 - rr)] = True
        codes, uniq = pd.factorize(pd.Series(groups[m]))
        counts = np.zeros((len(uniq), 9)); np.add.at(counts, (codes, cell[m]), w[m])
        totals = rng.gamma(1.0, 1.0, size=(n_resamples, len(uniq))) @ counts
        unit = float(N_h[h]) / max(len(uniq), 1)             # mean weight per CLUSTER: invariant to duplication
        totals[:, allowed] += rng.gamma(prior, unit, size=(n_resamples, int(allowed.sum())))
        out[h] = float(N_h[h]) * totals / totals.sum(axis=1, keepdims=True)
    return out


def _cell(yy: int, mm: int, rr: int) -> int:
    return (1 - yy) * 4 + (1 - mm) * 2 + (1 - rr)


def draws_to_rates(draws: dict, subset) -> dict:
    S = sum(draws[h] for h in subset if h in draws)
    m_tp = S[:, _cell(1, 1, 1)] + S[:, _cell(1, 1, 0)]; m_fp = S[:, _cell(0, 1, 1)] + S[:, _cell(0, 1, 0)]
    m_fn = S[:, _cell(1, 0, 1)] + S[:, _cell(1, 0, 0)]
    r_tp = S[:, _cell(1, 1, 1)] + S[:, _cell(1, 0, 1)]; r_fp = S[:, _cell(0, 1, 1)] + S[:, _cell(0, 0, 1)]
    r_fn = S[:, _cell(1, 1, 0)] + S[:, _cell(1, 0, 0)]
    with np.errstate(divide="ignore", invalid="ignore"):
        mp, mr = m_tp / (m_tp + m_fp), m_tp / (m_tp + m_fn)
        rp, rr = r_tp / (r_tp + r_fp), r_tp / (r_tp + r_fn)
        mf, rf = 2 * mp * mr / (mp + mr), 2 * rp * rr / (rp + rr)
        unresolved = S[:, 8] / S.sum(axis=1)
    return {"model": {"weighted_precision": mp, "weighted_recall": mr, "weighted_f1": mf},
            "rules": {"weighted_precision": rp, "weighted_recall": rr, "weighted_f1": rf},
            "delta": {"delta_weighted_precision": mp - rp, "delta_weighted_recall": mr - rr, "delta_weighted_f1": mf - rf},
            "unresolved_share": unresolved}


def evaluate_split(y_all, pred_model_all, pred_rules_all, strata_all, groups_all, regions_all, N_h, weights, *,
                   n_resamples: int, seed: int, level: float) -> dict:
    """Report block for one split. y_all: 1/0/-1 for EVERY non-repeat labelled case (INSUFFICIENT_EVIDENCE included).

    `weights` is the caller's design weight per row -- `design_weights(split_rows, N_h)`, i.e. the rate of the CELL
    each row was drawn from. It is passed in rather than derived here because it cannot be derived here: N_h and a
    stratum count give only the stratum AVERAGE, which the protocol forbids applying to any row. Both callers
    already compute it. One array feeds the point estimates and the bootstrap, so the estimate can never sit
    outside its own interval."""
    y_all = np.asarray(y_all, dtype=int); pm = _as_int_array(pred_model_all); pr = _as_int_array(pred_rules_all)
    strata_all = np.asarray(strata_all).astype(str); groups_all = np.asarray(groups_all).astype(str)
    regions_all = np.asarray(regions_all).astype(str)
    n_h = pd.Series(strata_all).value_counts().to_dict()
    missing = sorted(set(n_h) - set(N_h))
    if missing:
        raise ValueError(f"strata {missing} have labelled rows but no population count N_h")
    undrawn = sorted(h for h, n in N_h.items() if n > 0 and h not in n_h)
    if undrawn:
        raise ValueError(f"strata {undrawn} have population but no labelled row")
    rq = sorted(h for h in n_h if h.endswith(":rejected"))
    if not rq:
        raise ValueError("no labelled case in a rules-rejected stratum: the review queue cannot be estimated")
    w_all = np.asarray(weights, dtype=float)
    if len(w_all) != len(y_all):
        raise ValueError(f"weights has {len(w_all)} entries for {len(y_all)} labelled rows")
    for h in sorted(n_h):
        total = float(w_all[strata_all == h].sum())
        if not np.isclose(total, float(N_h[h]), rtol=1e-6):
            raise ValueError(f"the design weights of stratum {h!r} sum to {total:.2f}, which does not reconstruct "
                             f"its population N_h={N_h[h]}")
    draws = jeffreys_cluster_draws(y_all, pm, pr, strata_all, groups_all, N_h, w_all,
                                   n_resamples=n_resamples, seed=seed)

    def block(mask: np.ndarray, subset: list) -> dict:
        r = mask & (y_all >= 0)
        rates = draws_to_rates(draws, subset)
        model = weighted_metrics(y_all[r], pm[r], w_all[r]); rules = weighted_metrics(y_all[r], pr[r], w_all[r])
        model["ci"] = {k: _percentile_result(model[k], v, level).to_dict() for k, v in rates["model"].items()}
        rules["ci"] = {k: _percentile_result(rules[k], v, level).to_dict() for k, v in rates["rules"].items()}
        delta = {}
        for k, v in rates["delta"].items():
            base = k.replace("delta_", "")
            est = None if model[base] is None or rules[base] is None else model[base] - rules[base]
            delta[k] = _percentile_result(est, v, level).to_dict()
        pp = mask & (pm == 1)
        return {"n": int(r.sum()), "positives": int((y_all[r] == 1).sum()), "model": model, "rules": rules, "delta": delta,
                "unresolved_share": _percentile_result(float((w_all * (mask & (y_all < 0))).sum() / w_all[mask].sum()),
                                                       rates["unresolved_share"], level).to_dict(),
                "unresolved_share_among_model_positive": (float((w_all * (pp & (y_all < 0))).sum() / (w_all * pp).sum())
                                                          if pp.any() else None)}

    top = block(np.ones(len(y_all), dtype=bool), sorted(n_h))
    clusters = pd.DataFrame({"stratum": strata_all, "group_id": groups_all}).drop_duplicates()
    return {"n": top["n"], "positives": top["positives"], "negatives": int((y_all == 0).sum()),
            "weights": weights_table(N_h, strata_all), "model": top["model"], "rules": top["rules"],
            "delta_model_minus_rules": top["delta"], "unresolved_share": top["unresolved_share"],
            "unresolved_share_among_model_positive": top["unresolved_share_among_model_positive"],
            "review_queue": block(np.isin(strata_all, rq), rq),
            "per_region": [{"region": g, **block(regions_all == g, sorted(h for h in n_h if h.startswith(g + ":")))}
                           for g in sorted(set(regions_all.tolist()))],
            "per_stratum": [{"stratum": h, **block(strata_all == h, [h])} for h in sorted(n_h)],
            "bootstrap": {"method": BOOTSTRAP_METHOD, "n_resamples": n_resamples, "seed": seed, "level": level,
                          "n_clusters": int(len(clusters)),
                          "straddling_groups": int((clusters.groupby("group_id").size() > 1).sum())}}
