"""Design-weighted counts and P/R/F1. The weight is N_h / n_h over ALL drawn cases (INSUFFICIENT_EVIDENCE included),
checked against 1/inclusion_probability, and a Monte-Carlo test proves it estimates the RESOLVABLE population without
bias on a population whose unresolved share differs by stratum, where N_h / (resolved n_h) is visibly biased."""
import math

import numpy as np
import pandas as pd
import pytest

from ucc_ml.evaluation import (
    WeightedCounts,
    design_weights,
    precision_recall_f1,
    weighted_counts,
    weighted_metrics,
    weights_table,
    wilson,
)

# stratum: (N, INSUFFICIENT_EVIDENCE share, P(RELEVANT | resolvable), P(pred | RELEVANT), P(pred | NOT_RELEVANT), P(pred | IE))
POPULATION = {"CO:accepted": (2400, 0.05, 0.90, 0.85, 0.10, 0.30), "CO:rejected": (12000, 0.55, 0.05, 0.55, 0.004, 0.02),
              "CT:accepted": (600, 0.08, 0.80, 0.85, 0.12, 0.30), "CT:rejected": (5000, 0.50, 0.08, 0.50, 0.006, 0.02)}


def _population(seed: int = 11):
    rng = np.random.default_rng(seed)
    ys, preds, strata = [], [], []
    for stratum, (N, ie, prev, p_r, p_nr, p_ie) in POPULATION.items():
        y = np.where(rng.random(N) < ie, -1, (rng.random(N) < prev).astype(int))
        preds.append((rng.random(N) < np.select([y == 1, y == 0], [p_r, p_nr], p_ie)).astype(int))
        ys.append(y)
        strata.append(np.full(N, stratum))
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(strata)


def test_weighted_counts_and_prf_basic():
    c = weighted_counts(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]), np.array([2.0, 2.0, 3.0, 3.0]))
    assert c == WeightedCounts(tp=2.0, fp=3.0, fn=2.0, tn=3.0, n_tp=1, n_fp=1, n_fn=1, n_tn=1)
    p, r, f = precision_recall_f1(c)
    assert p == pytest.approx(2 / 5) and r == pytest.approx(0.5) and f == pytest.approx(2 * 0.4 * 0.5 / 0.9)
    p0, r0, f0 = precision_recall_f1(WeightedCounts(0, 0, 0, 4, 0, 0, 0, 4))
    assert math.isnan(p0) and math.isnan(r0) and math.isnan(f0)
    assert weighted_metrics([1, 0], [0, 0], [1.0, 1.0])["weighted_precision"] is None


def test_design_weights_are_per_cell_rates_that_reconstruct_the_population():
    """One stratum, two cells, different rates -- which is what the screened round actually draws."""
    rows = pd.DataFrame({"case_id": list("abcde"), "stratum": ["CO:accepted"] * 2 + ["CO:rejected"] * 3,
                         "inclusion_probability": [2 / 10, 2 / 10, 2 / 200, 1 / 100, 1 / 100]})
    N_h = {"CO:accepted": 10, "CO:rejected": 300}
    w = design_weights(rows, N_h)
    assert w.tolist() == pytest.approx([5, 5, 100, 100, 100])
    assert w[2:].sum() == pytest.approx(300)          # two cells, 200 + 100, reconstruct the stratum
    assert w[:2].sum() == pytest.approx(10)
    with pytest.raises(ValueError, match="CT:accepted"):
        design_weights(rows, {**N_h, "CT:accepted": 5})
    assert design_weights(rows, {**N_h, "CT:accepted": 0}).tolist() == pytest.approx([5, 5, 100, 100, 100])
    with pytest.raises(ValueError, match="no population count"):
        design_weights(rows, {"CO:accepted": 10})


def test_design_weights_refuse_a_sample_that_does_not_reconstruct_its_population():
    """The guard that replaced the row-equality check: a weight set that does not sum to N_h means the
    sample does not represent the population it is being weighted to, whatever the per-row rates say."""
    rows = pd.DataFrame({"case_id": list("abc"), "stratum": ["CO:rejected"] * 3,
                         "inclusion_probability": [3 / 300, 3 / 300, 3 / 900]})
    with pytest.raises(ValueError, match="does not reconstruct its population"):
        design_weights(rows, {"CO:rejected": 300})


def test_design_weights_estimate_the_resolvable_population_without_bias():
    y, pred, strata = _population()
    res = y >= 0
    truth = weighted_metrics(y[res], pred[res], np.ones(int(res.sum())))
    truth = np.array([truth["weighted_precision"], truth["weighted_recall"]])
    N_h = {h: int((strata == h).sum()) for h in POPULATION}
    members = {h: np.flatnonzero(strata == h) for h in POPULATION}
    rng = np.random.default_rng(1400)
    design, resolved_count = np.zeros((1000, 2)), np.zeros((1000, 2))
    for rep in range(1000):
        idx = np.concatenate([rng.choice(members[h], size=400, replace=False) for h in POPULATION])
        rows = pd.DataFrame({"case_id": idx.astype(str), "stratum": strata[idx],
                             "inclusion_probability": [400 / N_h[h] for h in strata[idx]]})
        ok = y[idx] >= 0
        m = weighted_metrics(y[idx][ok], pred[idx][ok], design_weights(rows, N_h)[ok])
        design[rep] = m["weighted_precision"], m["weighted_recall"]
        n_resolved = pd.Series(strata[idx][ok]).value_counts()
        wrong = np.array([N_h[h] / n_resolved[h] for h in strata[idx][ok]])     # N_h / resolved n_h: the retired weight
        m = weighted_metrics(y[idx][ok], pred[idx][ok], wrong)
        resolved_count[rep] = m["weighted_precision"], m["weighted_recall"]
    bias = design.mean(axis=0) - truth
    assert abs(bias[0]) < 0.005 and abs(bias[1]) < 0.005       # measured +0.0003 / +0.0003
    retired_bias = resolved_count.mean(axis=0) - truth
    assert retired_bias[0] < -0.008 and retired_bias[1] < -0.02    # measured -0.0118 / -0.0340


def test_weights_table_lists_every_population_stratum():
    strata = np.array(["CO:accepted", "CO:accepted", "CT:rejected"])
    rows = weights_table({"CO:accepted": 10, "CT:rejected": 300, "CT:accepted": 5}, strata)
    assert [r["stratum"] for r in rows] == ["CO:accepted", "CT:accepted", "CT:rejected"]
    assert rows[1] == {"stratum": "CT:accepted", "N_h": 5, "n_h": 0, "w_h": None, "unsampled": True}
    assert rows[2]["w_h"] == 300.0 and rows[2]["unsampled"] is False


def test_wilson_is_a_raw_count_interval():
    lo, hi = wilson(9, 10)
    assert 0.55 < lo < 0.6 and 0.98 < hi < 1.0
    assert wilson(0, 0) == (None, None)
