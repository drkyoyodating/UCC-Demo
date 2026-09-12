"""The threshold objective fixed by the protocol, the one-sided precision lower bound, calibration diagnostics and the
decision-region probability gate, the N_h-weighted unresolved prevalence, and the report block on the synthetic world."""
import numpy as np
import pandas as pd
import pytest

from ucc_ml.evaluation import (
    REPORT_KEYS,
    ThresholdSelection,
    decide_from_scores,
    design_weights,
    evaluate_split,
    precision_lower_bound,
    probability_gate,
    select_threshold,
    unresolved_prevalence,
    weighted_ece,
    weighted_metrics,
)


def _validation_like():
    scores = np.array([1 - i / 61 for i in range(60)])
    y = np.zeros(60, dtype=int)
    y[:35] = 1
    y[20] = 0
    y[45] = 1
    y[50] = 1
    return scores, y, np.ones(60)


def test_select_threshold_maximises_recall_under_the_precision_floor():
    scores, y, w = _validation_like()
    sel = select_threshold(scores, y, w, min_weighted_precision=0.95, min_predicted_positives=30)
    assert isinstance(sel, ThresholdSelection)
    assert sel.status == "production" and sel.fallback is None
    assert sel.threshold == pytest.approx(scores[34])
    assert sel.predicted_positives == 35
    assert sel.weighted_precision == pytest.approx(34 / 35)
    assert sel.weighted_recall == pytest.approx(34 / 36)
    assert sel.n == 60 and sel.positives == 36
    assert sel.objective == {"min_weighted_precision": 0.95, "min_predicted_positives": 30, "selected_on": "validation"}
    assert len(sel.curve) == 60 and sel.curve[0]["predicted_positives"] == 1
    assert decide_from_scores(scores, sel.threshold).sum() == 35
    d = sel.to_dict()
    assert d["status"] == "production" and d["threshold"] == pytest.approx(scores[34])


def test_select_threshold_falls_back_to_experimental_when_no_threshold_qualifies():
    scores, y, w = _validation_like()
    sel = select_threshold(scores, y, w, min_weighted_precision=0.99, min_predicted_positives=30)
    assert sel.status == "experimental"
    assert sel.fallback == "max_weighted_f1_with_min_predicted_positives"
    assert sel.threshold == pytest.approx(scores[34])
    assert sel.predicted_positives == 35


def test_select_threshold_with_too_few_cases_is_experimental_at_half():
    scores, y, w = _validation_like()
    sel = select_threshold(scores[:20], y[:20], w[:20], min_weighted_precision=0.95, min_predicted_positives=30)
    assert sel.status == "experimental"
    assert sel.fallback == "insufficient_predicted_positives"
    assert sel.threshold == 0.5


def test_weights_change_the_threshold_choice():
    scores, y, w = _validation_like()
    heavy = w.copy()
    heavy[20] = 40.0                      # the one negative inside the top-35 now weighs 40 → precision < 0.95 there
    sel = select_threshold(scores, y, heavy, min_weighted_precision=0.95, min_predicted_positives=30)
    assert sel.status == "experimental"


def _lower_bound_sample(rejected_relevant: int):
    """CO:accepted N=1,000: 100 drawn, 70 RELEVANT (66 predicted), 30 NOT_RELEVANT (3 predicted).
    CO:rejected N=100,000: 100 drawn, `rejected_relevant` RELEVANT (all predicted), the rest NOT_RELEVANT (none)."""
    y = np.array([1] * 70 + [0] * 30 + [1] * rejected_relevant + [0] * (100 - rejected_relevant))
    pm = np.array([1] * 66 + [0] * 4 + [1] * 3 + [0] * 27 + [1] * rejected_relevant + [0] * (100 - rejected_relevant))
    pr = np.array([1] * 100 + [0] * 100)
    strata = np.array(["CO:accepted"] * 100 + ["CO:rejected"] * 100)
    w = np.array([10.0] * 100 + [1000.0] * 100)
    return y, pm, pr, strata, np.arange(200).astype(str), {"CO:accepted": 1000, "CO:rejected": 100000}, w


def test_precision_lower_bound_downgrades_a_point_estimate_that_only_just_clears_the_floor():
    y, pm, pr, strata, groups, N_h, w = _lower_bound_sample(3)
    point = weighted_metrics(y, pm, w)["weighted_precision"]
    lcb = precision_lower_bound(y, pm, pr, strata, groups, N_h, w, n_resamples=2000, seed=0)
    assert point > 0.95 and lcb < 0.95
    y2, pm2, pr2, strata2, groups2, N_h2, w2 = _lower_bound_sample(40)
    lcb2 = precision_lower_bound(y2, pm2, pr2, strata2, groups2, N_h2, w2, n_resamples=2000, seed=0)
    assert lcb2 > lcb and weighted_metrics(y2, pm2, w2)["weighted_precision"] >= lcb2
    no_suggestion = precision_lower_bound(y, np.zeros(200, dtype=int), pr, strata, groups, N_h, w,
                                          n_resamples=2000, seed=0)
    assert no_suggestion < 0.5          # with no predicted positive only the pseudo-counts speak: nothing is certified


def test_weighted_ece_known_values():
    y = np.array([1, 0, 1, 0])
    assert weighted_ece(y, np.array([1.0, 0.0, 1.0, 0.0]), np.ones(4))["ece"] == pytest.approx(0.0)
    assert weighted_ece(y, np.array([0.5, 0.5, 0.5, 0.5]), np.ones(4))["ece"] == pytest.approx(0.0)
    d = weighted_ece(np.zeros(4, dtype=int), np.full(4, 0.9), np.ones(4))
    assert d["ece"] == pytest.approx(0.9) and d["n_bins"] == 10
    full = [b for b in d["bins"] if b["n"] > 0]
    assert len(full) == 1 and full[0]["observed_rate"] == 0.0 and full[0]["mean_score"] == pytest.approx(0.9)
    wd = weighted_ece(np.array([1, 0]), np.array([0.9, 0.9]), np.array([9.0, 1.0]))
    assert wd["ece"] == pytest.approx(0.0)


def _gate_sample(decision_positives: int):
    """400 heavily weighted cases scored 0.02 with 8 positives (calibrated), 60 cases scored 0.8 with
    `decision_positives` positives: 48 is calibrated, 33 is 25 points overconfident."""
    p = np.r_[np.full(400, 0.02), np.full(60, 0.8)]
    y = np.r_[np.ones(8), np.zeros(392), np.ones(decision_positives), np.zeros(60 - decision_positives)].astype(int)
    w = np.r_[np.full(400, 20.0), np.ones(60)]
    return y, p, w


def test_probability_gate_rejects_overconfidence_hidden_by_the_near_zero_bin():
    y, p, w = _gate_sample(33)
    gate = probability_gate(y, p, w)
    assert gate["weighted_ece"] <= 0.10                       # the overall ECE alone would call it a probability
    assert gate["decision_region_weighted_ece"] == pytest.approx(0.25)
    assert gate["decision_region_cases"] == 60 and gate["score_type"] == "raw_score"
    y, p, w = _gate_sample(48)
    assert probability_gate(y, p, w)["score_type"] == "calibrated_probability"
    assert probability_gate(y, p, w, min_decision_cases=61) == {
        **probability_gate(y, p, w, min_decision_cases=61), "decision_region_weighted_ece": None, "score_type": "raw_score"}


def test_unresolved_prevalence_weights_the_total_by_stratum_population():
    # 10 drawn from 100 and 10 from 900, so the inverse probabilities reconstruct each stratum exactly.
    # This fixture is single-rate, where the cell rate and the stratum average coincide -- the assertions
    # below are unchanged for that reason, and a multi-rate case is covered in test_bootstrap.py.
    table = pd.DataFrame({"region": ["CO"] * 20, "stratum": ["CO:accepted"] * 10 + ["CO:rejected"] * 10,
                          "inclusion_probability": [10 / 100] * 10 + [10 / 900] * 10,
                          "y": [np.nan] + [1.0] * 9 + [np.nan] * 5 + [0.0] * 5})
    pred = [1] * 5 + [0] * 5 + [1] + [0] * 4 + [1] + [0] * 4
    rows = unresolved_prevalence(table, {"CO:accepted": 100, "CO:rejected": 900}, pred=pred)
    assert rows[0] == {"region": "CO", "stratum": "CO:accepted", "n_labelled": 10, "n_unresolved": 1, "share_unresolved": 0.1}
    total = rows[-1]
    assert total["share_unresolved"] == pytest.approx((100 * 0.1 + 900 * 0.5) / 1000)      # unweighted pooling: 0.30
    assert total["share_unresolved_among_predicted_positive"] == pytest.approx((10 + 90) / (5 * 10 + 2 * 90))
    assert "share_unresolved_among_predicted_positive" not in unresolved_prevalence(table, {"CO:accepted": 100, "CO:rejected": 900})[-1]


def test_evaluate_split_report_on_the_synthetic_world():
    from ucc_ml.contracts import STRATA
    from ucc_ml.provenance import canonical_json_bytes
    from ucc_ml.synthetic import make_world
    from ucc_ml.training import build_model_table, split_populations

    world = make_world()
    table = build_model_table(world.candidates, world.splits, world.labels)
    val = table[table.split == "validation"].reset_index(drop=True)
    N_h = split_populations(world.candidates, world.splits)["validation"]
    w = design_weights(val, N_h)
    rng = np.random.default_rng(0)
    y_all = val.y.fillna(-1).to_numpy().astype(int)
    pred = np.where(y_all == 1, rng.random(len(val)) < 0.9, rng.random(len(val)) < 0.1).astype(int)
    report = evaluate_split(y_all, pred, val.baseline_qualifies.to_numpy().astype(int), val.stratum, val.group_id,
                            val.region, N_h, w, n_resamples=100, seed=0, level=0.95)
    assert set(report) == set(REPORT_KEYS)
    ok = y_all >= 0
    assert report["n"] == int(ok.sum())
    assert report["model"]["weighted_precision"] == pytest.approx(weighted_metrics(y_all[ok], pred[ok], w[ok])["weighted_precision"])
    assert {r["stratum"] for r in report["weights"]} == set(STRATA)
    assert report["review_queue"]["n"] == int((ok & val.stratum.str.endswith(":rejected").to_numpy()).sum())
    assert canonical_json_bytes(report)
