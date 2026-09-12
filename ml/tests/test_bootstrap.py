"""Stratified cluster Bayesian bootstrap with Jeffreys pseudo-counts (protocol §5). The interval must cover the truth at
about the nominal rate where false positives are rare and heavily weighted, stay uncertain when a heavy stratum shows no
false positive, not shrink when borrower rows are duplicated inside their group, and the paired delta must centre on
zero when model and rules agree."""
import json

import numpy as np
import pandas as pd
import pytest

from ucc_ml.evaluation import (
    BOOTSTRAP_METHOD,
    REPORT_KEYS,
    evaluate_split,
    jeffreys_cluster_draws,
    weighted_metrics,
)

N_SMALL = {"CO:accepted": 4000, "CO:rejected": 40000, "CT:accepted": 3000, "CT:rejected": 30000}
#: Every fixture draws each stratum in TWO cells at very different rates, as the screened round really does:
#: a small "boundary" pool sampled hard and a large "remainder" pool sampled lightly. A single-rate fixture
#: cannot fail this file's central property, because N_h/n_h and 1/inclusion_probability coincide exactly when
#: a stratum has one rate -- which is why the retired weight survived here unnoticed.
BOUNDARY_POOL_SHARE = 0.1


def _attach_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Give every row the rate of its own cell, so sum(1/pi) reconstructs N_h exactly, as the real design does."""
    df = df.reset_index(drop=True)
    pi = np.empty(len(df), dtype=float)
    for stratum, rows in df.groupby("stratum"):
        pools = {"boundary": BOUNDARY_POOL_SHARE * N_SMALL[stratum],
                 "remainder": (1.0 - BOUNDARY_POOL_SHARE) * N_SMALL[stratum]}
        for cell, cell_rows in rows.groupby("cell"):
            pi[cell_rows.index.to_numpy()] = len(cell_rows) / pools[cell]
    return df.assign(inclusion_probability=pi)


def _w(df: pd.DataFrame) -> np.ndarray:
    return (1.0 / df.inclusion_probability).to_numpy()


def test_interval_covers_rare_heavily_weighted_false_positives():
    rng = np.random.default_rng(2026)
    spec = {"CO:accepted": (13395, 0.05, 0.90, 0.95, 0.15), "CO:rejected": (166266, 0.45, 0.03, 0.40, 0.003),
            "CT:accepted": (1628, 0.07, 0.85, 0.93, 0.20), "CT:rejected": (61767, 0.40, 0.04, 0.35, 0.004)}
    ys, pm, hs = [], [], []
    for h, (N, ie, prev, p_r, p_nr) in spec.items():
        y = np.where(rng.random(N) < ie, -1, (rng.random(N) < prev).astype(int))
        ys.append(y); pm.append((rng.random(N) < np.select([y == 1, y == 0], [p_r, p_nr], 0.01)).astype(int)); hs.append(np.full(N, h))
    y, pm, hs = np.concatenate(ys), np.concatenate(pm), np.concatenate(hs)
    pr = np.array([h.endswith(":accepted") for h in hs]).astype(int); res = y >= 0
    truth = weighted_metrics(y[res], pm[res], np.ones(int(res.sum())))
    N_h = {h: int((hs == h).sum()) for h in spec}
    covered_p = covered_r = 0
    for rep in range(300):
        idx = np.concatenate([rng.choice(np.flatnonzero(hs == h), 200, replace=False) for h in spec])
        w = np.array([N_h[h] / 200.0 for h in hs[idx]])       # uniform 200 per stratum: one rate, w = N_h/n_h
        report = evaluate_split(y[idx], pm[idx], pr[idx], hs[idx], np.arange(len(idx)).astype(str),
                                np.array([h[:2] for h in hs[idx]]), N_h, w, n_resamples=1000, seed=rep, level=0.95)
        ci = report["model"]["ci"]
        covered_p += int(ci["weighted_precision"]["lower"] <= truth["weighted_precision"] <= ci["weighted_precision"]["upper"])
        covered_r += int(ci["weighted_recall"]["lower"] <= truth["weighted_recall"] <= ci["weighted_recall"]["upper"])
    assert covered_p / 300 >= 0.90   # measured 0.937 (draft group_bootstrap on the same samples: 0.627)
    assert covered_r / 300 >= 0.90   # measured 0.953


def _sample(seed: int = 3, groups: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(groups):
        stratum = ("CO:accepted", "CO:rejected", "CT:accepted", "CT:rejected")[g % 4]
        accepted = stratum.endswith(":accepted")
        cell = "boundary" if (g // 4) % 2 == 0 else "remainder"   # by GROUP, so a cluster never straddles cells
        # The outcome mix DIFFERS by cell, which is the whole point of screening and the ingredient a
        # fixture needs to tell the two weightings apart. Measured on the real held-out rounds: the
        # unscreened cell is 87-95% of the population but 27-74% of the sample and its relevant share is
        # ~0.00, while the boundary cells run 0.04-0.50. Equal-mix cells make N_h/n_h and the cell rate
        # agree to three decimals however far apart their weights are.
        base = 0.8 if accepted else 0.25
        p_relevant = min(base * 2.0, 0.95) if cell == "boundary" else base * 0.25
        y = -1 if rng.random() < 0.1 else int(rng.random() < p_relevant)
        # The model's ACCURACY differs by cell too, not only the prevalence. This is what actually separates
        # the two weightings: a rate only moves under reweighting if the rate itself differs by cell.
        # In the real round the false negatives concentrate in the boundary cells, which carry the LOWEST
        # weights (25-175 against 889-1119 in the unscreened cell), so a stratum average over-weights them
        # and drags recall down -- measured 0.789 against 0.969 on the real test split.
        hit = 0.55 if cell == "boundary" else 0.97
        for _ in range(int(rng.integers(1, 4))):
            rows.append((stratum, cell, stratum[:2], f"g{g}", y,
                         int(rng.random() < (hit if y == 1 else 0.1)), int(accepted)))
    return _attach_rates(pd.DataFrame(rows, columns=["stratum", "cell", "region", "group_id", "y", "pred", "rules"]))


def _report(df: pd.DataFrame, pred=None, **kw) -> dict:
    args = {"n_resamples": 400, "seed": 1, "level": 0.95, **kw}
    return evaluate_split(df.y, df.pred if pred is None else pred, df.rules, df.stratum, df.group_id, df.region,
                          N_SMALL, _w(df), **args)


def _width(report: dict, key: str) -> float:
    ci = report["model"]["ci"][key]
    return ci["upper"] - ci["lower"]


def test_report_block_shape_point_estimates_and_reproducibility():
    df = _sample()
    a, b, c = _report(df), _report(df), _report(df, seed=2)
    assert set(a) == set(REPORT_KEYS)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["model"]["ci"]["weighted_precision"]["lower"] != c["model"]["ci"]["weighted_precision"]["lower"]
    res = (df.y >= 0).to_numpy()
    w = _w(df)                      # the cell's rate, not N_h / n_h: that formula is what this file used to pin
    expected = weighted_metrics(df.y[res], df.pred[res], w[res])
    assert a["n"] == int(res.sum()) and a["negatives"] == int((df.y == 0).sum())
    assert a["model"]["weighted_precision"] == pytest.approx(expected["weighted_precision"])
    assert a["model"]["weighted_recall"] == pytest.approx(expected["weighted_recall"])
    for block in [a, a["review_queue"], *a["per_region"], *a["per_stratum"]]:
        for side in ("model", "rules"):
            assert set(block[side]["ci"]) == {"weighted_precision", "weighted_recall", "weighted_f1"}
        assert set(block["delta"] if "delta" in block else block["delta_model_minus_rules"]) == {
            "delta_weighted_precision", "delta_weighted_recall", "delta_weighted_f1"}
        assert 0.0 <= block["unresolved_share"]["estimate"] <= 1.0
    assert {r["region"] for r in a["per_region"]} == {"CO", "CT"}
    assert [r["stratum"] for r in a["per_stratum"]] == sorted(N_SMALL)
    assert a["review_queue"]["n"] == int((res & df.stratum.str.endswith(":rejected")).sum())
    assert a["bootstrap"]["method"] == BOOTSTRAP_METHOD and a["bootstrap"]["n_resamples"] == 400
    assert a["bootstrap"]["n_clusters"] == df[["stratum", "group_id"]].drop_duplicates().shape[0]


def test_draws_scale_every_stratum_to_its_population():
    df = _sample()
    draws = jeffreys_cluster_draws(df.y, df.pred, df.rules, df.stratum, df.group_id, N_SMALL, _w(df),
                                   n_resamples=50, seed=0)
    for h, totals in draws.items():
        assert totals.shape == (50, 9) and np.allclose(totals.sum(axis=1), N_SMALL[h])
    rejected = draws["CO:rejected"]
    assert np.allclose(rejected[:, [0, 2, 4, 6]], 0.0)          # rules decision is 0 in a rejected stratum


def test_a_heavy_stratum_with_no_sampled_false_positive_keeps_an_uncertain_precision():
    # CO:accepted N=1,000: 100 drawn, 70 RELEVANT (66 predicted), 30 NOT_RELEVANT (3 predicted)
    # CO:rejected N=100,000: 100 drawn, 4 RELEVANT (3 predicted), 96 NOT_RELEVANT (none predicted)
    y = np.array([1] * 70 + [0] * 30 + [1] * 4 + [0] * 96)
    pm = np.array([1] * 66 + [0] * 4 + [1] * 3 + [0] * 27 + [1] * 3 + [0] * 97)
    pr = np.array([1] * 100 + [0] * 100)
    strata = np.array(["CO:accepted"] * 100 + ["CO:rejected"] * 100)
    N_h = {"CO:accepted": 1000, "CO:rejected": 100000}
    w = np.where(strata == "CO:accepted", 1000 / 100, 100000 / 100).astype(float)
    report = evaluate_split(y, pm, pr, strata, np.arange(200).astype(str), np.full(200, "CO"), N_h, w,
                            n_resamples=2000, seed=0, level=0.95)
    rejected = [s for s in report["per_stratum"] if s["stratum"] == "CO:rejected"][0]["model"]
    assert rejected["weighted_precision"] == 1.0 and rejected["fp"] == 0
    assert rejected["ci"]["weighted_precision"]["lower"] < 0.9
    # a percentile bootstrap that only resamples the observed rows cannot see this: every replicate has 0 false positives
    rng = np.random.default_rng(0)
    predicted = np.flatnonzero((strata == "CO:rejected") & (pm == 1))
    replicate_fp = [(y[rng.choice(predicted, len(predicted))] == 0).sum() for _ in range(200)]
    assert max(replicate_fp) == 0
    overall = report["model"]
    assert overall["weighted_precision"] > 0.95 and overall["ci"]["weighted_precision"]["lower"] < 0.95


def test_duplicated_borrower_rows_do_not_shrink_the_interval():
    """Codex §8: repeating one borrower's rows is the same evidence, not more of it. Resampling (stratum, group)
    clusters keeps the width (measured ratio 0.98-1.01); treating each duplicate row as its own cluster collapses it
    by about 1/sqrt(5) (measured 0.37-0.50)."""
    df = _sample(seed=5, groups=400)
    base = _report(df, n_resamples=1000)
    # the same borrowers' rows five times. Rates are RECOMPUTED: five copies of one borrower are the same
    # evidence about the same population, so each copy carries a fifth of the weight and the total still
    # reconstructs N_h. Re-using the original rates would quintuple the population instead.
    tiled = _attach_rates(pd.concat([df] * 5, ignore_index=True))
    same_groups = _report(tiled, n_resamples=1000)
    one_cluster_per_row = _attach_rates(tiled.assign(group_id=[f"row{i}" for i in range(len(tiled))]))
    naive = _report(one_cluster_per_row, n_resamples=1000)
    for key in ("weighted_precision", "weighted_recall"):
        assert _width(same_groups, key) >= 0.95 * _width(base, key), key
        assert _width(naive, key) <= 0.65 * _width(base, key), key


def test_paired_delta_centres_on_zero_when_model_equals_rules_and_is_positive_when_strictly_better():
    df = _sample(seed=7, groups=200)
    same = _report(df, pred=df.rules, n_resamples=1000)
    for d in same["delta_model_minus_rules"].values():
        assert d["estimate"] == 0.0 and d["lower"] <= 0.0 <= d["upper"]
    perfect = (df.y == 1).astype(int)
    better = _report(df, pred=perfect, n_resamples=1000)
    assert better["delta_model_minus_rules"]["delta_weighted_precision"]["lower"] > 0


def test_evaluate_split_refuses_a_populated_stratum_without_rows_and_a_stratum_without_population():
    df = _sample()
    kept = _attach_rates(df[df.stratum != "CT:accepted"])
    with pytest.raises(ValueError, match="CT:accepted"):
        evaluate_split(kept.y, kept.pred, kept.rules, kept.stratum, kept.group_id, kept.region, N_SMALL, _w(kept),
                       n_resamples=10, seed=0, level=0.95)
    with pytest.raises(ValueError, match="no population count"):
        evaluate_split(df.y, df.pred, df.rules, df.stratum, df.group_id, df.region,
                       {k: v for k, v in N_SMALL.items() if k != "CO:accepted"}, _w(df),
                       n_resamples=10, seed=0, level=0.95)
    with pytest.raises(ValueError, match="does not reconstruct"):
        evaluate_split(df.y, df.pred, df.rules, df.stratum, df.group_id, df.region, N_SMALL, _w(df) * 2.0,
                       n_resamples=10, seed=0, level=0.95)


def test_point_estimates_use_the_per_row_cell_rate_not_the_stratum_average():
    """The defect this file used to pin. Both weightings reconstruct N_h identically, so a reconstruction
    guard is blind to the difference: only comparing the two answers separates them."""
    df = _sample(seed=11, groups=160)
    w_cell = _w(df)
    n_h = df.stratum.value_counts().to_dict()
    w_stratum = np.array([N_SMALL[h] / n_h[h] for h in df.stratum])
    for h in n_h:
        m = (df.stratum == h).to_numpy()
        assert w_cell[m].sum() == pytest.approx(N_SMALL[h])       # both reconstruct the population ...
        assert w_stratum[m].sum() == pytest.approx(N_SMALL[h])    # ... which is why the guard cannot catch it
    res = (df.y >= 0).to_numpy()
    by_cell = weighted_metrics(df.y[res], df.pred[res], w_cell[res])
    by_stratum = weighted_metrics(df.y[res], df.pred[res], w_stratum[res])
    assert abs(by_cell["weighted_recall"] - by_stratum["weighted_recall"]) > 0.02   # they really do differ
    report = _report(df)
    assert report["model"]["weighted_recall"] == pytest.approx(by_cell["weighted_recall"])
    assert report["model"]["weighted_recall"] != pytest.approx(by_stratum["weighted_recall"])


def test_interval_covers_the_truth_when_one_stratum_is_drawn_at_two_very_different_rates():
    """The screened design in miniature. Counting rows unweighted makes the bootstrap the retired estimator
    in disguise: its centre is the unweighted sample share scaled to N_h, so under a boundary-enriched
    sample the interval misses the population truth almost every time."""
    rng = np.random.default_rng(31)
    POP = {"CO:accepted": (6000, 0.90), "CO:rejected": (60000, 0.04)}
    BOUNDARY, DRAW = 0.1, 100
    ys, hs, cs = [], [], []
    for h, (N, prev) in POP.items():
        n_b = int(BOUNDARY * N)
        cell = np.array(["boundary"] * n_b + ["remainder"] * (N - n_b))
        # the boundary pool is enriched in RELEVANT cases, which is the entire point of screening
        p = np.where(cell == "boundary", min(prev * 4, 0.95), prev)
        ys.append((rng.random(N) < p).astype(int)); hs.append(np.full(N, h)); cs.append(cell)
    y, hs, cs = np.concatenate(ys), np.concatenate(hs), np.concatenate(cs)
    pm = np.where(rng.random(len(y)) < np.where(y == 1, 0.85, 0.05), 1, 0)
    pr = np.array([h.endswith(":accepted") for h in hs]).astype(int)
    N_h = {h: int((hs == h).sum()) for h in POP}
    truth = weighted_metrics(y, pm, np.ones(len(y)))

    covered = 0
    reps = 200
    for rep in range(reps):
        idx, w = [], []
        for h in POP:
            for cell in ("boundary", "remainder"):
                pool = np.flatnonzero((hs == h) & (cs == cell))
                take = rng.choice(pool, DRAW, replace=False)
                idx.append(take); w.append(np.full(DRAW, len(pool) / DRAW))
        idx, w = np.concatenate(idx), np.concatenate(w)
        report = evaluate_split(y[idx], pm[idx], pr[idx], hs[idx], np.arange(len(idx)).astype(str),
                                np.array([h[:2] for h in hs[idx]]), N_h, w,
                                n_resamples=600, seed=rep, level=0.95)
        ci = report["model"]["ci"]["weighted_recall"]
        covered += int(ci["lower"] <= truth["weighted_recall"] <= ci["upper"])
    assert covered / reps >= 0.85, covered / reps


def test_an_estimate_outside_its_own_interval_is_flagged_not_hidden():
    """The plug-in estimate is unsmoothed; the interval is smoothed by the Jeffreys pseudo-mass, so near
    the boundary the interval can sit below the estimate. Measured on the real validation split: precision
    0.9974 from one false positive in 207 rows, against an interval ending at 0.9937. At prior 0 the
    estimate is inside every time, so this is the prior working rather than a weighting error. Publishing
    that pair without saying so would be incoherent to a reader, so every interval carries the flag."""
    from ucc_ml.evaluation import _percentile_result

    outside = _percentile_result(0.99, np.linspace(0.10, 0.80, 400), 0.95).to_dict()
    assert outside["estimate_outside_interval"] is True
    inside = _percentile_result(0.50, np.linspace(0.10, 0.80, 400), 0.95).to_dict()
    assert inside["estimate_outside_interval"] is False
    none_est = _percentile_result(None, np.full(10, np.nan), 0.95).to_dict()
    assert none_est["estimate_outside_interval"] is False      # nothing to contradict

    df = _sample()
    report = _report(df)
    for block in [report, report["review_queue"], *report["per_region"], *report["per_stratum"]]:
        for side in ("model", "rules"):
            for ci in block[side]["ci"].values():
                assert "estimate_outside_interval" in ci
