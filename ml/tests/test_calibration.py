"""Explicit calibration on out-of-fold scores (CalibratedClassifierCV cannot take groups), with the
isotonic-or-sigmoid choice made on the Kish effective number of positives."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from ucc_ml.features import build_feature_frame, make_pipeline, transform_features
from ucc_ml.training import (
    CALIBRATION_METHODS,
    CalibratedModel,
    PlattCalibrator,
    choose_calibration_method,
    fit_calibrator,
    kish_effective_n,
    make_calibrator,
)


def _scores_and_labels(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    s = rng.normal(loc=np.where(y == 1, 1.5, -1.5), scale=1.0)
    w = np.where(y == 1, 1.0, 2.5)
    return s, y, w


def test_platt_is_monotone_and_bounded():
    s, y, w = _scores_and_labels()
    cal = PlattCalibrator().fit(s, y, sample_weight=w)
    grid = np.linspace(-6, 6, 25)
    p = cal.predict(grid)
    assert np.all(np.diff(p) >= 0)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert cal.a_ > 0


def test_isotonic_is_monotone_clipped_and_weighted():
    s, y, w = _scores_and_labels()
    cal = fit_calibrator("isotonic", s, y, w)
    assert isinstance(cal, IsotonicRegression)
    grid = np.linspace(-10, 10, 41)
    p = cal.predict(grid)
    assert np.all(np.diff(p) >= 0)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert not np.isnan(p).any()


def test_kish_effective_n():
    assert kish_effective_n(np.ones(10)) == pytest.approx(10.0)
    assert kish_effective_n(np.full(10, 7.5)) == pytest.approx(10.0)            # scale-free
    assert kish_effective_n([]) == 0.0
    assert kish_effective_n([100.0] + [1.0] * 9) == pytest.approx(109 ** 2 / 10009)


def test_choose_calibration_method_counts_effective_positives():
    y = np.array([1] * 300 + [0] * 100)
    flat = np.ones(400)
    assert choose_calibration_method("auto", y, flat, 200) == ("isotonic", pytest.approx(300.0))
    skewed = np.where(np.arange(400) < 30, 100.0, 1.0)     # 30 positives carry almost all of the positive weight
    method, n_eff = choose_calibration_method("auto", y, skewed, 200)
    assert method == "sigmoid" and n_eff == pytest.approx(3270 ** 2 / 300270)
    assert choose_calibration_method("sigmoid", y, flat, 200)[0] == "sigmoid"
    assert choose_calibration_method("isotonic", y[:5], flat[:5], 200)[0] == "isotonic"
    with pytest.raises(ValueError):
        choose_calibration_method("platt", y, flat, 200)
    assert CALIBRATION_METHODS == ("isotonic", "sigmoid")
    assert isinstance(make_calibrator("sigmoid"), PlattCalibrator)


def test_auto_rule_on_a_pilot_plus_main_training_sample_picks_sigmoid():
    """Real stratum sizes, 50 pilot rows (weight N_h/50) + 300 main rows per stratum: hundreds of raw positives,
    which a raw-count rule would send to isotonic, but far fewer effective ones."""
    rng = np.random.default_rng(6)
    populations = {"CO:accepted": 66976, "CO:rejected": 831328, "CT:accepted": 8141, "CT:rejected": 308834}
    ys, ws = [], []
    for stratum, N in populations.items():
        train_N = int(round(0.65 * N + 17.5))
        r = rng.normal(2.5, 1.5, 350) if stratum.endswith("accepted") else rng.normal(-4.0, 1.8, 350)
        logit = np.where(r < 0, 0.9 * r - 0.4, 1.4 * r - 0.4)
        ys.append((rng.random(350) < 1.0 / (1.0 + np.exp(-logit))).astype(int))
        ws.append(np.r_[np.full(50, N / 50), np.full(300, (train_N - 50) / 300)])
    y, w = np.concatenate(ys), np.concatenate(ws)
    assert int(y.sum()) >= 200
    method, n_eff = choose_calibration_method("auto", y, w, 200)
    assert method == "sigmoid" and n_eff < 200


def _tiny_model():
    rows = [("SMITH EXCAVATING LLC", "SMITH EXCAVATING", [], [], 1), ("JONES PAVING", "JONES PAVING", [], [], 1),
            ("GARCIA GRADING", "GARCIA GRADING", [], [], 1), ("DAVIS DEMOLITION", "DAVIS DEMOLITION", [], [], 1),
            ("ALPINE DENTAL", "ALPINE DENTAL", [], [], 0), ("SMITH LAW OFFICE", "SMITH LAW OFFICE", [], [], 0),
            ("MAIN STREET RESTAURANT", "MAIN STREET RESTAURANT", [], [], 0), ("SUNRISE BAKERY", "SUNRISE BAKERY", [], [], 0)]
    df = pd.DataFrame(rows, columns=["borrower_name_raw", "borrower_name_clean", "lender_names_raw", "lender_names_clean", "y"])
    frame = build_feature_frame(df)
    pipe = make_pipeline("borrower_only", C=1.0).fit(frame, df.y.to_numpy())
    raw = pipe.decision_function(frame)
    cal = fit_calibrator("sigmoid", raw, df.y.to_numpy(), np.ones(len(df)))
    return CalibratedModel(pipeline=pipe, calibrator=cal, variant="borrower_only", calibration_method="sigmoid"), frame


def test_calibrated_model_scores_and_matrix_path_agree():
    model, frame = _tiny_model()
    s = model.scores(frame)
    assert s.shape == (len(frame),) and s.min() >= 0 and s.max() <= 1
    raw, cal = model.scores_from_matrix(transform_features(model.pipeline, frame))
    assert np.allclose(raw, model.raw_scores(frame))
    assert np.allclose(cal, s)
    assert model.feature_policy_version == "features_v1"


def test_calibrated_model_joblib_roundtrip_is_exact(tmp_path):
    model, frame = _tiny_model()
    joblib.dump(model, tmp_path / "m.joblib")
    loaded = joblib.load(tmp_path / "m.joblib")
    assert isinstance(loaded, CalibratedModel)
    assert np.array_equal(loaded.scores(frame), model.scores(frame))
    assert loaded.variant == "borrower_only" and loaded.calibration_method == "sigmoid"
