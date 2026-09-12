"""Training, calibration and candidate freezing for the UCC relevance screener.

Part 1 (Task 4): calibrators fitted on out-of-fold scores and the model container pickled into
pipeline.joblib. CalibratedClassifierCV is not used because it cannot honour borrower groups; `train`
(Task 9) collects group-separated out-of-fold scores and the calibrator is fitted on them here. The
isotonic-or-sigmoid choice counts EFFECTIVE positives, (sum w)^2 / sum w^2 (Kish): the calibrator is
fitted with design weights, so a raw count overstates the information it has.

Nothing at module level imports mlflow, duckdb or Plan A's dataset / splitting / labeling modules: the
API unpickles a CalibratedModel through this module, and its image carries only the serving dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ucc_ml.features import FEATURE_POLICY_VERSION

CALIBRATION_METHODS = ("isotonic", "sigmoid")


class PlattCalibrator:
    """Sigmoid calibration p = 1 / (1 + exp(-(a*s + b))), fitted as a weighted
    logistic regression of y on the 1-D out-of-fold score (Platt 1999)."""

    def __init__(self) -> None:
        self.a_: float | None = None
        self.b_: float | None = None

    def fit(self, scores, y, sample_weight=None) -> "PlattCalibrator":
        s = np.asarray(scores, dtype=float).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(s, np.asarray(y, dtype=int), sample_weight=None if sample_weight is None else np.asarray(sample_weight, dtype=float))
        self.a_ = float(lr.coef_[0][0])
        self.b_ = float(lr.intercept_[0])
        return self

    def predict(self, scores) -> np.ndarray:
        if self.a_ is None or self.b_ is None:
            raise ValueError("PlattCalibrator is not fitted")
        z = self.a_ * np.asarray(scores, dtype=float) + self.b_
        return 1.0 / (1.0 + np.exp(-z))


def kish_effective_n(sample_weight) -> float:
    """(sum w)^2 / sum w^2: the number of equally weighted cases carrying the same information (0.0 when empty)."""
    w = np.asarray(sample_weight, dtype=float)
    square = float((w ** 2).sum())
    return float(w.sum() ** 2 / square) if len(w) and square > 0 else 0.0


def choose_calibration_method(configured: str, y, sample_weight, isotonic_min_effective_positives: int) -> tuple[str, float]:
    """(method, Kish effective number of positives). auto -> isotonic only with enough EFFECTIVE positives."""
    y = np.asarray(y, dtype=int)
    w = np.asarray(sample_weight, dtype=float)
    n_eff = kish_effective_n(w[y == 1])
    if configured == "auto":
        return ("isotonic" if n_eff >= isotonic_min_effective_positives else "sigmoid"), n_eff
    if configured in CALIBRATION_METHODS:
        return configured, n_eff
    raise ValueError(f"calibration.method must be auto|isotonic|sigmoid, got {configured!r}")


def make_calibrator(method: str):
    if method == "isotonic":
        return IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    if method == "sigmoid":
        return PlattCalibrator()
    raise ValueError(f"unknown calibration method {method!r}")


def fit_calibrator(method: str, scores: np.ndarray, y: np.ndarray, sample_weight: np.ndarray):
    cal = make_calibrator(method)
    cal.fit(np.asarray(scores, dtype=float), np.asarray(y, dtype=int), sample_weight=np.asarray(sample_weight, dtype=float))
    return cal


@dataclass
class CalibratedModel:
    """What pipeline.joblib contains: the fitted feature+LR pipeline and its calibrator.

    ``scores`` is always in [0, 1]. Whether it may be CALLED a probability is a
    separate decision recorded in threshold.json as ``score_type`` (Task 10).
    """

    pipeline: Pipeline
    calibrator: object
    variant: str
    calibration_method: str
    feature_policy_version: str = FEATURE_POLICY_VERSION

    def raw_scores(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.decision_function(frame), dtype=float)

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(self.calibrator.predict(np.asarray(raw, dtype=float)), dtype=float), 0.0, 1.0)

    def scores(self, frame: pd.DataFrame) -> np.ndarray:
        return self.calibrate(self.raw_scores(frame))

    def scores_from_matrix(self, X: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(self.pipeline.named_steps["clf"].decision_function(X), dtype=float)
        return raw, self.calibrate(raw)
