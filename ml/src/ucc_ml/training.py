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


# ---------------------------------------------------------------------------
# Part 2 — Plan A's artefacts (Plan A's readers, K7), label resolution (K3), the model table, weights
# ---------------------------------------------------------------------------
from collections.abc import Mapping  # noqa: E402

from ucc_ml.contracts import (  # noqa: E402
    ADJUDICATION_STATUSES,
    LABEL_DISCLOSURE,
    RESOLVED_ADJUDICATION_STATUSES,
    stratum_name,
)

LABEL_TO_Y: dict[str, float] = {"RELEVANT": 1.0, "NOT_RELEVANT": 0.0, "INSUFFICIENT_EVIDENCE": np.nan}
K3_LABEL_STATISTICS: tuple[str, ...] = ("disclosure", "counts_by_status", "counts_by_round", "pass_agreement",
                                        "founder_audit", "repeat_consistency")
MODEL_TABLE_COLUMNS: tuple[str, ...] = (
    "case_id", "region", "group_id", "split", "stratum", "sampling_stratum", "labelling_round", "baseline_qualifies",
    "baseline_route", "borrower_name_raw", "borrower_name_clean", "lender_names_raw", "lender_names_clean", "label", "y",
    "inclusion_probability", "sample_weight",
)


@dataclass(frozen=True)
class PlanAInputs:
    candidates: pd.DataFrame
    splits: pd.DataFrame
    labels: pd.DataFrame
    labels_manifest: dict


def read_plan_a_inputs(paths) -> PlanAInputs:
    """Plan A's four artefacts at their ArtefactPaths locations, through Plan A's readers (contract K7).

    The readers are imported here, not at module level: ucc_ml.dataset imports duckdb, which the serving image
    (which unpickles CalibratedModel through this module) does not carry."""
    from ucc_ml.dataset import read_candidates
    from ucc_ml.labeling import read_labels
    from ucc_ml.provenance import read_json
    from ucc_ml.splitting import read_splits

    return PlanAInputs(candidates=read_candidates(paths.candidates_parquet), splits=read_splits(paths.splits_parquet),
                       labels=read_labels(paths.labels_csv), labels_manifest=read_json(paths.labels_manifest))


def resolve_binary_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Every non-repeat row, one per case_id, y = 1 / 0 / NaN (INSUFFICIENT_EVIDENCE).

    blind_repeat rows are dropped; any status outside the K3 statuses stops the run."""
    unknown = sorted(set(labels.adjudication_status) - set(ADJUDICATION_STATUSES))
    if unknown:
        raise ValueError(f"labels: unknown adjudication_status values {unknown}; expected {ADJUDICATION_STATUSES}")
    # EVERY non-repeat row, not only the resolved ones. A blind_unresolved case -- both blind passes
    # answered and disagreed, and nobody adjudicated -- is INSUFFICIENT_EVIDENCE, and protocol section 2
    # keeps such a case in n_h: it is excluded from fitting by its NaN y, never by vanishing from the
    # table. Dropping it here shrinks every denominator silently and understates the unresolved share
    # the protocol requires beside every rate (main_v1 has 83 of them). Before blind_unresolved existed
    # these two filters selected the same set, which is why the narrower one read as correct.
    keep = labels[~labels.is_repeat.astype(bool)].copy()
    n_labels = keep.groupby("case_id").label.nunique()
    conflicts = n_labels[n_labels > 1].index.tolist()
    if conflicts:
        raise ValueError(f"conflicting resolved labels for case_ids {conflicts[:10]}")
    keep = keep.drop_duplicates(subset="case_id", keep="first").reset_index(drop=True)
    keep["y"] = keep.label.map(LABEL_TO_Y).astype(float)
    return keep


def build_model_table(candidates: pd.DataFrame, splits: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """One row per non-repeat labelled case with its candidate fields, split and K2 stratum.

    sample_weight is left NaN here and set per split: training_weights (TRAIN), design_weights (validation/test)."""
    res = resolve_binary_labels(labels)
    unknown = sorted(set(res.case_id) - set(candidates.case_id))
    if unknown:
        raise ValueError(f"labelled case_ids not in candidates: {unknown[:10]}")
    unsplit = sorted(set(res.case_id) - set(splits.case_id))
    if unsplit:
        raise ValueError(f"labelled case_ids not in splits: {unsplit[:10]}")
    cand_cols = ["case_id", "region", "baseline_qualifies", "baseline_route", "borrower_name_raw",
                 "borrower_name_clean", "lender_names_raw", "lender_names_clean"]
    table = (res[["case_id", "label", "y", "labelling_round", "sampling_stratum", "inclusion_probability"]]
             .merge(candidates[cand_cols], on="case_id", how="left")
             .merge(splits[["case_id", "group_id", "split"]], on="case_id", how="left"))
    table["stratum"] = [stratum_name(r, bool(b)) for r, b in zip(table.region, table.baseline_qualifies)]
    table["sample_weight"] = np.nan   # set per split: training_weights (TRAIN), design_weights (validation/test)
    return table[list(MODEL_TABLE_COLUMNS)].reset_index(drop=True)


def resolved(table: pd.DataFrame) -> pd.DataFrame:
    return table[table.y.notna()].reset_index(drop=True)


def check_stratum_alignment(table: pd.DataFrame) -> None:
    """Plan A's sampling_stratum must be this row's region x baseline_qualifies in the K2 colon form; otherwise
    a design weight would be attached to the wrong population."""
    bad = table[table.stratum != table.sampling_stratum]
    if len(bad):
        raise ValueError("sampling_stratum differs from region x baseline_qualifies for "
                         f"{len(bad)} labelled case(s): {bad[['case_id', 'stratum', 'sampling_stratum']].head(5).to_dict('records')}")


def split_populations(candidates: pd.DataFrame, splits: pd.DataFrame) -> dict[str, dict[str, int]]:
    """{split: {stratum: N_h}} over every candidate (Plan A's sampling.split_stratum_populations)."""
    from ucc_ml.sampling import split_stratum_populations

    return split_stratum_populations(candidates, splits)


def training_weights(table_all: pd.DataFrame) -> pd.Series:
    """Every TRAIN row carries the rate of the CELL it was actually drawn from: 1 / inclusion_probability,
    normalised to mean 1 over fitted rows.

    N_train_h / n_h is wrong here and was only ever invisible. It assumes pilot_v1 and main_v1 rows are one
    sample of a stratum's TRAIN population; measured in CO:rejected the pilot drew at 5.966e-05 and a main
    TRAIN draw in B4_remainder at 3.030e-04, and since the screened round there are six rates inside one
    TRAIN stratum. The two forms agree exactly when a stratum holds one cell drawn at one rate.

    The cost is real and accepted: inverse-probability weighting gives a pilot row about 9.2x a main row's
    weight and cuts the Kish effective sample. A noisier unbiased weight beats a tighter biased one, and the
    effective size is reported next to every figure that uses it."""
    train = table_all[table_all.split == "train"]
    w = (1.0 / train.inclusion_probability.astype(float))
    return w / w[train.y.notna()].mean()


def labels_summary(table: pd.DataFrame, split: str, labels_manifest: Mapping) -> dict:
    """The metrics `labels` block: the K3 statistics from labels_manifest.json plus this split's label counts."""
    missing = [k for k in K3_LABEL_STATISTICS if k not in labels_manifest]
    if missing:
        raise ValueError(f"labels_manifest.json lacks the K3 statistics {missing}")
    # The disclosure must TRUTHFULLY describe the labels, which is stricter than requiring one literal.
    # A file may hold rounds labelled under different arrangements: pilot_v1 was founder-adjudicated,
    # main_v1 retains its disagreements as unresolved. Demanding the single literal would refuse that
    # file outright and stop every metrics document; accepting any string would let a false provenance
    # claim through. So when the manifest states a per-round map, the pooled sentence is REBUILT from
    # it and must match exactly; otherwise the single literal is required, as before.
    by_round = labels_manifest.get("disclosure_by_round")
    stated = labels_manifest["disclosure"]
    if by_round:
        from ucc_ml.labeling import pooled_disclosure

        expected = pooled_disclosure(by_round)
        if stated != expected:
            raise ValueError(f"labels_manifest.json disclosure {stated!r} does not describe its own "
                             f"disclosure_by_round {by_round!r}; expected {expected!r}")
    elif stated != LABEL_DISCLOSURE:
        raise ValueError(f"labels_manifest.json disclosure must be {LABEL_DISCLOSURE!r}, got {stated!r}")
    sub = table[table.split == split]
    block = {**{k: labels_manifest[k] for k in K3_LABEL_STATISTICS}, "policy_version": labels_manifest["policy_version"],
             "split": split, "n_labelled": int(len(sub)), "n_resolved": int(sub.y.notna().sum()),
             "n_unresolved": int(sub.y.isna().sum()), "n_positive": int((sub.y == 1).sum()),
             "n_negative": int((sub.y == 0).sum())}
    if by_round:
        # Carried into every metrics document, so a reader sees each round's arrangement rather than
        # one phrase that covers only some of the rows.
        block["disclosure_by_round"] = dict(by_round)
    return block
