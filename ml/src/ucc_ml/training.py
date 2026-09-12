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


# ---------------------------------------------------------------------------
# Part 3 — grouped CV inside TRAIN, grid, OOF, refit, provenance, MLflow, run_train
# ---------------------------------------------------------------------------
import os  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import average_precision_score, log_loss  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from ucc_ml.config import artefact_paths, load_config  # noqa: E402
from ucc_ml.evaluation import design_weights, weighted_metrics  # noqa: E402
from ucc_ml.features import VARIANTS, build_feature_frame, feature_names, make_pipeline  # noqa: E402
from ucc_ml.provenance import finite_or_none, git_head, read_json, sha256_file, write_json  # noqa: E402

TRAIN_REPORT_NAME = "train-report.json"
TRAINING_WEIGHTING = "inverse_inclusion_probability_normalised"
SAMPLE_WEIGHT_NORMALISATION = "mean 1 within each fit"


@dataclass
class CVResult:
    variant: str
    C: float
    fold_weighted_ap: list[float]
    mean_weighted_ap: float
    std_weighted_ap: float
    oof_weighted_ap: float
    oof_weighted_logloss: float

    def to_dict(self) -> dict:
        return {"variant": self.variant, "C": self.C, "fold_weighted_ap": [finite_or_none(v) for v in self.fold_weighted_ap],
                "mean_weighted_ap": finite_or_none(self.mean_weighted_ap),
                "std_weighted_ap": finite_or_none(self.std_weighted_ap),
                "oof_weighted_ap": finite_or_none(self.oof_weighted_ap),
                "oof_weighted_logloss": finite_or_none(self.oof_weighted_logloss)}


def grouped_folds(y, groups, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in splitter.split(np.zeros((len(y), 1)), y, groups=np.asarray(groups))]


def cross_validate_variant(frame: pd.DataFrame, y, w, groups, variant: str, C: float, folds, *, seed: int,
                           max_iter: int) -> tuple[CVResult, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int); w = np.asarray(w, dtype=float)
    oof = np.full(len(y), np.nan); fold_ix = np.full(len(y), -1, dtype=int)
    fold_ap: list[float] = []
    for k, (tr, te) in enumerate(folds):
        pipe = make_pipeline(variant, C, max_iter=max_iter, seed=seed)
        pipe.fit(frame.iloc[tr], y[tr], clf__sample_weight=w[tr] / w[tr].mean())   # scale-free: keeps C meaningful
        oof[te] = pipe.decision_function(frame.iloc[te])
        fold_ix[te] = k
        fold_ap.append(float(average_precision_score(y[te], oof[te], sample_weight=w[te])))
    prob = 1.0 / (1.0 + np.exp(-oof))
    result = CVResult(variant=variant, C=float(C), fold_weighted_ap=fold_ap, mean_weighted_ap=float(np.mean(fold_ap)),
                      std_weighted_ap=float(np.std(fold_ap)),
                      oof_weighted_ap=float(average_precision_score(y, oof, sample_weight=w)),
                      oof_weighted_logloss=float(log_loss(y, prob, sample_weight=w, labels=[0, 1])))
    return result, oof, fold_ix


def run_grid(frame, y, w, groups, variant: str, c_grid, folds, *, seed: int, max_iter: int):
    results, oofs, fold_ixs = [], {}, {}
    for C in c_grid:
        res, oof, fx = cross_validate_variant(frame, y, w, groups, variant, float(C), folds, seed=seed, max_iter=max_iter)
        results.append(res); oofs[float(C)] = oof; fold_ixs[float(C)] = fx
    best = max(results, key=lambda r: (r.mean_weighted_ap, -r.C))     # ties → smaller C
    return results, best, oofs[best.C], fold_ixs[best.C]


def fit_full(frame, y, w, variant: str, C: float, *, seed: int, max_iter: int) -> Pipeline:
    pipe = make_pipeline(variant, C, max_iter=max_iter, seed=seed)
    w = np.asarray(w, dtype=float)
    pipe.fit(frame, np.asarray(y, dtype=int), clf__sample_weight=w / w.mean())   # scale-free: keeps C meaningful
    return pipe


def git_dirty(repo_root: Path) -> bool:
    """True when tracked files differ from HEAD (False outside a git work tree)."""
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
                             capture_output=True, text=True, check=False, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and bool(out.stdout.strip())


def provenance(cfg, paths) -> dict:
    """Digests of every input, the label policy and disclosure, the source commit and the package versions."""
    manifest = read_json(paths.labels_manifest)
    for key in ("policy_version", "disclosure"):
        if key not in manifest:
            raise ValueError(f"{paths.labels_manifest} must carry {key!r}")
    head = git_head(cfg.repo_root)
    return {
        "candidates_sha256": sha256_file(paths.candidates_parquet), "splits_sha256": sha256_file(paths.splits_parquet),
        "labels_sha256": sha256_file(paths.labels_csv), "labels_manifest_sha256": sha256_file(paths.labels_manifest),
        "policy_version": str(manifest["policy_version"]), "label_disclosure": str(manifest["disclosure"]),
        "config_sha256": cfg.config_sha256,
        "lock_sha256": sha256_file(paths.lock_file) if paths.lock_file.exists() else "missing",
        "source_commit": head or "unknown", "source_dirty": git_dirty(cfg.repo_root) if head else False,
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__, "feature_policy_version": FEATURE_POLICY_VERSION,
    }


class MlflowSession:
    """Local sqlite tracking at artefact_paths(cfg).mlflow_db (pack §6). MLflow is imported here, not at module import."""

    def __init__(self, cfg, paths) -> None:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow

        self._mlflow = mlflow
        paths.mlflow_db.parent.mkdir(parents=True, exist_ok=True)
        paths.mlflow_artifacts.mkdir(parents=True, exist_ok=True)
        self.tracking_uri = f"sqlite:///{paths.mlflow_db}"
        mlflow.set_tracking_uri(self.tracking_uri)
        name = cfg.section("mlflow")["experiment"]
        exp = mlflow.get_experiment_by_name(name)
        self.experiment_id = (exp.experiment_id if exp
                              else mlflow.create_experiment(name, artifact_location=paths.mlflow_artifacts.as_uri()))

    def log_run(self, name: str, *, params: dict, metrics: dict, tags: dict, artifacts=(), dicts: dict | None = None) -> str:
        with self._mlflow.start_run(experiment_id=self.experiment_id, run_name=name) as run:
            self._mlflow.log_params({k: str(v) for k, v in params.items()})
            self._mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
            self._mlflow.set_tags({k: str(v) for k, v in tags.items()})
            for p in artifacts:
                self._mlflow.log_artifact(str(p))
            for fname, obj in (dicts or {}).items():
                self._mlflow.log_dict(obj, fname)
            return run.info.run_id


def training_config(cfg) -> dict:
    """The `training` section, checked against the protocol (§2, §6)."""
    t = cfg.section("training")
    if t["class_weight"] not in (None, "none"):
        raise ValueError("training.class_weight must be null (context pack §6)")
    if t["selection_metric"] != "weighted_average_precision":
        raise ValueError("training.selection_metric must be weighted_average_precision (protocol §6)")
    if t["weighting"] != TRAINING_WEIGHTING:
        raise ValueError(f"training.weighting must be {TRAINING_WEIGHTING!r} (protocol §2: 1/inclusion_probability normalised, "
                         f"mean 1 within each fit), got {t['weighting']!r}")
    if t["candidate_variant"] not in VARIANTS:
        raise ValueError(f"training.candidate_variant must be one of {VARIANTS}, got {t['candidate_variant']!r}")
    return t


def run_train(config_path: Path) -> dict:
    """CLI `train`. Fits on TRAIN only and scores VALIDATION for the comparison rows; TEST is never scored."""
    started = time.time()
    cfg = load_config(config_path)
    paths = artefact_paths(cfg)
    tcfg = training_config(cfg)
    prov = provenance(cfg, paths)
    art_dir = cfg.repo_root / tcfg["artifacts_dir"]
    art_dir.mkdir(parents=True, exist_ok=True)

    inputs = read_plan_a_inputs(paths)
    table = build_model_table(inputs.candidates, inputs.splits, inputs.labels)
    check_stratum_alignment(table)
    populations = split_populations(inputs.candidates, inputs.splits)
    train_all = table[table.split == "train"].reset_index(drop=True)
    # training_weights takes only the table: the weight is 1/inclusion_probability per row, the rate of the
    # CELL it was drawn from. The populations argument belonged to the retired N_train_h/n_h form, which
    # protocol section 2 records as wrong for TRAIN because the pilot and the screened main round draw one
    # stratum at different rates. `populations` is still needed below for design_weights on VALIDATION.
    train_all["sample_weight"] = training_weights(train_all).to_numpy()
    train = resolved(train_all)
    if train.y.nunique() < 2:
        raise ValueError("TRAIN needs both classes after binary resolution")
    val_all = table[table.split == "validation"].reset_index(drop=True)
    w_val_all = design_weights(val_all, populations["validation"])
    keep = val_all.y.notna().to_numpy()
    val, w_val = val_all[keep].reset_index(drop=True), w_val_all[keep]

    y_tr = train.y.to_numpy().astype(int); w_tr = train.sample_weight.to_numpy(); g_tr = train.group_id.to_numpy()
    y_val = val.y.to_numpy().astype(int)
    frame_tr, frame_val = build_feature_frame(train), build_feature_frame(val)
    rules = {"train": weighted_metrics(y_tr, train.baseline_qualifies.to_numpy().astype(int), w_tr),
             "validation": weighted_metrics(y_val, val.baseline_qualifies.to_numpy().astype(int), w_val)}

    session = MlflowSession(cfg, paths)
    base_params = {**prov, "seed": tcfg["seed"], "cv_folds": tcfg["cv_folds"], "weighting": tcfg["weighting"],
                   "sample_weight_normalisation": SAMPLE_WEIGHT_NORMALISATION, "class_weight": "none",
                   "selection_metric": tcfg["selection_metric"]}
    folds = grouped_folds(y_tr, g_tr, int(tcfg["cv_folds"]), int(tcfg["seed"]))
    grid = [float(c) for c in tcfg["c_grid"]]
    variants_report: dict[str, dict] = {}
    for variant in VARIANTS:
        v0 = time.time()
        results, best, oof, fold_ix = run_grid(frame_tr, y_tr, w_tr, g_tr, variant, grid, folds,
                                               seed=int(tcfg["seed"]), max_iter=int(tcfg["max_iter"]))
        for r in results:
            session.log_run(f"cv-{variant}-C{r.C:g}", params={**base_params, "variant": variant, "C": r.C, "stage": "cv"},
                            metrics={"cv_mean_weighted_ap": r.mean_weighted_ap, "cv_std_weighted_ap": r.std_weighted_ap,
                                     "oof_weighted_ap": r.oof_weighted_ap, "oof_weighted_logloss": r.oof_weighted_logloss},
                            tags={"stage": "cv", "variant": variant})
        pipe = fit_full(frame_tr, y_tr, w_tr, variant, best.C, seed=int(tcfg["seed"]), max_iter=int(tcfg["max_iter"]))
        val_raw = pipe.decision_function(frame_val)
        val_ap = float(average_precision_score(y_val, val_raw, sample_weight=w_val)) if len(set(y_val.tolist())) == 2 else None
        vdir = art_dir / variant
        vdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, vdir / "pipeline.joblib")
        pd.DataFrame({"case_id": train.case_id.to_numpy(), "group_id": g_tr, "stratum": train.stratum.to_numpy(),
                      "region": train.region.to_numpy(), "labelling_round": train.labelling_round.to_numpy(), "y": y_tr,
                      "sample_weight": w_tr, "raw_score": oof, "fold": fold_ix}).to_parquet(vdir / "oof.parquet", index=False)
        cv_doc = {"variant": variant, "grid": [r.to_dict() for r in results], "best_C": best.C,
                  "best_C_at_grid_boundary": best.C in (min(grid), max(grid)),
                  "sample_weight_normalisation": SAMPLE_WEIGHT_NORMALISATION,
                  "selection_metric": tcfg["selection_metric"], "folds": int(tcfg["cv_folds"]), "seed": int(tcfg["seed"])}
        write_json(vdir / "cv.json", cv_doc)
        run_id = session.log_run(f"refit-{variant}", params={**base_params, "variant": variant, "C": best.C, "stage": "refit"},
                                 metrics={"cv_mean_weighted_ap": best.mean_weighted_ap, "oof_weighted_ap": best.oof_weighted_ap,
                                          "validation_weighted_ap": val_ap, "n_features": len(feature_names(pipe))},
                                 tags={"stage": "refit", "variant": variant},
                                 artifacts=[vdir / "pipeline.joblib", vdir / "cv.json"], dicts={"rules.json": rules})
        variants_report[variant] = {"best_C": best.C, "best_C_at_grid_boundary": cv_doc["best_C_at_grid_boundary"],
                                    "cv": cv_doc["grid"], "n_features": int(len(feature_names(pipe))),
                                    "intercept": float(pipe.named_steps["clf"].intercept_[0]),
                                    "validation": {"weighted_average_precision": val_ap, "n": int(len(val)),
                                                   "positives": int(y_val.sum())},
                                    "mlflow_run_id": run_id, "seconds": round(time.time() - v0, 2)}

    report = {"protocol": "ml/specs/evaluation_protocol_v1.md", "splits_used": ["train", "validation"],
              "weighting": tcfg["weighting"], "sample_weight_normalisation": SAMPLE_WEIGHT_NORMALISATION,
              "counts": {"train_labelled_rows": int(len(train_all)), "train_rows": int(len(train)),
                         "train_positives": int(y_tr.sum()), "train_groups": int(pd.Series(g_tr).nunique()),
                         "validation_labelled_rows": int(len(val_all)), "validation_rows": int(len(val)),
                         "validation_positives": int(y_val.sum()), "test_rows_touched": 0},
              "rules": rules, "variants": variants_report, "candidate_variant": tcfg["candidate_variant"],
              "provenance": prov, "mlflow": {"tracking_uri": session.tracking_uri, "experiment_id": session.experiment_id},
              "timings": {"seconds_total": round(time.time() - started, 2)}}
    write_json(art_dir / TRAIN_REPORT_NAME, report)
    return report
