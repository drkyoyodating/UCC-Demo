"""Plan B reads Plan A's run config (contract K4) and pre-registers its protocol before any model exists.

Plan B never writes ml/configs/v1.yaml. These tests fail when a section or key Plan B reads is
missing from it, or carries a value other than the one the protocol pre-registers.
"""
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = ML_ROOT / "configs" / "v1.yaml"
PROTOCOL = ML_ROOT / "specs" / "evaluation_protocol_v1.md"

PLAN_B_VALUES = {
    "mlflow": {"experiment": "ucc-ml-v1"},
    "training": {"seed": 20260912, "cv_folds": 5, "c_grid": [0.1, 1.0, 10.0], "class_weight": None,
                 "weighting": "inverse_inclusion_probability_normalised", "selection_metric": "weighted_average_precision",
                 "candidate_variant": "borrower_lender", "max_iter": 2000, "artifacts_dir": "ml/artifacts/train/v1"},
    "calibration": {"method": "auto", "isotonic_min_effective_positives": 200, "max_weighted_ece_for_probability": 0.10,
                    "n_bins": 10, "decision_region_min_score": 0.3, "max_decision_region_ece": 0.15,
                    "decision_region_min_cases": 20},
    "threshold": {"min_weighted_precision": 0.95, "min_predicted_positives": 30},
    "evaluation": {"bootstrap_resamples": 2000, "bootstrap_seed": 20260912, "ci_level": 0.95},
    "release": {"top_k_contributions": 10, "batch_chunk_rows": 20000},
}
PLAN_B_FEATURE_KEYS = {"policy_version", "lender_join", "word", "char"}


def test_real_config_carries_plan_b_sections_with_plan_b_values():
    from ucc_ml.config import load_config

    cfg = load_config(REAL_CONFIG)
    problems = []
    for section, expected in PLAN_B_VALUES.items():
        actual = cfg.section(section)
        for key, value in expected.items():
            if key not in actual:
                problems.append(f"{section}.{key} is missing")
            elif actual[key] != value:
                problems.append(f"{section}.{key} = {actual[key]!r}, Plan B needs {value!r}")
    missing_feature_keys = sorted(PLAN_B_FEATURE_KEYS - set(cfg.section("features")))
    if missing_feature_keys:
        problems.append(f"features lacks {missing_feature_keys}")
    assert not problems, "ml/configs/v1.yaml (owned by Plan A, contract K4): " + "; ".join(problems)


def _words(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def test_protocol_pre_registers_the_design_the_code_implements():
    text = _words(PROTOCOL)
    for phrase in (
        "Status: FROZEN on commit",
        "model-labelled, founder-adjudicated",
        "`CO:accepted`, `CO:rejected`, `CT:accepted`, `CT:rejected`",
        "`n_h` = ALL non-repeat labelled cases drawn in S with stratum h, INSUFFICIENT_EVIDENCE included; `w_h = N_h / n_h`",
        "equals 1/inclusion_probability",
        "RESOLVABLE population of S",
        "`1 / inclusion_probability`",
        "normalised to mean 1 over fitted rows",
        "`model_agreed`, `founder_confirmed` or `founder_adjudicated`",
        "`blind_repeat` rows",
        "Stratified cluster Bayesian bootstrap with Jeffreys pseudo-counts",
        "Gamma(0.5,1) pseudo-mass",
        "Kish effective number of out-of-fold positives",
        "the weighted ECE among scores ≥ 0.3 (≥ 20 cases) ≤ 0.15",
        "counted as the Kish effective number (sum w)^2 / sum w^2 of predicted positives",
        "one-sided 95% lower bound of weighted precision (section 5 bootstrap) ≥ 0.95",
        "Primary reported quantity: design-weighted precision of the review queue",
        "`docs/data/ml/frozen_candidate_v1.sha256` BEFORE the run",
        "VALIDATION_LOOKS.json",
        "not multiplicity-adjusted",
    ):
        assert phrase in text, phrase
    for retired in ("Stratified group bootstrap", "conditional on label resolution", "1 / inclusion_probability`.",
                    "(≥ 200 OOF positives)", "CO|accepted"):
        assert retired not in text, retired
