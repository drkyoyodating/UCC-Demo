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
        # Third amendment. Provenance is PER ROUND: 2400 of the 2600 non-repeat cases are main_v1, which
        # nobody adjudicated, so the old single phrase was false of 92% of the file. The document now names
        # each round and points at the function that builds the pooled sentence from disclosure_by_round.
        "Amended three times before any candidate was frozen",
        "All three amendments were made with no model trained and no candidate frozen",
        "Label provenance is PER ROUND",
        "Every report carries `disclosure_by_round`",
        "`ucc_ml.labeling.pooled_disclosure`",
        "`CO:accepted`, `CO:rejected`, `CT:accepted`, `CT:rejected`",
        # The sample-count definition is unchanged and still load-bearing; the weighting claim that used
        # to be bundled into the same sentence was retired by the section 2 amendment, and is forbidden
        # below so it cannot return.
        "`n_h` = ALL non-repeat labelled cases drawn in S with stratum h, INSUFFICIENT_EVIDENCE included",
        "The weight applied to a row is `1 / inclusion_probability`, the rate of the CELL it was drawn from",
        "is NOT applied to any row",
        "The run stops if the design weights of a validation or test stratum do not RECONSTRUCT its population",
        "`sum(1 / inclusion_probability)` over the stratum's drawn cases must equal `N_h`",
        "RESOLVABLE population of S",
        "`1 / inclusion_probability`",
        "normalised to mean 1 over fitted rows",
        "measured on the 2026-09-12 artefacts they sum to 2.53x `N_train_h`",
        "`model_agreed`, `founder_confirmed` or `founder_adjudicated`",
        "`blind_repeat` rows",
        "Stratified cluster Bayesian bootstrap with Jeffreys pseudo-counts",
        # The prior scales with the stratum's mean weight PER CLUSTER. Weighted totals sit on the
        # population scale (N_h ~ 1e5), where a fixed 0.5 smooths nothing: measured, a stratum's precision
        # lower bound went from 0.477 to 0.999, i.e. the uncertainty the method exists for disappeared.
        "`Gamma(0.5, N_h / number of clusters)` pseudo-mass",
        # And the resampled table accumulates DESIGN WEIGHTS. Counting rows and rescaling the shares to
        # N_h is algebraically w_h = N_h / n_h -- the retired estimator, rediscovered inside the bootstrap.
        "a row contributes its design weight `1 / inclusion_probability` to its cluster's cell total, never 1.0",
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
    # A pinning test that only tracks the current text is weaker than one that also refuses the design
    # it replaced: the stratum-level weight and its false equality must not come back unnoticed. The last
    # five entries are the third amendment's: the universal adjudication claim (two fragments, because the
    # sentence and its "every report says so" clause were separately false), the sample-scale prior, the
    # row-counting bootstrap and the guessed 1.5x. Each was false of the code by the time it was read.
    for retired in ("Stratified group bootstrap", "conditional on label resolution", "1 / inclusion_probability`.",
                    "(≥ 200 OOF positives)", "CO|accepted",
                    "which equals 1/inclusion_probability because validation and test contain no pilot case",
                    "The run stops if `N_h / n_h` disagrees with 1/inclusion_probability",
                    "Labels are model-labelled, founder-adjudicated", "and every report says so",
                    "Gamma(0.5,1)", "cell shares are scaled to N_h", "sum to about 1.5x the population"):
        assert retired not in text, retired
