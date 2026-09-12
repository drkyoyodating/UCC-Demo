"""freeze-candidate: calibrator fitted on OOF scores (Kish rule), threshold chosen on VALIDATION under the pre-registered
objective and production gate, the decision-region probability gate, the frozen bundle with SHA256SUMS and timestamp-free
manifests, and the validation-look counter."""
import json

import joblib
import pytest

from ucc_ml.features import build_feature_frame
from ucc_ml.provenance import read_json, sha256_file, verify_sha256sums
from ucc_ml.training import (
    FROZEN_FILES,
    GATE_FALLBACK,
    GUARD_FILE_NAME,
    VALIDATION_LOOKS_NAME,
    CalibratedModel,
    build_model_table,
    run_freeze_candidate,
    run_train,
)


@pytest.fixture(scope="module")
def world():
    from ucc_ml.synthetic import make_world

    return make_world()


@pytest.fixture(scope="module")
def frozen(world, tmp_path_factory):
    from ucc_ml.synthetic import write_config, write_world

    root = tmp_path_factory.mktemp("freeze")
    paths = write_world(world, root)
    config_path = write_config(root)
    run_train(config_path)
    return paths, config_path, root, run_freeze_candidate(config_path)


def test_frozen_bundle_files_and_checksums(frozen):
    paths, config_path, root, result = frozen
    assert sorted(p.name for p in paths.frozen_dir.iterdir()) == sorted([*FROZEN_FILES, "SHA256SUMS"])
    assert set(verify_sha256sums(paths.frozen_dir, required=FROZEN_FILES)) == set(FROZEN_FILES)
    assert result["frozen_dir"] == str(paths.frozen_dir)


def test_threshold_json_matches_the_protocol(frozen):
    paths, config_path, root, result = frozen
    t = read_json(paths.frozen_dir / "threshold.json")
    assert t["status"] in ("production", "experimental")
    assert t["objective"] == {"min_weighted_precision": 0.95, "min_predicted_positives": 30, "selected_on": "validation"}
    assert t["production_gate"] == {"precision_lower_bound_95_one_sided_at_least": 0.95,
                                    "kish_effective_predicted_positives_at_least": 30}
    assert t["score_type"] in ("calibrated_probability", "raw_score") and 0.0 <= t["threshold"] <= 1.0
    assert t["calibration"]["method"] in ("isotonic", "sigmoid") and t["protocol"] == "ml/specs/evaluation_protocol_v1.md"
    assert set(t["validation"]) == {"weighted_precision", "weighted_recall", "weighted_f1", "predicted_positives",
                                    "weighted_predicted_positives", "n", "positives", "precision_lower_bound_95_one_sided",
                                    "kish_effective_predicted_positives", "selection_bias"}
    assert t["validation"]["selection_bias"].startswith("optimistic")
    assert {"oof_effective_positives", "decision_region_weighted_ece", "isotonic_min_effective_positives"} <= set(t["calibration"])
    v = t["validation"]
    if t["status"] == "production":
        assert v["weighted_precision"] >= 0.95 and v["precision_lower_bound_95_one_sided"] >= 0.95
        assert v["kish_effective_predicted_positives"] >= 30 and t["fallback"] is None
    else:
        assert t["fallback"] in ("max_weighted_f1_with_min_predicted_positives", "insufficient_predicted_positives", GATE_FALLBACK)
    if t["score_type"] == "calibrated_probability":
        assert t["calibration"]["weighted_ece"] <= 0.10 and t["calibration"]["decision_region_weighted_ece"] <= 0.15


def test_model_manifest_has_no_timestamps_and_hashes_the_pipeline(frozen):
    paths, config_path, root, result = frozen
    m = read_json(paths.frozen_dir / "model-manifest.json")
    assert not any(k.endswith("_at") or "time" in k for k in m)
    assert m["pipeline_sha256"] == sha256_file(paths.frozen_dir / "pipeline.joblib")
    assert m["variant"] == "borrower_lender" and m["class_weight"] == "none" and m["c_grid"] == [0.1, 1.0, 10.0]
    assert m["weighting"] == "inverse_inclusion_probability_normalised" and m["sample_weight_normalisation"] == "mean 1 within each fit"
    assert m["candidates_sha256"] == sha256_file(paths.candidates_parquet)
    assert m["label_disclosure"] == "model-labelled, founder-adjudicated"
    assert m["oof_raw_score_sd"] > 0 and m["refit_train_raw_score_sd"] > 0


def test_frozen_pipeline_is_a_calibrated_model_scoring_in_unit_interval(frozen, world):
    paths, config_path, root, result = frozen
    model = joblib.load(paths.frozen_dir / "pipeline.joblib")
    assert isinstance(model, CalibratedModel) and model.variant == "borrower_lender"
    table = build_model_table(world.candidates, world.splits, world.labels)
    s = model.scores(build_feature_frame(table[table.split == "validation"]))
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_validation_metrics_report_block(frozen):
    paths, config_path, root, result = frozen
    vm = read_json(paths.frozen_dir / "validation-metrics.json")
    assert {"evaluation", "calibration", "threshold_curve", "unresolved", "labels", "threshold"} <= set(vm)
    assert {"review_queue", "unresolved_share", "unresolved_share_among_model_positive"} <= set(vm["evaluation"])
    assert vm["evaluation"]["bootstrap"]["n_resamples"] == 50
    assert vm["labels"]["disclosure"] == "model-labelled, founder-adjudicated" and "founder_audit" in vm["labels"]
    assert vm["unresolved"][-1]["weighting"] == "N_h-weighted over strata"
    assert "share_unresolved_among_predicted_positive" in vm["unresolved"][-1]
    assert vm["calibration"]["probability_gate"]["score_type"] == read_json(paths.frozen_dir / "threshold.json")["score_type"]


def test_refreeze_is_deterministic_counts_looks_and_is_refused_after_test_evaluation(frozen):
    paths, config_path, root, result = frozen
    before = (paths.frozen_dir / "SHA256SUMS").read_text()
    again = run_freeze_candidate(config_path)
    assert (paths.frozen_dir / "SHA256SUMS").read_text() == before
    assert again["validation_looks"] == result["validation_looks"] + 1
    assert read_json(paths.frozen_dir.parent / VALIDATION_LOOKS_NAME)["looks"] == again["validation_looks"]
    guard = paths.final_eval_dir / GUARD_FILE_NAME
    guard.parent.mkdir(parents=True, exist_ok=True)
    guard.write_text(json.dumps({"evaluated_at": "2026-09-12T00:00:00Z"}))
    with pytest.raises(RuntimeError, match="TEST_EVALUATED"):
        run_freeze_candidate(config_path)
    run_freeze_candidate(config_path, force=True)
    guard.unlink()


def test_an_impossible_precision_floor_is_experimental_and_a_relaxed_one_passes_the_gate(world, tmp_path):
    from ucc_ml.synthetic import write_config, write_world

    write_world(world, tmp_path)
    run_train(write_config(tmp_path))
    impossible = run_freeze_candidate(write_config(tmp_path, threshold={"min_weighted_precision": 1.01}))
    assert impossible["status"] == "experimental"
    relaxed = run_freeze_candidate(write_config(tmp_path, threshold={"min_weighted_precision": 0.5, "min_predicted_positives": 5}))
    t = read_json(tmp_path / "ml/artifacts/frozen/v1/threshold.json")
    assert relaxed["status"] == t["status"] == "production" and t["fallback"] is None
    assert t["validation"]["precision_lower_bound_95_one_sided"] >= 0.5
    assert t["validation"]["kish_effective_predicted_positives"] >= 5
