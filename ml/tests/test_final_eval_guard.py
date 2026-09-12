"""evaluate-final runs ONCE: a second invocation is refused and leaves the first result untouched; force_i_know keeps the
first result on disk and logs the override; the frozen bundle is checksum-verified before joblib.load; the verdict names
the pre-registered primary quantity and gives each secondary comparison a direction."""
import json

import pandas as pd
import pytest

from ucc_ml.evaluation import (
    FINAL_METRICS_NAME,
    FORCE_LOG_NAME,
    PRIMARY_QUANTITY,
    REPORT_KEYS,
    TEST_PREDICTIONS_NAME,
    FinalEvaluationRefused,
    final_verdict,
    run_evaluate_final,
)
from ucc_ml.provenance import BundleIntegrityError, read_json, sha256_file
from ucc_ml.training import GUARD_FILE_NAME, build_model_table, run_freeze_candidate, run_train


@pytest.fixture(scope="module")
def world():
    from ucc_ml.synthetic import make_world

    return make_world()


@pytest.fixture(scope="module")
def ready(world, tmp_path_factory):
    from ucc_ml.synthetic import write_config, write_world

    root = tmp_path_factory.mktemp("final")
    paths = write_world(world, root)
    config_path = write_config(root)
    run_train(config_path)
    run_freeze_candidate(config_path)
    return paths, config_path


def test_first_run_writes_metrics_predictions_and_guard(ready, world):
    paths, config_path = ready
    metrics = run_evaluate_final(config_path)
    fe = paths.final_eval_dir
    assert read_json(fe / FINAL_METRICS_NAME) == json.loads(json.dumps(metrics))
    assert metrics["split"] == "test" and metrics["forced"] is False
    assert set(metrics) == {"protocol", "split", "evaluated_at", "forced", "frozen", "evaluation", "calibration",
                            "unresolved", "labels", "verdict", "provenance"}
    assert set(metrics["evaluation"]) == set(REPORT_KEYS)
    assert metrics["evaluation"]["model"]["ci"]["weighted_recall"]["n_resamples"] == 50
    assert metrics["verdict"]["primary"] == PRIMARY_QUANTITY and "not multiplicity-adjusted" in metrics["verdict"]["text"]
    assert metrics["labels"]["disclosure"] == "model-labelled, founder-adjudicated"
    assert set(metrics["frozen"]["verified_files"]) == {"pipeline.joblib", "threshold.json", "model-manifest.json",
                                                        "validation-metrics.json"}
    guard = read_json(fe / GUARD_FILE_NAME)
    assert guard["metrics_sha256"] == sha256_file(fe / FINAL_METRICS_NAME)
    preds = pd.read_parquet(fe / TEST_PREDICTIONS_NAME)
    table = build_model_table(world.candidates, world.splits, world.labels)
    assert sorted(preds.case_id) == sorted(table.loc[table.split == "test", "case_id"])
    assert set(preds.columns) == {"case_id", "region", "stratum", "group_id", "labelling_round", "y", "sample_weight_eval",
                                  "score", "pred", "rules"}


def test_second_run_is_refused_and_changes_nothing(ready):
    paths, config_path = ready
    before = sha256_file(paths.final_eval_dir / FINAL_METRICS_NAME)
    with pytest.raises(FinalEvaluationRefused, match="REFUSED"):
        run_evaluate_final(config_path)
    assert sha256_file(paths.final_eval_dir / FINAL_METRICS_NAME) == before
    assert not (paths.final_eval_dir / FORCE_LOG_NAME).exists()


def test_forced_rerun_keeps_the_first_result_and_logs_the_override(ready):
    paths, config_path = ready
    fe = paths.final_eval_dir
    first = sha256_file(fe / FINAL_METRICS_NAME)
    assert run_evaluate_final(config_path, force_i_know=True)["forced"] is True
    backups = sorted(fe.glob("metrics.*.json"))
    assert len(backups) == 1 and sha256_file(backups[0]) == first
    log = (fe / FORCE_LOG_NAME).read_text()
    assert "force-i-know" in log and backups[0].name in log


def test_tampered_frozen_bundle_is_refused(world, tmp_path):
    from ucc_ml.synthetic import write_config, write_world

    paths = write_world(world, tmp_path)
    config_path = write_config(tmp_path)
    run_train(config_path)
    run_freeze_candidate(config_path)
    t = paths.frozen_dir / "threshold.json"
    doc = json.loads(t.read_text()); doc["threshold"] = 0.01
    t.write_text(json.dumps(doc))
    with pytest.raises(BundleIntegrityError, match="threshold.json"):
        run_evaluate_final(config_path)
    assert not (paths.final_eval_dir / GUARD_FILE_NAME).exists()


def _ci(estimate, lower, upper):
    return {"estimate": estimate, "lower": lower, "upper": upper, "n_resamples": 10, "n_failed": 0}


def test_final_verdict_gives_each_comparison_a_direction():
    report = {"delta_model_minus_rules": {"delta_weighted_recall": _ci(0.2, 0.1, 0.3),
                                          "delta_weighted_precision": _ci(-0.05, -0.08, -0.01)},
              "review_queue": {"model": {"ci": {"weighted_precision": _ci(0.9, 0.8, 0.95)}}}}
    v = final_verdict(report)
    assert (v["delta_recall"], v["delta_precision"]) == ("higher", "lower")
    assert v["review_queue_weighted_precision"]["lower"] == 0.8 and v["text"].startswith("review-queue precision 0.9 [0.8, 0.95]")
    report["delta_model_minus_rules"]["delta_weighted_recall"] = _ci(0.01, -0.02, 0.04)
    report["delta_model_minus_rules"]["delta_weighted_precision"] = _ci(None, None, None)
    v = final_verdict(report)
    assert (v["delta_recall"], v["delta_precision"]) == ("not distinguishable", "no interval")
