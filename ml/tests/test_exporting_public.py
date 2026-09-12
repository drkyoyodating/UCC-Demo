"""The public projection (plan C Task 2), run against a REAL release's metrics.json.

The page must never receive a private field, and a key Plan B renames must fail here — naming the
path — rather than reaching docs/data/ml."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from ucc_ml.contracts import ADJUDICATION_STATUSES, LABEL_DISCLOSURE, LABEL_DISCLOSURE_BLIND, STRATA
from ucc_ml.labeling import pooled_disclosure
from ucc_ml.exporting import (
    MODEL_MANIFEST_PUBLIC_KEYS,
    PROVENANCE_PUBLIC_KEYS,
    PublicProjectionError,
    build_model_card,
    evaluation_summary,
    load_manifest,
    new_manifest,
    open_release,
    parse_model_card_markdown,
    project_public_metrics,
    upsert_manifest_release,
    validate_manifest,
)
from ucc_ml.provenance import read_json, write_json

PRIVATE = ("per_case", "predictions", "frozen_dir", "sha256sums_sha256", "verified_files", "bins",
           "threshold_curve", "validation")


def expected_disclosure(metrics: dict) -> str:
    """The provenance sentence the public documents must carry for this release.

    Pinning the literal would be wrong for a labels file whose rounds were labelled under different
    arrangements -- the real one is exactly that -- and pinning whatever the file says would assert
    nothing. So this rebuilds the sentence from the round map with the same function the writer uses,
    and falls back to the literal only when the file states no map, which is itself the claim that
    one arrangement covers every row."""
    labels = metrics["test"]["labels"]
    by_round = labels.get("disclosure_by_round")
    return pooled_disclosure(by_round) if by_round else LABEL_DISCLOSURE


@pytest.fixture(scope="module")
def release(synthetic_release_dir: Path):
    return open_release(synthetic_release_dir)


@pytest.fixture(scope="module")
def public(release):
    return project_public_metrics(release.metrics, release.release_id, "2026-09-12T00:00:00Z")


def _walk(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            yield path, value
            yield from _walk(value, path)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from _walk(value, f"{prefix}[{index}]")


def test_the_projection_publishes_the_test_block_and_nothing_private(public, release):
    assert public["schema_version"] == 1
    assert public["release_id"] == release.release_id
    assert public["generated_at"] == "2026-09-12T00:00:00Z"
    assert public["label_provenance"] == expected_disclosure(release.metrics)
    assert public["latency"] is None
    assert set(public["test"]) == {"split", "evaluated_at", "forced", "protocol", "frozen", "evaluation",
                                   "calibration", "unresolved", "labels", "verdict", "provenance"}
    assert public["test"]["split"] == "test"
    assert set(public["test"]["frozen"]) == {"threshold", "status", "score_type", "variant", "calibration_method"}
    assert public["test"]["frozen"]["threshold"] == release.threshold
    assert set(public["test"]["provenance"]) == set(PROVENANCE_PUBLIC_KEYS)
    names = {path for path, _ in _walk(public)}
    for private in PRIVATE:
        assert not any(path == private or path.endswith("." + private) for path in names), private
    # Markers are spelled without a trailing slash on purpose: Task 1 Step 8 sanitises this plan
    # before publishing it, and "/Users/" written as a pattern would be rewritten with everything
    # else. Without the slash the pattern survives, and it matches strictly more.
    for path, value in _walk(public):
        if isinstance(value, str):
            assert "/Users" not in value and "/private" not in value and "/home" not in value, path


def test_the_evaluation_block_carries_the_design_and_the_intervals(public):
    evaluation = public["test"]["evaluation"]
    assert {row["stratum"] for row in evaluation["weights"]} == set(STRATA)
    assert all(set(row) >= {"stratum", "N_h", "n_h", "w_h"} for row in evaluation["weights"])
    for name in ("model", "rules"):
        block = evaluation[name]
        assert {"tp", "fp", "fn", "tn", "weighted_precision", "weighted_recall"} <= set(block)
        for metric in ("weighted_precision", "weighted_recall", "weighted_f1"):
            assert set(block["ci"][metric]) >= {"estimate", "lower", "upper"}
    assert set(evaluation["delta_model_minus_rules"]) == {"delta_weighted_precision", "delta_weighted_recall",
                                                          "delta_weighted_f1"}
    assert {row["region"] for row in evaluation["per_region"]} == {"CO", "CT"}
    assert {row["stratum"] for row in evaluation["per_stratum"]} == set(STRATA)
    assert evaluation["review_queue"]["n"] >= 0
    assert evaluation["bootstrap"]["n_resamples"] >= 1 and evaluation["bootstrap"]["level"] == 0.95
    total = [row for row in public["test"]["unresolved"] if row["region"] == "total"]
    assert len(total) == 1 and 0.0 <= total[0]["share_unresolved"] <= 1.0


def test_the_label_statistics_of_k3_are_published_in_full(public, release):
    labels = public["test"]["labels"]
    assert labels["disclosure"] == expected_disclosure(release.metrics)
    # The contract tuple, not a literal set: `blind_unresolved` is a status a round under the
    # "unresolved" disagreement policy actually produces, and a four-name set silently asserts that
    # no such round exists.
    assert set(labels["counts_by_status"]) == set(ADJUDICATION_STATUSES)
    # When the rounds differ, the map travels with the pooled sentence; when they do not, neither
    # key is written and the sentence is the whole truth.
    if labels.get("disclosure_by_round"):
        assert labels["disclosure"] == pooled_disclosure(labels["disclosure_by_round"])
        assert public["label_provenance_by_round"] == labels["disclosure_by_round"]
    else:
        assert "label_provenance_by_round" not in public
    assert set(labels["counts_by_round"]) == {"pilot_v1", "main_v1"}
    assert set(labels["pass_agreement"]) == {"pilot_v1", "main_v1"}
    assert set(labels["repeat_consistency"]) == {"pass_a", "pass_b"}
    assert labels["founder_audit"]
    assert labels["n_resolved"] + labels["n_unresolved"] == labels["n_labelled"]
    assert public["test"]["verdict"]["text"]
    assert set(public["test"]["verdict"]["review_queue_weighted_precision"]) >= {"estimate", "lower", "upper"}


def test_a_renamed_key_fails_loudly_naming_the_path(release):
    metrics = copy.deepcopy(release.metrics)
    del metrics["test"]["evaluation"]["model"]["ci"]["weighted_recall"]
    with pytest.raises(PublicProjectionError, match=r"test.evaluation.model.ci.weighted_recall lacks"):
        project_public_metrics(metrics, release.release_id, "t")
    metrics = copy.deepcopy(release.metrics)
    metrics["test"]["evaluation"]["per_stratum"] = []
    with pytest.raises(PublicProjectionError, match="test.evaluation.per_stratum is empty"):
        project_public_metrics(metrics, release.release_id, "t")
    metrics = copy.deepcopy(release.metrics)
    del metrics["test"]["labels"]["repeat_consistency"]
    with pytest.raises(PublicProjectionError, match=r"test.labels lacks the K3 statistics \['repeat_consistency'\]"):
        project_public_metrics(metrics, release.release_id, "t")
    metrics = copy.deepcopy(release.metrics)
    del metrics["test"]
    with pytest.raises(PublicProjectionError, match="lacks the 'test' block"):
        project_public_metrics(metrics, release.release_id, "t")


def test_the_projection_refuses_a_wrong_release_id_or_a_changed_disclosure(release):
    with pytest.raises(ValueError, match="but the bundle's id is 'other'"):
        project_public_metrics(release.metrics, "other", "t")
    # No per-round map means the file claims one arrangement covers every row, and then the literal
    # is the only honest sentence.
    metrics = copy.deepcopy(release.metrics)
    metrics["test"]["labels"].pop("disclosure_by_round", None)
    metrics["test"]["labels"]["disclosure"] = "hand-labelled"
    with pytest.raises(ValueError, match="must be the literal 'model-labelled, founder-adjudicated'"):
        project_public_metrics(metrics, release.release_id, "t")
    # A file whose rounds were labelled under DIFFERENT arrangements -- which the real labels are.
    # The pooled sentence is rebuilt from the map and published; a stated sentence the map does not
    # reproduce is refused rather than published, even when it is the project's own literal.
    by_round = {"main_v1": LABEL_DISCLOSURE_BLIND, "pilot_v1": LABEL_DISCLOSURE}
    metrics = copy.deepcopy(release.metrics)
    metrics["test"]["labels"]["disclosure_by_round"] = by_round
    metrics["test"]["labels"]["disclosure"] = LABEL_DISCLOSURE
    with pytest.raises(ValueError, match="does not describe its own disclosure_by_round"):
        project_public_metrics(metrics, release.release_id, "t")
    metrics["test"]["labels"]["disclosure"] = pooled_disclosure(by_round)
    mixed = project_public_metrics(metrics, release.release_id, "t")
    assert mixed["label_provenance"] == pooled_disclosure(by_round)
    assert mixed["label_provenance"].startswith("mixed by round -- ")
    assert mixed["label_provenance_by_round"] == by_round
    assert mixed["test"]["labels"]["disclosure_by_round"] == by_round


def test_nan_becomes_null_so_the_document_is_valid_json(release):
    metrics = copy.deepcopy(release.metrics)
    metrics["test"]["evaluation"]["model"]["ci"]["weighted_f1"]["lower"] = math.nan
    public = project_public_metrics(metrics, release.release_id, "t")
    assert public["test"]["evaluation"]["model"]["ci"]["weighted_f1"]["lower"] is None


def test_evaluation_summary_is_compact_and_names_its_population(public):
    summary = evaluation_summary(public)
    assert set(summary) == {"split", "evaluated_at", "label_provenance", "n", "positives", "threshold",
                            "model", "rules", "review_queue", "unresolved_share", "verdict", "labels"}
    assert set(summary["model"]) == {"weighted_precision", "weighted_recall"}
    assert summary["label_provenance"] == public["label_provenance"]
    assert {"disclosure", "counts_by_status", "counts_by_round", "pass_agreement",
            "founder_audit", "repeat_consistency"} <= set(summary["labels"])
    assert set(summary["labels"]) <= {"disclosure", "counts_by_status", "counts_by_round", "pass_agreement",
                                      "founder_audit", "repeat_consistency",
                                      "disclosure_by_round", "policy_version_by_round"}


def test_parse_model_card_markdown_sections():
    text = "# Title\n\n## Intended use\nScreening.\n\n## Limitations\n- one\n- two\n"
    assert parse_model_card_markdown(text) == [
        {"heading": "Intended use", "text": "Screening."},
        {"heading": "Limitations", "text": "- one\n- two"},
    ]
    assert parse_model_card_markdown("just prose\n") == [{"heading": "Preamble", "text": "just prose"}]


def test_the_model_card_whitelists_the_manifest_and_keeps_the_real_sections(release):
    card = build_model_card(release, "2026-09-12T00:00:00Z")
    assert card["release_id"] == release.release_id
    assert card["label_provenance"] == expected_disclosure(release.metrics)
    assert card["threshold"] == release.threshold
    assert set(card["metadata"]) <= set(MODEL_MANIFEST_PUBLIC_KEYS)
    assert "intercept" not in card["metadata"], "the fitted intercept is not a public fact"
    assert card["metadata"]["feature_policy_version"] == release.model_manifest["feature_policy_version"]
    headings = [section["heading"] for section in card["sections"]]
    assert "What it is" in headings and "Limitations" in headings
    assert all(section["text"] for section in card["sections"])


def test_manifest_upsert_and_validation(tmp_path: Path):
    manifest = new_manifest("t0")
    assert manifest == {"schema_version": 1, "generated_at": "t0", "current": None, "releases": []}
    manifest = upsert_manifest_release(manifest, "r1", "t1", None)
    assert manifest["current"] == "r1"
    assert manifest["releases"] == [{
        "release_id": "r1", "published_at": "t1",
        "metrics_url": "releases/r1/metrics.json", "examples_url": "releases/r1/examples.json",
        "model_card_url": "releases/r1/model-card.json", "api_url": None,
    }]
    manifest = upsert_manifest_release(manifest, "r1", "t2", "https://api.example.run.app/")
    assert manifest["releases"][0]["api_url"] == "https://api.example.run.app"
    manifest = upsert_manifest_release(manifest, "r1", "t3", None)
    assert manifest["releases"][0]["api_url"] == "https://api.example.run.app", "a re-export keeps the live URL"
    manifest = upsert_manifest_release(manifest, "r1", "t4", None, keep_api_url=False)
    assert manifest["releases"][0]["api_url"] is None and len(manifest["releases"]) == 1
    manifest = upsert_manifest_release(manifest, "r2", "t5", None)
    assert manifest["current"] == "r2" and [r["release_id"] for r in manifest["releases"]] == ["r2", "r1"]

    assert any("releases/r2/metrics.json" in problem for problem in validate_manifest(manifest, tmp_path))
    for release_id in ("r1", "r2"):
        for name in ("metrics", "examples", "model-card"):
            write_json(tmp_path / "releases" / release_id / f"{name}.json", {"release_id": release_id})
    assert validate_manifest(manifest, tmp_path) == []
    assert validate_manifest(dict(manifest, current="nope"), tmp_path) == ["current 'nope' is not a listed release"]
    bad_api = dict(manifest, releases=[dict(manifest["releases"][0], api_url="ftp://x")])
    assert any("api_url" in problem for problem in validate_manifest(bad_api, tmp_path))
    bad_url = dict(manifest, releases=[dict(manifest["releases"][0], metrics_url="/releases/r2/metrics.json")])
    assert any("must be relative" in problem for problem in validate_manifest(bad_url, tmp_path))
    assert load_manifest(tmp_path / "missing.json")["releases"] == []
    write_json(tmp_path / "manifest.json", manifest)
    assert load_manifest(tmp_path / "manifest.json")["current"] == "r2"
    assert read_json(tmp_path / "manifest.json")["schema_version"] == 1
