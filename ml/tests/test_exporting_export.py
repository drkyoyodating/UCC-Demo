"""export-public end to end (plan C Task 3) on the REAL synthetic release.

score-batch writes the predictions, export-public reads them, and the four public documents are
checked for the things that must never be wrong: no private string, a disclosure that is exactly
what this release's labels block states, the curated flags and a manifest that validates."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ucc_ml.contracts import LABEL_DISCLOSURE
from ucc_ml.exporting import export_public, validate_manifest
from ucc_ml.labeling import pooled_disclosure
from ucc_ml.inference import run_score_batch
from ucc_ml.provenance import read_json


@pytest.fixture(scope="module")
def exported(synthetic_release_dir: Path, synthetic_world_config: Path):
    run_score_batch(synthetic_world_config, synthetic_release_dir)
    written = export_public(synthetic_world_config, synthetic_release_dir,
                            generated_at="2026-09-12T00:00:00Z", measure=True)
    return synthetic_release_dir, written


def test_export_public_writes_exactly_four_documents(exported):
    release_dir, written = exported
    assert set(written) == {"metrics", "examples", "model_card", "manifest"}
    for path in written.values():
        assert path.is_file(), path
    public_dir = written["manifest"].parent
    assert sorted(p.name for p in (public_dir / "releases" / release_dir.name).iterdir()) == [
        "examples.json", "metrics.json", "model-card.json"]
    manifest = read_json(written["manifest"])
    assert manifest["current"] == release_dir.name
    assert manifest["releases"][0]["api_url"] is None
    assert "preview" not in manifest
    assert validate_manifest(manifest, public_dir) == []


def test_the_public_documents_name_nothing_private(exported):
    _, written = exported
    for name in ("metrics", "examples", "model_card", "manifest"):
        text = written[name].read_text(encoding="utf-8")
        # No trailing slash on the three path markers: Task 1 Step 8's sanitiser rewrites "/Users/"
        # and friends wherever they appear, test code included. Without the slash they survive it.
        for private in ("/Users", "/home", "/private", "frozen_dir", "ucc.duckdb", "labels.csv",
                        "mlflow", "per_case", "sha256sums_sha256"):
            assert private not in text, (name, private)
    # The model card's reproduction line names SHA256SUMS on purpose: it tells a reader how to
    # check the bundle they were given. That is an instruction, not a private path.
    assert "shasum -a 256 -c SHA256SUMS" in written["model_card"].read_text(encoding="utf-8")


def test_the_documents_agree_on_the_release_and_the_disclosure(exported):
    release_dir, written = exported
    metrics, examples, card = (read_json(written[name]) for name in ("metrics", "examples", "model_card"))
    assert metrics["release_id"] == examples["release_id"] == card["release_id"] == release_dir.name
    # Rebuilt from the release's own round map rather than pinned to a literal: this world states no
    # map, so the literal is what it must say -- and the day a round map appears, the expectation
    # moves with it instead of failing or, worse, passing on a false sentence.
    labels = metrics["test"]["labels"]
    by_round = labels.get("disclosure_by_round")
    expected = pooled_disclosure(by_round) if by_round else LABEL_DISCLOSURE
    assert metrics["label_provenance"] == examples["label_provenance"] == card["label_provenance"] == expected
    assert labels["disclosure"] == expected
    # K15: examples.json repeats the K3 statistics of the metrics it was exported beside, and the
    # round map with them whenever the metrics carry one.
    assert examples["label_statistics"] == {
        key: metrics["test"]["labels"][key]
        for key in ("disclosure", "disclosure_by_round", "counts_by_status", "counts_by_round",
                    "pass_agreement", "founder_audit", "repeat_consistency")
        if key in metrics["test"]["labels"]}


def test_the_curated_examples_are_bounded_flagged_and_never_from_train(exported):
    _, written = exported
    examples = read_json(written["examples"])
    assert examples["curated"] is True and examples["precomputed"] is True
    assert 50 <= len(examples["examples"]) <= 100
    assert examples["selection"]["pool_size"] >= len(examples["examples"])
    assert examples["selection"]["selected"] == len(examples["examples"])
    assert all(example["curated"] and example["precomputed"] for example in examples["examples"])
    assert {example["split"] for example in examples["examples"]} <= {"validation", "test"}
    assert all(example["label_source"] == examples["label_provenance"]
               for example in examples["examples"] if example["label"])
    assert len({example["cell"] for example in examples["examples"]}) > 1


def test_the_example_scores_are_the_batch_scores(exported, synthetic_release_dir: Path):
    import pandas as pd

    _, written = exported
    examples = read_json(written["examples"])
    predictions = pd.read_parquet(
        synthetic_release_dir.parents[3] / "ml" / "data" / "predictions" / f"{synthetic_release_dir.name}.parquet")
    scored = dict(zip(predictions["case_id"], predictions["score"]))
    for example in examples["examples"]:
        assert example["score"] == round(float(scored[example["case_id"]]), 6)


def test_the_latency_block_is_local_and_says_so(exported):
    _, written = exported
    latency = read_json(written["metrics"])["latency"]
    assert latency["warm_calls"] == 200 and latency["cold_load_and_first_call_ms"] > 0
    assert latency["warm_p50_ms"] >= 0 and latency["batch_rows"] > 0
    assert "not on Cloud Run" in latency["note"]


def test_a_re_export_keeps_an_api_url_written_by_the_deploy_tool(exported, synthetic_world_config: Path):
    release_dir, written = exported
    manifest = read_json(written["manifest"])
    manifest["releases"][0]["api_url"] = "https://ucc-ml-api-abc-uc.a.run.app"
    written["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    export_public(synthetic_world_config, release_dir, generated_at="2026-09-12T00:00:01Z")
    assert read_json(written["manifest"])["releases"][0]["api_url"] == "https://ucc-ml-api-abc-uc.a.run.app"


def test_export_public_refuses_a_release_without_batch_predictions(synthetic_world_config: Path, tmp_path: Path):
    from ucc_ml.config import artefact_paths, load_config

    paths = artefact_paths(load_config(synthetic_world_config))
    moved = tmp_path / "predictions"
    paths.predictions_dir.rename(moved)
    try:
        with pytest.raises(FileNotFoundError, match="run `python -m ucc_ml.cli score-batch"):
            export_public(synthetic_world_config, out_dir=tmp_path / "public")
    finally:
        moved.rename(paths.predictions_dir)


def test_the_cli_command_exports_the_same_documents(synthetic_release_dir: Path, synthetic_world_config: Path,
                                                    tmp_path: Path):
    out = tmp_path / "cli-public"
    result = subprocess.run(
        [sys.executable, "-m", "ucc_ml.cli", "export-public", "--config", str(synthetic_world_config),
         "--release-dir", str(synthetic_release_dir), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "manifest:" in result.stdout
    manifest = read_json(out / "manifest.json")
    assert manifest["current"] == synthetic_release_dir.name
    assert validate_manifest(manifest, out) == []
    help_text = subprocess.run([sys.executable, "-m", "ucc_ml.cli", "export-public", "--help"],
                               capture_output=True, text=True)
    assert help_text.returncode == 0
    for flag in ("--config", "--release-dir", "--out", "--api-url", "--measure-latency"):
        assert flag in help_text.stdout
