"""Opening a release for publication or serving (plan C Task 1).

SHA256SUMS is the contract between build-release and everything that serves or publishes a
bundle, and `open_release` is the only door plan C uses. These tests run against a REAL bundle
built by ucc_ml.synthetic.build_synthetic_release — never a hand-written one."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ucc_ml.contracts import LABEL_DISCLOSURE
from ucc_ml.exporting import (
    DECISION_NEGATIVE,
    DECISION_POSITIVE,
    MANIFEST_SCHEMA_VERSION,
    open_release,
    release_directory,
)
from ucc_ml.inference import RELEASE_FILES, compute_release_id
from ucc_ml.provenance import BundleIntegrityError, sha256_file


def _copy(bundle: Path, target: Path) -> Path:
    shutil.copytree(bundle, target)
    return target


def test_the_release_has_every_file_and_opens(synthetic_release_dir: Path):
    for name in RELEASE_FILES + ("SHA256SUMS",):
        assert (synthetic_release_dir / name).is_file(), name
    release = open_release(synthetic_release_dir)
    assert release.release_id == synthetic_release_dir.name == compute_release_id(synthetic_release_dir)
    assert 0.0 <= release.threshold <= 1.0
    assert release.score_type in ("calibrated_probability", "raw_score")
    assert release.threshold_status in ("production", "experimental")
    assert release.metrics["release_id"] == release.release_id
    assert release.model_manifest["pipeline_sha256"] == sha256_file(synthetic_release_dir / "pipeline.joblib")
    assert release.model_card_markdown.startswith("# Model card")
    assert release.metrics["test"]["labels"]["disclosure"] == LABEL_DISCLOSURE


def test_the_release_id_and_threshold_come_from_the_bundle_not_from_a_manifest(synthetic_release_dir: Path):
    release = open_release(synthetic_release_dir)
    assert "release_id" not in release.model_manifest, "model-manifest.json cannot name the id it is hashed into"
    assert release.threshold == release.bundle.threshold
    assert release.release_id == release.bundle.release_id


def test_a_tampered_file_is_refused_before_the_model_is_loaded(synthetic_release_dir: Path, tmp_path: Path):
    bundle = _copy(synthetic_release_dir, tmp_path / synthetic_release_dir.name)
    (bundle / "model-card.md").write_text("# edited after the fact\n", encoding="utf-8")
    with pytest.raises(BundleIntegrityError, match="model-card.md: sha256"):
        open_release(bundle)


def test_a_missing_listed_file_is_refused(synthetic_release_dir: Path, tmp_path: Path):
    bundle = _copy(synthetic_release_dir, tmp_path / synthetic_release_dir.name)
    (bundle / "metrics.json").unlink()
    with pytest.raises(BundleIntegrityError, match="lists metrics.json but it is missing"):
        open_release(bundle)


def test_a_renamed_release_directory_is_refused(synthetic_release_dir: Path, tmp_path: Path):
    bundle = _copy(synthetic_release_dir, tmp_path / "bundle")
    with pytest.raises(ValueError, match="does not match its computed release_id"):
        open_release(bundle)


def test_release_directory_finds_the_only_release_and_refuses_to_guess(synthetic_release_dir: Path, tmp_path: Path):
    releases = synthetic_release_dir.parent
    assert release_directory(releases) == synthetic_release_dir
    assert release_directory(releases, synthetic_release_dir) == synthetic_release_dir
    (releases / "staging-tmp").mkdir()
    assert release_directory(releases) == synthetic_release_dir, "a staging directory is not a release"
    second = _copy(synthetic_release_dir, releases / "0123456789ab")
    try:
        with pytest.raises(FileNotFoundError, match="holds 2 release"):
            release_directory(releases)
        assert release_directory(releases, second) == second
    finally:
        shutil.rmtree(second)
        (releases / "staging-tmp").rmdir()
    with pytest.raises(FileNotFoundError, match="holds 0 release"):
        release_directory(tmp_path / "nothing-here")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        release_directory(releases, tmp_path / "missing")


def test_the_bundle_feature_names_carry_plan_b_s_four_blocks(synthetic_release_dir: Path):
    """Feature names are `<block>__<token>`, and Task 3's `contribution_rows` splits on that
    separator. The naming belongs to Plan B, so it is pinned HERE against a real bundle -- if Plan B
    ever renames a block, this fails at the door rather than silently emptying the block field of
    every published feature contribution."""
    release = open_release(synthetic_release_dir)
    names = [str(name) for name in release.bundle.feature_names.tolist()]
    assert names, "a frozen pipeline with no feature names cannot be published"
    assert {name.partition("__")[0] for name in names} <= {
        "borrower_word", "borrower_char", "lender_word", "lender_char"}
    assert any("__" in name for name in names), "the block separator Task 3 splits on is gone"


def test_the_decision_strings_are_plan_b_s(synthetic_release_dir: Path):
    from ucc_ml.inference import CaseInput, predict_cases

    release = open_release(synthetic_release_dir)
    prediction = predict_cases([CaseInput(borrower_name="ACME EXCAVATING LLC", lender_names=[], region="CO")],
                               release.bundle)[0]
    assert prediction.decision in (DECISION_POSITIVE, DECISION_NEGATIVE)
    assert MANIFEST_SCHEMA_VERSION == 1
