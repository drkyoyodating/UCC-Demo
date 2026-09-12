"""The release bundle: nine files, release_id = first 12 hex of sha256 over the four manifests, timestamp-free
manifests (rebuild → same id), SHA256SUMS verified with every release file required before joblib.load, the K9
metrics shape, the model card's honesty statements, refusals, and build_synthetic_release (contract K10)."""
import hashlib
import json
import shutil

import pytest

from ucc_ml.evaluation import REPORT_KEYS, run_evaluate_final
from ucc_ml.inference import (
    RELEASE_FILES,
    RELEASE_ID_MANIFESTS,
    ReleaseBundle,
    compute_release_id,
    load_release_bundle,
    render_model_card,
    run_build_release,
)
from ucc_ml.provenance import BundleIntegrityError, read_json, sha256_file, verify_sha256sums, write_sha256sums
from ucc_ml.training import CalibratedModel, run_freeze_candidate, run_train


@pytest.fixture(scope="module")
def world():
    from ucc_ml.synthetic import make_world

    return make_world()


@pytest.fixture(scope="module")
def released(world, tmp_path_factory):
    from ucc_ml.synthetic import write_config, write_world

    root = tmp_path_factory.mktemp("release")
    paths = write_world(world, root)
    config_path = write_config(root)
    run_train(config_path)
    run_freeze_candidate(config_path)
    run_evaluate_final(config_path)
    return paths, config_path, run_build_release(config_path)


def test_bundle_has_exactly_the_nine_files_and_verifies(released):
    paths, config_path, rdir = released
    assert sorted(p.name for p in rdir.iterdir()) == sorted([*RELEASE_FILES, "SHA256SUMS"])
    assert set(verify_sha256sums(rdir, required=RELEASE_FILES)) == set(RELEASE_FILES)
    assert rdir.parent == paths.releases_dir


def test_release_id_is_first_12_hex_of_sha256_over_the_four_manifests(released):
    *_, rdir = released
    h = hashlib.sha256()
    for name in RELEASE_ID_MANIFESTS:
        h.update((rdir / name).read_bytes())
    assert rdir.name == h.hexdigest()[:12] == compute_release_id(rdir)
    assert RELEASE_ID_MANIFESTS == ("model-manifest.json", "threshold.json", "dataset-manifest.json", "split-manifest.json")


def test_manifests_and_the_k9_metrics_shape(released, world):
    paths, config_path, rdir = released
    ds = read_json(rdir / "dataset-manifest.json")
    assert ds["candidates_sha256"] == sha256_file(paths.candidates_parquet) and ds["candidates_rows"] == len(world.candidates)
    assert ds["policy_version"] == "label_policy_v1" and ds["label_disclosure"] == "model-labelled, founder-adjudicated"
    # The label file is per ROUND. A scalar policy_version is the claim that every round used it, so the
    # map is published beside it and the scalar is rebuilt from the map (one distinct value collapses to
    # itself). On the real labels that scalar reads label_policy_v1 while main_v1 -- 2,880 of 3,120 rows --
    # is label_policy_v2, and dataset-manifest.json is hashed into release_id, so it has to be right here.
    assert ds["policy_version_by_round"] == {r: "label_policy_v1" for r in world.labels_manifest["rounds"]}
    assert ds["founder_disagreements"] == world.labels_manifest["founder_disagreements"]
    assert ds["founder_audit"] == world.labels_manifest["founder_audit"]
    assert "disclosure_by_round" not in ds      # this world has one arrangement; it is not invented
    assert set(ds["strata_populations"]) == {"validation", "test"}
    assert set(ds["labelled_rows_by_split"]) == set(ds["resolved_rows_by_split"]) == {"train", "validation", "test"}
    sp = read_json(rdir / "split-manifest.json")
    assert sp["splits_sha256"] == sha256_file(paths.splits_parquet) and sp["group_scope"] == "region-scoped"
    assert sum(sp["rows_by_split"].values()) == len(world.splits)
    m = read_json(rdir / "metrics.json")
    assert set(m) == {"release_id", "created_at", "validation", "test"} and m["release_id"] == rdir.name
    assert set(m["test"]) == {"protocol", "split", "evaluated_at", "forced", "frozen", "evaluation", "calibration",
                              "unresolved", "labels", "verdict", "provenance"}
    assert set(m["test"]["evaluation"]) == set(REPORT_KEYS) and m["validation"]["split"] == "validation"
    assert m["test"]["frozen"]["threshold"] == read_json(rdir / "threshold.json")["threshold"]
    assert (rdir / "requirements-ml.lock.txt").read_bytes() == paths.lock_file.read_bytes()
    for name in RELEASE_ID_MANIFESTS:
        assert "created_at" not in (rdir / name).read_text()


def test_model_card_states_what_the_numbers_mean(released):
    *_, rdir = released
    card = (rdir / "model-card.md").read_text()
    for phrase in (rdir.name, "model-labelled, founder-adjudicated", "never replaces the frozen rules",
                   "Primary (pre-registered): review-queue weighted precision",
                   "Rates estimate performance on the RESOLVABLE population",
                   "selection-biased upward: the threshold was chosen on this sample",
                   "95% Jeffreys cluster-bootstrap intervals", "Validation was looked at 1 time(s)",
                   "not multiplicity-adjusted", "Cross-register (CO↔CT) entity linking", "Label statistics: by status",
                   # What the founder did is RENDERED from founder_disagreements / founder_audit, never asserted.
                   # This world has 97 pass disagreements, 30 of them undecided, and 85 audited of 85 selected.
                   "pass disagreements", "rows selected for the founder audit were audited",
                   # The applied weight, stated correctly.
                   "design-weighted by 1 / inclusion_probability"):
        assert phrase in card, phrase
    for retired in ("Butterfly", "conditional on label resolution", "group-bootstrap",
                    # The RETIRED estimator, published as the applied one. Protocol section 2 says N_h / n_h
                    # "is NOT applied to any row", and the real per-row weights span 21.25-1351.85 in
                    # validation with four distinct rates inside a single rejected stratum.
                    "w_h = N_h / n_h over all drawn cases",
                    # The hardcoded provenance claim. main_v1 is {"n": 83, "decided": 0, "undecided": 83}
                    # with 0 of 120 selected rows audited, and this sentence sat one clause after a
                    # disclosure saying the opposite.
                    "the founder decided every disagreement and audited"):
        assert retired not in card, retired


def test_the_model_card_renders_each_round_s_label_provenance_from_the_manifest(released):
    """The per-round path, against the REAL manifest's numbers.

    The synthetic world holds one arrangement and carries no disclosure_by_round (synthetic._labels_manifest
    passes only extra={"source": ...}), so the end-to-end bundle above exercises only the pooled form. The
    real labels hold two rounds under different arrangements, and every hardcoded sentence the card used to
    publish was false of the larger one. Driving render_model_card with those figures is what keeps the
    per-round branch honest until the fixture itself mirrors reality."""
    *_, rdir = released
    metrics, ds = read_json(rdir / "metrics.json"), read_json(rdir / "dataset-manifest.json")
    per_round = {
        **ds,
        "policy_version": "mixed by round -- main_v1: label_policy_v2; pilot_v1: label_policy_v1",
        "policy_version_by_round": {"main_v1": "label_policy_v2", "pilot_v1": "label_policy_v1"},
        "disclosure_by_round": {
            "main_v1": "model-labelled, two independent blind passes, disagreements retained as unresolved",
            "pilot_v1": "model-labelled, founder-adjudicated"},
        "founder_disagreements": {"main_v1": {"n": 83, "decided": 0, "undecided": 83},
                                  "pilot_v1": {"n": 5, "decided": 5, "undecided": 0}},
        "founder_audit": {"main_v1": {"n_selected": 120, "n_audited": 0},
                          "pilot_v1": {"n_selected": 40, "n_audited": 0}},
    }
    labels = {**metrics["test"]["labels"], "counts_by_round": {"main_v1": 2880, "pilot_v1": 240},
              "policy_version": per_round["policy_version"],
              "policy_version_by_round": per_round["policy_version_by_round"],
              "disclosure_by_round": per_round["disclosure_by_round"],
              "founder_audit": per_round["founder_audit"]}
    card = render_model_card(rdir.name, read_json(rdir / "model-manifest.json"), read_json(rdir / "threshold.json"),
                             per_round, read_json(rdir / "split-manifest.json"),
                             {**metrics["test"], "labels": labels}, 1)
    # Rounds are listed in registry order, each with its own policy, arrangement and adjudication numbers.
    assert card.index("`pilot_v1` (240 rows, policy `label_policy_v1`)") < card.index("`main_v1` (2,880 rows, policy `label_policy_v2`)")
    assert "the founder decided all 5 of the 5 pass disagreements" in card
    assert "all 83 pass disagreements were retained as unresolved" in card
    assert "0 of the 120 rows selected for the founder audit were audited" in card
    assert "0 of the 40 rows selected for the founder audit were audited" in card
    assert "disagreements retained as unresolved" in card and "label_policy_v2" in card
    # The claim this replaced, and the retired weight, cannot come back through this path either.
    assert "the founder decided every disagreement and audited" not in card
    assert "w_h = N_h / n_h over all drawn cases" not in card


def test_rebuild_is_idempotent(released):
    paths, config_path, rdir = released
    sums = (rdir / "SHA256SUMS").read_text()
    assert run_build_release(config_path) == rdir and (rdir / "SHA256SUMS").read_text() == sums
    assert [p.name for p in paths.releases_dir.iterdir()] == [rdir.name]


def test_load_release_bundle_verifies_and_exposes_the_model(released):
    *_, rdir = released
    b = load_release_bundle(rdir)
    assert isinstance(b, ReleaseBundle) and isinstance(b.model, CalibratedModel)
    assert b.release_id == rdir.name and 0.0 <= b.threshold <= 1.0
    assert b.score_type in ("calibrated_probability", "raw_score") and b.threshold_status in ("production", "experimental")
    assert len(b.feature_names) == len(b.coef) == b.model_manifest["n_features"]


def test_a_tampered_renamed_or_incompletely_listed_bundle_is_refused(released, tmp_path):
    *_, rdir = released
    tampered = tmp_path / "t" / rdir.name
    shutil.copytree(rdir, tampered)
    doc = json.loads((tampered / "threshold.json").read_text()); doc["threshold"] = 0.0
    (tampered / "threshold.json").write_text(json.dumps(doc))
    with pytest.raises(BundleIntegrityError, match="threshold.json"):
        load_release_bundle(tampered)
    renamed = tmp_path / "abcdef123456"
    shutil.copytree(rdir, renamed)
    with pytest.raises(ValueError, match="release_id"):
        load_release_bundle(renamed)
    unlisted = tmp_path / "u" / rdir.name
    shutil.copytree(rdir, unlisted)
    write_sha256sums(unlisted, [n for n in RELEASE_FILES if n != "model-card.md"])
    with pytest.raises(BundleIntegrityError, match="model-card.md"):
        load_release_bundle(unlisted)


def test_build_release_refuses_without_final_eval_after_refreeze_or_with_edited_metrics(world, tmp_path):
    from ucc_ml.synthetic import write_config, write_world

    paths = write_world(world, tmp_path)
    config_path = write_config(tmp_path)
    run_train(config_path)
    run_freeze_candidate(config_path)
    with pytest.raises(RuntimeError, match="evaluate-final"):
        run_build_release(config_path)
    run_evaluate_final(config_path)
    metrics = paths.final_eval_dir / "metrics.json"
    original = metrics.read_bytes()
    metrics.write_bytes(original.replace(b'"forced": false', b'"forced": true'))
    with pytest.raises(RuntimeError, match="changed after evaluate-final"):
        run_build_release(config_path)
    metrics.write_bytes(original)
    run_freeze_candidate(write_config(tmp_path, threshold={"min_weighted_precision": 1.01}), force=True)
    with pytest.raises(RuntimeError, match="changed after the test"):
        run_build_release(config_path)


def test_build_synthetic_release_runs_the_real_pipeline(tmp_path):
    from ucc_ml.synthetic import build_synthetic_release

    rdir = build_synthetic_release(tmp_path / "world")
    assert rdir.parent == tmp_path / "world" / "ml" / "artifacts" / "releases"
    assert (tmp_path / "world" / "ml" / "configs" / "v1.yaml").is_file()
    assert load_release_bundle(rdir).release_id == rdir.name
