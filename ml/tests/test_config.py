"""load_config: paths resolve against the repo root implied by ml/configs/<file>.yaml; sections pass through."""
import hashlib
from pathlib import Path

import pytest

REAL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "v1.yaml"

MINIMAL = """\
version:
  dataset_version: v1
  label_policy_version: label_policy_v1
  baseline_version: heavy_filter_v1
seed: 20260912
paths:
  source_duckdb: ucc.duckdb
  snapshot_dir: ml/data/snapshots/v1
  candidates_dir: ml/data/candidates/v1
  pilot_dir: ml/data/pilot/v1
  main_round_dir: ml/data/main/v1
  splits_dir: ml/data/splits/v1
  labels_dir: ml/data/labels/v1
  predictions_dir: ml/data/predictions
  artifacts_dir: ml/artifacts
  mlflow_dir: ml/mlflow
  reports_dir: ml/reports
  public_data_dir: docs/data/ml
  specs_dir: ml/specs
eligibility:
  year_min: "1990"
  junk_addresses: ["", "NONE", "N/A"]
sampling:
  strata: ["CO:accepted", "CO:rejected", "CT:accepted", "CT:rejected"]
  pilot_per_stratum: 50
  repeat_fraction: 0.10
splits:
  train: 0.65
  validation: 0.15
  test: 0.20
labelling:
  labels: [RELEVANT, NOT_RELEVANT, INSUFFICIENT_EVIDENCE]
  reason_max_chars: 300
  disclosure: "model-labelled, founder-adjudicated"
  chunk_size: 200
  founder_audit_per_split_stratum: 10
budget:
  main_round: {train: 1200, validation: 400, test: 800}
features:
  word_ngram_range: [1, 2]
"""


def _write(tmp_path: Path, text: str) -> Path:
    cfg = tmp_path / "ml" / "configs" / "v1.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(text)
    return cfg


def test_load_config_resolves_paths_and_hashes(tmp_path):
    from ucc_ml.config import load_config

    cfg_path = _write(tmp_path, MINIMAL)
    cfg = load_config(cfg_path)
    assert cfg.repo_root == tmp_path.resolve()
    assert cfg.path("snapshot_dir") == (tmp_path / "ml/data/snapshots/v1").resolve()
    assert cfg.path("main_round_dir") == (tmp_path / "ml/data/main/v1").resolve()
    assert cfg.path("predictions_dir") == (tmp_path / "ml/data/predictions").resolve()
    assert cfg.seed == 20260912 and cfg.eligibility.year_min == "1990" and cfg.splits.test == 0.20
    assert cfg.labelling.chunk_size == 200 and cfg.labelling.founder_audit_per_split_stratum == 10
    assert cfg.config_sha256 == hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    assert "release_id" not in type(cfg.version).model_fields          # K4: the release id is computed
    with pytest.raises(KeyError, match="config has no path 'nope'"):
        cfg.path("nope")


def test_section_returns_a_copy_and_names_a_missing_section(tmp_path):
    from ucc_ml.config import load_config

    cfg = load_config(_write(tmp_path, MINIMAL))
    features = cfg.section("features")
    assert features == {"word_ngram_range": [1, 2]}
    features["word_ngram_range"].append(3)
    assert cfg.section("features") == {"word_ngram_range": [1, 2]}
    assert cfg.section("budget") == {"main_round": {"train": 1200, "validation": 400, "test": 800}}
    with pytest.raises(KeyError, match="config has no section 'service'"):
        cfg.section("service")


def test_split_ratios_must_sum_to_one(tmp_path):
    from ucc_ml.config import load_config

    with pytest.raises(ValueError, match="sum to 1"):
        load_config(_write(tmp_path, MINIMAL.replace("test: 0.20", "test: 0.30")))


def test_year_min_must_be_a_four_digit_string(tmp_path):
    from ucc_ml.config import load_config

    with pytest.raises(ValueError):
        load_config(_write(tmp_path, MINIMAL.replace('year_min: "1990"', "year_min: 1990")))


def test_release_id_and_unknown_path_keys_are_refused(tmp_path):
    from ucc_ml.config import load_config

    with pytest.raises(ValueError, match="release_id"):
        load_config(_write(tmp_path, MINIMAL.replace("  baseline_version: heavy_filter_v1\n",
                                                     "  baseline_version: heavy_filter_v1\n  release_id: null\n")))
    with pytest.raises(ValueError, match="frozen_dir"):
        load_config(_write(tmp_path, MINIMAL.replace("  specs_dir: ml/specs\n",
                                                     "  specs_dir: ml/specs\n  frozen_dir: ml/artifacts/frozen/v1\n")))


def test_disclosure_is_the_literal(tmp_path):
    from ucc_ml.config import DISCLOSURE, load_config

    assert DISCLOSURE == "model-labelled, founder-adjudicated"
    with pytest.raises(ValueError, match="disclosure"):
        load_config(_write(tmp_path, MINIMAL.replace('disclosure: "model-labelled, founder-adjudicated"',
                                                     'disclosure: "human-labelled"')))


@pytest.mark.parametrize("extra", [
    "features:\n  x: 1\n",                                   # a top-level section twice
    "seed: 1\n",                                              # a scalar twice
])
def test_duplicate_yaml_keys_are_refused(tmp_path, extra):
    from ucc_ml.config import load_config

    with pytest.raises(ValueError, match="duplicate key"):
        load_config(_write(tmp_path, MINIMAL + extra))


def test_nested_duplicate_yaml_key_is_refused(tmp_path):
    from ucc_ml.config import load_config

    text = MINIMAL.replace("  pilot_per_stratum: 50\n", "  pilot_per_stratum: 50\n  pilot_per_stratum: 60\n")
    with pytest.raises(ValueError, match="duplicate key 'pilot_per_stratum'"):
        load_config(_write(tmp_path, text))


def test_config_must_live_under_ml_configs(tmp_path):
    from ucc_ml.config import load_config

    loose = tmp_path / "v1.yaml"
    loose.write_text(MINIMAL)
    with pytest.raises(ValueError, match="ml/configs"):
        load_config(loose)


def test_artefact_paths_are_the_k5_locations(tmp_path):
    from ucc_ml.config import ArtefactPaths, artefact_paths, load_config

    cfg = load_config(_write(tmp_path, MINIMAL))
    p = artefact_paths(cfg)
    root = tmp_path.resolve()
    assert isinstance(p, ArtefactPaths)
    expected = {
        "repo_root": root,
        "snapshot_manifest": root / "ml/data/snapshots/v1/manifest.json",
        "candidates_parquet": root / "ml/data/candidates/v1/candidates.parquet",
        "candidates_manifest": root / "ml/data/candidates/v1/candidates_manifest.json",
        "reconciliation": root / "ml/data/candidates/v1/reconciliation.json",
        "pilot_cases": root / "ml/data/pilot/v1/pilot_cases.parquet",
        "pilot_manifest": root / "ml/data/pilot/v1/pilot_manifest.json",
        "main_cases": root / "ml/data/main/v1/main_cases.parquet",
        "main_manifest": root / "ml/data/main/v1/main_manifest.json",
        "ablation_cases": root / "ml/data/ablation/v1/ablation_cases.parquet",
        "ablation_manifest": root / "ml/data/ablation/v1/ablation_manifest.json",
        "screen_cells": root / "ml/data/screening/v1/screen_cells.parquet",
        "screening_manifest": root / "ml/data/screening/v1/screening_manifest.json",
        "splits_parquet": root / "ml/data/splits/v1/splits.parquet",
        "split_manifest": root / "ml/data/splits/v1/split_manifest.json",
        "labels_csv": root / "ml/data/labels/v1/labels.csv",
        "labels_manifest": root / "ml/data/labels/v1/labels_manifest.json",
        "mlflow_db": root / "ml/mlflow/mlflow.db",
        "mlflow_artifacts": root / "ml/mlflow/artifacts",
        "frozen_dir": root / "ml/artifacts/frozen/v1",
        "final_eval_dir": root / "ml/artifacts/final_eval/v1",
        "releases_dir": root / "ml/artifacts/releases",
        "predictions_dir": root / "ml/data/predictions",
        "public_data_dir": root / "docs/data/ml",
        "lock_file": root / "ml/requirements-ml.lock.txt",
    }
    assert {k: getattr(p, k) for k in expected} == expected
    assert set(ArtefactPaths.__dataclass_fields__) == set(expected)
    with pytest.raises(AttributeError):
        p.labels_csv = root / "elsewhere.csv"                         # frozen


def _walk(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)
    else:
        yield value


def test_real_config_carries_every_section_plans_b_and_c_read():
    from ucc_ml.config import artefact_paths, load_config

    cfg = load_config(REAL_CONFIG)
    for name in ("budget", "mlflow", "features", "training", "calibration", "threshold", "evaluation",
                 "release", "service", "export"):
        assert isinstance(cfg.section(name), dict), name
    assert cfg.section("threshold") == {"min_weighted_precision": 0.95, "min_predicted_positives": 30}
    assert cfg.section("evaluation") == {"bootstrap_resamples": 2000, "bootstrap_seed": 20260912, "ci_level": 0.95}
    assert cfg.section("training")["c_grid"] == [0.1, 1.0, 10.0] and cfg.section("training")["cv_folds"] == 5
    assert cfg.section("calibration")["max_weighted_ece_for_probability"] == 0.10
    assert cfg.section("release") == {"top_k_contributions": 10, "batch_chunk_rows": 20000}
    assert cfg.section("service")["allowed_origins"] == [
        "https://drkyoyodating.github.io", "http://127.0.0.1:8080", "http://localhost:8080"]
    assert cfg.section("service")["max_body_bytes"] == 16384
    assert cfg.section("export")["curated_examples"] == {"per_cell_cap": 9, "error_cap_per_cell": 3,
                                                         "minimum": 50, "maximum": 100}
    assert cfg.section("budget") == {"main_round": {"train": 1200, "validation": 400, "test": 800}}
    assert cfg.labelling.disclosure == "model-labelled, founder-adjudicated"
    # K5: no section repeats a path that artefact_paths owns
    owned = {str(v.relative_to(cfg.repo_root)) for k, v in vars(artefact_paths(cfg)).items() if k != "repo_root"}
    for name in ("mlflow", "training", "calibration", "threshold", "evaluation", "release", "service", "export"):
        assert not owned & {str(v) for v in _walk(cfg.section(name))}, name
    assert all(p.is_absolute() for p in vars(artefact_paths(cfg)).values())
