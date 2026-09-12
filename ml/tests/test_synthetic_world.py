"""The synthetic world (contract K10) passes Plan A's readers and models, follows the K12/K13 design with the
real stratum-size ratios, uses only the K3 statuses, and writes a config Plan A's load_config accepts."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REAL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "v1.yaml"


@pytest.fixture(scope="module")
def world():
    from ucc_ml.synthetic import make_world

    return make_world()


@pytest.fixture(scope="module")
def written(world, tmp_path_factory):
    from ucc_ml.synthetic import write_world

    root = tmp_path_factory.mktemp("world")
    return root, write_world(world, root)


def test_world_files_pass_plan_a_readers_and_models(world, written):
    from ucc_ml.contracts import CASE_COLUMNS, LABEL_COLUMNS, SPLIT_COLUMNS, Label
    from ucc_ml.dataset import read_candidates, validate_cases
    from ucc_ml.labeling import read_labels
    from ucc_ml.splitting import read_splits

    _, paths = written
    candidates = read_candidates(paths.candidates_parquet)
    assert list(candidates.columns) == list(CASE_COLUMNS) and len(candidates) == len(world.candidates)
    assert validate_cases(world.candidates) == len(world.candidates)
    splits = read_splits(paths.splits_parquet)
    assert list(splits.columns) == list(SPLIT_COLUMNS) and set(splits.case_id) == set(candidates.case_id)
    labels = read_labels(paths.labels_csv)
    assert list(labels.columns) == list(LABEL_COLUMNS) and len(labels) == len(world.labels)
    assert labels.is_repeat.dtype == bool and labels.inclusion_probability.dtype == float
    for row in world.labels.to_dict("records"):
        Label.model_validate(row)


def test_baseline_columns_come_from_the_frozen_rules(world):
    from ucc_ml import legacy

    c = world.candidates
    assert c.baseline_qualifies.tolist() == [legacy.baseline_qualifies(b, l) for b, l in zip(c.borrower_name_raw, c.lender_names_raw)]
    assert c.baseline_route.tolist() == [legacy.baseline_route(b, l) for b, l in zip(c.borrower_name_raw, c.lender_names_raw)]
    assert c.groupby("group_id").region.nunique().max() == 1


def test_strata_have_the_real_size_ratios_and_every_cell_is_populated(world):
    from ucc_ml.contracts import stratum_name

    c = world.candidates
    sizes = pd.Series([stratum_name(r, q) for r, q in zip(c.region, c.baseline_qualifies)]).value_counts()
    assert 8 <= sizes["CO:rejected"] / sizes["CO:accepted"] <= 16
    assert 25 <= sizes["CT:rejected"] / sizes["CT:accepted"] <= 50
    originals = world.labels[~world.labels.is_repeat]
    cells = originals[originals.label != "INSUFFICIENT_EVIDENCE"].groupby(["sampling_stratum", "label"]).size()
    assert len(cells) == 8 and (cells >= 3).all(), cells


def test_design_follows_k12_and_k13(world):
    from ucc_ml.sampling import split_stratum_populations

    d, splits = world.design, world.splits
    pilot = d[d.labelling_round == "pilot_v1"]
    pilot_groups = set(world.candidates.loc[world.candidates.case_id.isin(pilot.case_id), "group_id"])
    assert set(splits.loc[splits.group_id.isin(pilot_groups), "split"]) == {"train"}
    everyone = world.candidates.region.groupby([world.candidates.region, world.candidates.baseline_qualifies]).size()
    assert np.allclose(pilot.inclusion_probability, pilot.n_h / pilot.N_h) and set(pilot.n_h) == {50}
    assert sorted(pilot.N_h.unique().tolist()) == sorted(everyone.tolist())
    main = d[d.labelling_round == "main_v1"]
    assert np.allclose(main.inclusion_probability, main.n_h / main.pool_h)
    populations = split_stratum_populations(world.candidates, splits)
    held_out = main[main.split.isin(["validation", "test"])]
    assert (held_out.pool_h == held_out.N_h).all()
    assert all(r.N_h == populations[r.split][r.sampling_stratum] for r in held_out.itertuples())
    labelled = world.labels[~world.labels.is_repeat].merge(d[["case_id", "split"]], on="case_id")
    for (split, stratum), group in labelled[labelled.split != "train"].groupby(["split", "sampling_stratum"]):
        assert np.allclose(populations[split][stratum] / len(group), 1.0 / group.inclusion_probability), (split, stratum)
    train = labelled[labelled.split == "train"]
    ratio = train.groupby("labelling_round").inclusion_probability.mean()
    assert ratio["pilot_v1"] < ratio["main_v1"]          # pilot rows were drawn at a lower rate than main rows


def test_labels_use_only_the_k3_statuses_with_stratum_dependent_unresolved_shares(world):
    from ucc_ml.contracts import ADJUDICATION_STATUSES, LABELLING_ROUNDS

    labels = world.labels
    assert set(labels.adjudication_status) == set(ADJUDICATION_STATUSES)
    assert ((labels.adjudication_status == "blind_repeat") == labels.is_repeat).all()
    # LABELLING_ROUNDS is the registry of rounds that MAY exist. The world labels the two that carry
    # rows in the real labels.csv (labelling.rounds_in_labels); demanding all five would require the
    # fixture to invent rounds the pipeline never produces.
    assert set(labels.labelling_round) == {"pilot_v1", "main_v1"} <= set(LABELLING_ROUNDS)
    originals = labels[~labels.is_repeat]
    assert originals.case_id.is_unique
    share = (originals.label == "INSUFFICIENT_EVIDENCE").groupby(originals.sampling_stratum.str.endswith(":rejected")).mean()
    assert share[True] > 0.3 and share[False] < 0.15
    repeats = labels[labels.is_repeat]
    assert 0.08 <= len(repeats) / len(originals) <= 0.12


def test_labels_manifest_carries_the_k3_statistics_and_the_csv_digest(world, written, tmp_path):
    from ucc_ml.labeling import write_csv
    from ucc_ml.provenance import read_json, sha256_file

    _, paths = written
    manifest = read_json(paths.labels_manifest)
    for key in ("policy_version", "disclosure", "labels_sha256", "rows", "counts_by_status", "counts_by_round",
                "pass_agreement", "founder_audit", "repeat_consistency"):
        assert key in manifest, key
    assert manifest["disclosure"] == "model-labelled, founder-adjudicated"
    assert manifest["labels_sha256"] == sha256_file(paths.labels_csv) == write_csv(world.labels, tmp_path / "labels.csv")
    assert set(manifest["counts_by_round"]) == {"pilot_v1", "main_v1"} and manifest["rows"] == len(world.labels)


def test_write_config_loads_under_plan_a_with_the_fast_profile_and_merges_overrides(tmp_path):
    from ucc_ml.config import load_config
    from ucc_ml.synthetic import write_config

    cfg = load_config(write_config(tmp_path))
    assert cfg.section("training")["cv_folds"] == 3 and cfg.section("evaluation")["bootstrap_resamples"] == 50
    cfg = load_config(write_config(tmp_path, threshold={"min_weighted_precision": 1.01}))
    assert cfg.section("threshold") == {"min_weighted_precision": 1.01, "min_predicted_positives": 30}


def test_synthetic_config_mirrors_the_real_config_outside_the_fast_profile():
    from ucc_ml.config import load_config
    from ucc_ml.synthetic import FAST_PROFILE, synthetic_config

    real = load_config(REAL_CONFIG)
    synthetic = synthetic_config()
    for section in ("mlflow", "features", "training", "calibration", "threshold", "evaluation", "release", "service",
                    "export", "budget"):
        assert section in synthetic, section
        for key, value in real.section(section).items():
            if key in synthetic[section] and key not in FAST_PROFILE.get(section, {}):
                assert synthetic[section][key] == value, f"{section}.{key}"
    for section in ("version", "paths", "eligibility", "sampling", "splits", "labelling"):
        assert synthetic[section] == real.model_dump(mode="json")[section], section


def test_make_world_is_deterministic_and_write_world_needs_no_config_first(tmp_path):
    from ucc_ml.config import load_config
    from ucc_ml.synthetic import make_world, write_config, write_world

    a, b = make_world(seed=3, groups_per_region=40), make_world(seed=3, groups_per_region=40)
    pd.testing.assert_frame_equal(a.candidates, b.candidates)
    pd.testing.assert_frame_equal(a.labels, b.labels)
    assert make_world(seed=4, groups_per_region=40).candidates.case_id.tolist() != a.candidates.case_id.tolist()
    paths = write_world(a, tmp_path)
    assert (tmp_path / "ml/configs/v1.yaml").exists() and paths.labels_csv.exists()
    assert write_config(tmp_path) == tmp_path / "ml/configs/v1.yaml"
    assert load_config(tmp_path / "ml/configs/v1.yaml").repo_root == tmp_path.resolve()
