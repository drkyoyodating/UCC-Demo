"""`train`: grouped CV strictly inside TRAIN with scale-free design weights, a grid over C, OOF scores for every TRAIN
row, a refit, MLflow records with digests; no VALIDATION or TEST row ever enters a fit."""
import os

import numpy as np
import pandas as pd
import pytest

from ucc_ml.features import build_feature_frame
from ucc_ml.provenance import read_json, sha256_file
from ucc_ml.training import (
    TRAIN_REPORT_NAME,
    CVResult,
    build_model_table,
    cross_validate_variant,
    fit_full,
    grouped_folds,
    provenance,
    resolved,
    run_grid,
    run_train,
    split_populations,
    training_weights,
)


@pytest.fixture(scope="module")
def world():
    from ucc_ml.synthetic import make_world

    return make_world()


@pytest.fixture(scope="module")
def trained(world, tmp_path_factory):
    from ucc_ml.synthetic import write_config, write_world

    root = tmp_path_factory.mktemp("train")
    paths = write_world(world, root)
    config_path = write_config(root)
    return paths, config_path, root, run_train(config_path)


def _train_rows(world) -> pd.DataFrame:
    table = build_model_table(world.candidates, world.splits, world.labels)
    train_all = table[table.split == "train"].reset_index(drop=True)
    train_all["sample_weight"] = training_weights(table).to_numpy()
    return resolved(train_all)


def test_grouped_folds_never_share_a_group_and_are_seeded(world):
    train = _train_rows(world)
    folds = grouped_folds(train.y.to_numpy().astype(int), train.group_id.to_numpy(), 3, seed=1)
    assert len(folds) == 3
    for tr, te in folds:
        assert not set(train.group_id.iloc[tr]) & set(train.group_id.iloc[te])
        assert len(set(train.y.iloc[tr])) == 2
    again = grouped_folds(train.y.to_numpy().astype(int), train.group_id.to_numpy(), 3, seed=1)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again))


def test_cross_validate_produces_oof_for_every_row_and_run_grid_picks_by_mean_ap(world):
    train = _train_rows(world)
    frame = build_feature_frame(train)
    y = train.y.to_numpy().astype(int); w = train.sample_weight.to_numpy(); g = train.group_id.to_numpy()
    folds = grouped_folds(y, g, 3, seed=0)
    res, oof, fold_ix = cross_validate_variant(frame, y, w, g, "borrower_only", 1.0, folds, seed=0, max_iter=500)
    assert isinstance(res, CVResult) and not np.isnan(oof).any() and set(fold_ix) == {0, 1, 2}
    results, best, best_oof, _ = run_grid(frame, y, w, g, "borrower_only", [0.1, 1.0, 10.0], folds, seed=0, max_iter=500)
    assert [r.C for r in results] == [0.1, 1.0, 10.0]
    assert best.mean_weighted_ap == max(r.mean_weighted_ap for r in results)
    assert best.C == min(r.C for r in results if r.mean_weighted_ap == best.mean_weighted_ap)
    assert best_oof.shape == y.shape


def test_the_refit_is_scale_free_in_the_sample_weights(world):
    """LogisticRegression multiplies C by the weights' scale; normalising to mean 1 keeps the pre-registered grid meaningful."""
    train = _train_rows(world)
    frame = build_feature_frame(train)
    y, w = train.y.to_numpy().astype(int), train.sample_weight.to_numpy()
    a = fit_full(frame, y, w, "borrower_only", 1.0, seed=0, max_iter=2000)
    b = fit_full(frame, y, w * 1432.2, "borrower_only", 1.0, seed=0, max_iter=2000)
    assert np.allclose(a.named_steps["clf"].coef_, b.named_steps["clf"].coef_, atol=1e-5)


def test_run_train_artifacts_and_report(trained):
    paths, config_path, root, report = trained
    art = root / "ml/artifacts/train/v1"
    assert (art / TRAIN_REPORT_NAME).exists()
    for variant in ("borrower_only", "borrower_lender"):
        assert (art / variant / "pipeline.joblib").exists()
        cv = read_json(art / variant / "cv.json")
        assert [r["C"] for r in cv["grid"]] == [0.1, 1.0, 10.0] and cv["best_C"] in (0.1, 1.0, 10.0)
        assert cv["best_C_at_grid_boundary"] == (cv["best_C"] in (0.1, 10.0))
        assert cv["sample_weight_normalisation"] == "mean 1 within each fit"
        oof = pd.read_parquet(art / variant / "oof.parquet")
        assert set(oof.columns) == {"case_id", "group_id", "stratum", "region", "labelling_round", "y", "sample_weight",
                                    "raw_score", "fold"}
        assert oof.raw_score.notna().all() and oof.sample_weight.mean() == pytest.approx(1.0, rel=0.05)
    assert report["splits_used"] == ["train", "validation"] and report["weighting"] == "inverse_inclusion_probability_normalised"
    assert set(report["variants"]) == {"borrower_only", "borrower_lender"}
    assert set(report["rules"]) == {"train", "validation"}
    assert report["provenance"]["candidates_sha256"] == sha256_file(paths.candidates_parquet)
    assert report["provenance"]["lock_sha256"] == sha256_file(paths.lock_file)
    assert report["variants"]["borrower_lender"]["validation"]["weighted_average_precision"] is not None


def test_run_train_never_touches_validation_or_test_rows(trained, world):
    paths, config_path, root, report = trained
    table = build_model_table(world.candidates, world.splits, world.labels)
    train_ids = set(table.loc[table.split == "train", "case_id"])
    for variant in ("borrower_only", "borrower_lender"):
        oof = pd.read_parquet(root / "ml/artifacts/train/v1" / variant / "oof.parquet")
        assert set(oof.case_id) <= train_ids and len(oof) == len(resolved(table[table.split == "train"]))
    assert report["counts"]["test_rows_touched"] == 0


def test_run_train_logs_to_mlflow_with_digests(trained):
    paths, config_path, root, report = trained
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{paths.mlflow_db}")
    runs = mlflow.search_runs(experiment_names=["ucc-ml-v1"])
    assert len(runs) == 8                                     # 3 C x 2 variants + 2 refits
    assert (runs["params.candidates_sha256"] == sha256_file(paths.candidates_parquet)).all()
    assert set(runs["params.variant"]) == {"borrower_only", "borrower_lender"}
    assert set(runs["params.weighting"]) == {"inverse_inclusion_probability_normalised"}
    assert "metrics.cv_mean_weighted_ap" in runs.columns and paths.mlflow_artifacts.exists()


def test_run_train_is_deterministic(trained, world, tmp_path):
    from ucc_ml.synthetic import write_config, write_world

    paths, config_path, root, report = trained
    write_world(world, tmp_path)
    run_train(write_config(tmp_path))
    for variant in ("borrower_only", "borrower_lender"):
        assert sha256_file(root / "ml/artifacts/train/v1" / variant / "pipeline.joblib") == \
               sha256_file(tmp_path / "ml/artifacts/train/v1" / variant / "pipeline.joblib"), variant
    r1, r2 = (read_json(r / "ml/artifacts/train/v1" / TRAIN_REPORT_NAME) for r in (root, tmp_path))
    for r in (r1, r2):
        r.pop("timings"); r.pop("mlflow")
        for v in r["variants"].values():
            v.pop("mlflow_run_id"); v.pop("seconds")
    assert r1 == r2


def test_run_train_refuses_another_weighting_scheme(world, tmp_path):
    from ucc_ml.synthetic import write_config, write_world

    write_world(world, tmp_path)
    with pytest.raises(ValueError, match="training.weighting"):
        run_train(write_config(tmp_path, training={"weighting": "inverse_inclusion_probability"}))


def test_provenance_survives_a_non_git_root(trained):
    from ucc_ml.config import artefact_paths, load_config

    paths, config_path, root, report = trained
    cfg = load_config(config_path)
    p = provenance(cfg, artefact_paths(cfg))
    assert p["source_commit"] == "unknown" and p["source_dirty"] is False
    assert p["policy_version"] == "label_policy_v1" and p["feature_policy_version"] == "features_v1"
    assert p["label_disclosure"] == "model-labelled, founder-adjudicated"
