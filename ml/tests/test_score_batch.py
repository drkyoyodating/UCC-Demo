"""score-batch: every candidate row lands in the predictions file; chunking does not change a value; oversized names go
to review with a recorded error instead of being truncated; batch rows equal predict_cases one at a time (the parity
Plan C's API test extends); run_score_batch takes the config path and an optional release directory (contract K8); and the read itself is
chunked, so the whole candidate table is never materialised."""
import json
import shutil

import pandas as pd
import pytest

from ucc_ml.inference import CaseInput, load_release_bundle, predict_cases, run_score_batch, score_batch
from ucc_ml.provenance import read_json


@pytest.fixture(scope="module")
def released(tmp_path_factory):
    from ucc_ml.config import artefact_paths, load_config
    from ucc_ml.synthetic import build_synthetic_release, make_world

    root = tmp_path_factory.mktemp("batch")
    rdir = build_synthetic_release(root)
    config_path = root / "ml" / "configs" / "v1.yaml"
    return make_world(), artefact_paths(load_config(config_path)), config_path, rdir


def _load(path) -> pd.DataFrame:
    return pd.read_parquet(path).sort_values("case_id").reset_index(drop=True)


def test_batch_covers_every_candidate_and_chunking_is_invisible(released, tmp_path):
    world, paths, config_path, rdir = released
    b = load_release_bundle(rdir, top_k=10)
    one = score_batch(b, paths.candidates_parquet, tmp_path / "one", chunk_rows=100000)
    many = score_batch(b, paths.candidates_parquet, tmp_path / "many", chunk_rows=997)
    a, m = _load(one), _load(many)
    assert len(a) == len(world.candidates) and one.name == f"{b.release_id}.parquet"
    pd.testing.assert_frame_equal(a.drop(columns=["scored_at"]), m.drop(columns=["scored_at"]))
    assert set(a.columns) == {"case_id", "region", "release_id", "score", "score_type", "decision", "threshold",
                              "baseline_qualifies", "baseline_route", "input_hash", "top_feature_contributions",
                              "validation_error", "scored_at"}
    assert a.validation_error.isna().all() and a.score.between(0, 1).all()
    assert (a.decision == "suggest_relevant").equals(a.score >= b.threshold)
    first = json.loads(a.top_feature_contributions.iloc[0])
    assert first and isinstance(first[0], list)
    summary = read_json(tmp_path / "one" / f"{b.release_id}.summary.json")
    assert summary["rows"] == len(world.candidates) and summary["invalid_rows"] == 0
    assert sum(r["n"] for r in summary["by_region_decision_baseline"]) == len(world.candidates)
    assert summary["review_queue_rules_rejected_suggested"] == int(((a.decision == "suggest_relevant") & (~a.baseline_qualifies)).sum())


def test_batch_rows_equal_predict_cases_one_at_a_time(released, tmp_path):
    world, paths, config_path, rdir = released
    b = load_release_bundle(rdir, top_k=10)
    out = _load(score_batch(b, paths.candidates_parquet, tmp_path / "p", chunk_rows=5000))
    for r in world.candidates.sample(n=8, random_state=3).itertuples():
        single = predict_cases([CaseInput(borrower_name=r.borrower_name_raw, lender_names=list(r.lender_names_raw), region=r.region,
                                          case_id=r.case_id, borrower_name_clean=r.borrower_name_clean,
                                          lender_names_clean=list(r.lender_names_clean))], b)[0]
        row = out[out.case_id == r.case_id].iloc[0]
        assert row.score == single.score and row.decision == single.decision and row.input_hash == single.input_hash
        assert json.loads(row.top_feature_contributions) == [list(t) for t in single.top_feature_contributions]


def test_oversized_names_go_to_review_with_an_error(released, tmp_path):
    from ucc_ml.dataset import write_candidates

    world, paths, config_path, rdir = released
    b = load_release_bundle(rdir)
    cand = world.candidates.copy()
    cand.at[cand.index[0], "borrower_name_raw"] = "X" * 301
    cand.at[cand.index[1], "lender_names_raw"] = [f"LENDER {i:02d}" for i in range(21)]
    write_candidates(cand, tmp_path / "cand.parquet")
    out = _load(score_batch(b, tmp_path / "cand.parquet", tmp_path / "o", chunk_rows=5000))
    bad = out[out.validation_error.notna()]
    assert len(bad) == 2 and (bad.decision == "review_needed").all() and bad.score.isna().all()
    assert set(bad.case_id) == {cand.case_id.iloc[0], cand.case_id.iloc[1]} and len(out) == len(cand)


def test_run_score_batch_resolves_paths_from_config(released, tmp_path):
    world, paths, config_path, rdir = released
    expected = paths.predictions_dir / f"{rdir.name}.parquet"
    assert run_score_batch(config_path) == expected and expected.exists()
    assert run_score_batch(config_path, rdir) == expected
    with pytest.raises(FileNotFoundError):
        run_score_batch(config_path, paths.releases_dir / "000000000000")
    second = paths.releases_dir / "ffffffffffff"
    shutil.copytree(rdir, second)
    try:
        with pytest.raises(FileNotFoundError, match="2 release"):
            run_score_batch(config_path)
    finally:
        shutil.rmtree(second)


def test_score_batch_chunks_at_the_read_and_never_materialises_the_table(released, tmp_path, monkeypatch):
    """The defect this task was written with: read_candidates built every row of all 21 columns as Python objects,
    and .iloc slicing kept that frame alive for the whole run. score_batch must use the streaming reader only."""
    from ucc_ml import dataset

    world, paths, config_path, rdir = released
    b = load_release_bundle(rdir, top_k=10)
    monkeypatch.setattr(dataset, "read_candidates",
                        lambda *a, **k: pytest.fail("score_batch read the whole candidate table"))
    real, sizes = dataset.iter_candidates, []

    def spy(path, *, chunk_rows):
        for chunk in real(path, chunk_rows=chunk_rows):
            sizes.append(len(chunk))
            yield chunk

    monkeypatch.setattr(dataset, "iter_candidates", spy)
    out = _load(score_batch(b, paths.candidates_parquet, tmp_path / "s", chunk_rows=500))
    assert len(out) == len(world.candidates) and sum(sizes) == len(world.candidates)
    assert len(sizes) >= 2 and max(sizes) <= 500


def test_iter_candidates_is_read_candidates_in_chunks(released, tmp_path):
    from ucc_ml.dataset import iter_candidates, read_candidates, write_candidates

    world, paths, config_path, rdir = released
    whole = read_candidates(paths.candidates_parquet)
    for chunk_rows in (97, 1000, 10 ** 9):
        chunks = list(iter_candidates(paths.candidates_parquet, chunk_rows=chunk_rows))
        assert max(len(c) for c in chunks) <= chunk_rows
        pd.testing.assert_frame_equal(whole, pd.concat(chunks, ignore_index=True))
    write_candidates(world.candidates.iloc[:0], tmp_path / "empty.parquet")
    assert list(iter_candidates(tmp_path / "empty.parquet")) == []
    assert len(read_candidates(tmp_path / "empty.parquet")) == 0


def test_iter_candidates_catches_a_duplicate_whose_twin_is_in_an_earlier_chunk(released, tmp_path):
    """A per-chunk uniqueness check would pass this file. K7 says the whole file is unique, so the streaming
    reader carries case_ids across chunks."""
    from ucc_ml.dataset import iter_candidates, read_candidates, write_candidates

    world, paths, config_path, rdir = released
    write_candidates(pd.concat([world.candidates, world.candidates.head(1)], ignore_index=True),
                     tmp_path / "dup.parquet")
    with pytest.raises(ValueError, match="duplicate"):
        list(iter_candidates(tmp_path / "dup.parquet", chunk_rows=50))
    with pytest.raises(ValueError, match="duplicate"):
        read_candidates(tmp_path / "dup.parquet")


def test_empty_candidate_table_still_writes_a_predictions_file(released, tmp_path):
    """Streaming an empty file yields no chunks at all; the writer is opened before the loop, so the run still
    produces a valid empty parquet and a summary rather than nothing."""
    from ucc_ml.dataset import write_candidates

    world, paths, config_path, rdir = released
    b = load_release_bundle(rdir, top_k=10)
    write_candidates(world.candidates.iloc[:0], tmp_path / "none.parquet")
    out = score_batch(b, tmp_path / "none.parquet", tmp_path / "n")
    assert out.exists() and len(pd.read_parquet(out)) == 0
    assert read_json(tmp_path / "n" / f"{b.release_id}.summary.json")["rows"] == 0
