"""Curated-example selection (plan C Task 3) is a written, deterministic rule — never random.

These tests use inline frames only: no model, no bundle, no world."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ucc_ml.contracts import LABEL_DISCLOSURE, LABEL_DISCLOSURE_BLIND
from ucc_ml.labeling import pooled_disclosure
from ucc_ml.exporting import (
    DECISION_NEGATIVE,
    DECISION_POSITIVE,
    SELECTION_RULE,
    as_list,
    build_example_pool,
    contribution_rows,
    build_examples_document,
    example_cell,
    example_priority,
    read_predictions,
    resolve_labels,
    select_curated_examples,
)


#: The rounds of the REAL labels file, which were labelled under different arrangements. The fixture
#: mirrors that rather than a single-arrangement world, because the single-arrangement case is the
#: one the synthetic release already exercises end to end.
_DISCLOSURE_BY_ROUND = {"main_v1": LABEL_DISCLOSURE_BLIND, "pilot_v1": LABEL_DISCLOSURE}


def test_contribution_rows_split_plan_b_feature_names():
    """Plan B names a feature `<block>__<token>`; the API publishes block and feature separately.
    A name WITHOUT the separator keeps an empty block rather than losing the token -- dropping it
    would silently delete a feature from a published explanation."""
    rows = contribution_rows([("borrower_word__concrete", 0.31), ("lender_char__ CAT", -0.02),
                              ("bare", 1.0)])
    assert rows == [
        {"block": "borrower_word", "feature": "concrete", "contribution": 0.31},
        {"block": "lender_char", "feature": " CAT", "contribution": -0.02},
        {"block": "", "feature": "bare", "contribution": 1.0},
    ]
    assert contribution_rows(None) == []


def _label_statistics() -> dict:
    """The K3 statistics exactly as `project_public_metrics` publishes them (contract K15)."""
    return {
        "disclosure": pooled_disclosure(_DISCLOSURE_BY_ROUND),
        "disclosure_by_round": dict(_DISCLOSURE_BY_ROUND),
        # All five statuses of contracts.ADJUDICATION_STATUSES: a round under the "unresolved"
        # disagreement policy produces blind_unresolved rows, and a four-status fixture would let
        # code that cannot handle the fifth pass here and fail on the real file.
        "counts_by_status": {"model_agreed": 3, "founder_confirmed": 1, "founder_adjudicated": 1,
                             "blind_unresolved": 1, "blind_repeat": 1},
        "counts_by_round": {"pilot_v1": 3, "main_v1": 4},
        "pass_agreement": {"pilot_v1": {"n": 2, "agreed": 2, "rate": 1.0},
                           "main_v1": {"n": 4, "agreed": 3, "rate": 0.75}},
        "founder_audit": {"pilot_v1": {"n_audited": 1, "n_confirmed": 1, "n_overturned": 0,
                                       "agreement_rate": 1.0},
                          "main_v1": {"n_audited": 2, "n_confirmed": 1, "n_overturned": 1,
                                      "agreement_rate": 0.5}},
        "repeat_consistency": {"pass_a": {"n": 1, "consistent": 1, "rate": 1.0},
                               "pass_b": {"n": 1, "consistent": 1, "rate": 1.0}},
    }


def _labels() -> pd.DataFrame:
    return pd.DataFrame([
        {"case_id": "a", "label": "RELEVANT", "adjudication_status": "model_agreed", "is_repeat": False},
        {"case_id": "b", "label": "NOT_RELEVANT", "adjudication_status": "founder_confirmed", "is_repeat": False},
        {"case_id": "c", "label": "INSUFFICIENT_EVIDENCE", "adjudication_status": "founder_adjudicated", "is_repeat": False},
        {"case_id": "a", "label": "NOT_RELEVANT", "adjudication_status": "blind_repeat", "is_repeat": True},
    ])


def test_resolve_labels_drops_blind_repeats_and_keeps_the_resolved_statuses():
    assert resolve_labels(_labels()).to_dict("records") == [
        {"case_id": "a", "label": "RELEVANT"},
        {"case_id": "b", "label": "NOT_RELEVANT"},
        {"case_id": "c", "label": "INSUFFICIENT_EVIDENCE"},
    ]


def test_resolve_labels_refuses_unknown_values_missing_columns_and_double_labels():
    with pytest.raises(KeyError, match=r"lacks columns \['adjudication_status', 'is_repeat'\]"):
        resolve_labels(pd.DataFrame([{"case_id": "a", "label": "RELEVANT"}]))
    bad = _labels()
    bad.loc[0, "label"] = "MAYBE"
    with pytest.raises(ValueError, match=r"unknown label values: \['MAYBE'\]"):
        resolve_labels(bad)
    twice = pd.concat([_labels(), _labels().head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="resolves 1 case"):
        resolve_labels(twice)


def test_as_list_handles_none_string_and_arrays():
    assert as_list(None) == []
    assert as_list(float("nan")) == []
    assert as_list("X") == ["X"]
    assert as_list(np.array(["B", "A"])) == ["B", "A"]
    assert as_list(["A"]) == ["A"]


def test_example_cell_and_priority():
    assert example_cell("CO", True, DECISION_POSITIVE) == "CO/accepted/suggest"
    assert example_cell("CT", False, DECISION_NEGATIVE) == "CT/rejected/review"
    assert example_priority("RELEVANT", DECISION_NEGATIVE) == 0
    assert example_priority("NOT_RELEVANT", DECISION_POSITIVE) == 0
    assert example_priority("RELEVANT", DECISION_POSITIVE) == 1
    assert example_priority("NOT_RELEVANT", DECISION_NEGATIVE) == 1
    assert example_priority("INSUFFICIENT_EVIDENCE", DECISION_POSITIVE) == 2
    assert example_priority(None, DECISION_POSITIVE) == 3
    assert example_priority(float("nan"), DECISION_NEGATIVE) == 3


def _world(n: int = 400) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates, splits, labels, predictions = [], [], [], []
    for i in range(n):
        case_id = f"{i:04d}"
        region = "CO" if i % 2 == 0 else "CT"
        accepted = (i % 4) < 2
        score = ((i * 37) % 100) / 100
        decision = DECISION_POSITIVE if score >= 0.5 else DECISION_NEGATIVE
        split = "train" if i % 5 == 0 else ("validation" if i % 5 == 1 else "test")
        candidates.append({"case_id": case_id, "region": region, "borrower_name_raw": f"BORROWER {i}",
                           "lender_names_raw": ["A BANK"], "borrower_city": "DENVER", "borrower_state": region,
                           "baseline_qualifies": accepted, "baseline_route": "borrower" if accepted else "neither"})
        splits.append({"case_id": case_id, "group_id": f"{region}:g{i}", "split": split})
        if i % 4 != 3:
            labels.append({"case_id": case_id, "label": ["RELEVANT", "NOT_RELEVANT", "INSUFFICIENT_EVIDENCE"][i % 3]})
        predictions.append({"case_id": case_id, "score": score, "score_type": "calibrated_probability",
                            "decision": decision, "threshold": 0.5, "baseline_qualifies": accepted,
                            "baseline_route": "borrower" if accepted else "neither", "release_id": "r1"})
    return (pd.DataFrame(candidates), pd.DataFrame(splits), pd.DataFrame(labels), pd.DataFrame(predictions))


def test_the_pool_excludes_train_caps_unlabelled_test_and_carries_the_batch_score():
    candidates, splits, labels, predictions = _world()
    pool = build_example_pool(candidates, splits, labels, predictions, unlabelled_cap=5)
    assert "train" not in set(pool["split"])
    unlabelled = pool[pool["label"].isna()]
    assert len(unlabelled) == 5 and set(unlabelled["split"]) == {"test"}
    assert pool["case_id"].is_unique and list(pool["case_id"]) == sorted(pool["case_id"])
    assert set(pool["score_type"]) == {"calibrated_probability"}
    held_out = labels.merge(splits, on="case_id")
    held_out = held_out[held_out["split"] != "train"]
    assert set(pool[pool["label"].notna()]["case_id"]) == set(held_out["case_id"])


def test_the_pool_aborts_on_a_baseline_disagreement_or_a_missing_score():
    candidates, splits, labels, predictions = _world(40)
    pooled = build_example_pool(candidates, splits, labels, predictions)["case_id"].iloc[0]
    flipped = predictions.copy()
    row = flipped.index[flipped["case_id"] == pooled][0]
    flipped.loc[row, "baseline_qualifies"] = not bool(flipped.loc[row, "baseline_qualifies"])
    with pytest.raises(ValueError, match="baseline_qualifies differs"):
        build_example_pool(candidates, splits, labels, flipped)
    with pytest.raises(ValueError, match="not in the batch predictions"):
        build_example_pool(candidates, splits, labels, predictions.head(3))


def test_read_predictions_checks_the_release_and_drops_unscored_rows(tmp_path):
    _, _, _, predictions = _world(20)
    predictions.loc[0, "score"] = None
    path = tmp_path / "r1.parquet"
    predictions.to_parquet(path, index=False)
    assert len(read_predictions(path, "r1")) == 19
    with pytest.raises(ValueError, match=r"holds predictions for release\(s\) \['r1'\], not r2"):
        read_predictions(path, "r2")
    with pytest.raises(FileNotFoundError, match="run `python -m ucc_ml.cli score-batch"):
        read_predictions(tmp_path / "missing.parquet", "r1")


def test_selection_is_deterministic_stratified_and_bounded():
    candidates, splits, labels, predictions = _world()
    pool = build_example_pool(candidates, splits, labels, predictions)
    first = select_curated_examples(pool)
    shuffled = select_curated_examples(pool.sample(frac=1, random_state=3))
    pd.testing.assert_frame_equal(first.reset_index(drop=True), shuffled.reset_index(drop=True))
    assert 50 <= len(first) <= 100
    per_cell = first.groupby("cell").size()
    assert len(per_cell) == 8 and per_cell.max() <= 9
    assert first[first["priority"] == 0].groupby("cell").size().max() <= 3
    for _, group in first.groupby("cell"):
        assert list(group["priority"]) == sorted(group["priority"])
        for _, same in group.groupby("priority"):
            assert list(same["case_id"]) == sorted(same["case_id"])


def test_selection_fills_to_the_minimum_and_respects_the_maximum():
    candidates, splits, labels, predictions = _world()
    pool = build_example_pool(candidates, splits, labels, predictions)
    co_only = pool[pool["region"] == "CO"].reset_index(drop=True)
    assert len(select_curated_examples(co_only)) == 50
    assert len(select_curated_examples(pool, maximum=60)) == 60
    tiny = pool.head(12)
    assert len(select_curated_examples(tiny)) == len(tiny)
    assert len(select_curated_examples(pool.head(0))) == 0


def test_the_examples_document_marks_everything_curated_and_precomputed():
    candidates, splits, labels, predictions = _world()
    pool = build_example_pool(candidates, splits, labels, predictions)
    selected = select_curated_examples(pool)
    document = build_examples_document(selected, "r1", "t", dict(SELECTION_RULE), _label_statistics())
    assert document["curated"] is True and document["precomputed"] is True
    # The pooled sentence of the rounds this file actually holds -- never one round's phrase.
    assert document["label_provenance"] == pooled_disclosure(_DISCLOSURE_BY_ROUND)
    assert document["label_provenance"].startswith("mixed by round -- ")
    assert document["label_provenance_by_round"] == _DISCLOSURE_BY_ROUND
    # K15: the disclosure alone is not enough -- the K3 statistics travel with the examples, and so
    # does the round map the pooled sentence was built from.
    assert set(document["label_statistics"]) == {"disclosure", "disclosure_by_round", "counts_by_status",
                                                 "counts_by_round", "pass_agreement", "founder_audit",
                                                 "repeat_consistency"}
    assert document["label_statistics"] == _label_statistics()
    assert document["selection"]["method"] == "deterministic stratified selection, never random"
    assert "not to measure it" in document["selection"]["not_a_benchmark"]
    first = document["examples"][0]
    assert set(first) == set(document["columns"])
    assert first["curated"] is True and first["precomputed"] is True
    assert "borrower_name_raw" not in first and "priority" not in first
    for example in document["examples"]:
        # The file-level pooled sentence: these rows carry no labelling round, so a per-row
        # arrangement would be a claim the document cannot support.
        assert example["label_source"] == (document["label_provenance"] if example["label"] else None)
        assert example["lender_names"] == ["A BANK"]
