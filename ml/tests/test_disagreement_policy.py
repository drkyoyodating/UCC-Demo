"""A round may resolve a pass disagreement by founder decision, or by keeping it unresolved.

What these tests defend:
  * "unresolved" never invents an answer -- the row is INSUFFICIENT_EVIDENCE, which is what it means
    when two careful independent readers cannot agree, and it is excluded from fitting and evaluation;
  * a decision a person actually made always wins, under either policy, and is never discarded;
  * "founder" still refuses, so no round silently loses its gate;
  * the policy is per round, so one round's arrangement never restates another's.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ucc_ml.contracts import (LABELLER_AGREED, RESOLVED_ADJUDICATION_STATUSES, UNRESOLVED_REASON_CODE)
from ucc_ml.labeling import (FOUNDER_DECISION_COLUMNS, PASS_COLUMNS, RESOLVED_LABELS, UndecidedDisagreements,
                             assemble_round_labels)

A, B = "a" * 64, "b" * 64
CODE = {"RELEVANT": "R_BORROWER_TRADE_WORD", "NOT_RELEVANT": "N_OTHER_INDUSTRY_EXPLICIT",
        "INSUFFICIENT_EVIDENCE": "I_GENERIC_NAME"}


def _pass(letter: str, answers: dict[str, str]) -> pd.DataFrame:
    rows = [{"queue_case_id": cid, "case_id": cid, "is_repeat": False, "part": 1, "label": label,
             "reason_code": CODE[label], "reason": f"pass {letter} saw this", "labeller_id": f"claude_blind_pass_{letter}",
             "labelled_at": "2026-09-12T10:00:00Z", "sampling_stratum": "CO:rejected",
             "inclusion_probability": 0.25, "labelling_round": "main_v1"}
            for cid, label in answers.items()]
    return pd.DataFrame(rows, columns=list(PASS_COLUMNS))


def _no_decisions() -> pd.DataFrame:
    return pd.DataFrame(columns=list(FOUNDER_DECISION_COLUMNS), dtype=object)


def _decision(case_id: str, label: str, kind: str = "adjudicated") -> pd.DataFrame:
    return pd.DataFrame([{"case_id": case_id, "review_type": "disagreement", "founder_label": label,
                          "founder_note": "the founder decided this one", "decision": kind,
                          "reviewed_at": "2026-09-12T11:00:00Z"}], columns=list(FOUNDER_DECISION_COLUMNS))


DISAGREE_A = _pass("a", {A: "RELEVANT", B: "RELEVANT"})
DISAGREE_B = _pass("b", {A: "NOT_RELEVANT", B: "RELEVANT"})


def test_unresolved_keeps_the_disagreement_instead_of_inventing_an_answer():
    labels = assemble_round_labels(DISAGREE_A, DISAGREE_B, _no_decisions(), "main_v1", "label_policy_v2",
                                   disagreement_policy="unresolved")
    row = labels[labels.case_id == A].iloc[0]
    assert row.adjudication_status == "blind_unresolved"
    assert row.label == "INSUFFICIENT_EVIDENCE"          # never one of the two passes' answers
    assert row.reason_code == UNRESOLVED_REASON_CODE
    assert row.labeller_id == LABELLER_AGREED
    assert "RELEVANT" in row.reason and "NOT_RELEVANT" in row.reason   # the reason records both answers
    # the case both passes agreed on is untouched by the policy
    assert labels[labels.case_id == B].iloc[0].adjudication_status == "model_agreed"


def test_an_unresolved_row_is_never_fitted_or_evaluated():
    labels = assemble_round_labels(DISAGREE_A, DISAGREE_B, _no_decisions(), "main_v1", "label_policy_v2",
                                   disagreement_policy="unresolved")
    row = labels[labels.case_id == A].iloc[0]
    assert row.adjudication_status not in RESOLVED_ADJUDICATION_STATUSES
    assert row.label not in RESOLVED_LABELS


def test_founder_policy_still_refuses_an_undecided_disagreement():
    with pytest.raises(UndecidedDisagreements) as exc:
        assemble_round_labels(DISAGREE_A, DISAGREE_B, _no_decisions(), "main_v1", "label_policy_v2")
    assert exc.value.case_ids == [A]


@pytest.mark.parametrize("policy", ["founder", "unresolved"])
def test_a_decision_a_person_actually_made_always_wins(policy):
    """Switching a round to "unresolved" must never throw away real human judgment."""
    labels = assemble_round_labels(DISAGREE_A, DISAGREE_B, _decision(A, "RELEVANT"), "main_v1",
                                   "label_policy_v2", disagreement_policy=policy)
    row = labels[labels.case_id == A].iloc[0]
    assert row.adjudication_status == "founder_adjudicated" and row.label == "RELEVANT"
    assert row.labeller_id == "founder"


def test_the_policy_is_per_round_and_defaults_to_founder():
    from ucc_ml.config import load_config
    from ucc_ml.labeling import disagreement_policy_for_round, disclosure_for_round

    cfg = load_config("ml/configs/v1.yaml")
    assert disagreement_policy_for_round(cfg, "pilot_v1") == "founder"
    assert disagreement_policy_for_round(cfg, "main_v1") == "unresolved"
    assert disagreement_policy_for_round(cfg, "queue_v1") == "founder"        # absent -> unchanged behaviour
    assert disclosure_for_round(cfg, "pilot_v1") != disclosure_for_round(cfg, "main_v1")
    with pytest.raises(ValueError, match="unknown labelling round"):
        disagreement_policy_for_round(cfg, "not_a_round")


def test_the_config_refuses_an_unknown_round_or_policy(tmp_path):
    from pydantic import ValidationError

    from ucc_ml.config import Labelling

    ok = dict(labels=["RELEVANT"], reason_max_chars=300, disclosure="model-labelled, founder-adjudicated")
    Labelling(**ok, rounds_in_labels=["pilot_v1", "main_v1"], disagreement_policy_by_round={"main_v1": "unresolved"})
    with pytest.raises(ValidationError, match="rounds_in_labels"):
        Labelling(**ok, rounds_in_labels=["not_a_round"])
    with pytest.raises(ValidationError, match="disagreement_policy_by_round"):
        Labelling(**ok, disagreement_policy_by_round={"main_v1": "whatever"})


def test_a_merged_file_never_states_one_round_s_arrangement_as_if_it_covered_all():
    from ucc_ml.contracts import LABEL_DISCLOSURE
    from ucc_ml.labeling import pooled_disclosure

    assert pooled_disclosure({}) == LABEL_DISCLOSURE
    assert pooled_disclosure({"pilot_v1": "X", "main_v1": "X"}) == "X"          # agreeing rounds: just true
    mixed = pooled_disclosure({"pilot_v1": "X", "main_v1": "Y"})
    assert mixed.startswith("mixed by round")
    assert "pilot_v1: X" in mixed and "main_v1: Y" in mixed                     # neither is hidden
    assert mixed not in ("X", "Y")
