"""Identity functions have known vectors; models reject malformed rows; the K3 label vocabulary is exact."""
import typing
from datetime import date, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

BK = "d3251935b935fbdaf6c21e84068eaee5b97ee17b175faaaf2e5768dd73489558"
CID = "7f059dac0cbd9c293deef455d86baf07c0aae1fa51aec078c67eeb89c3e1c98b"
GID_CO = "5da75097cb2af1dc9e5c8784bddbdbbe5a94eefe0883bc09db84ba21daa7965c"
GID_CT = "9fa888671defbc09875f19aa622b8213532e10f750d5f5645f6d602311f22d23"


def test_known_vectors():
    from ucc_ml.contracts import make_borrower_key, make_case_id, make_group_id

    bk = make_borrower_key("Acme Excavation LLC ", "1 Main St", "Denver", "co", "80202")
    assert bk == BK
    assert make_case_id("CO", "F1", bk) == CID
    assert make_group_id("CO", "ACME EXCAVATION", "ignored") == GID_CO
    assert make_group_id("CT", "ACME EXCAVATION", "ignored") == GID_CT  # region-scoped
    assert GID_CO != GID_CT


def test_norm_key_treats_non_strings_as_blank():
    from ucc_ml.contracts import canonical_json, make_borrower_key, make_group_id, norm_key, sha256_hex

    assert norm_key(None) == "" and norm_key(float("nan")) == "" and norm_key("  x ") == "X"
    assert make_borrower_key("A", "B", None, float("nan"), "") == sha256_hex(canonical_json(["A", "B", "", "", ""]))
    # a name that is pure legal form has no name_clean; the raw string keeps it grouped by itself
    assert make_group_id("CO", None, " llc ") == sha256_hex(canonical_json(["CO", "RAW:LLC"]))


def test_canonical_json_is_compact_and_unicode():
    from ucc_ml.contracts import canonical_json

    assert canonical_json(["CO", "1", "é", []]) == '["CO","1","é",[]]'


def test_stratum_name_uses_the_colon_form():
    from ucc_ml.contracts import STRATA, stratum_name

    assert stratum_name("CO", True) == "CO:accepted" and stratum_name("CT", False) == "CT:rejected"
    assert STRATA == ("CO:accepted", "CO:rejected", "CT:accepted", "CT:rejected")


def test_reason_codes_cover_every_label_and_are_unique():
    from ucc_ml.contracts import ALL_REASON_CODES, FOUNDER_REASON_CODE, LABELS, REASON_CODES

    assert tuple(REASON_CODES) == LABELS
    assert len(set(ALL_REASON_CODES)) == len(ALL_REASON_CODES) == 11
    assert FOUNDER_REASON_CODE in ALL_REASON_CODES
    assert all(code.startswith(("R_", "N_", "I_")) for label in LABELS for code in REASON_CODES[label])


def test_label_vocabulary_is_the_k3_contract():
    from ucc_ml import contracts as c
    from ucc_ml.config import DISCLOSURE

    assert typing.get_args(c.AdjudicationStatus) == c.ADJUDICATION_STATUSES == (
        "model_agreed", "founder_confirmed", "founder_adjudicated", "blind_repeat")
    assert c.RESOLVED_ADJUDICATION_STATUSES == ("model_agreed", "founder_confirmed", "founder_adjudicated")
    assert typing.get_args(c.LabellingRound) == c.LABELLING_ROUNDS == ("pilot_v1", "main_v1")
    assert c.LABELLER_PASS_A == "claude_blind_pass_a" and c.LABELLER_PASS_B == "claude_blind_pass_b"
    assert c.LABELLER_AGREED == "claude_blind_pass_a+claude_blind_pass_b" and c.LABELLER_FOUNDER == "founder"
    assert c.PASS_LABELLERS == {"a": "claude_blind_pass_a", "b": "claude_blind_pass_b"}
    assert c.LABEL_DISCLOSURE == DISCLOSURE == "model-labelled, founder-adjudicated"
    assert "pending" not in c.ADJUDICATION_STATUSES and "adjudicated" not in c.ADJUDICATION_STATUSES


def _case(**over):
    from ucc_ml.contracts import make_borrower_key, make_case_id, make_group_id

    bk = make_borrower_key("ACME EXCAVATION LLC", "1 MAIN ST", "DENVER", "CO", "80202")
    base = dict(
        case_id=make_case_id("CO", "F1", bk), dataset_version="v1", region="CO", file_id="F1",
        lineage_id="M1", borrower_key=bk, borrower_name_raw="ACME EXCAVATION LLC",
        borrower_name_clean="ACME EXCAVATION", borrower_suffix="LLC",
        lender_names_raw=["CATERPILLAR FINANCIAL", "WELLS FARGO BANK"],
        lender_names_clean=["CATERPILLAR FINANCIAL", "WELLS FARGO BANK"],
        earliest_observed_date=date(2001, 2, 3), latest_observed_date=date(2001, 2, 3),
        borrower_city="DENVER", borrower_state="CO", borrower_zip="80202",
        source_row_count=1, source_filing_type="ucc", source_status="false",
        baseline_qualifies=True, baseline_route="both",
        group_id=make_group_id("CO", "ACME EXCAVATION", "ACME EXCAVATION LLC"),
    )
    base.update(over)
    return base


def test_case_accepts_a_valid_row_and_orders_columns():
    from ucc_ml.contracts import CASE_COLUMNS, Case

    c = Case.model_validate(_case())
    assert c.split is None
    assert CASE_COLUMNS[0] == "case_id" and CASE_COLUMNS[-1] == "group_id" and "split" not in CASE_COLUMNS
    assert len(CASE_COLUMNS) == 22


@pytest.mark.parametrize("bad", [
    dict(lender_names_raw=["WELLS FARGO BANK", "CATERPILLAR FINANCIAL"]),   # unsorted
    dict(lender_names_raw=["A", "A"]),                                      # duplicate
    dict(lender_names_raw=["A", ""]),                                       # blank
    dict(case_id="0" * 64),                                                  # does not recompute
    dict(baseline_qualifies=False),                                          # route says both
    dict(baseline_route="neither"),                                          # qualifies says True
    dict(earliest_observed_date=date(2002, 1, 1)),                           # earliest > latest
    dict(region="NY"),
    dict(source_row_count=0),
    dict(extra_field=1),
])
def test_case_rejects_malformed_rows(bad):
    from ucc_ml.contracts import Case

    with pytest.raises(ValidationError):
        Case.model_validate(_case(**bad))


def test_case_allows_empty_lender_set_and_null_dates():
    from ucc_ml.contracts import Case

    c = Case.model_validate(_case(lender_names_raw=[], lender_names_clean=[], baseline_route="borrower",
                                  earliest_observed_date=None, latest_observed_date=None))
    assert c.lender_names_raw == []


def _label(**over):
    from ucc_ml.contracts import LABELLER_AGREED

    base = dict(
        case_id="a" * 64, label="RELEVANT", reason_code="R_BORROWER_EQUIPMENT_WORD",
        reason="EXCAVATION in the borrower name", labeller_id=LABELLER_AGREED,
        labelled_at=datetime(2026, 9, 12, 10, 0, 0), policy_version="label_policy_v1",
        sampling_stratum="CO:rejected", inclusion_probability=0.001,
        adjudication_status="model_agreed", is_repeat=False, labelling_round="pilot_v1",
    )
    base.update(over)
    return base


def test_label_columns_are_derived_from_the_model():
    from ucc_ml.contracts import LABEL_COLUMNS

    assert LABEL_COLUMNS == ("case_id", "label", "reason_code", "reason", "labeller_id", "labelled_at",
                             "policy_version", "sampling_stratum", "inclusion_probability",
                             "adjudication_status", "is_repeat", "labelling_round")


@pytest.mark.parametrize("good", [
    dict(),                                                                               # model_agreed
    dict(adjudication_status="founder_confirmed", labeller_id="founder"),
    dict(adjudication_status="founder_adjudicated", labeller_id="founder", reason_code="ADJUDICATED",
         label="NOT_RELEVANT", reason="founder: a law office"),
    dict(adjudication_status="blind_repeat", is_repeat=True, labeller_id="claude_blind_pass_a"),
    dict(adjudication_status="blind_repeat", is_repeat=True, labeller_id="claude_blind_pass_b",
         label="INSUFFICIENT_EVIDENCE", reason_code="I_GENERIC_NAME", labelling_round="main_v1"),
    dict(labelled_at="2026-09-12T10:00:00Z", inclusion_probability=1.0),
])
def test_label_accepts_each_k3_status(good):
    from ucc_ml.contracts import Label

    Label.model_validate(_label(**good))


@pytest.mark.parametrize("bad", [
    dict(reason_code="N_OTHER_INDUSTRY_EXPLICIT"),                                        # wrong family
    dict(sampling_stratum="CO:maybe"),
    dict(sampling_stratum="CO|accepted"),                                                 # pipe form is not K2
    dict(inclusion_probability=0.0),
    dict(reason="   "),
    dict(adjudication_status="founder_adjudicated", labeller_id="founder"),               # needs ADJUDICATED
    dict(adjudication_status="founder_adjudicated", reason_code="ADJUDICATED", labeller_id="claude_blind_pass_a"),
    dict(adjudication_status="founder_confirmed", labeller_id="founder", reason_code="ADJUDICATED"),
    dict(adjudication_status="founder_confirmed"),                                        # labeller must be founder
    dict(labeller_id="founder"),                                                          # model_agreed by founder
    dict(is_repeat=True),                                                                 # model_agreed repeat
    dict(adjudication_status="blind_repeat", is_repeat=False, labeller_id="claude_blind_pass_a"),
    dict(adjudication_status="blind_repeat", is_repeat=True),                             # agreed labeller id
    dict(adjudication_status="pending"),
    dict(adjudication_status="adjudicated"),
    dict(labelling_round="pilot_v2"),
    dict(case_id="A" * 64),
])
def test_label_rejects_rows_that_break_k3(bad):
    from ucc_ml.contracts import Label

    with pytest.raises(ValidationError):
        Label.model_validate(_label(**bad))


def test_split_validation():
    from ucc_ml.contracts import SPLIT_COLUMNS, Split

    Split.model_validate({"case_id": "a" * 64, "group_id": "b" * 64, "split": "test"})
    assert SPLIT_COLUMNS == ("case_id", "group_id", "split")
    with pytest.raises(ValidationError):
        Split.model_validate({"case_id": "a" * 64, "group_id": "b" * 64, "split": "holdout"})


def test_check_column_names_source_column_and_values():
    from ucc_ml.contracts import check_column, is_sha256_hex

    assert is_sha256_hex("a" * 64) and not is_sha256_hex("A" * 64) and not is_sha256_hex(None)
    frame = pd.DataFrame({"case_id": ["a" * 64, "nope", None]})
    check_column("x.csv", frame.head(1), "case_id", is_sha256_hex)
    with pytest.raises(ValueError, match=r"labels\.csv: column 'case_id' has 2 invalid value\(s\): .*'nope'"):
        check_column("labels.csv", frame, "case_id", is_sha256_hex)
    with pytest.raises(ValueError, match="missing required column 'split'"):
        check_column("splits.parquet", frame, "split", lambda v: True)
