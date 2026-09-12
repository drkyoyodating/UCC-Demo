"""predict_cases: one path for batch and API. Decision = score >= threshold; the baseline comes from ucc_ml.legacy on
the candidate table's canonical lender set; lender order and blank lenders never matter; bounds are enforced by the input
model and equal the service config; contributions are real coef*x terms from the bundle."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from ucc_ml import legacy
from ucc_ml.inference import (
    MAX_LENDERS,
    MAX_NAME_CHARS,
    CaseInput,
    Prediction,
    canonical_lenders,
    cases_to_frame,
    input_hash_for,
    load_release_bundle,
    predict_cases,
)

REAL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "v1.yaml"


@pytest.fixture(scope="module")
def released(tmp_path_factory):
    from ucc_ml.synthetic import build_synthetic_release, make_world

    return make_world(), load_release_bundle(build_synthetic_release(tmp_path_factory.mktemp("infer")), top_k=5)


def test_case_input_bounds():
    CaseInput(borrower_name="A", lender_names=[], region="CO")
    for bad in (dict(borrower_name=""), dict(borrower_name="X" * 301), dict(lender_names=["L"] * 21),
                dict(lender_names=["L" * 301]), dict(region="NY"), dict(score=0.9)):
        with pytest.raises(ValidationError):
            CaseInput(**{"borrower_name": "A", "lender_names": [], "region": "CO", **bad})


def test_case_input_bounds_equal_the_service_config():
    from ucc_ml.config import load_config

    service = load_config(REAL_CONFIG).section("service")
    assert (MAX_NAME_CHARS, MAX_LENDERS, MAX_NAME_CHARS) == (service["max_name_chars"], service["max_lenders"],
                                                            service["max_lender_chars"])


def test_canonical_lenders_is_the_candidate_table_rule():
    from ucc_ml.dataset import canonical_lender_set

    for values in (["B BANK", "A FIN", "B BANK"], [" A FIN ", "", "   ", None, float("nan")], [], None,
                   ["WELLS FARGO BANK NA", "wells fargo bank na"]):
        assert canonical_lenders(values) == canonical_lender_set(values), values


def test_input_hash_is_order_and_whitespace_insensitive_but_region_sensitive():
    a = input_hash_for("Smith Excavating LLC", ["B BANK", "A FIN"], "CO")
    b = input_hash_for("Smith Excavating LLC ", ["A FIN", "B BANK", "B BANK"], "CO")
    c = input_hash_for("Smith Excavating LLC", ["B BANK", "A FIN"], "CT")
    assert a == b and a != c and len(a) == 64


def test_cases_to_frame_uses_the_vendored_normaliser_when_clean_names_are_not_supplied():
    frame = cases_to_frame([CaseInput(borrower_name="Colorado Foliage, Inc.", lender_names=["Wells Fargo Bank N.A.", " "], region="CO")])
    assert frame.borrower_name_clean.iloc[0] == "COLORADO FOLIAGE"
    assert frame.lender_names_raw.iloc[0] == ["Wells Fargo Bank N.A."]
    assert frame.lender_names_clean.iloc[0] == ["WELLS FARGO BANK"]
    given = cases_to_frame([CaseInput(borrower_name="X", lender_names=["Y"], region="CO", borrower_name_clean="X", lender_names_clean=["Y"])])
    assert given.borrower_name_clean.iloc[0] == "X" and given.lender_names_clean.iloc[0] == ["Y"]


def test_predict_cases_fields_decision_baseline_and_contributions(released):
    world, b = released
    assert predict_cases([], b) == []
    preds = predict_cases([CaseInput(borrower_name="SMITH EXCAVATING LLC", lender_names=["WELLS FARGO BANK NA"], region="CO", case_id="c1"),
                           CaseInput(borrower_name="ALPINE DENTAL PC", lender_names=[], region="CT")], b)
    assert len(preds) == 2 and all(isinstance(p, Prediction) for p in preds)
    p = preds[0]
    assert p.case_id == "c1" and p.release_id == b.release_id and p.threshold == b.threshold and p.score_type == b.score_type
    assert 0.0 <= p.score <= 1.0
    assert p.decision == ("suggest_relevant" if p.score >= b.threshold else "review_needed")
    assert (p.baseline_qualifies, p.baseline_route) == (True, "borrower")
    assert p.baseline_qualifies == legacy.baseline_qualifies("SMITH EXCAVATING LLC", ["WELLS FARGO BANK NA"])
    assert 1 <= len(p.top_feature_contributions) <= 5
    assert all(name.split("__", 1)[0] in {"borrower_word", "borrower_char", "lender_word", "lender_char"} for name, _ in p.top_feature_contributions)
    weights = [abs(w) for _, w in p.top_feature_contributions]
    assert weights == sorted(weights, reverse=True)
    assert preds[1].case_id is None and preds[1].baseline_qualifies is False


def test_lender_order_and_blank_lenders_do_not_change_the_prediction(released):
    world, b = released
    x = predict_cases([CaseInput(borrower_name="MILLER GRADING CO", lender_names=["US BANK NA", "KOMATSU FINANCIAL LIMITED PARTNERSHIP"], region="CO")], b)[0]
    y = predict_cases([CaseInput(borrower_name="MILLER GRADING CO", lender_names=["KOMATSU FINANCIAL LIMITED PARTNERSHIP", "", "US BANK NA", "  "], region="CO")], b)[0]
    assert x.score == y.score and x.input_hash == y.input_hash and x.top_feature_contributions == y.top_feature_contributions
    assert (x.baseline_qualifies, x.baseline_route) == (y.baseline_qualifies, y.baseline_route) == (True, "both")


def test_supplied_clean_names_reproduce_the_dataset_path(released):
    world, b = released
    rows = world.candidates.sample(n=25, random_state=1)
    with_clean = [CaseInput(borrower_name=r.borrower_name_raw, lender_names=list(r.lender_names_raw), region=r.region, case_id=r.case_id,
                            borrower_name_clean=r.borrower_name_clean, lender_names_clean=list(r.lender_names_clean)) for r in rows.itertuples()]
    computed = [CaseInput(borrower_name=r.borrower_name_raw, lender_names=list(r.lender_names_raw), region=r.region, case_id=r.case_id)
                for r in rows.itertuples()]
    a, c = predict_cases(with_clean, b), predict_cases(computed, b)
    assert [p.score for p in a] == [p.score for p in c]
    assert [p.baseline_qualifies for p in a] == rows.baseline_qualifies.tolist()
    assert [p.baseline_route for p in a] == rows.baseline_route.tolist()
