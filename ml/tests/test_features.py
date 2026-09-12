"""Feature policy v1: text preparation, the four TF-IDF blocks, namespace separation,
lender-order invariance, fit-only vocabularies, linear contributions, and parity with the
`features` section of Plan A's run config."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from ucc_ml.features import (
    FEATURE_POLICY_VERSION,
    LENDER_JOIN,
    TEXT_COLUMNS,
    as_str_list,
    build_feature_frame,
    feature_names,
    lender_text,
    make_feature_transformer,
    make_pipeline,
    normalize_raw_for_chars,
    top_contributions,
    transform_features,
)

REAL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "v1.yaml"


def _cases() -> pd.DataFrame:
    rows = [
        ("SMITH EXCAVATING LLC", "SMITH EXCAVATING", ["WELLS FARGO BANK NA"], ["WELLS FARGO BANK"], 1),
        ("jones paving inc", "JONES PAVING", ["CATERPILLAR FINANCIAL SERVICES CORPORATION"], ["CATERPILLAR FINANCIAL SERVICES"], 1),
        ("GARCIA CONCRETE PUMPING", "GARCIA CONCRETE PUMPING", [], [], 1),
        ("MILLER GRADING CO", "MILLER GRADING", ["US BANK NA", "KOMATSU FINANCIAL LIMITED PARTNERSHIP"], ["US BANK", "KOMATSU FINANCIAL LIMITED PARTNERSHIP"], 1),
        ("DAVIS DEMOLITION", "DAVIS DEMOLITION", ["FIRSTBANK"], ["FIRSTBANK"], 1),
        ("LOPEZ EARTHMOVING LLC", "LOPEZ EARTHMOVING", ["VOLVO FINANCIAL SERVICES"], ["VOLVO FINANCIAL SERVICES"], 1),
        ("ALPINE DENTAL PC", "ALPINE DENTAL", ["WELLS FARGO BANK NA"], ["WELLS FARGO BANK"], 0),
        ("SMITH LAW OFFICE", "SMITH LAW OFFICE", ["US BANK NA"], ["US BANK"], 0),
        ("MAIN STREET RESTAURANT", "MAIN STREET RESTAURANT", [], [], 0),
        ("GREEN ACRES FARMS", "GREEN ACRES FARMS", ["FIRSTBANK", "US BANK NA"], ["FIRSTBANK", "US BANK"], 0),
        ("FIRST BAPTIST CHURCH", "FIRST BAPTIST CHURCH", ["LIBERTY BANK"], ["LIBERTY BANK"], 0),
        ("SUNRISE BAKERY", "SUNRISE BAKERY", ["PEOPLES UNITED BANK"], ["PEOPLES UNITED BANK"], 0),
    ]
    return pd.DataFrame(rows, columns=["borrower_name_raw", "borrower_name_clean", "lender_names_raw", "lender_names_clean", "y"])


def test_normalize_raw_for_chars():
    assert normalize_raw_for_chars("  smith   Excavating  ") == "SMITH EXCAVATING"
    assert normalize_raw_for_chars(None) == ""
    assert normalize_raw_for_chars(float("nan")) == ""
    assert normalize_raw_for_chars("A\tB\nC") == "A B C"


def test_as_str_list_accepts_numpy_none_nan():
    assert as_str_list(np.array(["A", "B"], dtype=object)) == ["A", "B"]
    assert as_str_list(None) == []
    assert as_str_list(float("nan")) == []
    assert as_str_list(["x", None]) == ["x"]


def test_lender_text_sorted_unique_joined():
    raw = np.array(["wells fargo bank na", "CATERPILLAR FINANCIAL", "WELLS FARGO   BANK NA"], dtype=object)
    assert lender_text(raw, char=True) == "CATERPILLAR FINANCIAL | WELLS FARGO BANK NA"
    assert lender_text(["B", "A", "B"], char=False) == "A | B"
    assert lender_text([], char=False) == ""
    assert lender_text(None, char=True) == ""


def test_build_feature_frame_columns_and_nulls():
    df = _cases().copy()
    df.loc[2, "borrower_name_clean"] = None
    frame = build_feature_frame(df)
    assert list(frame.columns) == list(TEXT_COLUMNS)
    assert frame.index.equals(df.index)
    assert frame.loc[2, "borrower_word_text"] == ""
    assert frame.loc[2, "borrower_char_text"] == "GARCIA CONCRETE PUMPING"
    assert frame.loc[3, "lender_word_text"] == "KOMATSU FINANCIAL LIMITED PARTNERSHIP | US BANK"
    assert frame.loc[1, "borrower_char_text"] == "JONES PAVING INC"


def test_variants_have_the_right_blocks():
    assert [n for n, _, _ in make_feature_transformer("borrower_only").transformers] == ["borrower_word", "borrower_char"]
    assert [n for n, _, _ in make_feature_transformer("borrower_lender").transformers] == [
        "borrower_word", "borrower_char", "lender_word", "lender_char"]
    with pytest.raises(ValueError):
        make_feature_transformer("lender_only")
    assert FEATURE_POLICY_VERSION == "features_v1"


def test_feature_constants_match_the_run_config():
    from ucc_ml.config import load_config

    section = load_config(REAL_CONFIG).section("features")
    blocks = dict((name, vec) for name, vec, _ in make_feature_transformer("borrower_lender").transformers)
    word, char = blocks["borrower_word"], blocks["borrower_char"]
    assert section["policy_version"] == FEATURE_POLICY_VERSION and section["lender_join"] == LENDER_JOIN
    assert section["word"] == {"analyzer": word.analyzer, "ngram_range": list(word.ngram_range),
                               "lowercase": word.lowercase, "token_pattern": word.token_pattern}
    assert section["char"] == {"analyzer": char.analyzer, "ngram_range": list(char.ngram_range), "lowercase": char.lowercase}


def test_pipeline_fits_names_are_namespaced_and_output_is_sparse():
    df = _cases()
    frame = build_feature_frame(df)
    pipe = make_pipeline("borrower_lender", C=1.0)
    pipe.fit(frame, df.y.to_numpy(), clf__sample_weight=np.ones(len(df)))
    names = feature_names(pipe)
    prefixes = {n.split("__", 1)[0] for n in names}
    assert prefixes == {"borrower_word", "borrower_char", "lender_word", "lender_char"}
    assert "borrower_word__excavating" in set(names)
    assert "lender_word__bank" in set(names)
    X = transform_features(pipe, frame)
    assert isinstance(X, sparse.csr_matrix)
    assert X.shape == (len(df), len(names))
    assert pipe.named_steps["clf"].class_weight is None


def test_borrower_only_ignores_lenders_entirely():
    df = _cases()
    frame = build_feature_frame(df)
    pipe = make_pipeline("borrower_only", C=1.0).fit(frame, df.y.to_numpy())
    other = frame.copy()
    other["lender_word_text"] = "ZZZ QQQ"
    other["lender_char_text"] = "ZZZ QQQ"
    assert np.allclose(pipe.decision_function(frame), pipe.decision_function(other))


def test_lender_order_does_not_change_the_score():
    df = _cases()
    pipe = make_pipeline("borrower_lender", C=1.0).fit(build_feature_frame(df), df.y.to_numpy())
    a = df.iloc[[3]].copy()
    b = a.copy()
    b["lender_names_raw"] = [list(reversed(a.lender_names_raw.iloc[0]))]
    b["lender_names_clean"] = [list(reversed(a.lender_names_clean.iloc[0]))]
    assert np.allclose(pipe.decision_function(build_feature_frame(a)), pipe.decision_function(build_feature_frame(b)))


def test_vocabulary_comes_only_from_the_fitting_frame():
    df = _cases()
    pipe = make_pipeline("borrower_lender", C=1.0).fit(build_feature_frame(df), df.y.to_numpy())
    unseen = pd.DataFrame([("ZEBRA QUARRYWORKS", "ZEBRA QUARRYWORKS", [], [], 1)], columns=df.columns)
    names = set(feature_names(pipe))
    assert "borrower_word__zebra" not in names
    X = transform_features(pipe, build_feature_frame(unseen))
    assert X.shape[1] == len(names)


def test_top_contributions_sum_to_decision_function_minus_intercept():
    df = _cases()
    frame = build_feature_frame(df)
    pipe = make_pipeline("borrower_lender", C=10.0).fit(frame, df.y.to_numpy())
    contribs = top_contributions(pipe, frame, k=100000)
    dec = pipe.decision_function(frame)
    intercept = float(pipe.named_steps["clf"].intercept_[0])
    for i, row in enumerate(contribs):
        assert abs(sum(w for _, w in row) + intercept - dec[i]) < 1e-9
        weights = [abs(w) for _, w in row]
        assert weights == sorted(weights, reverse=True)
    top3 = top_contributions(pipe, frame.iloc[[0]], k=3)[0]
    assert len(top3) == 3
    assert all(isinstance(n, str) and isinstance(w, float) for n, w in top3)
