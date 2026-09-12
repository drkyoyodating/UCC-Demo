"""Feature policy v1 — four TF-IDF blocks over borrower and lender names.

Text columns produced by ``build_feature_frame`` (pack §6 Plan B):

  borrower_word_text  borrower_name_clean (None -> "")                     word  (1,2)
  borrower_char_text  borrower_name_raw, upper-cased, whitespace-collapsed char_wb (3,5)
  lender_word_text    sorted unique lender_names_clean joined by ' | '      word  (1,2)
  lender_char_text    sorted unique upper/ws-normalised lender_names_raw    char_wb (3,5)

Borrower and lender blocks are separate ColumnTransformer entries so identical
tokens keep their role (``borrower_word__bank`` vs ``lender_word__bank``).
Vocabularies and IDF are learned only from whatever frame ``fit`` receives.
Region, file ids, addresses, dates, rule outputs and notes are never features
(Codex §4). The constants below are mirrored in the `features` section of
ml/configs/v1.yaml (Plan A, contract K4) and a test keeps the two equal.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

FEATURE_POLICY_VERSION = "features_v1"
VARIANTS = ("borrower_only", "borrower_lender")
TEXT_COLUMNS = ("borrower_word_text", "borrower_char_text", "lender_word_text", "lender_char_text")
LENDER_JOIN = " | "


def _is_null(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value)) if np.isscalar(value) else False
    except (TypeError, ValueError):
        return False


def normalize_raw_for_chars(raw) -> str:
    """Deterministic case/whitespace normalisation for the character blocks."""
    if _is_null(raw):
        return ""
    return " ".join(str(raw).upper().split())


def text_or_empty(value) -> str:
    return "" if _is_null(value) else str(value)


def as_str_list(value) -> list[str]:
    """Parquet list columns arrive as numpy arrays, lists, None or NaN."""
    if _is_null(value):
        return []
    return [str(v) for v in list(value) if not _is_null(v)]


def lender_text(names, *, char: bool) -> str:
    items = as_str_list(names)
    if char:
        items = [normalize_raw_for_chars(x) for x in items]
    return LENDER_JOIN.join(sorted({x for x in items if x != ""}))


def build_feature_frame(cases: pd.DataFrame) -> pd.DataFrame:
    """Needs borrower_name_raw, borrower_name_clean, lender_names_raw, lender_names_clean."""
    return pd.DataFrame(
        {
            "borrower_word_text": [text_or_empty(v) for v in cases["borrower_name_clean"]],
            "borrower_char_text": [normalize_raw_for_chars(v) for v in cases["borrower_name_raw"]],
            "lender_word_text": [lender_text(v, char=False) for v in cases["lender_names_clean"]],
            "lender_char_text": [lender_text(v, char=True) for v in cases["lender_names_raw"]],
        },
        index=cases.index,
    )


def _word_vectorizer() -> TfidfVectorizer:
    # token_pattern keeps single-character tokens ("J & J EXCAVATING", "160 CORP") —
    # the sklearn default drops them and initials matter for person-vs-business.
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True,
                           token_pattern=r"(?u)\b\w+\b", dtype=np.float64)


def _char_vectorizer() -> TfidfVectorizer:
    # lowercase=False: the text is already upper-cased by normalize_raw_for_chars,
    # so the normalisation is explicit and identical for batch and API input.
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False, dtype=np.float64)


def make_feature_transformer(variant: str) -> ColumnTransformer:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    blocks = [
        ("borrower_word", _word_vectorizer(), "borrower_word_text"),
        ("borrower_char", _char_vectorizer(), "borrower_char_text"),
    ]
    if variant == "borrower_lender":
        blocks += [
            ("lender_word", _word_vectorizer(), "lender_word_text"),
            ("lender_char", _char_vectorizer(), "lender_char_text"),
        ]
    return ColumnTransformer(blocks, remainder="drop", sparse_threshold=1.0,
                             verbose_feature_names_out=True)


def make_pipeline(variant: str, C: float, *, max_iter: int = 2000, seed: int = 0) -> Pipeline:
    # sklearn 1.9 deprecates the `penalty` argument; L2 is the default (l1_ratio=0.0).
    clf = LogisticRegression(C=C, class_weight=None, solver="lbfgs", max_iter=max_iter, random_state=seed)
    return Pipeline([("features", make_feature_transformer(variant)), ("clf", clf)])


def feature_names(pipeline: Pipeline) -> np.ndarray:
    return pipeline.named_steps["features"].get_feature_names_out()


def transform_features(pipeline: Pipeline, frame: pd.DataFrame) -> sparse.csr_matrix:
    return sparse.csr_matrix(pipeline.named_steps["features"].transform(frame))


def top_contributions_from_matrix(X: sparse.csr_matrix, coef: np.ndarray, names: np.ndarray,
                                  k: int) -> list[list[tuple[str, float]]]:
    """Per row: the k largest |coef_j * x_j| as (feature, weight), largest first.

    Linear contributions describe the model's calculation, not independent evidence
    about the business (Codex §9). The intercept is not a feature and is reported in
    model-manifest.json instead.
    """
    X = sparse.csr_matrix(X)
    out: list[list[tuple[str, float]]] = []
    for i in range(X.shape[0]):
        start, end = X.indptr[i], X.indptr[i + 1]
        cols = X.indices[start:end]
        weights = coef[cols] * X.data[start:end]
        order = np.argsort(-np.abs(weights), kind="stable")[:k]
        out.append([(str(names[cols[j]]), float(weights[j])) for j in order])
    return out


def top_contributions(pipeline: Pipeline, frame: pd.DataFrame, k: int = 10) -> list[list[tuple[str, float]]]:
    X = transform_features(pipeline, frame)
    coef = pipeline.named_steps["clf"].coef_[0]
    return top_contributions_from_matrix(X, coef, feature_names(pipeline), k)
