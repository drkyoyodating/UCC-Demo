"""The screen: four disjoint sampling cells over the rules-rejected pool (amendment §3).

THE SCREEN SETS SAMPLING RATES. It is never a model feature, never evidence, and never shown to a
labeller. The validity of every published estimate rests on ``B4_remainder`` being sampled at a
known positive rate, not on this screen being any good: a class of job-site firms invisible to all
four cells is still estimated through B4's design weight. A better screen buys a narrower interval;
it never changes what the interval is around.

Why not one hand-written word list. Three reasonable definitions of "the boundary" select nearly
disjoint sets, and five careful readings of one twelve-class probe spanned 30%. A list is therefore
a sampling decision, not a frame, so it is demoted to one cell of four and published verbatim.

The four cells, assigned in PRIORITY order over rules-rejected cases:

``B1_maker_group``    the borrower's region-scoped group was financed by a named maker on ANOTHER
                      filing. Taken from the register's own lender data; no vocabulary at all.
``B2_score_top``      top ``score_quantile`` within (region, split) by a fitted score on the borrower
                      name with the rules' own vocabulary MASKED OUT, anchored on cases the rules
                      accepted for their LENDER alone -- a set whose selection provably never looked
                      at the borrower name. Fitted on TRAIN groups only.
``B3_declared_words`` the declared, published, pre-registered word list. One cell among four.
``B4_remainder``      everything else. Never assumed empty; sampled at a hard floor per split.

The accepted pool splits two ways, tied to the world-knowledge ruling:

``A1_lender_only``    ``baseline_route == "lender"`` -- decidable ONLY from the lender, which is
                      exactly where withdrawing the manufacturer roster bites.
``A2_name_decidable`` ``baseline_route in {"borrower", "both"}`` -- the borrower's own name carries a
                      qualifying word.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ucc_ml.vendor.heavy_filter_v1 import BORROWER_RE, LENDER_RE

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sklearn.pipeline import Pipeline

#: Cells over the rules-rejected pool, in the order they are assigned.
SCREEN_CELLS: tuple[str, ...] = ("B1_maker_group", "B2_score_top", "B3_declared_words", "B4_remainder")
#: Cells over the rules-accepted pool.
ACCEPTED_CELLS: tuple[str, ...] = ("A1_lender_only", "A2_name_decidable")
ALL_CELLS: tuple[str, ...] = ACCEPTED_CELLS + SCREEN_CELLS

#: Routes that mean a named maker, captive or franchised dealer financed the filing.
MAKER_ROUTES: tuple[str, ...] = ("lender", "both")

#: B3, published verbatim so no reader mistakes it for a sampling frame. Whole-word throughout.
DECLARED_CLASSES: dict[str, str] = {
    "CONTRACTING": r"\bCONTRACTING\b|\bCONTRACTORS?\b",
    "BUILDERS": r"\bBUILDERS?\b|\bBUILDING\b",
    "EQUIP_RENTAL": r"\bEQUIPMENT\s+RENTALS?\b",
    "HEAVY_EQUIP": r"\bHEAVY\s+EQUIPMENT\b|\bHEAVY\s+MACHINERY\b|\bMACHINERY\b",
    "UNDERGROUND": r"\bUNDERGROUND\b|\bUTILITIES\b|\bUTILITY\b",
    "HAULING": r"\bHAULING\b",
    "SNOWPLOW": r"\bSNOW\s*PLOW\w*\b|\bSNOWPLOW\w*\b",
    "SANDGRAVEL": r"\bSAND\b|\bGRAVEL\b|\bAGGREGATES?\b",
    "SEPTIC": r"\bSEPTIC\b|\bSEWER\b|\bDRAINAGE\b",
    "DIRT": r"\bDIRT\b",
    "TRENCHLESS": r"\bTRENCHLESS\b",
    "STEEL_ERECT": r"\bSTEEL\s+ERECTION\b",
}
DECLARED_RE = re.compile("|".join(DECLARED_CLASSES.values()))

_WS = re.compile(r"\s+")


def mask_rules_vocabulary(name: str | None) -> str:
    """Strip every rules word before vectorising, so the screen cannot relearn the word list.

    Without this the fitted score reaches the rules' vocabulary through co-occurrence and the screen
    becomes the thing it is meant to look past.
    """
    s = BORROWER_RE.sub(" ", str(name or "").upper())
    s = LENDER_RE.sub(" ", s)
    return _WS.sub(" ", s).strip()


def maker_groups(cases: pd.DataFrame) -> set[str]:
    """Region-scoped groups financed by a named maker on at least one filing.

    From the register's own lender data. B1 is the intersection of these groups with the rejected
    pool, so a B1 case is a filing the rules rejected by a borrower the rules accepted elsewhere.
    """
    hit = cases.baseline_route.isin(MAKER_ROUTES)
    return set(cases.loc[hit, "group_id"].unique())


def accepted_cell(route: str) -> str:
    return "A1_lender_only" if route == "lender" else "A2_name_decidable"


@dataclass(frozen=True)
class Screener:
    """A fitted screen. ``pipeline`` scores a masked borrower name; nothing else is ever scored."""

    pipeline: "Pipeline"
    n_positive: int
    n_negative: int
    fit_split: str

    def score(self, names: pd.Series) -> np.ndarray:
        masked = [mask_rules_vocabulary(n) for n in names]
        return self.pipeline.decision_function(masked)


def fit_screener(cases: pd.DataFrame, splits: pd.DataFrame, *, fit_split: str = "train",
                 seed: int = 20260912) -> Screener:
    """Fit the B2 score on ``fit_split`` GROUPS only.

    positives  cases the rules accepted for their LENDER ALONE (``baseline_route == "lender"``).
               By construction of ``legacy.baseline_route`` that means ``is_heavy_borrower`` was
               False, so these are real heavy-construction borrowers whose names the Route-B lists
               do not match -- exactly the population a review queue has to find, and a set whose
               selection never looked at the borrower name.
    negatives  rejected cases in the same split whose group has NO maker lender, so no group appears
               on both sides.

    No label, no validation row and no test row is touched.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, make_union

    d = cases.merge(splits[["case_id", "split"]], on="case_id", how="inner")
    d = d[d.split == fit_split]
    makers = maker_groups(cases)
    pos = d[d.baseline_route == "lender"]
    neg = d[(~d.baseline_qualifies.astype(bool)) & (~d.group_id.isin(makers))]
    if pos.empty or neg.empty:
        raise ValueError(f"screener needs both classes in {fit_split}: {len(pos)} positive, {len(neg)} negative")

    text = [mask_rules_vocabulary(n) for n in pd.concat([pos.borrower_name_raw, neg.borrower_name_raw])]
    y = np.r_[np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)]
    features = make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=10, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=20, sublinear_tf=True),
    )
    model = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)
    pipeline = Pipeline([("features", features), ("model", model)]).fit(text, y)
    return Screener(pipeline=pipeline, n_positive=len(pos), n_negative=len(neg), fit_split=fit_split)


def strongest_features(screener: Screener, k: int = 200) -> list[str]:
    """The k strongest positive features, for the test that proves no rules word survived masking."""
    names = screener.pipeline.named_steps["features"].get_feature_names_out()
    coef = screener.pipeline.named_steps["model"].coef_[0]
    return [str(names[i]) for i in np.argsort(coef)[::-1][:k]]


def assign_screen_cells(cases: pd.DataFrame, splits: pd.DataFrame, scores: pd.Series | None = None,
                        *, score_quantile: float = 0.02) -> pd.Series:
    """One cell per case, indexed like ``cases``. Accepted cases get an A cell, rejected a B cell.

    Priority is B1, then B2, then B3, then B4: a case that would match several lands in the first.
    ``scores`` is indexed by case_id; when it is None, B2 is empty and its cases fall to B3/B4.
    """
    d = cases.merge(splits[["case_id", "split"]], on="case_id", how="left")
    cell = pd.Series(pd.NA, index=d.index, dtype="object")
    accepted = d.baseline_qualifies.astype(bool)
    cell[accepted] = [accepted_cell(r) for r in d.loc[accepted, "baseline_route"]]

    rejected = ~accepted
    makers = maker_groups(cases)
    b1 = rejected & d.group_id.isin(makers)
    cell[b1] = "B1_maker_group"

    remaining = rejected & cell.isna()
    if scores is not None and remaining.any():
        s = d.loc[remaining, "case_id"].map(scores)
        frame = pd.DataFrame({"score": s.to_numpy(), "region": d.loc[remaining, "region"].to_numpy(),
                              "split": d.loc[remaining, "split"].to_numpy()}, index=s.index)
        picked: list = []
        for _, grp in frame.groupby(["region", "split"], dropna=False):
            take = int(round(score_quantile * len(grp)))
            if take:
                picked.extend(grp.score.nlargest(take).index.tolist())
        cell[picked] = "B2_score_top"

    remaining = rejected & cell.isna()
    names = d.loc[remaining, "borrower_name_raw"].fillna("").str.upper()
    cell[remaining & names.str.contains(DECLARED_RE, regex=True).reindex(cell.index, fill_value=False)] = \
        "B3_declared_words"
    cell[rejected & cell.isna()] = "B4_remainder"
    cell.index = cases.index
    return cell.astype("string")


def cell_counts(cases: pd.DataFrame, splits: pd.DataFrame, cells: pd.Series) -> pd.DataFrame:
    """Per (region, cell, split) populations: the N_h every design weight is computed from."""
    d = cases[["case_id", "region"]].copy()
    d["cell"] = cells.to_numpy()
    d = d.merge(splits[["case_id", "split"]], on="case_id", how="left")
    out = (d.groupby(["region", "cell", "split"], dropna=False).size().unstack("split", fill_value=0)
           .reset_index())
    for col in ("train", "validation", "test"):
        if col not in out.columns:
            out[col] = 0
    out["total"] = out[["train", "validation", "test"]].sum(axis=1)
    return out[["region", "cell", "train", "validation", "test", "total"]].sort_values(["region", "cell"])


def screening_manifest(screener: Screener, counts: pd.DataFrame, *, score_quantile: float,
                       cells_sha256: str, extra: dict | None = None) -> dict:
    """What the screen was, in enough detail that the draw it produced can be re-derived.

    Committed BEFORE any label of a screened round exists, so the sampling frame is provably fixed
    in advance. The declared word list is published here verbatim rather than hidden, because a
    reader must be able to see that it is one cell of four and not the frame.
    """
    return {
        "purpose": "sampling cells over the rules-rejected pool (amendment §3)",
        "not_a_feature": ("The screen sets sampling rates only. Every cell, including the unscreened "
                          "remainder, was sampled at a known positive rate, so the estimates do not "
                          "depend on the screen being right; a worse screen would only have widened "
                          "the intervals."),
        "cells": list(ALL_CELLS),
        "priority_order": list(SCREEN_CELLS),
        "fit_split": screener.fit_split,
        "fit_positive_route": "lender",
        "fit_positives": screener.n_positive,
        "fit_negatives": screener.n_negative,
        "score_quantile": score_quantile,
        "declared_classes": dict(DECLARED_CLASSES),
        "populations": counts.to_dict(orient="records"),
        "screen_cells_sha256": cells_sha256,
        **(extra or {}),
    }


def build_screen(config_path) -> dict:
    """Fit the screen, assign every case a cell, and write the side artefact plus its manifest.

    The cells are keyed by ``case_id`` in their own file: ``candidates.parquet`` is never rewritten,
    so its committed digest stays valid and the screen can never leak into the model's features.
    """
    from ucc_ml.config import artefact_paths, load_config
    from ucc_ml.dataset import read_candidates
    from ucc_ml.provenance import git_head, sha256_file, utc_now_iso, write_json
    from ucc_ml.splitting import read_splits

    cfg = load_config(config_path)
    paths = artefact_paths(cfg)
    settings = cfg.section("screening")
    cases = read_candidates(paths.candidates_parquet)
    splits = read_splits(paths.splits_parquet)

    screener = fit_screener(cases, splits, fit_split=settings["fit_split"], seed=settings["seed"])
    makers = maker_groups(cases)
    rejected = cases[~cases.baseline_qualifies.astype(bool)]
    to_score = rejected[~rejected.group_id.isin(makers)]
    scores = pd.Series(screener.score(to_score.borrower_name_raw), index=to_score.case_id)
    cells = assign_screen_cells(cases, splits, scores, score_quantile=float(settings["score_quantile"]))
    counts = cell_counts(cases, splits, cells)

    paths.screen_cells.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"case_id": cases.case_id.to_numpy(), "screen_cell": cells.to_numpy()})
    frame.to_parquet(paths.screen_cells, index=False)
    manifest = screening_manifest(
        screener, counts, score_quantile=float(settings["score_quantile"]),
        cells_sha256=sha256_file(paths.screen_cells),
        extra={"created_at": utc_now_iso(), "git_head": git_head(cfg.repo_root),
               "config_sha256": cfg.config_sha256,
               "candidates_sha256": sha256_file(paths.candidates_parquet),
               "splits_sha256": sha256_file(paths.splits_parquet),
               "rules_words_in_top_200_features": sum(
                   1 for f in strongest_features(screener, k=200)
                   if BORROWER_RE.search(f.upper()) or LENDER_RE.search(f.upper()))})
    write_json(paths.screening_manifest, manifest)
    return {"counts": counts, "manifest": manifest, "cells": paths.screen_cells}
