"""Frozen grouped splits (Codex §6, context pack §6, contracts K7 and K12).

Split by borrower GROUP, never by row: group_id = sha256([region, name_clean]) so repeated filings by
one borrower stay together and no group spans CO and CT. Every group that contains a pilot case is
assigned TRAIN (K12): validation (which sets the weighted-precision threshold) and test must each be a
probability sample of its own split population with known inclusion probabilities, and the pilot was
drawn from the whole candidate population at different rates. The other groups are ordered by a seeded
sha256 rank and fill test (round(test * all groups)), then validation (round(validation * all groups)),
then train. The grouping is independent of labels and predictions. A digest over the sorted
"case_id,group_id,split" lines is committed under docs/data/ml/ at freeze time.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from ucc_ml.contracts import (
    SPLIT_COLUMNS,
    SPLITS,
    STRATA,
    Split,
    canonical_json,
    check_column,
    is_sha256_hex,
    norm_key,
    sha256_hex,
    stratum_name,
)
from ucc_ml.dataset import read_frame
from ucc_ml.provenance import utc_now_iso

PILOT_RULE = "every group containing a pilot case is assigned train (contract K12)"


def group_rank(seed: int, group_id: str) -> str:
    return sha256_hex(canonical_json(["split_v1", seed, group_id]))


def assign_splits(group_ids: Iterable[str], pilot_groups: set[str], ratios: Mapping[str, float],
                  seed: int) -> dict[str, str]:
    groups = sorted(set(group_ids))
    n = len(groups)
    n_test = round(ratios["test"] * n)
    n_val = round(ratios["validation"] * n)
    assignment: dict[str, str] = {}
    filled = 0
    for g in sorted(groups, key=lambda group: group_rank(seed, group)):
        if g in pilot_groups:
            assignment[g] = "train"
            continue
        assignment[g] = "test" if filled < n_test else ("validation" if filled < n_test + n_val else "train")
        filled += 1
    return assignment


def split_digest(splits: pd.DataFrame) -> str:
    ordered = splits.sort_values("case_id", kind="mergesort")
    text = "\n".join(f"{r.case_id},{r.group_id},{r.split}" for r in ordered.itertuples(index=False)) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit_splits(cases: pd.DataFrame, splits: pd.DataFrame, pilot_case_ids: Iterable[str] = ()) -> dict:
    merged = cases[["case_id", "region", "group_id", "borrower_name_raw", "borrower_name_clean"]].merge(
        splits[["case_id", "split"]], on="case_id", how="inner", validate="one_to_one")
    pilot = set(pilot_case_ids)
    in_pilot = merged.case_id.isin(pilot)
    merged["norm_name"] = [norm_key(n) for n in merged.borrower_name_raw]
    named = merged[merged.borrower_name_clean.notna()]
    audit = {
        "group_overlap": int((merged.groupby("group_id").split.nunique() > 1).sum()),
        "case_overlap": int(splits.case_id.duplicated().sum()),
        "pilot_cases_in_test": int(((merged.split == "test") & in_pilot).sum()),
        "pilot_cases_not_in_train": int(((merged.split != "train") & in_pilot).sum()),
        "same_norm_name_across_splits_within_region": int(
            (merged.groupby(["region", "norm_name"]).split.nunique() > 1).sum()),
        "same_name_clean_across_regions_and_splits": int((named.groupby("borrower_name_clean").split.nunique() > 1).sum()),
        "note": "the last figure is expected and by design: groups are region-scoped, so the same cleaned "
                "name in CO and CT is two groups and may land in two splits",
    }
    failed = {key: audit[key] for key in ("group_overlap", "case_overlap", "pilot_cases_in_test",
                                          "pilot_cases_not_in_train", "same_norm_name_across_splits_within_region")
              if audit[key] != 0}
    if failed:                       # a real raise, not assert: `python -O` must not skip the audit
        raise RuntimeError(f"split audit failed: {failed}")
    return audit


def freeze_splits(cases: pd.DataFrame, pilot_case_ids: Iterable[str], ratios: Mapping[str, float], seed: int,
                  label_policy_version: str, candidates_sha256: str) -> tuple[pd.DataFrame, dict]:
    pilot_ids = set(pilot_case_ids)
    unknown = pilot_ids - set(cases.case_id)
    if unknown:
        raise ValueError(f"{len(unknown)} pilot case_ids not in candidates, e.g. {sorted(unknown)[:3]}")
    pilot_groups = set(cases.loc[cases.case_id.isin(pilot_ids), "group_id"])
    assignment = assign_splits(cases.group_id, pilot_groups, ratios, seed)
    splits = pd.DataFrame({
        "case_id": cases.case_id, "group_id": cases.group_id,
        "split": [assignment[g] for g in cases.group_id],
    }).sort_values("case_id", kind="mergesort").reset_index(drop=True)[list(SPLIT_COLUMNS)]
    for row in splits.head(1000).to_dict("records"):
        Split.model_validate(row)
    audit = audit_splits(cases, splits, pilot_ids)
    groups_by_split = Counter(assignment.values())
    cases_by_split = Counter(splits.split)
    pilot_by_split = Counter(splits.loc[splits.case_id.isin(pilot_ids), "split"])
    # Ratios are applied to GROUPS, so each split's share of CASES drifts from 65/15/20 by stratum.
    # Record the (split, stratum) populations: they are K13's N_h for the main-round allocation.
    stratum_of = {cid: stratum_name(r, bool(q))
                  for cid, r, q in zip(cases.case_id, cases.region, cases.baseline_qualifies)}
    by_split_stratum = Counter((s, stratum_of[c]) for c, s in zip(splits.case_id, splits.split))
    manifest = {
        "seed": seed, "ratios": dict(ratios), "pilot_rule": PILOT_RULE,
        "label_policy_version": label_policy_version, "candidates_sha256": candidates_sha256,
        "groups": {"total": len(assignment), "pilot": len(pilot_groups),
                   "by_split": {s: int(groups_by_split.get(s, 0)) for s in SPLITS}},
        "cases": {"total": int(len(splits)), "by_split": {s: int(cases_by_split.get(s, 0)) for s in SPLITS},
                  "pilot_by_split": {s: int(pilot_by_split.get(s, 0)) for s in SPLITS},
                  "by_split_stratum": {s: {h: int(by_split_stratum.get((s, h), 0)) for h in STRATA}
                                       for s in SPLITS}},
        "audit": audit, "digest": split_digest(splits), "created_at": utc_now_iso(),
    }
    return splits, manifest


def read_splits(path: Path) -> pd.DataFrame:
    """The only reader of splits.parquet (contract K7)."""
    path = Path(path)
    df = read_frame(path)
    if list(df.columns) != list(SPLIT_COLUMNS):
        raise ValueError(f"{path}: columns {list(df.columns)} != {list(SPLIT_COLUMNS)}")
    check_column(path, df, "case_id", is_sha256_hex)
    check_column(path, df, "group_id", is_sha256_hex)
    check_column(path, df, "split", lambda v: v in SPLITS)
    if not df.case_id.is_unique:
        dupes = sorted(set(df.case_id[df.case_id.duplicated()]))
        raise ValueError(f"{path}: column 'case_id' has {len(dupes)} duplicate value(s): {dupes[:5]}")
    per_group = df.groupby("group_id").split.nunique()
    spanning = sorted(per_group[per_group > 1].index)
    if spanning:
        raise ValueError(f"{path}: column 'split' differs within {len(spanning)} group_id value(s): {spanning[:5]}")
    return df
