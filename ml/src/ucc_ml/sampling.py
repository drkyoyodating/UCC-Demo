"""Seeded, design-aware sampling. Strata = region x baseline_qualifies (context pack §6).

Every draw is a deterministic SHA-256 rank, never a library RNG, so the same seed reproduces the
same sample under any pandas / numpy version. For stratum h with population N_h and n_h drawn
from a pool of pool_h eligible cases, inclusion_probability = n_h / pool_h; the design weight
Plan B uses is 1 / inclusion_probability. For the pilot nothing is excluded, so pool_h == N_h.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from ucc_ml.contracts import STRATA, canonical_json, sha256_hex, stratum_name

#: WARNING: N_h and n_h differ only by case, and DuckDB resolves identifiers case-insensitively --
#: read_parquet() renames the second to n_h_1 and `SELECT n_h` then returns the stratum POPULATION.
#: Read pilot_cases.parquet / main_cases.parquet with dataset.read_frame (pyarrow) ONLY, never DuckDB.
#: test_cli_make_pilot_writes_parquet_and_manifest pins that behaviour so it cannot surprise anybody.
SAMPLE_COLUMNS: tuple[str, ...] = (
    "case_id", "sampling_stratum", "N_h", "pool_h", "n_h", "inclusion_probability", "draw_rank",
)


def draw_rank(seed: int, purpose: str, case_id: str) -> str:
    return sha256_hex(canonical_json([purpose, seed, case_id]))


def add_stratum(cases: pd.DataFrame) -> pd.DataFrame:
    out = cases.copy()
    out["sampling_stratum"] = [stratum_name(r, bool(q)) for r, q in zip(out.region, out.baseline_qualifies)]
    return out


def stratum_sizes(cases: pd.DataFrame) -> dict[str, int]:
    s = cases if "sampling_stratum" in cases.columns else add_stratum(cases)
    counts = s.sampling_stratum.value_counts()
    return {k: int(counts.get(k, 0)) for k in STRATA}


def draw_stratified(cases: pd.DataFrame, n_per_stratum: Mapping[str, int], seed: int, purpose: str,
                    exclude_case_ids: Iterable[str] = ()) -> pd.DataFrame:
    s = cases if "sampling_stratum" in cases.columns else add_stratum(cases)
    sizes = stratum_sizes(s)
    excluded = set(exclude_case_ids)
    parts: list[pd.DataFrame] = []
    for stratum, n_h in n_per_stratum.items():
        if stratum not in STRATA:
            raise ValueError(f"unknown stratum {stratum!r}; expected one of {STRATA}")
        pool = s[(s.sampling_stratum == stratum) & ~s.case_id.isin(excluded)]
        ranked = sorted((draw_rank(seed, purpose, cid), cid) for cid in pool.case_id)
        if len(ranked) < n_h:
            raise ValueError(f"{stratum}: asked for {n_h} but only {len(ranked)} cases are available after exclusions")
        chosen = ranked[:n_h]
        parts.append(pd.DataFrame({
            "case_id": [cid for _, cid in chosen],
            "sampling_stratum": stratum,
            "N_h": int(sizes[stratum]),
            "pool_h": int(len(ranked)),
            "n_h": int(n_h),
            "inclusion_probability": (n_h / len(ranked)) if n_h else 0.0,
            "draw_rank": [rank for rank, _ in chosen],
        }))
    if not parts:
        return pd.DataFrame(columns=list(SAMPLE_COLUMNS))
    return pd.concat(parts, ignore_index=True)[list(SAMPLE_COLUMNS)]
def make_pilot(cases: pd.DataFrame, per_stratum: int, seed: int, purpose: str = "pilot_v1") -> pd.DataFrame:
    """The development pilot: per_stratum cases from each of the four strata, full case columns attached."""
    sample = draw_stratified(cases, {s: per_stratum for s in STRATA}, seed=seed, purpose=purpose)
    merged = sample.merge(cases, on="case_id", how="left", validate="one_to_one")
    case_cols = [c for c in cases.columns if c != "case_id"]
    ordered = ["case_id"] + case_cols + [c for c in SAMPLE_COLUMNS if c != "case_id"]
    merged["_order"] = [STRATA.index(s) for s in merged.sampling_stratum]
    merged = merged.sort_values(["_order", "draw_rank"], kind="mergesort").drop(columns="_order")
    return merged[ordered].reset_index(drop=True)
