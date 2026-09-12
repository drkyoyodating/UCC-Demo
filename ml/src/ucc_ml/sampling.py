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


# ============================================================================= the main round
# The estimation stratum stays the four-value K2 form. A screen CELL sub-divides a stratum for
# SAMPLING only: its whole effect on estimation is carried by the per-row inclusion_probability, so
# cells live in the round's cases file and never in labels.csv.

#: Design columns a screened round carries in addition to the case columns.
MAIN_ROUND_DESIGN_COLUMNS: tuple[str, ...] = SAMPLE_COLUMNS + ("screen_cell",)


def split_cell_populations(cases: pd.DataFrame, splits: pd.DataFrame,
                           cells: pd.Series | pd.DataFrame) -> dict[str, dict[tuple[str, str], int]]:
    """N_h for every (split, stratum, cell): the population each design weight is computed from."""
    frame = cells if isinstance(cells, pd.DataFrame) else pd.DataFrame(
        {"case_id": cases.case_id.to_numpy(), "screen_cell": cells.to_numpy()})
    d = (add_stratum(cases)[["case_id", "sampling_stratum"]]
         .merge(splits[["case_id", "split"]], on="case_id", how="inner")
         .merge(frame[["case_id", "screen_cell"]], on="case_id", how="inner"))
    out: dict[str, dict[tuple[str, str], int]] = {}
    for (split, stratum, cell), n in d.groupby(["split", "sampling_stratum", "screen_cell"]).size().items():
        out.setdefault(str(split), {})[(str(stratum), str(cell))] = int(n)
    return out


def propose_main_allocation(populations: Mapping[str, Mapping[tuple[str, str], int]],
                            totals: Mapping[str, int], *, b4_floor_per_split: int,
                            overrides: Mapping[str, Mapping[tuple[str, str], int]] | None = None,
                            floor_cell: str = "B4_remainder") -> dict[str, dict[tuple[str, str], int]]:
    """A default allocation over the cells of each split, for the founder to accept or change.

    The rule is deliberately dull and explainable: before the yield probe measures anything, no cell
    is assumed more valuable than another, so each cell of a split gets an equal base share. Shares
    are then capped by the cell's population, the shortfall is redistributed to cells with room, and
    the unscreened cell is raised to its floor last, taking from the largest other cells.

    Refuses rather than returning a quietly broken design:
      * any cell at n_h = 0 -- a cell with no draw has no inclusion probability and silently drops
        its whole population out of every estimate;
      * the floor cell below ``b4_floor_per_split`` -- that floor is the entire reason the estimates
        are design-unbiased regardless of the screen, so it is a refusal condition, not a guideline.
    """
    out: dict[str, dict[tuple[str, str], int]] = {}
    for split, total in totals.items():
        pops = dict(populations.get(split, {}))
        if not pops:
            raise ValueError(f"{split}: no cell populations")
        keys = sorted(pops)

        # a founder override replaces that cell's count exactly; the rest share what is left
        fixed = {k: int(v) for k, v in (overrides or {}).get(split, {}).items()}
        for k, v in fixed.items():
            if k not in pops:
                raise ValueError(f"{split}: override names unknown cell {k}")
            if v > pops[k]:
                raise ValueError(f"{split}: override asks for {v} of {pops[k]} in cell {k}")
        free = [k for k in keys if k not in fixed]
        remaining = total - sum(fixed.values())
        if remaining < 0:
            raise ValueError(f"{split}: overrides total {sum(fixed.values())}, more than the split's {total}")
        if not free and remaining:
            raise ValueError(f"{split}: every cell is overridden but {remaining} of the total is unassigned")

        alloc = dict(fixed)
        share = remaining // len(free) if free else 0
        for k in free:
            alloc[k] = min(share, pops[k])

        def room(k: tuple[str, str]) -> int:
            return 0 if k in fixed else pops[k] - alloc[k]

        # redistribute whatever the caps left over, largest-population cell first
        leftover = total - sum(alloc.values())
        for k in sorted(free, key=lambda k: -pops[k]):
            if leftover <= 0:
                break
            take = min(leftover, room(k))
            alloc[k] += take
            leftover -= take

        # raise the unscreened cell to its floor, taking from the largest non-fixed allocations
        floor_keys = [k for k in keys if k[1] == floor_cell]
        shortfall = b4_floor_per_split - sum(alloc[k] for k in floor_keys)
        for k in floor_keys:
            if k in fixed:
                continue
            while shortfall > 0 and room(k) > 0:
                donor = max((j for j in free if j not in floor_keys and alloc[j] > 1),
                            key=lambda j: alloc[j], default=None)
                if donor is None:
                    break
                alloc[donor] -= 1
                alloc[k] += 1
                shortfall -= 1

        for k in keys:
            if alloc[k] <= 0:
                raise ValueError(f"{split}: cell {k} would be drawn {alloc[k]} times; a cell with no "
                                 f"draw drops its whole population out of every estimate")
            if alloc[k] > pops[k]:
                raise ValueError(f"{split}: cell {k} asks for {alloc[k]} of {pops[k]}")
        drawn_floor = sum(alloc[k] for k in floor_keys)
        if drawn_floor < b4_floor_per_split:
            raise ValueError(f"{split}: {floor_cell} would be drawn {drawn_floor} times, below the floor of "
                             f"{b4_floor_per_split}; that floor is why the estimates do not depend on the screen")
        out[split] = alloc
    return out


def draw_main_round(cases: pd.DataFrame, splits: pd.DataFrame, cells: pd.DataFrame,
                    allocation: Mapping[str, Mapping[tuple[str, str], int]], *, seed: int,
                    exclude_case_ids: Iterable[str] = (), purpose: str = "main_v1") -> pd.DataFrame:
    """Draw each (split, stratum, cell) independently, so every row carries its own cell's rate.

    ``draw_stratified`` needs no change: a cell is a filter on the frame handed to it, never a key,
    so it already computes pool_h and inclusion_probability per cell. Pilot cases are excluded, which
    is why pool_h can differ from N_h.
    """
    frame = (add_stratum(cases)
             .merge(splits[["case_id", "split"]], on="case_id", how="inner")
             .merge(cells[["case_id", "screen_cell"]], on="case_id", how="inner"))
    parts: list[pd.DataFrame] = []
    for split, cell_alloc in allocation.items():
        for (stratum, cell), n_h in sorted(cell_alloc.items()):
            pool = frame[(frame.split == split) & (frame.sampling_stratum == stratum)
                         & (frame.screen_cell == cell)]
            drawn = draw_stratified(pool, {stratum: n_h}, seed=seed,
                                    purpose=f"{purpose}_{split}_{cell}", exclude_case_ids=exclude_case_ids)
            drawn["screen_cell"] = cell
            drawn["split"] = split
            parts.append(drawn)
    if not parts:
        return pd.DataFrame(columns=list(MAIN_ROUND_DESIGN_COLUMNS) + ["split"])
    sample = pd.concat(parts, ignore_index=True)
    merged = sample.merge(cases, on="case_id", how="left", validate="one_to_one")
    case_cols = [c for c in cases.columns if c != "case_id"]
    ordered = (["case_id"] + case_cols
               + [c for c in MAIN_ROUND_DESIGN_COLUMNS if c != "case_id"] + ["split"])
    return merged.sort_values(["split", "sampling_stratum", "screen_cell", "draw_rank"],
                              kind="mergesort")[ordered].reset_index(drop=True)
