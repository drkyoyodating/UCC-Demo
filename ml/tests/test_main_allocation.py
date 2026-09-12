"""The screened main round: populations per cell, an allocation that refuses a broken design, and a
draw that gives every row its own cell's inclusion probability.

What these tests defend:
  * a cell is a SAMPLING filter, never an estimation key -- the stratum stays the four-value K2 form;
  * no cell is ever drawn zero times, because a cell with no draw silently drops its whole
    population out of every population estimate;
  * the unscreened cell is always drawn at its floor, which is the entire reason the estimates do
    not depend on the screen being any good;
  * inclusion_probability is computed per cell, not per stratum, or every design weight is wrong.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ucc_ml.sampling import (MAIN_ROUND_DESIGN_COLUMNS, draw_main_round, propose_main_allocation,
                             split_cell_populations)

CELLS = ("A1_lender_only", "A2_name_decidable", "B1_maker_group", "B2_score_top",
         "B3_declared_words", "B4_remainder")


#: Cell sizes in the real pool are wildly unequal -- the unscreened cell is ~1M cases and the
#: declared list ~12k -- and that inequality is exactly what makes the design weights differ.
UNEQUAL = {"A1_lender_only": 40, "A2_name_decidable": 35, "B1_maker_group": 70,
           "B2_score_top": 25, "B3_declared_words": 15, "B4_remainder": 400}


def world(per_cell: int = 60, sizes: dict[str, int] | None = None
          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A tiny two-region world with every cell populated in every split."""
    rows, cell_rows, split_rows = [], [], []
    i = 0
    for region in ("CO", "CT"):
        for split in ("train", "validation", "test"):
            for cell in CELLS:
                per_cell_here = (sizes or {}).get(cell, per_cell)
                accepted = cell.startswith("A")
                for _ in range(per_cell_here):
                    cid = f"{i:064x}"
                    i += 1
                    rows.append({"case_id": cid, "region": region, "group_id": f"g{i}",
                                 "borrower_name_raw": f"FIRM {i}",
                                 "baseline_qualifies": accepted,
                                 "baseline_route": "lender" if cell == "A1_lender_only"
                                 else ("borrower" if accepted else "neither")})
                    cell_rows.append({"case_id": cid, "screen_cell": cell})
                    split_rows.append({"case_id": cid, "group_id": f"g{i}", "split": split})
    return pd.DataFrame(rows), pd.DataFrame(split_rows), pd.DataFrame(cell_rows)


def test_populations_are_keyed_by_split_stratum_and_cell():
    cases, splits, cells = world(per_cell=5)
    pops = split_cell_populations(cases, splits, cells)
    assert set(pops) == {"train", "validation", "test"}
    assert pops["train"][("CO:accepted", "A1_lender_only")] == 5
    assert pops["train"][("CO:rejected", "B4_remainder")] == 5
    # 2 regions x 6 cells, and an accepted cell never lands in a rejected stratum
    assert len(pops["train"]) == 12
    assert all(("accepted" in s) == c.startswith("A") for s, c in pops["train"])


def test_allocation_refuses_a_cell_it_would_never_draw():
    pops = {"train": {("CO:accepted", "A1_lender_only"): 100, ("CO:rejected", "B4_remainder"): 100}}
    with pytest.raises(ValueError, match="drops its whole population"):
        propose_main_allocation(pops, {"train": 1}, b4_floor_per_split=1)


def test_allocation_refuses_to_trade_away_the_unscreened_floor():
    """The floor is why the estimates survive a bad screen, so it is a refusal, not a guideline."""
    pops = {"train": {("CO:rejected", "B2_score_top"): 500, ("CO:rejected", "B4_remainder"): 3}}
    with pytest.raises(ValueError, match="below the floor"):
        propose_main_allocation(pops, {"train": 100}, b4_floor_per_split=50)


def test_allocation_honours_the_floor_and_the_total():
    cases, splits, cells = world(per_cell=500)
    pops = split_cell_populations(cases, splits, cells)
    alloc = propose_main_allocation(pops, {"train": 1200, "validation": 400, "test": 800},
                                    b4_floor_per_split=200)
    for split, total in (("train", 1200), ("validation", 400), ("test", 800)):
        assert sum(alloc[split].values()) == total
        assert all(n > 0 for n in alloc[split].values())
        floor = sum(n for (_, cell), n in alloc[split].items() if cell == "B4_remainder")
        assert floor >= 200, (split, floor)


def test_allocation_never_asks_for_more_than_a_cell_holds():
    cases, splits, cells = world(per_cell=30)
    pops = split_cell_populations(cases, splits, cells)
    alloc = propose_main_allocation(pops, {"train": 300}, b4_floor_per_split=50)
    for key, n in alloc["train"].items():
        assert n <= pops["train"][key]


def test_the_draw_gives_every_row_its_own_cells_inclusion_probability():
    cases, splits, cells = world(sizes=UNEQUAL)
    pops = split_cell_populations(cases, splits, cells)
    alloc = propose_main_allocation(pops, {"train": 240}, b4_floor_per_split=40)
    drawn = draw_main_round(cases, splits, cells, {"train": alloc["train"]}, seed=20260912)

    assert len(drawn) == 240
    assert set(MAIN_ROUND_DESIGN_COLUMNS) <= set(drawn.columns)
    assert drawn.case_id.is_unique
    # the rate is per cell: two cells drawn at different n_h must not share a probability
    by_cell = drawn.groupby(["sampling_stratum", "screen_cell"]).inclusion_probability.nunique()
    assert (by_cell == 1).all()                       # one rate within a cell
    # cells of different size drawn at the same n_h MUST carry different rates, or the design
    # weights are wrong: a 400-case cell and a 15-case cell are not sampled alike.
    assert drawn.inclusion_probability.nunique() > 1
    for (stratum, cell), n_h in alloc["train"].items():
        got = drawn[(drawn.sampling_stratum == stratum) & (drawn.screen_cell == cell)]
        assert len(got) == n_h
        assert (got.inclusion_probability == n_h / pops["train"][(stratum, cell)]).all()


def test_the_draw_is_reproducible_from_the_seed_and_excludes_the_pilot():
    cases, splits, cells = world(per_cell=100)
    pops = split_cell_populations(cases, splits, cells)
    alloc = propose_main_allocation(pops, {"train": 120}, b4_floor_per_split=20)
    first = draw_main_round(cases, splits, cells, {"train": alloc["train"]}, seed=20260912)
    again = draw_main_round(cases, splits, cells, {"train": alloc["train"]}, seed=20260912)
    assert first.case_id.tolist() == again.case_id.tolist()
    assert draw_main_round(cases, splits, cells, {"train": alloc["train"]},
                           seed=1).case_id.tolist() != first.case_id.tolist()

    excluded = set(first.case_id.head(10))
    after = draw_main_round(cases, splits, cells, {"train": alloc["train"]}, seed=20260912,
                            exclude_case_ids=excluded)
    assert not (set(after.case_id) & excluded)
    # pool_h records the exclusion, so the design weight reflects what could actually be drawn
    assert after.pool_h.min() < after.N_h.max()


def test_the_stratum_stays_the_four_value_contract_form():
    """A cell must never become an estimation key: Label.sampling_stratum still takes only K2 values."""
    from ucc_ml.contracts import STRATA

    cases, splits, cells = world(per_cell=50)
    pops = split_cell_populations(cases, splits, cells)
    alloc = propose_main_allocation(pops, {"train": 120}, b4_floor_per_split=20)
    drawn = draw_main_round(cases, splits, cells, {"train": alloc["train"]}, seed=20260912)
    assert set(drawn.sampling_stratum) <= set(STRATA)


def test_a_founder_override_replaces_one_cell_and_the_rest_share_what_is_left():
    cases, splits, cells = world(sizes=UNEQUAL)
    pops = split_cell_populations(cases, splits, cells)
    key = ("CO:rejected", "B3_declared_words")
    alloc = propose_main_allocation(pops, {"train": 240}, b4_floor_per_split=40,
                                    overrides={"train": {key: 15}})
    assert alloc["train"][key] == 15
    assert sum(alloc["train"].values()) == 240


def test_an_override_may_not_exceed_the_cell_it_names():
    cases, splits, cells = world(sizes=UNEQUAL)
    pops = split_cell_populations(cases, splits, cells)
    key = ("CO:rejected", "B3_declared_words")
    with pytest.raises(ValueError, match="override asks for"):
        propose_main_allocation(pops, {"train": 240}, b4_floor_per_split=40,
                                overrides={"train": {key: 10_000}})
    with pytest.raises(ValueError, match="unknown cell"):
        propose_main_allocation(pops, {"train": 240}, b4_floor_per_split=40,
                                overrides={"train": {("CO:rejected", "B9_nope"): 5}})
