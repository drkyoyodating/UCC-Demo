"""The screen is a SAMPLING artefact: disjoint cells, priority order, and no rules vocabulary in it.

What these tests defend:
  * the mask really removes the rules' own words, so the fitted score cannot relearn the word list;
  * the cells partition the population exactly once, so every case has one known inclusion rate;
  * B4_remainder is never empty -- it is the entire reason any estimate is design-unbiased;
  * the fit touches only its own split's groups, so no validation or test row informs the screen.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ucc_ml.screening import (ACCEPTED_CELLS, ALL_CELLS, DECLARED_RE, SCREEN_CELLS, assign_screen_cells,
                              cell_counts, fit_screener, maker_groups, mask_rules_vocabulary,
                              strongest_features)


def case(case_id: str, region: str, group: str, name: str, route: str) -> dict:
    return {"case_id": case_id, "region": region, "group_id": group, "borrower_name_raw": name,
            "baseline_route": route, "baseline_qualifies": route != "neither"}


def frames(rows: list[dict], splits: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = pd.DataFrame(rows)
    split = pd.DataFrame({"case_id": cases.case_id, "group_id": cases.group_id,
                          "split": [(splits or {}).get(c, "train") for c in cases.case_id]})
    return cases, split


def test_mask_removes_the_rules_own_vocabulary():
    assert "EXCAVATING" not in mask_rules_vocabulary("BOBS EXCAVATING LLC").upper()
    assert "CATERPILLAR" not in mask_rules_vocabulary("CATERPILLAR FINANCIAL SERVICES").upper()
    assert "CONCRETE" not in mask_rules_vocabulary("HERNANDEZ CONCRETE").upper()
    # what is NOT a rules word survives, or the screen would have nothing to learn from
    assert "HOLDINGS" in mask_rules_vocabulary("ACME HOLDINGS LLC").upper()
    assert mask_rules_vocabulary(None) == ""
    assert mask_rules_vocabulary("   ") == ""


def test_declared_list_is_whole_word():
    assert DECLARED_RE.search("JCO UNDERGROUND INC")
    assert DECLARED_RE.search("PLATEAU MACHINERY INC")
    assert not DECLARED_RE.search("SANDWICH SHOP")          # SAND must not match inside SANDWICH
    assert not DECLARED_RE.search("REBUILDING SOCIETY")     # BUILDING must not match inside REBUILDING


def test_maker_groups_come_from_the_registers_own_lender_data():
    rows = [case("a" * 64, "CO", "g1", "ALPHA LLC", "lender"),
            case("b" * 64, "CO", "g1", "ALPHA LLC", "neither"),
            case("c" * 64, "CO", "g2", "BETA LLC", "neither")]
    cases, _ = frames(rows)
    assert maker_groups(cases) == {"g1"}


def test_cells_partition_every_case_exactly_once():
    rows = [case("a" * 64, "CO", "g1", "ALPHA LLC", "lender"),
            case("b" * 64, "CO", "g1", "ALPHA LLC", "neither"),          # B1: group has a maker
            case("c" * 64, "CO", "g2", "JCO UNDERGROUND INC", "neither"),  # B3: declared word
            case("d" * 64, "CO", "g3", "QUIET HOLDINGS LLC", "neither"),   # B4
            case("e" * 64, "CT", "g4", "SMITH EXCAVATING", "borrower")]
    cases, splits = frames(rows)
    cells = assign_screen_cells(cases, splits)
    assert len(cells) == len(cases)
    assert cells.notna().all()
    assert set(cells) <= set(ALL_CELLS)
    assert cells.tolist() == ["A1_lender_only", "B1_maker_group", "B3_declared_words",
                              "B4_remainder", "A2_name_decidable"]


def test_accepted_cells_follow_the_route_and_never_take_a_B_cell():
    rows = [case("a" * 64, "CO", "g1", "ALPHA LLC", "lender"),
            case("b" * 64, "CO", "g2", "BOBS EXCAVATING", "borrower"),
            case("c" * 64, "CO", "g3", "GAMMA CRANES", "both")]
    cases, splits = frames(rows)
    cells = assign_screen_cells(cases, splits)
    assert cells.tolist() == ["A1_lender_only", "A2_name_decidable", "A2_name_decidable"]
    assert not set(cells) & set(SCREEN_CELLS)


def test_b1_beats_the_declared_word_list():
    """Priority matters: a maker-financed group that also carries a declared word is B1, not B3."""
    rows = [case("a" * 64, "CO", "g1", "ALPHA LLC", "lender"),
            case("b" * 64, "CO", "g1", "ALPHA UNDERGROUND UTILITIES", "neither")]
    cases, splits = frames(rows)
    assert assign_screen_cells(cases, splits).tolist() == ["A1_lender_only", "B1_maker_group"]


def test_b4_is_never_empty_when_a_rejected_case_matches_nothing():
    rows = [case("a" * 64, "CO", "g1", "QUIET HOLDINGS LLC", "neither")]
    cases, splits = frames(rows)
    assert assign_screen_cells(cases, splits).tolist() == ["B4_remainder"]


def _fit_rows() -> list[dict]:
    """Enough repetition to clear the vectorisers' min_df, with a signal the mask cannot erase."""
    rows: list[dict] = []
    for i in range(40):
        rows.append(case(f"{i:064x}", "CO", f"p{i}", f"PIONEER RESOURCES {i} COMPANY", "lender"))
    for i in range(60):
        rows.append(case(f"{i + 500:064x}", "CO", f"n{i}", f"QUIET HOLDINGS {i} COMPANY", "neither"))
    return rows


def test_fit_uses_only_its_own_split_and_both_classes():
    rows = _fit_rows()
    held_out = {rows[0]["case_id"]: "test", rows[-1]["case_id"]: "validation"}
    cases, splits = frames(rows, held_out)
    s = fit_screener(cases, splits, fit_split="train")
    assert s.fit_split == "train"
    assert s.n_positive == 39 and s.n_negative == 59      # the held-out rows are not in the fit
    scores = s.score(pd.Series(["PIONEER RESOURCES 1 COMPANY", "QUIET HOLDINGS 1 COMPANY"]))
    assert scores[0] > scores[1]


def test_fit_refuses_when_a_class_is_missing():
    rows = [case(f"{i:064x}", "CO", f"n{i}", f"QUIET HOLDINGS {i}", "neither") for i in range(20)]
    cases, splits = frames(rows)
    with pytest.raises(ValueError, match="both classes"):
        fit_screener(cases, splits)


def test_no_rules_word_survives_into_the_strongest_features():
    """The mask is the difference between a screen that looks past the rules and one that relearns
    them. If a rules word reaches the features, the cell it selects is the rules' own vocabulary."""
    from ucc_ml.vendor.heavy_filter_v1 import BORROWER_RE, LENDER_RE

    rows = _fit_rows()
    rows += [case(f"{i + 900:064x}", "CO", f"x{i}", f"PIONEER EXCAVATING {i} COMPANY", "lender")
             for i in range(20)]
    cases, splits = frames(rows)
    s = fit_screener(cases, splits)
    for feature in strongest_features(s, k=50):
        upper = feature.upper()
        assert not BORROWER_RE.search(upper), feature
        assert not LENDER_RE.search(upper), feature


def test_cell_counts_report_every_split_column():
    rows = [case("a" * 64, "CO", "g1", "ALPHA LLC", "lender"),
            case("b" * 64, "CO", "g2", "QUIET HOLDINGS", "neither")]
    cases, splits = frames(rows, {"b" * 64: "test"})
    counts = cell_counts(cases, splits, assign_screen_cells(cases, splits))
    assert list(counts.columns) == ["region", "cell", "train", "validation", "test", "total"]
    assert int(counts.total.sum()) == len(cases)
