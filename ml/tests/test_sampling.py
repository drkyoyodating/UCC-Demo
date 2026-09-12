"""Seeded stratified draws are reproducible, honour exclusions, and record N_h / n_h / inclusion."""
import re

import pandas as pd
import pytest


def _cases():
    rows = []
    for region, qual, n in [("CO", True, 30), ("CO", False, 50), ("CT", True, 12), ("CT", False, 40)]:
        for i in range(n):
            rows.append({"case_id": f"{region}{int(qual)}{i:03d}".ljust(64, "0"), "region": region,
                         "baseline_qualifies": qual})
    return pd.DataFrame(rows)


def test_stratum_sizes_and_add_stratum():
    from ucc_ml.sampling import add_stratum, stratum_sizes

    assert stratum_sizes(_cases()) == {"CO:accepted": 30, "CO:rejected": 50, "CT:accepted": 12, "CT:rejected": 40}
    s = add_stratum(_cases())
    assert s.sampling_stratum.iloc[0] == "CO:accepted" and "sampling_stratum" not in _cases().columns


def test_draw_is_deterministic_and_correctly_sized():
    from ucc_ml.sampling import SAMPLE_COLUMNS, draw_rank, draw_stratified

    want = {"CO:accepted": 5, "CO:rejected": 5, "CT:accepted": 5, "CT:rejected": 5}
    a = draw_stratified(_cases(), want, seed=20260912, purpose="pilot_v1")
    b = draw_stratified(_cases(), want, seed=20260912, purpose="pilot_v1")
    c = draw_stratified(_cases(), want, seed=1, purpose="pilot_v1")
    assert list(a.columns) == list(SAMPLE_COLUMNS)
    assert a.case_id.tolist() == b.case_id.tolist() and a.case_id.tolist() != c.case_id.tolist()
    assert len(a) == 20 and a.case_id.is_unique
    assert a.groupby("sampling_stratum").size().to_dict() == want
    row = a[a.sampling_stratum == "CT:accepted"].iloc[0]
    assert row.N_h == 12 and row.pool_h == 12 and row.n_h == 5 and row.inclusion_probability == pytest.approx(5 / 12)
    assert re.fullmatch(r"[0-9a-f]{64}", row.draw_rank)
    assert row.draw_rank == draw_rank(20260912, "pilot_v1", row.case_id)
    # within a stratum the drawn ranks are the smallest in the population
    ct = a[a.sampling_stratum == "CT:accepted"].draw_rank.tolist()
    all_ct = sorted(draw_rank(20260912, "pilot_v1", cid) for cid in _cases().query("region == 'CT' and baseline_qualifies").case_id)
    assert sorted(ct) == all_ct[:5]


def test_exclusions_change_pool_not_population():
    from ucc_ml.sampling import draw_stratified

    first = draw_stratified(_cases(), {"CT:accepted": 5}, seed=7, purpose="p")
    second = draw_stratified(_cases(), {"CT:accepted": 5}, seed=7, purpose="p", exclude_case_ids=first.case_id)
    assert not set(first.case_id) & set(second.case_id)
    assert second.N_h.iloc[0] == 12 and second.pool_h.iloc[0] == 7 and second.inclusion_probability.iloc[0] == pytest.approx(5 / 7)


def test_shortage_and_unknown_stratum_raise():
    from ucc_ml.sampling import draw_stratified

    with pytest.raises(ValueError, match="CT:accepted"):
        draw_stratified(_cases(), {"CT:accepted": 13}, seed=1, purpose="p")
    with pytest.raises(ValueError, match="unknown stratum"):
        draw_stratified(_cases(), {"CT:maybe": 1}, seed=1, purpose="p")
