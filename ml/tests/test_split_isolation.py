"""Zero group overlap, zero case overlap, every pilot group in train (K12), ratios honoured, digest stable."""
import hashlib
import json

import pandas as pd
import pytest

from conftest import balanced_case_rows, make_case_row, write_candidates_fixture
from test_config import MINIMAL

RATIOS = {"train": 0.65, "validation": 0.15, "test": 0.20}


def _cases():
    rows = balanced_case_rows(100)                                   # 400 cases, 400 groups
    # 20 extra filings for an existing borrower -> one group with 21 cases
    rows += [make_case_row("CO", f"COX{i:03d}", "ACME EXCAVATION 7 LLC", ["FIRST BANK"]) for i in range(20)]
    return pd.DataFrame(rows)


def test_assign_splits_ratios_and_pilot_groups_go_to_train():
    from ucc_ml.splitting import assign_splits

    groups = [f"g{i:04d}" for i in range(1000)]
    pilot = set(groups[:40])
    a = assign_splits(groups, pilot, RATIOS, seed=20260912)
    assert set(a) == set(groups)
    counts = pd.Series(a).value_counts()
    assert (counts["test"], counts["validation"], counts["train"]) == (200, 150, 650)
    assert all(a[g] == "train" for g in pilot)                      # contract K12
    assert a == assign_splits(groups, pilot, RATIOS, seed=20260912)
    assert a != assign_splits(groups, pilot, RATIOS, seed=1)


def test_freeze_splits_isolation_and_digest():
    from ucc_ml.splitting import audit_splits, freeze_splits, split_digest

    cases = _cases()
    pilot_ids = cases.case_id.iloc[:30].tolist()                    # includes ACME EXCAVATION 7 LLC (21 cases)
    splits, manifest = freeze_splits(cases, pilot_ids, RATIOS, seed=20260912,
                                     label_policy_version="label_policy_v1", candidates_sha256="abc")
    assert list(splits.columns) == ["case_id", "group_id", "split"]
    assert splits.case_id.tolist() == sorted(cases.case_id)
    assert (splits.groupby("group_id").split.nunique() == 1).all()  # zero group overlap
    assert splits.case_id.is_unique                                  # zero case overlap
    pilot_groups = set(cases[cases.case_id.isin(pilot_ids)].group_id)
    assert (splits[splits.group_id.isin(pilot_groups)].split == "train").all()
    big = cases[cases.borrower_name_raw == "ACME EXCAVATION 7 LLC"].case_id
    assert splits[splits.case_id.isin(big)].split.unique().tolist() == ["train"]
    audit = audit_splits(cases, splits, pilot_ids)
    for key in ("group_overlap", "case_overlap", "pilot_cases_in_test", "pilot_cases_not_in_train",
                "same_norm_name_across_splits_within_region"):
        assert audit[key] == 0, key
    assert manifest["groups"]["total"] == 400 and manifest["groups"]["pilot"] == len(pilot_groups)
    assert manifest["cases"]["pilot_by_split"] == {"train": 30, "validation": 0, "test": 0}
    by_ss = manifest["cases"]["by_split_stratum"]                 # K13 N_h per (split, stratum)
    assert set(by_ss) == {"train", "validation", "test"}
    assert all(sum(by_ss[s].values()) == manifest["cases"]["by_split"][s] for s in by_ss)
    assert manifest["pilot_rule"] == "every group containing a pilot case is assigned train (contract K12)"
    assert manifest["digest"] == split_digest(splits)
    lines = "\n".join(f"{r.case_id},{r.group_id},{r.split}" for r in splits.itertuples()) + "\n"
    assert manifest["digest"] == hashlib.sha256(lines.encode()).hexdigest()
    changed = splits.copy()
    changed.loc[0, "split"] = "test" if changed.loc[0, "split"] != "test" else "train"
    assert split_digest(changed) != manifest["digest"]


def test_cli_freeze_splits(tmp_path, capsys):
    from ucc_ml.cli import main
    from ucc_ml.dataset import write_frame
    from ucc_ml.splitting import read_splits

    cfg = tmp_path / "ml/configs/v1.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(MINIMAL)
    cand = tmp_path / "ml/data/candidates/v1/candidates.parquet"
    cand.parent.mkdir(parents=True)
    cases = write_candidates_fixture(cand, balanced_case_rows(50))
    pilot = cases.head(12).copy()
    pilot["sampling_stratum"] = "CO:accepted"
    write_frame(pilot, tmp_path / "ml/data/pilot/v1/pilot_cases.parquet")
    assert main(["freeze-splits", "--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "groups=200" in out and "pilot_cases_in_test=0" in out and "pilot_cases_not_in_train=0" in out
    splits = read_splits(tmp_path / "ml/data/splits/v1/splits.parquet")
    assert len(splits) == 200
    assert set(splits[splits.case_id.isin(pilot.case_id)].split) == {"train"}
    m = json.loads((tmp_path / "ml/data/splits/v1/split_manifest.json").read_text())
    digest_file = (tmp_path / "docs/data/ml/splits_v1.sha256").read_text().splitlines()
    assert digest_file[0].split()[0] == hashlib.sha256((tmp_path / "ml/data/splits/v1/splits.parquet").read_bytes()).hexdigest()
    assert digest_file[1].split()[0] == m["digest"]
    assert digest_file[2].startswith("#")


def test_freeze_splits_refuses_unknown_pilot_ids():
    from ucc_ml.splitting import freeze_splits

    with pytest.raises(ValueError, match="pilot case_ids not in candidates"):
        freeze_splits(_cases(), ["f" * 64], RATIOS, seed=1, label_policy_version="p", candidates_sha256="x")


def _break_one_group(splits: pd.DataFrame) -> pd.DataFrame:
    sizes = splits.group_id.value_counts()
    group = sizes[sizes > 1].index[0]
    out = splits.copy()
    row = out.index[out.group_id == group][0]
    out.loc[row, "split"] = "validation" if out.loc[row, "split"] != "validation" else "test"
    return out


@pytest.mark.parametrize("mutate, message", [
    (lambda s: s.assign(split=["holdout"] + s.split.tolist()[1:]), "column 'split'.*'holdout'"),
    (lambda s: s.assign(case_id=["nope"] + s.case_id.tolist()[1:]), "column 'case_id'.*'nope'"),
    (lambda s: s.assign(group_id=["G1"] + s.group_id.tolist()[1:]), "column 'group_id'.*'G1'"),
    (lambda s: pd.concat([s, s.head(1)], ignore_index=True), "duplicate"),
    (_break_one_group, "differs within"),
])
def test_read_splits_names_file_column_and_values(tmp_path, mutate, message):
    from ucc_ml.dataset import write_frame
    from ucc_ml.splitting import freeze_splits, read_splits

    splits, _ = freeze_splits(_cases(), [], RATIOS, seed=1, label_policy_version="p", candidates_sha256="x")
    write_frame(splits, tmp_path / "good.parquet")
    assert read_splits(tmp_path / "good.parquet").case_id.tolist() == splits.case_id.tolist()
    write_frame(mutate(splits), tmp_path / "bad.parquet")
    with pytest.raises(ValueError, match=message) as exc:
        read_splits(tmp_path / "bad.parquet")
    assert "bad.parquet" in str(exc.value)
