"""The blind queue carries five columns and nothing else; chunks hold their own hidden repeats; order is seeded."""
import hashlib
import re

import pandas as pd
import pytest

from conftest import balanced_case_rows, write_candidates_fixture
from test_config import MINIMAL


def _pilot(per_stratum=10):
    from ucc_ml.sampling import make_pilot

    return make_pilot(pd.DataFrame(balanced_case_rows(per_stratum + 2)), per_stratum=per_stratum, seed=20260912)


def test_queue_shape_blindness_and_hidden_repeats():
    from ucc_ml.labeling import KEY_COLUMNS, QUEUE_COLUMNS, build_blind_queue

    pilot = _pilot()                                                  # 40 cases
    chunks, key = build_blind_queue(pilot, "pilot_v1", repeat_fraction=0.10, seed=20260912, chunk_size=15)
    assert [len(c) for c in chunks] == [17, 17, 11]                  # 15+2, 15+2, 10+1 (round half up)
    assert list(key.columns) == list(KEY_COLUMNS) and len(key) == 45
    assert int(key.is_repeat.sum()) == 5 and sorted(key[~key.is_repeat].case_id) == sorted(pilot.case_id)
    for part, chunk in enumerate(chunks, start=1):
        assert list(chunk.columns) == list(QUEUE_COLUMNS) == ["case_id", "borrower_name_raw", "lender_names_raw", "city", "state"]
        assert chunk.case_id.is_unique                                 # aliases hide the repeats
        assert chunk.case_id.tolist() == key[key.part == part].queue_case_id.tolist()
    repeats = key[key.is_repeat]
    assert all(re.fullmatch(r"[0-9a-f]{64}", a) for a in repeats.queue_case_id)
    assert not set(repeats.queue_case_id) & set(pilot.case_id)
    part_of_original = key[~key.is_repeat].set_index("case_id").part
    assert all(part_of_original[c] == p for c, p in zip(repeats.case_id, repeats.part))   # repeats stay in their chunk
    assert chunks[0].lender_names_raw.iloc[0] == "FIRST BANK"
    assert set(pd.concat(chunks).state) <= {"CO", "CT"}


def test_queue_is_seeded_and_round_scoped():
    from ucc_ml.labeling import build_blind_queue, queue_rank, repeat_alias

    pilot = _pilot()
    c1, k1 = build_blind_queue(pilot, "pilot_v1", 0.10, seed=20260912, chunk_size=15)
    c2, _ = build_blind_queue(pilot, "pilot_v1", 0.10, seed=20260912, chunk_size=15)
    c3, _ = build_blind_queue(pilot, "pilot_v1", 0.10, seed=99, chunk_size=15)
    assert [c.case_id.tolist() for c in c1] == [c.case_id.tolist() for c in c2]
    assert [c.case_id.tolist() for c in c1] != [c.case_id.tolist() for c in c3]
    for _, rows in k1.groupby("part"):
        assert rows.queue_rank.tolist() == sorted(rows.queue_rank)
    assert k1.queue_rank.iloc[0] == queue_rank(20260912, "pilot_v1", k1.queue_case_id.iloc[0])
    cid = pilot.case_id.iloc[0]
    assert repeat_alias(20260912, "pilot_v1", cid) != repeat_alias(20260912, "main_v1", cid)
    assert k1.sampling_stratum.tolist() != sorted(k1.sampling_stratum)   # not grouped by stratum
    with pytest.raises(ValueError, match="unknown labelling round"):
        build_blind_queue(pilot, "pilot_v2", 0.10, seed=1, chunk_size=15)


def test_write_csv_and_readers_round_trip_and_refuse_bad_files(tmp_path):
    from ucc_ml.labeling import build_blind_queue, read_key, read_queue, write_csv

    chunks, key = build_blind_queue(_pilot(), "pilot_v1", 0.10, seed=20260912, chunk_size=40)
    a = write_csv(chunks[0], tmp_path / "q1.csv")
    b = write_csv(chunks[0], tmp_path / "q2.csv")
    assert a == b == hashlib.sha256((tmp_path / "q1.csv").read_bytes()).hexdigest()
    text = (tmp_path / "q1.csv").read_text(encoding="utf-8")
    assert text.splitlines()[0] == "case_id,borrower_name_raw,lender_names_raw,city,state" and "\r" not in text
    assert read_queue(tmp_path / "q1.csv").case_id.tolist() == chunks[0].case_id.tolist()
    write_csv(key, tmp_path / "k.csv")
    kb = read_key(tmp_path / "k.csv")
    assert kb.is_repeat.dtype == bool and int(kb.is_repeat.sum()) == 4 and kb.part.tolist() == [1] * 44
    (tmp_path / "bad.csv").write_text(text.replace("borrower_name_raw", "borrower", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="bad.csv"):
        read_queue(tmp_path / "bad.csv")
    bad_key = key.copy()
    bad_key["is_repeat"] = ["no" if not r else "True" for r in key.is_repeat]
    write_csv(bad_key, tmp_path / "k2.csv")
    with pytest.raises(ValueError, match="column 'is_repeat'.*'no'"):
        read_key(tmp_path / "k2.csv")


def test_round_paths_name_every_file(tmp_path):
    from ucc_ml.config import load_config
    from ucc_ml.labeling import round_paths

    cfg_path = tmp_path / "ml/configs/v1.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(MINIMAL)
    cfg = load_config(cfg_path)
    root = tmp_path.resolve()
    pilot, main = round_paths(cfg, "pilot_v1"), round_paths(cfg, "main_v1")
    assert pilot.cases == root / "ml/data/pilot/v1/pilot_cases.parquet"
    assert main.cases == root / "ml/data/main/v1/main_cases.parquet"
    assert pilot.queue_part(1) == root / "ml/data/pilot/v1/queue_pilot_v1_part_001.csv"
    assert main.queue_part(12).name == "queue_main_v1_part_012.csv" and main.key.name == "queue_key_main_v1.csv"
    assert main.raw_output("b", 7) == root / "ml/data/labels/v1/raw/labeller_output_main_v1_pass_b_part_007.csv"
    assert pilot.preregistration == root / "docs/data/ml/pilot_v1.sha256"
    assert main.passes_digest.name == "main_v1_passes.sha256" and pilot.founder_digest.name == "pilot_v1_founder_review.sha256"
    assert main.report == root / "ml/data/main/v1/main_report.json" and pilot.report.name == "pilot_report.json"
    with pytest.raises(ValueError, match="unknown labelling round"):
        round_paths(cfg, "main_v2")


def _repo_with_pilot(tmp_path, chunk_size=15):
    from ucc_ml.cli import main

    cfg = tmp_path / "ml/configs/v1.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(MINIMAL.replace("pilot_per_stratum: 50", "pilot_per_stratum: 10")
                   .replace("chunk_size: 200", f"chunk_size: {chunk_size}"))
    cand = tmp_path / "ml/data/candidates/v1/candidates.parquet"
    cand.parent.mkdir(parents=True)
    write_candidates_fixture(cand, balanced_case_rows(12))
    assert main(["make-pilot", "--config", str(cfg)]) == 0
    return cfg


def test_cli_label_blind_writes_chunks_key_and_preregistration(tmp_path, capsys):
    from ucc_ml.cli import main

    cfg = _repo_with_pilot(tmp_path)
    capsys.readouterr()
    assert main(["label-blind", "--config", str(cfg), "--round", "pilot_v1"]) == 0
    assert "round=pilot_v1 cases=40 repeats=5 parts=3 chunk_size=15" in capsys.readouterr().out
    pilot_dir = tmp_path / "ml/data/pilot/v1"
    names = ["pilot_cases.parquet", "pilot_manifest.json", "queue_key_pilot_v1.csv",
             "queue_pilot_v1_part_001.csv", "queue_pilot_v1_part_002.csv", "queue_pilot_v1_part_003.csv"]
    lines = (tmp_path / "docs/data/ml/pilot_v1.sha256").read_text().splitlines()
    assert lines[:6] == [f"{hashlib.sha256((pilot_dir / n).read_bytes()).hexdigest()}  {n}" for n in names]
    assert lines[6].startswith("# Committed BEFORE any pilot_v1 label exists")
    assert "model-labelled, founder-adjudicated" in "\n".join(lines[6:])
    assert main(["label-blind", "--config", str(cfg), "--round", "pilot_v1", "--chunk-size", "40"]) == 0
    assert "parts=1" in capsys.readouterr().out
    assert sorted(p.name for p in pilot_dir.glob("queue_pilot_v1_part_*.csv")) == ["queue_pilot_v1_part_001.csv"]


def test_cli_label_blind_refuses_once_a_label_exists(tmp_path, capsys):
    from ucc_ml.cli import main

    cfg = _repo_with_pilot(tmp_path)
    raw = tmp_path / "ml/data/labels/v1/raw"
    raw.mkdir(parents=True)
    (raw / "labeller_output_pilot_v1_pass_a_part_001.csv").write_text("case_id,label,reason_code,reason\n")
    assert main(["label-blind", "--config", str(cfg), "--round", "pilot_v1"]) == 1
    assert "REFUSED" in capsys.readouterr().out
