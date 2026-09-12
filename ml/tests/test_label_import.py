"""import-labels: both passes checked against the policy's reason codes and de-aliased through the private key."""
import hashlib
import re

import pytest

from conftest import labelling_repo, run_blind_passes


@pytest.fixture(scope="module")
def labelled_pilot(tmp_path_factory):
    cfg = labelling_repo(tmp_path_factory.mktemp("repo"))
    run_blind_passes(cfg, "pilot_v1")
    return cfg


def _inputs(cfg):
    from ucc_ml.config import load_config
    from ucc_ml.labeling import queue_parts, read_key, read_queue, read_raw_output, round_paths

    rp = round_paths(load_config(cfg), "pilot_v1")
    key = read_key(rp.key)
    parts = queue_parts(key)
    chunks = {p: read_queue(rp.queue_part(p)) for p in parts}
    raws = {p: read_raw_output(rp.raw_output("a", p)) for p in parts}
    stamps = {p: "2026-09-12T10:00:00Z" for p in parts}
    return rp, key, chunks, raws, stamps


def test_import_pass_de_aliases_hidden_repeats(labelled_pilot):
    from ucc_ml.contracts import LABELLER_PASS_A
    from ucc_ml.labeling import PASS_COLUMNS, import_pass

    _, key, chunks, raws, stamps = _inputs(labelled_pilot)
    df = import_pass(key, chunks, raws, "a", "pilot_v1", stamps, 300)
    assert list(df.columns) == list(PASS_COLUMNS) and len(df) == len(key)
    originals, repeats = df[~df.is_repeat], df[df.is_repeat]
    assert originals.case_id.is_unique and (originals.queue_case_id == originals.case_id).all()
    assert len(repeats) == int(key.is_repeat.sum()) > 0
    assert set(repeats.case_id) <= set(originals.case_id)
    assert not set(repeats.queue_case_id) & set(originals.case_id)
    assert (df.labeller_id == LABELLER_PASS_A).all() and (df.labelling_round == "pilot_v1").all()
    assert (df.labelled_at == "2026-09-12T10:00:00Z").all()
    by_case = originals.set_index("case_id")
    for r in repeats.itertuples():
        assert r.label == by_case.at[r.case_id, "label"]
        assert r.sampling_stratum == by_case.at[r.case_id, "sampling_stratum"]
    expected = key.set_index("queue_case_id").inclusion_probability
    assert all(abs(p - expected[q]) < 1e-12 for q, p in zip(df.queue_case_id, df.inclusion_probability))


@pytest.mark.parametrize("breakage, message", [
    ("wrong_family", "pilot_v1 pass a part 001: row 1: reason_code 'I_GENERIC_NAME' is not valid for RELEVANT"),
    ("missing_part", "labeller output missing for part(s) [2]"),
    ("extra_part", "labeller output for unknown part(s) [9]"),
    ("reordered", "part 001: case_id sequence differs from the queue chunk"),
    ("url", "reason contains a URL"),
    ("chunk_key_mismatch", "part 001: the queue chunk and the private key disagree"),
])
def test_import_pass_refuses_bad_outputs(labelled_pilot, breakage, message):
    from ucc_ml.labeling import import_pass

    _, key, chunks, raws, stamps = _inputs(labelled_pilot)
    raws, chunks = dict(raws), dict(chunks)
    first = raws[1].copy()
    if breakage == "wrong_family":
        first.loc[0, "label"] = "RELEVANT"
        first.loc[0, "reason_code"] = "I_GENERIC_NAME"
        raws[1] = first
    elif breakage == "missing_part":
        del raws[2]
    elif breakage == "extra_part":
        raws[9] = first
    elif breakage == "reordered":
        raws[1] = first.iloc[::-1].reset_index(drop=True)
    elif breakage == "url":
        first.loc[0, "reason"] = "www.example.com says so"
        raws[1] = first
    elif breakage == "chunk_key_mismatch":
        chunks[1] = chunks[1].iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match=re.escape(message)):
        import_pass(key, chunks, raws, "a", "pilot_v1", stamps, 300)


def test_read_pass_file_round_trip_and_refusal(labelled_pilot, tmp_path):
    from ucc_ml.labeling import import_pass, read_pass_file, write_csv

    _, key, chunks, raws, stamps = _inputs(labelled_pilot)
    df = import_pass(key, chunks, raws, "a", "pilot_v1", stamps, 300)
    write_csv(df, tmp_path / "pass_a.csv")
    back = read_pass_file(tmp_path / "pass_a.csv")
    assert back.is_repeat.dtype == bool and back.part.dtype.kind == "i"
    assert back.case_id.tolist() == df.case_id.tolist() and back.label.tolist() == df.label.tolist()
    write_csv(df.assign(labelling_round="pilot_v9"), tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="column 'labelling_round'.*'pilot_v9'"):
        read_pass_file(tmp_path / "bad.csv")


def test_cli_import_labels_writes_passes_and_digest(tmp_path, capsys):
    from ucc_ml.cli import main
    from ucc_ml.config import load_config
    from ucc_ml.labeling import read_pass_file, round_paths

    cfg = labelling_repo(tmp_path)
    rp = round_paths(load_config(cfg), "pilot_v1")
    capsys.readouterr()
    assert main(["import-labels", "--config", str(cfg), "--round", "pilot_v1"]) == 1
    assert "REFUSED: pilot_v1: labeller output missing" in capsys.readouterr().out
    run_blind_passes(cfg, "pilot_v1")
    capsys.readouterr()
    assert main(["import-labels", "--config", str(cfg), "--round", "pilot_v1"]) == 0
    out = capsys.readouterr().out
    assert "pass a: cases=40 repeats=5" in out and "pass b: cases=40 repeats=5" in out
    a, b = read_pass_file(rp.pass_file("a")), read_pass_file(rp.pass_file("b"))
    assert a.case_id.tolist() == b.case_id.tolist()
    lines = rp.passes_digest.read_text().splitlines()
    names = [f"labeller_output_pilot_v1_pass_{letter}_part_{part:03d}.csv" for letter in "ab" for part in (1, 2, 3)]
    names += ["pass_a_pilot_v1.csv", "pass_b_pilot_v1.csv"]
    assert [line.split("  ", 1)[1] for line in lines[:8]] == names
    assert lines[0].split("  ", 1)[0] == hashlib.sha256(rp.raw_output("a", 1).read_bytes()).hexdigest()
    assert lines[8].startswith("# The pilot_v1 blind labeller outputs")
