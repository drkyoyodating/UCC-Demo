"""K14: a brief carries the policy and one chunk inline; structured output is validated before a raw CSV exists."""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from conftest import fake_structured_output, labelling_repo, run_blind_passes

SPECS = Path(__file__).resolve().parents[1] / "specs"


@pytest.fixture(scope="module")
def pilot_repo(tmp_path_factory):
    from ucc_ml.config import load_config
    from ucc_ml.labeling import read_key, read_queue, round_paths

    cfg = labelling_repo(tmp_path_factory.mktemp("repo"))
    rp = round_paths(load_config(cfg), "pilot_v1")
    return cfg, rp, read_queue(rp.queue_part(1)), read_key(rp.key)


def test_schema_in_code_matches_the_prompt_document():
    from ucc_ml.contracts import LABELS, REASON_CODES
    from ucc_ml.labeling import LABELLER_OUTPUT_SCHEMA

    text = (SPECS / "labeller_prompt_v1.md").read_text(encoding="utf-8")
    block = re.search(r"```json\n(.*?)\n```", text, re.S).group(1)
    assert json.loads(block) == LABELLER_OUTPUT_SCHEMA
    item = LABELLER_OUTPUT_SCHEMA["properties"]["rows"]["items"]["properties"]
    assert item["label"]["enum"] == list(LABELS)
    assert set(item["reason_code"]["enum"]) == {c for label in LABELS for c in REASON_CODES[label]}
    assert "ADJUDICATED" not in item["reason_code"]["enum"] and item["reason"]["maxLength"] == 300


def test_brief_is_identical_for_both_passes_and_carries_only_the_chunk(pilot_repo):
    from ucc_ml.labeling import render_labeller_brief

    _, _, chunk, key = pilot_repo
    prompt = (SPECS / "labeller_prompt_v1.md").read_text(encoding="utf-8")
    policy = "policy_version: label_policy_v1\nstatus: FROZEN\nThe rules.\n"
    brief = render_labeller_brief(prompt, policy, chunk)
    assert brief == render_labeller_brief(prompt, policy, chunk)
    assert brief.startswith("You are a blind labeller") and "The rules." in brief
    assert "case_id,borrower_name_raw,lender_names_raw,city,state" in brief
    assert all(queue_id in brief for queue_id in key[key.part == 1].queue_case_id)
    assert "How the orchestrator runs a blind pass" not in brief
    for private in ("queue_rank", "inclusion_probability", "baseline_qualifies", "sampling_stratum", "is_repeat"):
        assert private not in brief, private
    with pytest.raises(ValueError, match="BRIEF"):
        render_labeller_brief("no marker here", policy, chunk)
    with pytest.raises(ValueError, match="exactly the columns"):
        render_labeller_brief(prompt, policy, chunk.assign(stratum="CO:accepted"))


def test_rows_from_structured_output_is_strict():
    from ucc_ml.labeling import RAW_OUTPUT_COLUMNS, rows_from_structured_output

    row = {"case_id": "a" * 64, "label": "RELEVANT", "reason_code": "R_BORROWER_TRADE_WORD", "reason": "CONCRETE"}
    df = rows_from_structured_output({"rows": [row]})
    assert list(df.columns) == list(RAW_OUTPUT_COLUMNS) and df.iloc[0].reason == "CONCRETE"
    for bad in ([row], {"rows": [row], "note": "x"}, {"rows": [{"case_id": "a" * 64, "label": "RELEVANT"}]},
                {"rows": [dict(row, note="x")]}, {"rows": [dict(row, reason=3)]}):
        with pytest.raises(ValueError):
            rows_from_structured_output(bad)


def _first(values, value):
    return [value] + list(values)[1:]


@pytest.mark.parametrize("mutate, problem", [
    (lambda r: r.assign(label=_first(r.label, "RELEVANT"), reason_code=_first(r.reason_code, "N_OTHER_INDUSTRY_EXPLICIT")),
     "not valid for RELEVANT"),
    (lambda r: r.assign(label=_first(r.label, "MAYBE")), "label 'MAYBE'"),
    (lambda r: r.assign(reason=_first(r.reason, " ")), "reason is blank"),
    (lambda r: r.assign(reason=_first(r.reason, "x" * 301)), "301 characters"),
    (lambda r: r.assign(reason=_first(r.reason, "see https://example.com")), "URL"),
    (lambda r: r.assign(reason=_first(r.reason, "two\nlines")), "line break"),
    (lambda r: r.iloc[::-1].reset_index(drop=True), "differs from the queue chunk"),
    (lambda r: r.iloc[1:].reset_index(drop=True), "missing ['"),
    (lambda r: pd.concat([r, r.head(1)], ignore_index=True), "duplicated ['"),
])
def test_validate_labeller_rows_names_each_problem(pilot_repo, mutate, problem):
    from ucc_ml.labeling import rows_from_structured_output, validate_labeller_rows

    _, _, chunk, _ = pilot_repo
    rows = rows_from_structured_output(fake_structured_output(chunk))
    assert validate_labeller_rows(rows, chunk.case_id.tolist(), 300) == []
    problems = validate_labeller_rows(mutate(rows), chunk.case_id.tolist(), 300)
    assert any(problem in p for p in problems), problems


def test_write_raw_output_writes_the_k14_header_and_never_overwrites(pilot_repo, tmp_path):
    from ucc_ml.labeling import read_raw_output, rows_from_structured_output, write_raw_output

    _, _, chunk, _ = pilot_repo
    rows = rows_from_structured_output(fake_structured_output(chunk))
    target = tmp_path / "labeller_output_pilot_v1_pass_a_part_001.csv"
    with pytest.raises(ValueError, match="labeller_output_pilot_v1_pass_a_part_001.csv"):
        write_raw_output(rows.iloc[1:], target, chunk.case_id.tolist(), 300)
    assert not target.exists()
    write_raw_output(rows, target, chunk.case_id.tolist(), 300)
    assert target.read_text(encoding="utf-8").splitlines()[0] == "case_id,label,reason_code,reason"
    assert read_raw_output(target).case_id.tolist() == chunk.case_id.tolist()
    with pytest.raises(FileExistsError):
        write_raw_output(rows, target, chunk.case_id.tolist(), 300)


def test_cli_brief_write_raw_labels_and_status(tmp_path, capsys):
    from ucc_ml.cli import main
    from ucc_ml.config import load_config
    from ucc_ml.labeling import LABELLER_OUTPUT_SCHEMA, queue_parts, read_key, round_paths

    cfg = labelling_repo(tmp_path)
    rp = round_paths(load_config(cfg), "pilot_v1")
    assert queue_parts(read_key(rp.key)) == [1, 2, 3]
    capsys.readouterr()
    assert main(["labelling-status", "--config", str(cfg), "--round", "pilot_v1"]) == 1
    assert "INCOMPLETE" in capsys.readouterr().out
    assert main(["labeller-brief", "--config", str(cfg), "--round", "pilot_v1", "--part", "1"]) == 0
    assert rp.brief(1).read_text(encoding="utf-8").startswith("You are a blind labeller")
    assert json.loads((rp.briefs_dir / "labeller_output_schema_v1.json").read_text()) == LABELLER_OUTPUT_SCHEMA
    assert main(["labeller-brief", "--config", str(cfg), "--round", "pilot_v1", "--part", "9"]) == 1
    run_blind_passes(cfg, "pilot_v1")
    capsys.readouterr()
    assert main(["labelling-status", "--config", str(cfg), "--round", "pilot_v1"]) == 0
    assert "COMPLETE" in capsys.readouterr().out
    again = rp.structured_output("a", 1)
    assert main(["write-raw-labels", "--config", str(cfg), "--round", "pilot_v1", "--pass", "a", "--part", "1",
                 "--structured", str(again)]) == 1
    assert "REJECTED" in capsys.readouterr().out
    policy = tmp_path / "ml/specs/label_policy_v1.md"
    policy.write_text(policy.read_text().replace("status: FROZEN", "status: DRAFT"))
    assert main(["labeller-brief", "--config", str(cfg), "--round", "pilot_v1", "--part", "1"]) == 1
    assert "REFUSED" in capsys.readouterr().out
