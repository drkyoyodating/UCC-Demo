"""validate-labels: one labels.csv for every round (K3), read_labels (K7), the manifest and the round report."""
import json

import pandas as pd
import pytest

from conftest import imported_pilot_repo, reviewed_pilot_repo

A, B, C, D, E = ("a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64)
CODES = {"RELEVANT": "R_BORROWER_TRADE_WORD", "NOT_RELEVANT": "N_OTHER_INDUSTRY_EXPLICIT",
         "INSUFFICIENT_EVIDENCE": "I_GENERIC_NAME"}
LABELLERS = {"model_agreed": "claude_blind_pass_a+claude_blind_pass_b", "founder_confirmed": "founder",
             "founder_adjudicated": "founder", "blind_repeat": "claude_blind_pass_a"}


def _row(case_id, status="model_agreed", round_name="pilot_v1", label="RELEVANT", stratum="CO:accepted", labeller=None):
    return {"case_id": case_id, "label": label,
            "reason_code": "ADJUDICATED" if status == "founder_adjudicated" else CODES[label],
            "reason": "the deciding words", "labeller_id": labeller or LABELLERS[status],
            "labelled_at": "2026-09-12T10:00:00Z", "policy_version": "label_policy_v1", "sampling_stratum": stratum,
            "inclusion_probability": 0.25, "adjudication_status": status, "is_repeat": status == "blind_repeat",
            "labelling_round": round_name}


def _frame(rows):
    from ucc_ml.contracts import LABEL_COLUMNS

    return pd.DataFrame(rows, columns=list(LABEL_COLUMNS))


PILOT = _frame([_row(A), _row(B, "founder_confirmed", stratum="CT:rejected"),
                _row(C, "founder_adjudicated", label="NOT_RELEVANT", stratum="CT:rejected"),
                _row(A, "blind_repeat"), _row(A, "blind_repeat", labeller="claude_blind_pass_b")])
MAIN = _frame([_row(D, round_name="main_v1", label="INSUFFICIENT_EVIDENCE", stratum="CO:rejected"),
               _row(E, "founder_adjudicated", round_name="main_v1", stratum="CO:rejected")])


def test_build_write_and_read_labels_round_trip(tmp_path):
    from ucc_ml.contracts import LABEL_COLUMNS, Label
    from ucc_ml.labeling import build_labels, read_labels, write_csv

    labels = build_labels({"main_v1": MAIN, "pilot_v1": PILOT})
    assert labels.labelling_round.tolist() == ["pilot_v1"] * 5 + ["main_v1"] * 2
    write_csv(labels, tmp_path / "labels.csv")
    assert (tmp_path / "labels.csv").read_text().splitlines()[0] == ",".join(LABEL_COLUMNS)
    back = read_labels(tmp_path / "labels.csv")
    assert list(back.columns) == list(LABEL_COLUMNS)
    assert back.is_repeat.dtype == bool and back.inclusion_probability.dtype == float
    assert back.is_repeat.tolist() == [False, False, False, True, True, False, False]
    assert back.case_id.tolist() == labels.case_id.tolist()
    for record in back.to_dict("records"):
        Label.model_validate(record)


@pytest.mark.parametrize("frames, message", [
    ({"pilot_v1": PILOT, "main_v1": pd.concat([MAIN, _frame([_row(A, round_name="main_v1")])])}, "labelled more than once"),
    ({"pilot_v1": PILOT.iloc[1:]}, "without an original row"),
    ({"pilot_v1": pd.concat([PILOT, PILOT.tail(1)])}, "repeats a blind_repeat row"),
    ({"pilot_v2": PILOT}, "unknown labelling round"),
    ({"pilot_v1": MAIN}, "carry labelling_round"),
])
def test_build_labels_refuses_inconsistent_rounds(frames, message):
    from ucc_ml.labeling import build_labels

    with pytest.raises(ValueError, match=message):
        build_labels(frames)


@pytest.mark.parametrize("column, value, message", [
    ("label", "MAYBE", "column 'label'.*'MAYBE'"),
    ("adjudication_status", "pending", "column 'adjudication_status'.*'pending'"),
    ("adjudication_status", "adjudicated", "column 'adjudication_status'.*'adjudicated'"),
    ("labelling_round", "pilot_v2", "column 'labelling_round'.*'pilot_v2'"),
    ("is_repeat", "yes", "column 'is_repeat'.*'yes'"),
    ("case_id", "nope", "column 'case_id'.*'nope'"),
    ("inclusion_probability", "0", "column 'inclusion_probability'.*'0'"),
    ("sampling_stratum", "CO|accepted", r"column 'sampling_stratum'.*'CO\|accepted'"),
    ("is_repeat", "True", "disagrees with adjudication_status"),
])
def test_read_labels_names_file_column_and_values(tmp_path, column, value, message):
    from ucc_ml.labeling import build_labels, read_labels, write_csv

    labels = build_labels({"pilot_v1": PILOT}).astype({"is_repeat": object, "inclusion_probability": object})
    labels.loc[0, column] = value
    write_csv(labels, tmp_path / "labels.csv")
    with pytest.raises(ValueError, match=message) as exc:
        read_labels(tmp_path / "labels.csv")
    assert "labels.csv" in str(exc.value)


AGREEMENT = {"pass_agreement": {"n": 3, "agreed": 2, "rate": 2 / 3, "by_stratum": {}},
             "repeat_consistency": {"pass_a": {"n": 1, "consistent": 1, "rate": 1.0},
                                    "pass_b": {"n": 1, "consistent": 0, "rate": 0.0}}}
FOUNDER = {"disagreements": {"n": 1, "decided": 1, "undecided": 0},
           "audit": {"n_selected": 2, "n_audited": 2, "n_confirmed": 1, "n_overturned": 1, "agreement_rate": 0.5}}


def test_labels_manifest_carries_the_k3_statistics():
    from ucc_ml.labeling import build_labels, labels_manifest

    labels = build_labels({"pilot_v1": PILOT, "main_v1": MAIN})
    manifest = labels_manifest(labels, "f" * 64, {"pilot_v1": AGREEMENT, "main_v1": AGREEMENT},
                               {"pilot_v1": FOUNDER, "main_v1": FOUNDER}, "label_policy_v1", extra={"git_head": None})
    for key in ("policy_version", "disclosure", "labels_sha256", "rows", "counts_by_status", "counts_by_round",
                "pass_agreement", "founder_audit", "repeat_consistency"):
        assert key in manifest, key
    assert manifest["disclosure"] == "model-labelled, founder-adjudicated" and manifest["rows"] == 7
    assert manifest["counts_by_status"] == {"model_agreed": 2, "founder_confirmed": 1, "founder_adjudicated": 2,
                                            "blind_unresolved": 0, "blind_repeat": 2}
    assert manifest["counts_by_round"] == {"pilot_v1": 5, "main_v1": 2} and manifest["cases_by_round"] == {"pilot_v1": 3, "main_v1": 2}
    assert manifest["pass_agreement"]["main_v1"] == {"n": 3, "agreed": 2, "rate": 2 / 3}
    assert manifest["founder_audit"]["pilot_v1"]["agreement_rate"] == 0.5
    assert manifest["repeat_consistency"] == {"pass_a": {"n": 2, "consistent": 2, "rate": 1.0},
                                              "pass_b": {"n": 2, "consistent": 0, "rate": 0.0}}
    assert manifest["git_head"] is None
    json.dumps(manifest, allow_nan=False)


def test_round_report_is_k13_1():
    from ucc_ml.labeling import build_labels, round_report

    report = round_report(build_labels({"pilot_v1": PILOT, "main_v1": MAIN}), AGREEMENT, FOUNDER, "pilot_v1")
    assert report["cases"] == 3 and report["disclosure"] == "model-labelled, founder-adjudicated"
    assert report["counts_by_status"] == {"model_agreed": 1, "founder_confirmed": 1, "founder_adjudicated": 1,
                                          "blind_unresolved": 0, "blind_repeat": 2}
    assert report["labelability"]["CT:rejected"] == {"n": 2, "insufficient": 0, "insufficient_share": 0.0}
    assert report["relevant_prevalence"]["CT:rejected"] == {"n": 2, "relevant": 1, "relevant_share": 0.5,
                                                            "resolved": 2, "relevant_share_of_resolved": 0.5}
    assert report["labelability"]["CO:rejected"] == {"n": 0, "insufficient": 0, "insufficient_share": None}
    assert report["founder_audit"] == FOUNDER["audit"] and report["repeat_consistency"] == AGREEMENT["repeat_consistency"]
    json.dumps(report, allow_nan=False)


def test_cli_validate_labels_writes_labels_manifest_report_and_digest(tmp_path, capsys):
    import hashlib

    from ucc_ml.cli import main
    from ucc_ml.config import artefact_paths, load_config
    from ucc_ml.labeling import read_labels

    cfg = reviewed_pilot_repo(tmp_path)
    capsys.readouterr()
    assert main(["validate-labels", "--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "pilot_v1: cases=40" in out and "model-labelled, founder-adjudicated" in out
    paths = artefact_paths(load_config(cfg))
    labels = read_labels(paths.labels_csv)
    counts = labels.adjudication_status.value_counts().to_dict()
    assert counts == {"model_agreed": 35, "founder_adjudicated": 4, "founder_confirmed": 1, "blind_repeat": 10}
    assert set(labels.labelling_round) == {"pilot_v1"} and len(labels) == 50
    manifest = json.loads(paths.labels_manifest.read_text())
    assert manifest["labels_sha256"] == hashlib.sha256(paths.labels_csv.read_bytes()).hexdigest()
    assert manifest["counts_by_status"]["blind_repeat"] == 10 and manifest["rounds"] == ["pilot_v1"]
    assert manifest["founder_audit"]["pilot_v1"]["n_audited"] == 2
    assert set(manifest["inputs_sha256"]) == {"ml/data/labels/v1/passes/pass_a_pilot_v1.csv",
                                              "ml/data/labels/v1/passes/pass_b_pilot_v1.csv",
                                              "ml/data/labels/v1/founder_review_pilot_v1.json",
                                              "ml/data/labels/v1/raw/founder_decisions_pilot_v1.csv"}
    report = json.loads((tmp_path / "ml/data/pilot/v1/pilot_report.json").read_text())
    assert {"labelability", "relevant_prevalence", "pass_agreement", "founder_audit", "repeat_consistency"} <= set(report)
    digest = (paths.public_data_dir / "labels_v1.sha256").read_text().splitlines()
    assert digest[0] == f"{manifest['labels_sha256']}  labels.csv" and digest[1].endswith("  labels_manifest.json")
    assert "model-labelled, founder-adjudicated" in digest[2]


def test_cli_validate_labels_refuses_undecided_or_unreviewed_rounds(tmp_path, capsys):
    from ucc_ml.cli import main

    undecided = reviewed_pilot_repo(tmp_path / "undecided", decide_all=False)
    capsys.readouterr()
    assert main(["validate-labels", "--config", str(undecided)]) == 1
    assert "REFUSED: pilot_v1: 1 disagreement(s) have no founder decision" in capsys.readouterr().out
    assert not (tmp_path / "undecided/ml/data/labels/v1/labels.csv").exists()
    unreviewed, _ = imported_pilot_repo(tmp_path / "unreviewed")
    capsys.readouterr()
    assert main(["validate-labels", "--config", str(unreviewed)]) == 1
    assert "founder review manifest" in capsys.readouterr().out
