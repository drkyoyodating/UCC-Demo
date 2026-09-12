"""The live-API decision record (plan C Task 11): deploy needs a budget, skip takes none, nothing
private is written to the public repository, and --require is Task 12's precondition."""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("record_deploy_decision", REPO / "ml" / "tools" / "record_deploy_decision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_record_for_both_decisions(tool):
    deploy = tool.build_record("deploy", 10, "2026-09-12T10:00:00Z")
    assert deploy["decision"] == "deploy" and deploy["budget_alert_usd_per_month"] == 10.0
    assert deploy["cloud_run"] == {"service": "ucc-ml-api", "region": "us-central1", "memory": "1Gi", "cpu": 1,
                                   "min_instances": 0, "max_instances": 2, "concurrency": 8}
    skip = tool.build_record("skip", None, "2026-09-12T10:00:00Z")
    assert skip["decision"] == "skip" and "api_url null" in skip["note"]
    with pytest.raises(ValueError, match="--budget-usd greater than 0"):
        tool.build_record("deploy", None, "t")
    with pytest.raises(ValueError, match="--budget-usd greater than 0"):
        tool.build_record("deploy", 0, "t")
    with pytest.raises(ValueError, match="takes no --budget-usd"):
        tool.build_record("skip", 5, "t")
    with pytest.raises(ValueError, match="deploy or skip"):
        tool.build_record("later", None, "t")


def test_the_written_record_holds_nothing_private(tool, tmp_path):
    path = tmp_path / "deploy_decision.json"
    assert tool.main(["deploy", "--budget-usd", "25", "--path", str(path)]) == 0
    text = path.read_text(encoding="utf-8")
    assert set(json.loads(text)) == {"decision", "decided_at", "budget_alert_usd_per_month", "cloud_run", "note"}
    for private in ("project_id", "billing_account", "email", "@"):
        assert private not in text, private


def test_require_is_the_gate_task_12_checks(tool, tmp_path):
    path = tmp_path / "deploy_decision.json"
    assert tool.main(["--require", "deploy", "--path", str(path)]) == 3
    assert tool.main(["skip", "--path", str(path)]) == 0
    assert tool.main(["--require", "deploy", "--path", str(path)]) == 3
    assert tool.main(["--require", "skip", "--path", str(path)]) == 0
    assert tool.main(["deploy", "--path", str(path)]) == 2
    with pytest.raises(SystemExit):
        tool.main(["--require", "deploy", "skip", "--path", str(path)])
    with pytest.raises(SystemExit):
        tool.main(["--path", str(path)])
