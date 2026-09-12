"""Record the founder's live-API decision (plan C Task 11) in ml/deploy_decision.json.

    ./.venv-ml/bin/python ml/tools/record_deploy_decision.py skip
    ./.venv-ml/bin/python ml/tools/record_deploy_decision.py deploy --budget-usd 10
    ./.venv-ml/bin/python ml/tools/record_deploy_decision.py --require deploy     # Task 12's precondition

The repository is public, so the record never holds the Google Cloud project id, the billing
account or the alert recipients; those stay in the Cloud Console and the local gcloud config.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DECISION_FILE = REPO_ROOT / "ml" / "deploy_decision.json"
CLOUD_RUN = {"service": "ucc-ml-api", "region": "us-central1", "memory": "1Gi", "cpu": 1,
             "min_instances": 0, "max_instances": 2, "concurrency": 8}


def build_record(decision: str, budget_usd: float | None, decided_at: str) -> dict:
    if decision == "deploy":
        if budget_usd is None or not budget_usd > 0:
            raise ValueError("a deploy decision needs --budget-usd greater than 0 (the monthly budget alert amount)")
        return {
            "decision": "deploy",
            "decided_at": decided_at,
            "budget_alert_usd_per_month": float(budget_usd),
            "cloud_run": dict(CLOUD_RUN),
            "note": ("Budget alerts and max instances limit risk but are not hard spending caps. The project id, "
                     "billing account and alert recipients are deliberately not recorded in this public repository."),
        }
    if decision == "skip":
        if budget_usd is not None:
            raise ValueError("a skip decision takes no --budget-usd")
        return {
            "decision": "skip",
            "decided_at": decided_at,
            "note": ("The live API is not deployed. docs/data/ml/manifest.json keeps api_url null and the Lab page "
                     "says the API is unavailable."),
        }
    raise ValueError(f"decision must be deploy or skip, got {decision!r}")


def require(path: Path, expected: str) -> int:
    if not path.is_file():
        print(f"{path} does not exist: run the plan C Task 11 founder gate first", file=sys.stderr)
        return 3
    from ucc_ml.provenance import read_json

    actual = read_json(path).get("decision")
    if actual != expected:
        print(f"the recorded decision is {actual!r}, not {expected!r}", file=sys.stderr)
        return 3
    print(f"recorded decision: {actual}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="record_deploy_decision", description="Record or check the live-API decision.")
    parser.add_argument("decision", nargs="?", choices=("deploy", "skip"))
    parser.add_argument("--budget-usd", type=float, default=None, help="monthly budget alert amount (deploy only)")
    parser.add_argument("--require", choices=("deploy", "skip"), default=None,
                        help="check the recorded decision instead of writing one; exit 3 when it differs")
    parser.add_argument("--path", type=Path, default=DECISION_FILE)
    ns = parser.parse_args(argv)
    if ns.require is not None:
        if ns.decision is not None or ns.budget_usd is not None:
            parser.error("--require takes no decision and no --budget-usd")
        return require(ns.path, ns.require)
    if ns.decision is None:
        parser.error("give a decision (deploy or skip) or --require")
    from ucc_ml.provenance import utc_now_iso, write_json

    try:
        record = build_record(ns.decision, ns.budget_usd, utc_now_iso())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    write_json(ns.path, record)
    print(f"recorded {record['decision']} in {ns.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
