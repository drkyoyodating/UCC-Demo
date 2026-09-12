"""ml/DEPLOY.md is the operating record for the public side of the Lab (plan C Task 7). These checks
pin the commands that must not drift: the Pages source check, the scoped gh login, the release
export, the preview outside docs/, the explicit-path commit and a rollback with nothing to fill in."""
from pathlib import Path

DOC = Path(__file__).resolve().parents[2] / "ml" / "DEPLOY.md"


def test_deploy_doc_pins_the_static_publishing_commands():
    text = DOC.read_text(encoding="utf-8")
    for required in (
        "gh api repos/drkyoyodating/UCC-Demo/pages --jq '.status, .source.branch, .source.path, .html_url'",
        "It prints `built`, `main`, `/docs` and `https://drkyoyodating.github.io/UCC-Demo/`",
        "gh auth login --hostname github.com --git-protocol https --web --scopes workflow",
        "./.venv-ml/bin/python ml/tools/make_lab_preview.py --out /tmp/ucc-lab-preview",
        'RELEASE_ID=$(sed -n \'s/^release_id=//p\' docs/data/ml/release_v1.sha256)',
        './.venv-ml/bin/python -m ucc_ml.cli export-public --config ml/configs/v1.yaml --release-dir "ml/artifacts/releases/$RELEASE_ID"',
        "never `git add -A`",
        'git revert --no-edit "$(git log -1 --format=%H -- docs/data/ml/manifest.json)"',
        "https://drkyoyodating.github.io/UCC-Demo/ml/",
    ):
        assert required in text, required


def test_deploy_doc_names_nothing_private():
    text = DOC.read_text(encoding="utf-8")
    assert "UCC_DEMO_RUNBOOK" not in text
    # No trailing slash: the marker must survive Task 1 Step 8's sanitiser (and it matches more).
    assert "/Users" not in text
