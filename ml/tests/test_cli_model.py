"""The five Plan B commands through `python -m ucc_ml.cli` on the synthetic world: registered at Plan A's markers with lazy
imports (contract K11), exit code 2 when TEST was already evaluated, --force-i-know, the release id on build-release's last
line, and score-batch with and without --release-dir."""
import subprocess
import sys
from pathlib import Path

PLAN_B_COMMANDS = ("build-release", "evaluate-final", "freeze-candidate", "score-batch", "train")
SUBCOMMAND_MARKER = "    # --- subcommands are registered below this line by later tasks (keep alphabetical) ---\n"
CMD_MARKER = "# --- cmd_<name> functions are added above this line by later tasks; each imports its implementation lazily ---\n"


def _cli(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "ucc_ml.cli", *args], cwd=cwd, capture_output=True, text=True)


def test_help_lists_the_five_commands_and_the_markers_survive(tmp_path):
    import ucc_ml.cli as cli

    out = _cli(tmp_path, "--help")
    assert out.returncode == 0
    for command in PLAN_B_COMMANDS:
        assert command in out.stdout
    text = Path(cli.__file__).read_text(encoding="utf-8")
    assert text.count(SUBCOMMAND_MARKER) == 1 and text.count(CMD_MARKER) == 1
    for command in PLAN_B_COMMANDS:
        assert text.index(f"def cmd_{command.replace('-', '_')}(ns: argparse.Namespace) -> int:") < text.index(CMD_MARKER)


def test_end_to_end_via_the_cli(tmp_path):
    from ucc_ml.synthetic import make_world, write_config, write_world

    write_world(make_world(), tmp_path)
    cfg = str(write_config(tmp_path))
    train = _cli(tmp_path, "train", "--config", cfg)
    assert train.returncode == 0, train.stderr
    assert "borrower_lender: best C=" in train.stdout and "rules validation: wP=" in train.stdout
    freeze = _cli(tmp_path, "freeze-candidate", "--config", cfg)
    assert freeze.returncode == 0 and '"status"' in freeze.stdout and '"validation_looks": 1' in freeze.stdout
    first = _cli(tmp_path, "evaluate-final", "--config", cfg)
    assert first.returncode == 0 and "TEST (n=" in first.stdout and "review-queue precision" in first.stdout
    second = _cli(tmp_path, "evaluate-final", "--config", cfg)
    assert second.returncode == 2 and "REFUSED" in second.stdout
    assert _cli(tmp_path, "evaluate-final", "--config", cfg, "--force-i-know").returncode == 0
    build = _cli(tmp_path, "build-release", "--config", cfg)
    assert build.returncode == 0, build.stderr
    release_id = build.stdout.strip().splitlines()[-1]
    assert len(release_id) == 12 and int(release_id, 16) >= 0
    release_dir = tmp_path / "ml/artifacts/releases" / release_id
    explicit = _cli(tmp_path, "score-batch", "--config", cfg, "--release-dir", str(release_dir))
    assert explicit.returncode == 0 and explicit.stdout.strip().endswith(f"{release_id}.parquet")
    assert _cli(tmp_path, "score-batch", "--config", cfg).returncode == 0
    assert (tmp_path / "ml/data/predictions" / f"{release_id}.summary.json").exists()
