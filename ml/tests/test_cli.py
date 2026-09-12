"""The CLI is importable, prints usage on --help, runs as `python -m ucc_ml.cli`, and imports nothing heavy."""
import subprocess
import sys
from pathlib import Path

import pytest

HEAVY_MODULES = ("pandas", "pyarrow", "duckdb", "pydantic", "yaml", "openpyxl", "sklearn", "mlflow", "dagster", "fastapi")
SUBCOMMAND_MARKER = "    # --- subcommands are registered below this line by later tasks (keep alphabetical) ---\n"
CMD_MARKER = "# --- cmd_<name> functions are added above this line by later tasks; each imports its implementation lazily ---\n"


def test_help_exits_zero_and_prints_usage(capsys):
    from ucc_ml.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "usage: ucc_ml" in capsys.readouterr().out


def test_no_command_prints_help_and_returns_2(capsys):
    from ucc_ml.cli import main

    assert main([]) == 2
    assert "usage: ucc_ml" in capsys.readouterr().out


def test_module_entrypoint_runs():
    proc = subprocess.run(
        [sys.executable, "-m", "ucc_ml.cli", "--version"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "ucc_ml 0.1.0"


def test_building_the_parser_imports_nothing_heavy():
    """Contract K11: every cmd_<name> imports its implementation lazily, so --help needs no data stack."""
    code = (
        "import sys\n"
        "from ucc_ml.cli import build_parser\n"
        "build_parser().format_help()\n"
        f"print(','.join(m for m in {HEAVY_MODULES!r} if m in sys.modules))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == ""


def test_anchor_markers_are_present_exactly_once():
    """Plans A, B and C anchor their cli.py Replace steps on these two lines; they are never edited."""
    import ucc_ml.cli as cli

    text = Path(cli.__file__).read_text(encoding="utf-8")
    assert text.count(SUBCOMMAND_MARKER) == 1
    assert text.count(CMD_MARKER) == 1
    assert text.index(SUBCOMMAND_MARKER) < text.index(CMD_MARKER) < text.index("def main(argv: list[str] | None = None) -> int:")
