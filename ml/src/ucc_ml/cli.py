"""Command line for the UCC ML Lab.

    ./.venv-ml/bin/python -m ucc_ml.cli <command> --config ml/configs/v1.yaml

Every subcommand is a thin wrapper over a tested function in another module; nothing is computed
here. Two marker comments are the anchors every plan edits against, and neither is ever changed
(contract K11):

* inside ``build_parser`` a plan replaces the subcommand marker line with
  ``sp = sub.add_parser("<name>", help="...")``, ``_add_config_arg(sp)``,
  ``sp.set_defaults(func=cmd_<name>)`` followed by the same marker line, unchanged;
* above ``main`` a plan replaces the cmd marker line with ``def cmd_<name>(ns) -> int:`` followed by
  the same marker line, unchanged. A ``cmd_<name>`` imports its implementation inside the function,
  so ``--help`` never imports pandas, pydantic, sklearn, mlflow or fastapi.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ucc_ml import __version__


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", required=True, type=Path,
        help="path to the run config, e.g. ml/configs/v1.yaml",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ucc_ml",
        description="UCC ML Lab: candidate dataset, blind labels, splits, model, evaluation.",
    )
    parser.add_argument("--version", action="version", version=f"ucc_ml {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    # --- subcommands are registered below this line by later tasks (keep alphabetical) ---
    return parser


# --- cmd_<name> functions are added above this line by later tasks; each imports its implementation lazily ---


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    if ns.command is None:
        parser.print_help()
        return 2
    return int(ns.func(ns))


if __name__ == "__main__":
    sys.exit(main())
