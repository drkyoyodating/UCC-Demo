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
    sp = sub.add_parser("build-candidates", help="snapshot Parquet -> candidates.parquet + reconciliation")
    _add_config_arg(sp)
    sp.set_defaults(func=cmd_build_candidates)
    sp = sub.add_parser("make-pilot", help="draw the 200-case development pilot (per_stratum x 4 strata)")
    _add_config_arg(sp)
    sp.set_defaults(func=cmd_make_pilot)
    # --- subcommands are registered below this line by later tasks (keep alphabetical) ---
    return parser


def cmd_build_candidates(ns: argparse.Namespace) -> int:
    from ucc_ml import dataset, legacy
    from ucc_ml.config import artefact_paths, load_config
    from ucc_ml.provenance import git_head

    cfg = load_config(ns.config)
    paths = artefact_paths(cfg)
    vendor = legacy.verify_vendor_hashes()
    manifest = dataset.build_candidates(
        snapshot_dir=cfg.path("snapshot_dir"), out_dir=cfg.path("candidates_dir"),
        dataset_version=cfg.version.dataset_version, year_min=cfg.eligibility.year_min,
        junk=cfg.eligibility.junk_addresses,
        provenance={"config_sha256": cfg.config_sha256, "config_path": str(cfg.config_path),
                    "git_head": git_head(cfg.repo_root), "vendor": vendor},
    )
    c = manifest["candidates"]
    print(f"cases={c['rows']} by_region={c['by_region']} strata={c['by_stratum']} routes={c['by_route']}")
    print(f"parity: {'OK' if manifest['parity_ok'] else 'FAILED -- read reconciliation.json'}")
    print(f"wrote {paths.candidates_parquet} sha256={c['sha256']}")
    return 0 if manifest["parity_ok"] else 1


def cmd_make_pilot(ns: argparse.Namespace) -> int:
    import hashlib

    from ucc_ml import dataset, sampling
    from ucc_ml.config import artefact_paths, load_config
    from ucc_ml.provenance import git_head, sha256_file, utc_now_iso, write_json

    cfg = load_config(ns.config)
    paths = artefact_paths(cfg)
    cases = dataset.read_candidates(paths.candidates_parquet)
    pilot = sampling.make_pilot(cases, per_stratum=cfg.sampling.pilot_per_stratum, seed=cfg.seed, purpose="pilot_v1")
    sha = dataset.write_frame(pilot, paths.pilot_cases)
    strata = {s: {"N_h": int(g.N_h.iloc[0]), "pool_h": int(g.pool_h.iloc[0]), "n_h": int(g.n_h.iloc[0]),
                  "inclusion_probability": float(g.inclusion_probability.iloc[0])}
              for s, g in pilot.groupby("sampling_stratum", sort=False)}
    manifest = {
        "purpose": "pilot_v1", "dataset_version": cfg.version.dataset_version, "seed": cfg.seed,
        "per_stratum": cfg.sampling.pilot_per_stratum, "strata": strata, "rows": int(len(pilot)),
        "by_region": {k: int(v) for k, v in sorted(pilot.region.value_counts().items())},
        "case_id_digest": hashlib.sha256(("\n".join(sorted(pilot.case_id)) + "\n").encode("utf-8")).hexdigest(),
        "pilot_cases_sha256": sha, "candidates_sha256": sha256_file(paths.candidates_parquet),
        "created_at": utc_now_iso(), "git_head": git_head(cfg.repo_root), "config_sha256": cfg.config_sha256,
    }
    write_json(paths.pilot_manifest, manifest)
    print(f"pilot rows={len(pilot)} by_region={manifest['by_region']} strata={ {k: v['n_h'] for k, v in strata.items()} }")
    print(f"wrote {paths.pilot_cases} sha256={sha}")
    return 0


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
