"""Build a SYNTHETIC preview of the ML Lab page, outside docs/ (plan C Task 5).

    ./.venv-ml/bin/python ml/tools/make_lab_preview.py --out /tmp/ucc-lab-preview
    ./.venv-ml/bin/python -m http.server 8080 --bind 127.0.0.1 --directory /tmp/ucc-lab-preview/site
    open http://127.0.0.1:8080/ml/

The numbers come from the REAL pipeline (train, freeze-candidate, evaluate-final, build-release via
ucc_ml.synthetic.build_synthetic_release, then score-batch and export-public) run on the fictitious
synthetic world, so the layout is reviewed against exactly the document shapes a real release
publishes. The copied manifest carries "preview": true, which makes the page show a SYNTHETIC banner.
Nothing is ever written under docs/.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE_FILES = (
    ("docs/ml/index.html", "ml/index.html"),
    ("docs/assets/ml-lab.js", "assets/ml-lab.js"),
    ("docs/assets/ml-lab.css", "assets/ml-lab.css"),
)


def find_world_root(bundle_dir: Path) -> Path:
    """The synthetic world root: the nearest ancestor holding ml/configs/v1.yaml."""
    for parent in Path(bundle_dir).resolve().parents:
        if (parent / "ml" / "configs" / "v1.yaml").is_file():
            return parent
    raise FileNotFoundError(f"no ml/configs/v1.yaml above {bundle_dir}")


def make_preview(out_dir: Path, *, repo_root: Path = REPO_ROOT, bundle_dir: Path | None = None) -> Path:
    """Write <out_dir>/site/{ml/index.html, assets/ml-lab.{js,css}, data/ml/...} and return the site dir."""
    out_dir = Path(out_dir).resolve()
    docs_dir = (Path(repo_root) / "docs").resolve()
    if out_dir == docs_dir or out_dir.is_relative_to(docs_dir):
        raise ValueError(f"the preview must be written outside docs/, got {out_dir}")

    from ucc_ml import cli
    from ucc_ml.config import artefact_paths, load_config
    from ucc_ml.inference import run_score_batch
    from ucc_ml.provenance import read_json, write_json

    if bundle_dir is None:
        from ucc_ml.synthetic import build_synthetic_release

        world_dir = out_dir / "world"
        if world_dir.exists():
            shutil.rmtree(world_dir)
        bundle_dir = build_synthetic_release(world_dir)
    bundle_dir = Path(bundle_dir).resolve()
    config_path = find_world_root(bundle_dir) / "ml" / "configs" / "v1.yaml"

    run_score_batch(config_path, bundle_dir)
    code = cli.main(["export-public", "--config", str(config_path), "--release-dir", str(bundle_dir)])
    if code != 0:
        raise RuntimeError(f"export-public exited {code}")
    public_dir = artefact_paths(load_config(config_path)).public_data_dir

    site = out_dir / "site"
    if site.exists():
        shutil.rmtree(site)
    for source, target in PAGE_FILES:
        destination = site / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(repo_root) / source, destination)
    shutil.copytree(public_dir, site / "data" / "ml")
    manifest_path = site / "data" / "ml" / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["preview"] = True
    write_json(manifest_path, manifest)
    return site


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make_lab_preview", description="SYNTHETIC ML Lab preview, written outside docs/.")
    parser.add_argument("--out", type=Path, required=True, help="output directory, e.g. /tmp/ucc-lab-preview")
    parser.add_argument("--bundle-dir", type=Path, default=None,
                        help="reuse a release built by ucc_ml.synthetic.build_synthetic_release (its world holds ml/configs/v1.yaml)")
    ns = parser.parse_args(argv)
    site = make_preview(ns.out, bundle_dir=ns.bundle_dir)
    print(f"preview site: {site}")
    print(f"serve: ./.venv-ml/bin/python -m http.server 8080 --bind 127.0.0.1 --directory {site}")
    print("open:  http://127.0.0.1:8080/ml/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
