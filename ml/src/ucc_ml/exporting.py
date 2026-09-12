"""Plan C — the public export of a release bundle (contracts K6, K8, K9, K15).

Everything the Lab page reads under ``docs/data/ml/`` is written by this module and by nothing
else. The public metrics document is a **whitelist projection** of Plan B's release
``metrics.json``: per-case rows, local paths and the frozen candidate's directory never pass
through it, and a key this module does not name is dropped rather than published by accident.

Ownership (contract v1): hashing, canonical JSON and ``SHA256SUMS`` belong to
``ucc_ml.provenance`` (K6); the release bundle, its id, its threshold and the one inference path
belong to ``ucc_ml.inference`` (K8); the artefact locations belong to ``ucc_ml.config`` (K5); the
candidate, split and label readers belong to Plan A (K7). This module defines none of them again.

Sections, in file order: the open release (Task 1) · the public projection, model card and
manifest (Task 2) · curated examples, latency, ``export_public`` and the CLI (Task 3).

``ucc_ml.dataset``, ``ucc_ml.splitting`` and ``ucc_ml.labeling`` are imported INSIDE the functions
that need them: ``ucc_ml.dataset`` imports duckdb, which the Cloud Run serving image does not
carry, and ``ucc_ml.api`` imports this module.
"""
from __future__ import annotations

import argparse
import copy
import platform
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ucc_ml.contracts import LABEL_DISCLOSURE, LABELS, RESOLVED_ADJUDICATION_STATUSES
from ucc_ml.inference import (
    RELEASE_FILES,
    CaseInput,
    Prediction,
    load_release_bundle,
    predict_cases,
)
from ucc_ml.provenance import finite_or_none, read_json, utc_now_iso, write_json

MANIFEST_SCHEMA_VERSION = 1
PUBLIC_DOCUMENTS = ("metrics", "examples", "model_card")
# Codex §7's two decisions. Plan B's Prediction.decision is a Literal of exactly these.
DECISION_POSITIVE = "suggest_relevant"
DECISION_NEGATIVE = "review_needed"


@dataclass(frozen=True)
class OpenRelease:
    """One verified release bundle plus the documents the public export reads from it.

    ``bundle`` comes from ``ucc_ml.inference.load_release_bundle`` (K8), which verifies
    ``SHA256SUMS`` with every ``RELEASE_FILES`` name required, checks the directory name against
    the computed release id and checks the pipeline digest before ``joblib.load``. The release id
    and the threshold are the bundle's; nothing here reads them out of a manifest.
    """

    bundle: object
    metrics: dict
    model_manifest: dict
    model_card_markdown: str

    @property
    def directory(self) -> Path:
        return self.bundle.directory

    @property
    def release_id(self) -> str:
        return self.bundle.release_id

    @property
    def threshold(self) -> float:
        return float(self.bundle.threshold)

    @property
    def score_type(self) -> str:
        return str(self.bundle.score_type)

    @property
    def threshold_status(self) -> str:
        return str(self.bundle.threshold_status)


def open_release(directory: Path, *, top_k: int = 10) -> OpenRelease:
    """Verify and open a release bundle for publication or serving."""
    directory = Path(directory)
    bundle = load_release_bundle(directory, top_k=top_k)
    return OpenRelease(
        bundle=bundle,
        metrics=read_json(directory / "metrics.json"),
        model_manifest=copy.deepcopy(bundle.model_manifest),
        model_card_markdown=(directory / "model-card.md").read_text(encoding="utf-8"),
    )


def release_directory(releases_dir: Path, release_dir: Path | None = None) -> Path:
    """``release_dir`` when given, else the only release under ``releases_dir``.

    Release directories are named after their content id (12 lowercase hex), so a staging or
    scratch directory beside them is never mistaken for a release; several releases are an
    ambiguity to resolve explicitly rather than to guess at.
    """
    if release_dir is not None:
        release_dir = Path(release_dir)
        if not release_dir.is_dir():
            raise FileNotFoundError(f"release directory {release_dir} does not exist")
        return release_dir
    releases_dir = Path(releases_dir)
    found = sorted(
        path for path in (releases_dir.iterdir() if releases_dir.is_dir() else [])
        if path.is_dir() and len(path.name) == 12 and set(path.name) <= set("0123456789abcdef")
    )
    if len(found) != 1:
        raise FileNotFoundError(
            f"{releases_dir} holds {len(found)} release(s); pass --release-dir to say which one"
        )
    return found[0]
