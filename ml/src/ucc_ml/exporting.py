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


# ---------------------------------------------------------------------------------------------
# Section 2 — the public projection, the model card and the manifest (Task 2)
# ---------------------------------------------------------------------------------------------

# model-manifest.json keys that may appear publicly. Everything else — the fitted intercept, any
# key a later Plan B task adds — is dropped rather than published by accident.
MODEL_MANIFEST_PUBLIC_KEYS: tuple[str, ...] = (
    "model_version", "variant", "C", "c_grid", "best_C_at_grid_boundary", "class_weight",
    "cv_folds", "seed", "selection_metric", "weighting", "sample_weight_normalisation",
    "calibration_method", "feature_policy_version", "labels_policy_version", "label_disclosure",
    "n_features", "train_rows", "train_groups", "train_positives", "oof_positives",
    "oof_effective_positives", "oof_raw_score_sd", "refit_train_raw_score_sd",
    "source_commit", "source_dirty", "python_version", "sklearn_version", "numpy_version",
    "config_sha256", "lock_sha256", "candidates_sha256", "splits_sha256", "labels_sha256",
    "pipeline_sha256",
)
# K9: of `provenance`, only commit SHAs and package versions are public. The artefact digests,
# the config digest and the policy version stay private to the bundle.
PROVENANCE_PUBLIC_KEYS: tuple[str, ...] = ("source_commit", "python_version", "sklearn_version", "numpy_version")
_CI_REQUIRED = ("estimate", "lower", "upper")
_CI_OPTIONAL = ("n_resamples", "n_failed")
_BLOCK_REQUIRED = ("n", "positives", "predicted_positives", "tp", "fp", "fn", "tn",
                   "w_tp", "w_fp", "w_fn", "w_tn", "weighted_predicted_positives",
                   "weighted_precision", "weighted_recall", "weighted_f1")
_BLOCK_CI = ("weighted_precision", "weighted_recall", "weighted_f1")
_DELTA_KEYS = ("delta_weighted_precision", "delta_weighted_recall", "delta_weighted_f1")
_SUBSET_REQUIRED = ("n", "positives", "unresolved_share_among_model_positive")
_LABEL_STATISTICS = ("disclosure", "counts_by_status", "counts_by_round", "pass_agreement",
                     "founder_audit", "repeat_consistency")
# Optional, and published verbatim whenever present. training.labels_summary writes these maps into
# test.labels exactly when the file's rounds were labelled under DIFFERENT arrangements or under
# different label policies, and they are then the truth `disclosure` and `policy_version` are pooled
# from. They travel with the pooled scalars so a reader sees each round's arrangement rather than one
# phrase that is true of only some of the rows.
_LABEL_STATISTICS_BY_ROUND = ("disclosure_by_round", "policy_version_by_round")
_LABEL_COUNTS = ("policy_version", "split", "n_labelled", "n_resolved", "n_unresolved",
                 "n_positive", "n_negative")
_LATENCY_KEYS = ("measured_at", "hardware", "cold_load_and_first_call_ms", "warm_calls",
                 "warm_p50_ms", "warm_p95_ms", "batch_rows", "batch_rows_per_second",
                 "peak_rss_mb", "note")


class PublicProjectionError(KeyError):
    """Plan B's metrics.json does not carry a key the public document must publish."""


def _clean(value):
    """Floats become JSON-safe (NaN and infinities become null); everything else is unchanged."""
    if isinstance(value, float):
        return finite_or_none(value)
    return value


def _pick(src, where: str, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> dict:
    """Whitelist copy. A missing required key raises, naming the dotted path — this function is
    the one place Plan B's metrics.json is reconciled with the public document."""
    if src is None:
        src = {}   # an absent block reports which keys it lacks, not "must be an object"
    if not isinstance(src, dict):
        raise PublicProjectionError(f"release metrics.json: {where} must be an object, got {type(src).__name__}")
    missing = [key for key in required if key not in src]
    if missing:
        raise PublicProjectionError(f"release metrics.json: {where} lacks {missing}")
    return {key: _clean(src[key]) for key in (*required, *optional) if key in src}


def _ci(src, where: str) -> dict:
    return _pick(src, where, _CI_REQUIRED, _CI_OPTIONAL)


def _block(src, where: str) -> dict:
    out = _pick(src, where, _BLOCK_REQUIRED)
    cis = src.get("ci")
    if not isinstance(cis, dict):
        raise PublicProjectionError(f"release metrics.json: {where}.ci lacks {list(_BLOCK_CI)}")
    out["ci"] = {key: _ci(cis.get(key), f"{where}.ci.{key}") for key in _BLOCK_CI}
    return out


def _delta(src, where: str) -> dict:
    return {key: _ci((src or {}).get(key), f"{where}.{key}") for key in _DELTA_KEYS}


def _subset(src, where: str, *, key: str) -> dict:
    """A `review_queue`, `per_region[i]` or `per_stratum[i]` block: the same shape, one label key."""
    out = _pick(src, where, (key, *_SUBSET_REQUIRED)) if key else _pick(src, where, _SUBSET_REQUIRED)
    out["model"] = _block((src or {}).get("model"), f"{where}.model")
    out["rules"] = _block((src or {}).get("rules"), f"{where}.rules")
    out["delta"] = _delta((src or {}).get("delta"), f"{where}.delta")
    out["unresolved_share"] = _ci((src or {}).get("unresolved_share"), f"{where}.unresolved_share")
    return out


def _rows(src, where: str) -> list:
    if not isinstance(src, list) or not src:
        raise PublicProjectionError(f"release metrics.json: {where} is empty")
    return src


def _evaluation(src, where: str) -> dict:
    out = _pick(src, where, ("n", "positives", "negatives", "unresolved_share_among_model_positive"))
    out["weights"] = [
        _pick(row, f"{where}.weights[{i}]", ("stratum", "N_h", "n_h", "w_h"), ("unsampled",))
        for i, row in enumerate(_rows((src or {}).get("weights"), f"{where}.weights"))
    ]
    out["model"] = _block((src or {}).get("model"), f"{where}.model")
    out["rules"] = _block((src or {}).get("rules"), f"{where}.rules")
    out["delta_model_minus_rules"] = _delta((src or {}).get("delta_model_minus_rules"),
                                            f"{where}.delta_model_minus_rules")
    out["unresolved_share"] = _ci((src or {}).get("unresolved_share"), f"{where}.unresolved_share")
    out["review_queue"] = _subset((src or {}).get("review_queue"), f"{where}.review_queue", key="")
    out["per_region"] = [
        _subset(row, f"{where}.per_region[{i}]", key="region")
        for i, row in enumerate(_rows((src or {}).get("per_region"), f"{where}.per_region"))
    ]
    out["per_stratum"] = [
        _subset(row, f"{where}.per_stratum[{i}]", key="stratum")
        for i, row in enumerate(_rows((src or {}).get("per_stratum"), f"{where}.per_stratum"))
    ]
    out["bootstrap"] = _pick((src or {}).get("bootstrap"), f"{where}.bootstrap",
                             ("method", "n_resamples", "seed", "level"),
                             ("n_clusters", "straddling_groups"))
    return out


def project_public_metrics(release_metrics: dict, release_id: str, generated_at: str) -> dict:
    """The public metrics document: Plan B's TEST block, whitelisted key by key (K9).

    The VALIDATION block is never published — the threshold was chosen on it, so its rates are
    biased upward and only TEST is an unbiased estimate. `frozen.frozen_dir` (an absolute path on
    the machine that trained the model), the artefact digests under `provenance`, the calibration
    bin table and every per-case row are dropped. The label statistics of contract K3 are published
    in full, with the disclosure of K15 as the release's own labels block states it -- validated
    below, never asserted.
    """
    if not isinstance(release_metrics, dict):
        raise PublicProjectionError(f"release metrics.json must be an object, got {type(release_metrics).__name__}")
    if release_metrics.get("release_id") != release_id:
        raise ValueError(
            f"release metrics.json names release {release_metrics.get('release_id')!r}, "
            f"but the bundle's id is {release_id!r}"
        )
    test = release_metrics.get("test")
    if not isinstance(test, dict):
        raise PublicProjectionError("release metrics.json lacks the 'test' block: run evaluate-final and build-release")
    labels = test.get("labels")
    if not isinstance(labels, dict):
        raise PublicProjectionError("release metrics.json: test.labels must be an object")
    # The disclosure must TRUTHFULLY describe the labels, which is both stricter and less strict
    # than demanding one literal -- the same rule, for the same reason, as training.labels_summary
    # and inference.run_build_release. A labels file may hold rounds labelled under different
    # arrangements, and the real one does: main_v1 (2,880 of 3,120 rows) retains its disagreements
    # as unresolved while pilot_v1 (240) was founder-adjudicated. Demanding the literal would refuse
    # that file and publish NO public document at all; accepting any string would let a false
    # provenance claim onto a public page. So a stated map is rebuilt and must match exactly, and
    # only a file that states no map is held to the single literal.
    from ucc_ml.labeling import pooled_disclosure

    by_round = labels.get("disclosure_by_round")
    stated = labels.get("disclosure")
    if by_round:
        expected = pooled_disclosure(by_round)
        if stated != expected:
            raise ValueError(
                f"release metrics.json: test.labels.disclosure {stated!r} does not describe its own "
                f"disclosure_by_round {by_round!r}; expected {expected!r}. Refusing to publish it."
            )
    elif stated != LABEL_DISCLOSURE:
        raise ValueError(
            f"release metrics.json: test.labels.disclosure must be the literal {LABEL_DISCLOSURE!r}, "
            f"got {stated!r}"
        )
    public_test = _pick(test, "test", ("split", "evaluated_at", "forced"), ("protocol",))
    public_test["frozen"] = _pick(test.get("frozen"), "test.frozen",
                                  ("threshold", "status", "score_type"), ("variant", "calibration_method"))
    public_test["evaluation"] = _evaluation(test.get("evaluation"), "test.evaluation")
    calibration = _pick(test.get("calibration"), "test.calibration", ("ece", "n_bins"))
    calibration["probability_gate"] = _pick(
        (test.get("calibration") or {}).get("probability_gate"), "test.calibration.probability_gate",
        ("weighted_ece", "score_type"),
        ("decision_region_weighted_ece", "decision_region_cases", "decision_score"),
    )
    public_test["calibration"] = calibration
    public_test["unresolved"] = [
        _pick(row, f"test.unresolved[{i}]", ("region", "stratum", "n_labelled", "n_unresolved", "share_unresolved"),
              ("weighting", "share_unresolved_among_predicted_positive"))
        for i, row in enumerate(_rows(test.get("unresolved"), "test.unresolved"))
    ]
    public_test["labels"] = {
        **{key: copy.deepcopy(labels[key]) for key in _LABEL_STATISTICS if key in labels},
        **{key: copy.deepcopy(labels[key]) for key in _LABEL_STATISTICS_BY_ROUND if labels.get(key)},
        **_pick(labels, "test.labels", _LABEL_COUNTS),
    }
    missing_statistics = [key for key in _LABEL_STATISTICS if key not in public_test["labels"]]
    if missing_statistics:
        raise PublicProjectionError(f"release metrics.json: test.labels lacks the K3 statistics {missing_statistics}")
    verdict = _pick(test.get("verdict"), "test.verdict", ("primary", "delta_precision", "delta_recall", "text"))
    verdict["review_queue_weighted_precision"] = _ci(
        (test.get("verdict") or {}).get("review_queue_weighted_precision"),
        "test.verdict.review_queue_weighted_precision",
    )
    public_test["verdict"] = verdict
    public_test["provenance"] = _pick(test.get("provenance"), "test.provenance", PROVENANCE_PUBLIC_KEYS)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "release_id": release_id,
        "generated_at": generated_at,
        "created_at": release_metrics.get("created_at"),
        "label_provenance": stated,
        **({"label_provenance_by_round": copy.deepcopy(by_round)} if by_round else {}),
        "test": public_test,
        "latency": None,
    }


def evaluation_summary(public_metrics: dict) -> dict:
    """The compact block `/v1/model` returns: what the model was measured at, and on what."""
    test = public_metrics["test"]
    evaluation = test["evaluation"]
    return {
        "split": test["split"],
        "evaluated_at": test["evaluated_at"],
        "label_provenance": public_metrics["label_provenance"],
        "n": evaluation["n"],
        "positives": evaluation["positives"],
        "threshold": {key: test["frozen"][key] for key in ("threshold", "status", "score_type")},
        "model": {key: evaluation["model"]["ci"][key] for key in ("weighted_precision", "weighted_recall")},
        "rules": {key: evaluation["rules"]["ci"][key] for key in ("weighted_precision", "weighted_recall")},
        "review_queue": {"n": evaluation["review_queue"]["n"],
                         "weighted_precision": evaluation["review_queue"]["model"]["ci"]["weighted_precision"]},
        "unresolved_share": evaluation["unresolved_share"],
        "verdict": test["verdict"]["text"],
        "labels": {key: test["labels"][key]
                   for key in (*_LABEL_STATISTICS, *_LABEL_STATISTICS_BY_ROUND) if key in test["labels"]},
    }


def parse_model_card_markdown(text: str) -> list[dict]:
    """Split model-card.md on `## ` headings. The `# ` title line is dropped; anything before the
    first `## ` becomes a 'Preamble' section."""
    sections: list[dict] = []
    heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if heading is not None or body:
            sections.append({"heading": heading or "Preamble", "text": body})

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            heading, buffer = line[3:].strip(), []
        elif line.startswith("# ") and heading is None and not any(l.strip() for l in buffer):
            continue
        else:
            buffer.append(line)
    flush()
    return sections


def build_model_card(release: OpenRelease, generated_at: str) -> dict:
    """The public model card: the bundle's own model-card.md as sections, plus whitelisted
    metadata and the threshold the bundle actually carries.

    The provenance comes from the release's own labels block, like every other public document here:
    `project_public_metrics` has already refused a sentence that block's round map contradicts, and
    the literal is the fallback only for a bundle whose metrics state no disclosure at all."""
    labels = ((release.metrics or {}).get("test") or {}).get("labels") or {}
    by_round = labels.get("disclosure_by_round")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "release_id": release.release_id,
        "generated_at": generated_at,
        "label_provenance": labels.get("disclosure") or LABEL_DISCLOSURE,
        **({"label_provenance_by_round": copy.deepcopy(by_round)} if by_round else {}),
        "metadata": {key: release.model_manifest[key] for key in MODEL_MANIFEST_PUBLIC_KEYS
                     if key in release.model_manifest},
        "threshold": release.threshold,
        "threshold_status": release.threshold_status,
        "score_type": release.score_type,
        "sections": parse_model_card_markdown(release.model_card_markdown),
    }


def new_manifest(generated_at: str) -> dict:
    return {"schema_version": MANIFEST_SCHEMA_VERSION, "generated_at": generated_at, "current": None, "releases": []}


def load_manifest(path: Path) -> dict:
    path = Path(path)
    return read_json(path) if path.is_file() else new_manifest(utc_now_iso())


def _release_entry(release_id: str, published_at: str, api_url: str | None) -> dict:
    return {
        "release_id": release_id,
        "published_at": published_at,
        "metrics_url": f"releases/{release_id}/metrics.json",
        "examples_url": f"releases/{release_id}/examples.json",
        "model_card_url": f"releases/{release_id}/model-card.json",
        "api_url": api_url.rstrip("/") if isinstance(api_url, str) and api_url else None,
    }


def upsert_manifest_release(manifest: dict, release_id: str, published_at: str, api_url: str | None,
                            *, keep_api_url: bool = True) -> dict:
    """Replace or add the release entry, move it to the front, and make it `current`.

    With keep_api_url=True an api_url written by ml/tools/set_manifest_api_url.py (Task 12)
    survives a re-export that passes None, so republishing a release never silently takes the
    live form away."""
    releases = [dict(entry) for entry in manifest.get("releases", [])]
    existing = next((entry for entry in releases if entry.get("release_id") == release_id), None)
    if api_url is None and keep_api_url and existing is not None:
        api_url = existing.get("api_url")
    entry = _release_entry(release_id, published_at, api_url)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": published_at,
        "current": release_id,
        "releases": [entry] + [r for r in releases if r.get("release_id") != release_id],
    }


def validate_manifest(manifest: dict, base_dir: Path) -> list[str]:
    """Return a list of problems (empty = valid): shape, unique ids, relative URLs that exist
    under base_dir, and an api_url that is null, https, or a loopback http URL."""
    problems: list[str] = []
    base_dir = Path(base_dir)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        problems.append(f"schema_version must be {MANIFEST_SCHEMA_VERSION}")
    releases = manifest.get("releases")
    if not isinstance(releases, list) or not releases:
        return problems + ["releases must be a non-empty list"]
    ids = [entry.get("release_id") for entry in releases]
    if len(set(ids)) != len(ids):
        problems.append("release_id values must be unique")
    if manifest.get("current") not in ids:
        problems.append(f"current {manifest.get('current')!r} is not a listed release")
    for entry in releases:
        release_id = entry.get("release_id")
        if not isinstance(release_id, str) or not release_id:
            problems.append("every release needs a string release_id")
            continue
        if not isinstance(entry.get("published_at"), str):
            problems.append(f"{release_id}: published_at must be a string")
        for key in ("metrics_url", "examples_url", "model_card_url"):
            url = entry.get(key)
            if not isinstance(url, str) or url.startswith("/") or "://" in url or ".." in url:
                problems.append(f"{release_id}: {key} must be relative to the manifest directory, got {url!r}")
                continue
            if not (base_dir / url).is_file():
                problems.append(f"{release_id}: {url} does not exist under {base_dir}")
        api_url = entry.get("api_url")
        if api_url is not None and not (
            isinstance(api_url, str)
            and (api_url.startswith("https://") or api_url.startswith("http://127.0.0.1")
                 or api_url.startswith("http://localhost"))
        ):
            problems.append(f"{release_id}: api_url must be null, https://…, or a loopback http URL, got {api_url!r}")
    return problems
