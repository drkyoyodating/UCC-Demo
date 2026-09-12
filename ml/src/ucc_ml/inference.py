"""Release bundle (build / verify / load) and the shared prediction path (contract K8).

Part 1 (Task 12): the immutable release directory
  ml/artifacts/releases/<release_id>/{pipeline.joblib, model-manifest.json, threshold.json, dataset-manifest.json,
  split-manifest.json, metrics.json, model-card.md, requirements-ml.lock.txt, SHA256SUMS}
release_id = first 12 hex of sha256 over the concatenated bytes of the four manifests, in RELEASE_ID_MANIFESTS order.
Manifests carry no timestamps, so the same frozen candidate always yields the same id. joblib.load happens only after
SHA256SUMS verifies with every RELEASE_FILES name required (Codex §8, contract K8). Nothing at module level imports
duckdb, mlflow or Plan A's dataset / splitting / labeling modules: the API imports this module.
"""
from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ucc_ml.contracts import LABELLING_ROUNDS, SPLITS
from ucc_ml.evaluation import FINAL_METRICS_NAME
from ucc_ml.features import feature_names as _feature_names
from ucc_ml.provenance import (
    BundleIntegrityError,
    read_json,
    sha256_file,
    utc_now_iso,
    verify_sha256sums,
    write_json,
    write_sha256sums,
)
from ucc_ml.training import FROZEN_FILES, GUARD_FILE_NAME, PROTOCOL_PATH, VALIDATION_LOOKS_NAME, CalibratedModel

RELEASE_FILES = ("pipeline.joblib", "model-manifest.json", "threshold.json", "dataset-manifest.json",
                 "split-manifest.json", "metrics.json", "model-card.md", "requirements-ml.lock.txt")
RELEASE_ID_MANIFESTS = ("model-manifest.json", "threshold.json", "dataset-manifest.json", "split-manifest.json")


def compute_release_id(directory: Path) -> str:
    h = hashlib.sha256()
    for name in RELEASE_ID_MANIFESTS:
        h.update((Path(directory) / name).read_bytes())
    return h.hexdigest()[:12]


def _fmt(x, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def _interval(c: dict) -> str:
    """An estimate with its interval, and a mark when the estimate falls OUTSIDE its own interval.

    The point estimate is the design-weighted plug-in; the interval is smoothed by the Jeffreys
    pseudo-mass, which deliberately keeps a cell holding no observed false positive uncertain. Near the
    boundary the two disagree -- a precision of 1.000 against an interval ending at 0.997 -- and that is
    the prior working, not a weighting error. evaluation.BootstrapResult.to_dict has always computed
    this flag and metrics.json has always carried it; the card simply never read it, so the one
    document a reader actually opens showed the pair with nothing to say they were computed differently.
    """
    text = f"{_fmt(c['estimate'])} [{_fmt(c['lower'])}, {_fmt(c['upper'])}]"
    return f"{text} (estimate outside interval)" if c.get("estimate_outside_interval") else text


def _ci(block: dict, key: str) -> str:
    return _interval(block["ci"][key])


def _ordered_rounds(by_round) -> list:
    """Labelling rounds in LABELLING_ROUNDS order, so the card reads chronologically; unknown names last."""
    known = [r for r in LABELLING_ROUNDS if r in by_round]
    return known + sorted(r for r in by_round if r not in known)


def _pool_counts(by_round, keys: tuple) -> dict:
    return {k: sum(int((row or {}).get(k) or 0) for row in by_round.values()) for k in keys}


def _adjudication_clause(disagreements, audit) -> str:
    """What the founder actually did with this round's pass disagreements and its audit sample.

    RENDERED, never asserted. The card used to publish the flat sentence "the founder decided every
    disagreement and audited a sample of agreed rows". On the real labels main_v1 is
    {"n": 83, "decided": 0, "undecided": 83} with 0 of 120 selected rows audited, so that sentence was
    false of 2,880 of the 3,120 label rows -- and it sat one clause after a disclosure saying the
    opposite. Every clause below comes out of founder_disagreements / founder_audit, so the card cannot
    drift away from the file it describes."""
    parts = []
    if disagreements:
        n, decided = int(disagreements.get("n") or 0), int(disagreements.get("decided") or 0)
        undecided = int(disagreements.get("undecided") or 0)
        if n == 0:
            parts.append("the two blind passes agreed on every case")
        elif undecided == 0:
            parts.append(f"the founder decided all {decided} of the {n} pass disagreements")
        elif decided == 0:
            parts.append(f"all {n} pass disagreements were retained as unresolved (INSUFFICIENT_EVIDENCE, "
                         "counted in n_h) and none was adjudicated")
        else:
            parts.append(f"the founder decided {decided} of the {n} pass disagreements and {undecided} "
                         "were retained as unresolved")
    if audit:
        n_selected, n_audited = int(audit.get("n_selected") or 0), int(audit.get("n_audited") or 0)
        parts.append(f"{n_audited} of the {n_selected} rows selected for the founder audit were audited"
                     if n_selected else "no rows were selected for a founder audit")
    return "; ".join(parts) if parts else "no adjudication statistics were recorded"


def _label_provenance_lines(labels: dict, dataset_manifest: dict) -> list[str]:
    """The "Labels:" bullet -- one bullet per ROUND when the rounds were labelled under different
    arrangements, because no single sentence is true of the whole file in that case.

    `labels` is the metrics document's block (training.labels_summary), which already rebuilds the pooled
    policy from policy_version_by_round; `dataset_manifest` carries the per-round maps and the founder
    counts. When the manifest states no disclosure map, one arrangement really does cover the file and the
    single-sentence form is the honest rendering -- but its adjudication clause is still rendered from the
    pooled counts rather than asserted."""
    by_round = labels.get("disclosure_by_round") or dataset_manifest.get("disclosure_by_round")
    policy_by_round = labels.get("policy_version_by_round") or dataset_manifest.get("policy_version_by_round") or {}
    disagreements = dataset_manifest.get("founder_disagreements") or {}
    audits = labels.get("founder_audit") or dataset_manifest.get("founder_audit") or {}
    counts = labels.get("counts_by_round") or {}
    head = (f"- Labels: policy `{labels.get('policy_version') or dataset_manifest['policy_version']}`; "
            f"**{dataset_manifest['label_disclosure']}** — two independent blind Claude passes labelled every "
            "case under the written policy; nothing was labelled by looking anything up.")
    if not by_round:
        pooled_d = _pool_counts(disagreements, ("n", "decided", "undecided")) if disagreements else None
        pooled_a = _pool_counts(audits, ("n_selected", "n_audited")) if audits else None
        return [f"{head} Across the rounds, {_adjudication_clause(pooled_d, pooled_a)}."]
    out = [f"{head} The rounds were NOT labelled under one arrangement, so each is stated separately:"]
    for r in _ordered_rounds(by_round):
        rows = counts.get(r)
        out.append(f"  - `{r}` ({rows:,} rows, policy `{policy_by_round.get(r, 'unrecorded')}`): " if isinstance(rows, int)
                   else f"  - `{r}` (policy `{policy_by_round.get(r, 'unrecorded')}`): ")
        out[-1] += f"{by_round[r]}; {_adjudication_clause(disagreements.get(r), audits.get(r))}."
    return out


def render_model_card(release_id: str, model_manifest: dict, threshold_doc: dict, dataset_manifest: dict,
                      split_manifest: dict, test_metrics: dict, validation_looks: int | None) -> str:
    ev = test_metrics["evaluation"]; m, r = ev["model"], ev["rules"]; d = ev["delta_model_minus_rules"]
    rq, labels, verdict = ev["review_queue"], test_metrics["labels"], test_metrics["verdict"]
    val, cal = threshold_doc["validation"], threshold_doc["calibration"]
    sd_oof, sd_refit = model_manifest["oof_raw_score_sd"], model_manifest["refit_train_raw_score_sd"]
    lines = [
        f"# Model card — UCC heavy-construction relevance screener, release `{release_id}`", "",
        "## What it is",
        "A scikit-learn TF-IDF + logistic-regression screener over UCC borrower and lender NAMES "
        "(word 1–2-grams on the normalised name, character 3–5-grams on the raw name, separate borrower "
        "and lender blocks). It is a **review queue over rules-rejected cases plus a second opinion on "
        "rules-accepted ones**. It never replaces the frozen rules, never changes `scope_all`, the headline "
        "counts or the existing map. A score indicates evidence of relevance under the written screening "
        "policy; it does not establish collateral, equipment ownership, loan amount, active debt or creditworthiness.", "",
        "## Data and labels",
        f"- Candidates: {dataset_manifest['candidates_rows']:,} filing-borrower cases (dataset `{dataset_manifest['dataset_version']}`, "
        f"sha256 `{dataset_manifest['candidates_sha256'][:12]}…`), by region {dataset_manifest['candidates_rows_by_region']}.",
        *_label_provenance_lines(labels, dataset_manifest),
        f"- Label statistics: by status {labels['counts_by_status']}; by round {labels['counts_by_round']}; pass agreement "
        f"{labels['pass_agreement']}; founder audit {labels['founder_audit']}; blind-repeat consistency {labels['repeat_consistency']}.",
        f"- Labelled cases by split {dataset_manifest['labelled_rows_by_split']}, of which resolved (RELEVANT / NOT_RELEVANT) "
        f"{dataset_manifest['resolved_rows_by_split']}.",
        f"- Groups are {split_manifest['group_scope']} borrower groups; splits {split_manifest['rows_by_split']} rows / "
        f"{split_manifest['groups_by_split']} groups.", "",
        "## Training",
        f"- Variant `{model_manifest['variant']}`, C = {model_manifest['C']} (grid {model_manifest['c_grid']}"
        + (", **at a grid boundary: widening the grid is a founder decision**" if model_manifest["best_C_at_grid_boundary"] else "")
        + f"), class_weight none, sample weight 1/inclusion_probability per row ({model_manifest['sample_weight_normalisation']}), "
        f"seed {model_manifest['seed']}, StratifiedGroupKFold({model_manifest['cv_folds']}) inside TRAIN, selected by "
        f"{model_manifest['selection_metric']}.",
        f"- Calibration `{model_manifest['calibration_method']}` on group-separated out-of-fold TRAIN scores "
        f"({_fmt(cal['oof_effective_positives'], 1)} Kish effective positives); validation weighted ECE {_fmt(cal['weighted_ece'])}, "
        f"decision-region (score ≥ {cal['decision_region_min_score']}) weighted ECE {_fmt(cal['decision_region_weighted_ece'])} "
        f"→ score_type `{threshold_doc['score_type']}`.",
        f"- Out-of-fold raw-score SD {_fmt(sd_oof)} vs refit TRAIN raw-score SD {_fmt(sd_refit)} (ratio "
        f"{_fmt(sd_refit / sd_oof if sd_oof else None)}): the calibrator learned the fold models' score scale, so a ratio "
        "above 1 makes the refit's probabilities slightly overconfident.",
        f"- {model_manifest['n_features']:,} features; feature policy `{model_manifest['feature_policy_version']}`; "
        f"scikit-learn {model_manifest['sklearn_version']}, Python {model_manifest['python_version']}.", "",
        "## Threshold (chosen on VALIDATION, before TEST)",
        f"- Objective: max weighted recall s.t. weighted precision ≥ {threshold_doc['objective']['min_weighted_precision']} "
        f"with ≥ {threshold_doc['objective']['min_predicted_positives']} predicted positives counted as the Kish effective "
        "number, and a one-sided 95% lower bound of weighted precision at or above the same floor.",
        f"- Threshold {_fmt(threshold_doc['threshold'], 4)} → status **{threshold_doc['status']}**"
        + (f" (fallback: {threshold_doc['fallback']}; the rules remain authoritative)" if threshold_doc["status"] != "production" else "")
        + ".",
        f"- Validation: wP {_fmt(val['weighted_precision'])} (one-sided 95% lower bound "
        f"{_fmt(val['precision_lower_bound_95_one_sided'])}), wR {_fmt(val['weighted_recall'])}, Kish effective predicted "
        f"positives {_fmt(val['kish_effective_predicted_positives'], 1)}, "
        f"{threshold_doc['validation']['predicted_positives']} predicted positives of n = {threshold_doc['validation']['n']} (selection-biased upward: the threshold was chosen on this sample; the TEST rows below are the unbiased estimate).",
        f"- Validation was looked at {validation_looks if validation_looks is not None else 'an unrecorded number of'} "
        "time(s) before this release.", "",
        "## Test result (one run, held-out borrower groups, design-weighted, 95% Jeffreys cluster-bootstrap intervals)",
        f"- Primary (pre-registered): review-queue weighted precision (model suggestions among rules-rejected cases) "
        f"{_ci(rq['model'], 'weighted_precision')} over {rq['n']} resolved rules-rejected cases.",
        f"- n = {ev['n']} resolved cases ({ev['positives']} positive); design-weighted unresolved share {_fmt(ev['unresolved_share']['estimate'])} [{_fmt(ev['unresolved_share']['lower'])}, {_fmt(ev['unresolved_share']['upper'])}]; unresolved share among model suggestions {_fmt(ev['unresolved_share_among_model_positive'])}.",
        "", "| | weighted precision | weighted recall | weighted F1 | TP/FP/FN/TN |", "|---|---|---|---|---|",
        f"| model | {_ci(m, 'weighted_precision')} | {_ci(m, 'weighted_recall')} | {_ci(m, 'weighted_f1')} | {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']} |",
        f"| frozen rules | {_ci(r, 'weighted_precision')} | {_ci(r, 'weighted_recall')} | {_ci(r, 'weighted_f1')} | {r['tp']}/{r['fp']}/{r['fn']}/{r['tn']} |",
        "",
        f"- Secondary: paired delta (model − rules) precision {_interval(d['delta_weighted_precision'])} "
        f"({verdict['delta_precision']}), recall {_interval(d['delta_weighted_recall'])} ({verdict['delta_recall']}); "
        "two comparisons, not multiplicity-adjusted.",
        "- Per region:",
    ]
    for row in ev["per_region"]:
        lines.append(f"  - {row['region']}: n={row['n']}, model wP {_ci(row['model'], 'weighted_precision')}, wR "
                     f"{_ci(row['model'], 'weighted_recall')}; rules wP {_ci(row['rules'], 'weighted_precision')}, wR "
                     f"{_ci(row['rules'], 'weighted_recall')}")
    lines += [
        "", "## Limitations",
        "- A figure marked `(estimate outside interval)` is not a contradiction to reconcile: the point "
        "estimate is the design-weighted plug-in, while the interval is smoothed by a Jeffreys prior that "
        "keeps a cell with no observed error uncertain, so at the boundary the smoothed interval ends "
        "below an unsmoothed 1.000. Read the interval, not the point, wherever the mark appears.",
        "- Inputs are names only: no collateral text (CO's field is a 124-value category list, CT has none), no documents.",
        "- Rates estimate performance on the RESOLVABLE population (cases a screener could label RELEVANT or NOT_RELEVANT), design-weighted by 1 / inclusion_probability — the sampling rate of the (split, stratum, screen cell) each case was actually drawn from, since the boundary screen puts several rates inside one stratum. N_h / n_h is reported per stratum as that stratum's AVERAGE weight and is applied to no row. The INSUFFICIENT_EVIDENCE share of the population and of the model's suggestions is reported above with its interval.",
        "- The split is by borrower group within this snapshot; it is not a chronological forecast and not a transfer claim to other states.",
        "- Linear feature contributions describe the model's calculation, not independent evidence about the business.",
        "- Cross-register (CO↔CT) entity linking, entity resolution and active-loan status are out of scope (v1 screens historical observations).",
        "", "## Reproducibility",
        f"- Protocol `{PROTOCOL_PATH}`; source commit `{model_manifest['source_commit']}` (dirty: {model_manifest['source_dirty']}); "
        f"config sha256 `{model_manifest['config_sha256'][:12]}…`; lock sha256 `{model_manifest['lock_sha256'][:12]}…`.",
        f"- Digests: candidates `{model_manifest['candidates_sha256'][:12]}…`, splits `{model_manifest['splits_sha256'][:12]}…`, "
        f"labels `{model_manifest['labels_sha256'][:12]}…`, pipeline `{model_manifest['pipeline_sha256'][:12]}…`.",
        "- Verify: `shasum -a 256 -c SHA256SUMS` in this directory; `cat model-manifest.json threshold.json dataset-manifest.json "
        "split-manifest.json | shasum -a 256 | cut -c1-12` reproduces the release id.", "",
    ]
    return "\n".join(lines)


def run_build_release(config_path: Path) -> Path:
    """CLI `build-release`. Requires a frozen candidate and its ONE final TEST result; returns the release directory."""
    from ucc_ml.config import artefact_paths, load_config
    from ucc_ml.training import build_model_table, read_plan_a_inputs, split_populations

    cfg = load_config(config_path)
    paths = artefact_paths(cfg)
    fdir, fe_dir = paths.frozen_dir, paths.final_eval_dir
    verify_sha256sums(fdir, required=FROZEN_FILES)
    if not (fe_dir / GUARD_FILE_NAME).exists() or not (fe_dir / FINAL_METRICS_NAME).exists():
        raise RuntimeError(f"no final test result in {fe_dir}: run evaluate-final before build-release")
    guard = read_json(fe_dir / GUARD_FILE_NAME)
    if guard["frozen_sha256sums_sha256"] != sha256_file(fdir / "SHA256SUMS"):
        raise RuntimeError("the frozen candidate changed after the test was evaluated; the test result does not "
                           "describe these weights. Refusing to build. A new candidate needs a fresh benchmark.")
    if guard["metrics_sha256"] != sha256_file(fe_dir / FINAL_METRICS_NAME):
        raise RuntimeError(f"{fe_dir / FINAL_METRICS_NAME} changed after evaluate-final wrote it; refusing to build")
    test_metrics = read_json(fe_dir / FINAL_METRICS_NAME)
    validation_doc = read_json(fdir / "validation-metrics.json")
    model_manifest, threshold_doc = read_json(fdir / "model-manifest.json"), read_json(fdir / "threshold.json")

    inputs = read_plan_a_inputs(paths)
    table = build_model_table(inputs.candidates, inputs.splits, inputs.labels)
    populations = split_populations(inputs.candidates, inputs.splits)
    labelled = table.groupby("split").size()
    resolved_rows = table[table.y.notna()].groupby("split").size()
    lm = inputs.labels_manifest

    # THE LABEL FILE IS PER ROUND, and this manifest is hashed into release_id -- so it has to be right
    # BEFORE the first real bundle, not after. The scalar policy_version is the config default (cli.py
    # passes cfg.version.label_policy_version and never overrides it): on the real labels it reads
    # label_policy_v1 while policy_version_by_round puts main_v1 -- 2,880 of 3,120 rows -- under
    # label_policy_v2. Publishing that scalar alone states the smaller round's policy as if it covered
    # the file. So the map is published, and the scalar is rebuilt from it by the same rule
    # training.labels_summary already applies: one distinct policy collapses to that policy, several are
    # named per round. A manifest that states no map is itself the claim that every round used its
    # scalar, so the map is written out from `rounds` rather than left silent.
    policy_by_round = {str(k): str(v) for k, v in (lm.get("policy_version_by_round") or {}).items()}
    if not policy_by_round:
        policy_by_round = {str(r): str(lm["policy_version"]) for r in lm.get("rounds") or []}
    from ucc_ml.labeling import pooled_policy_version

    policy_version = pooled_policy_version(policy_by_round, str(lm["policy_version"]))
    # The DISCLOSURE map is deliberately not invented when it is absent: one arrangement really does
    # cover some label files, and the card's single-sentence form is the honest rendering for those.
    # When the map IS stated, the pooled sentence is rebuilt from it and must agree exactly -- the same
    # check training.labels_summary makes -- so a bundle can never publish a provenance sentence its own
    # per-round map contradicts.
    disclosure_by_round = {str(k): str(v) for k, v in (lm.get("disclosure_by_round") or {}).items()}
    label_disclosure = str(lm["disclosure"])
    if disclosure_by_round:
        from ucc_ml.labeling import pooled_disclosure

        expected = pooled_disclosure(disclosure_by_round)
        if label_disclosure != expected:
            raise RuntimeError(f"{paths.labels_manifest}: disclosure {label_disclosure!r} does not describe its own "
                               f"disclosure_by_round; expected {expected!r}. Refusing to publish it.")
    dataset_manifest = {
        "dataset_version": str(inputs.candidates.dataset_version.iloc[0]) if len(inputs.candidates) else "unknown",
        "candidates_sha256": sha256_file(paths.candidates_parquet), "candidates_rows": int(len(inputs.candidates)),
        "candidates_rows_by_region": {str(k): int(v) for k, v in inputs.candidates.groupby("region").size().items()},
        "labels_sha256": sha256_file(paths.labels_csv), "labels_rows_raw": int(len(inputs.labels)),
        "labels_manifest_sha256": sha256_file(paths.labels_manifest),
        "policy_version": policy_version, "policy_version_by_round": policy_by_round,
        "labelled_rows_by_split": {s: int(labelled.get(s, 0)) for s in SPLITS},
        "resolved_rows_by_split": {s: int(resolved_rows.get(s, 0)) for s in SPLITS},
        "strata_populations": {s: populations[s] for s in ("validation", "test")},
        "label_disclosure": label_disclosure,
        # The model card renders what the founder actually did from these, instead of asserting it.
        "founder_disagreements": {str(k): dict(v) for k, v in (lm.get("founder_disagreements") or {}).items()},
        "founder_audit": {str(k): dict(v) for k, v in (lm.get("founder_audit") or {}).items()},
        **({"disclosure_by_round": disclosure_by_round} if disclosure_by_round else {}),
    }
    split_manifest = {
        "splits_sha256": sha256_file(paths.splits_parquet),
        "rows_by_split": {str(k): int(v) for k, v in inputs.splits.groupby("split").size().items()},
        "groups_by_split": {str(k): int(v) for k, v in inputs.splits.groupby("split").group_id.nunique().items()},
        "group_scope": "region-scoped", "protocol": PROTOCOL_PATH,
    }
    looks_path = fdir.parent / VALIDATION_LOOKS_NAME
    looks = int(read_json(looks_path)["looks"]) if looks_path.exists() else None

    paths.releases_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="staging-", dir=paths.releases_dir))
    try:
        for name in ("pipeline.joblib", "model-manifest.json", "threshold.json"):
            shutil.copyfile(fdir / name, staging / name)
        write_json(staging / "dataset-manifest.json", dataset_manifest)
        write_json(staging / "split-manifest.json", split_manifest)
        release_id = compute_release_id(staging)
        final_dir = paths.releases_dir / release_id
        if final_dir.exists():
            verify_sha256sums(final_dir, required=RELEASE_FILES)
            if not all((final_dir / n).read_bytes() == (staging / n).read_bytes() for n in RELEASE_ID_MANIFESTS):
                raise RuntimeError(f"{final_dir} exists with different manifests but the same id — impossible unless tampered")
            return final_dir
        write_json(staging / "metrics.json", {"release_id": release_id, "created_at": utc_now_iso(),
                                              "validation": validation_doc, "test": test_metrics})
        (staging / "model-card.md").write_text(render_model_card(release_id, model_manifest, threshold_doc, dataset_manifest,
                                                                 split_manifest, test_metrics, looks), encoding="utf-8")
        if not paths.lock_file.exists():
            raise RuntimeError(f"lock file {paths.lock_file} is missing; the release must pin its environment")
        shutil.copyfile(paths.lock_file, staging / "requirements-ml.lock.txt")
        write_sha256sums(staging, list(RELEASE_FILES))
        staging.rename(final_dir)
        return final_dir
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


@dataclass
class ReleaseBundle:
    release_id: str
    directory: Path
    model: CalibratedModel
    threshold: float
    threshold_status: str
    score_type: str
    model_manifest: dict
    threshold_manifest: dict
    feature_names: np.ndarray
    coef: np.ndarray
    top_k: int = 10


def load_release_bundle(directory: Path, *, verify: bool = True, top_k: int = 10) -> ReleaseBundle:
    """Verify SHA256SUMS (every RELEASE_FILES name required), the directory name and the pipeline digest BEFORE joblib.load."""
    directory = Path(directory)
    if verify:
        verify_sha256sums(directory, required=RELEASE_FILES)
    release_id = compute_release_id(directory)
    if directory.name != release_id:
        raise ValueError(f"directory {directory.name} does not match its computed release_id {release_id}")
    manifest = read_json(directory / "model-manifest.json")
    if manifest["pipeline_sha256"] != sha256_file(directory / "pipeline.joblib"):
        raise BundleIntegrityError("pipeline.joblib digest does not match model-manifest.json")
    model = joblib.load(directory / "pipeline.joblib")
    if not isinstance(model, CalibratedModel):
        raise ValueError("pipeline.joblib is not a CalibratedModel")
    threshold_doc = read_json(directory / "threshold.json")
    names = _feature_names(model.pipeline)
    coef = np.asarray(model.pipeline.named_steps["clf"].coef_[0], dtype=float)
    return ReleaseBundle(release_id=release_id, directory=directory, model=model, threshold=float(threshold_doc["threshold"]),
                         threshold_status=str(threshold_doc["status"]), score_type=str(threshold_doc["score_type"]),
                         model_manifest=manifest, threshold_manifest=threshold_doc, feature_names=names, coef=coef,
                         top_k=int(top_k))


# ---------------------------------------------------------------------------
# Part 2 — predict_cases: the single inference path (Codex §9, contract K8)
# ---------------------------------------------------------------------------
import json  # noqa: E402
from typing import Annotated, Literal  # noqa: E402

from pydantic import BaseModel, ConfigDict, Field, StringConstraints  # noqa: E402

from ucc_ml import legacy  # noqa: E402
from ucc_ml.features import build_feature_frame, top_contributions_from_matrix, transform_features  # noqa: E402

MAX_NAME_CHARS = 300
MAX_LENDERS = 20
BoundedName = Annotated[str, StringConstraints(min_length=1, max_length=MAX_NAME_CHARS)]
LenderName = Annotated[str, StringConstraints(max_length=MAX_NAME_CHARS)]


class CaseInput(BaseModel):
    """One borrower and its lender names. ``region`` is context, never a feature."""
    model_config = ConfigDict(extra="forbid")
    borrower_name: BoundedName
    lender_names: list[LenderName] = Field(default_factory=list, max_length=MAX_LENDERS)
    region: Literal["CO", "CT"]
    case_id: str | None = None
    borrower_name_clean: str | None = None          # batch passes the dataset's value; the API leaves None
    lender_names_clean: list[str] | None = None


class Prediction(BaseModel):
    case_id: str | None
    input_hash: str
    release_id: str
    score: float
    score_type: Literal["calibrated_probability", "raw_score"]
    decision: Literal["suggest_relevant", "review_needed"]
    threshold: float
    baseline_qualifies: bool
    baseline_route: Literal["lender", "borrower", "both", "neither"]
    top_feature_contributions: list[tuple[str, float]]   # ("<block>__<token>", weight), largest |weight| first


def canonical_lenders(values) -> list[str]:
    """Sorted unique trimmed non-blank lender names: the candidate table's lender-set rule (context pack §3,
    ucc_ml.dataset.canonical_lender_set). Restated here because ucc_ml.dataset imports duckdb, which the serving image
    does not carry; test_inference proves the two agree."""
    out: set[str] = set()
    for value in values or []:
        if isinstance(value, str) and value.strip():
            out.add(value.strip())
    return sorted(out)


def input_hash_for(borrower_name: str, lender_names: list[str], region: str) -> str:
    payload = {"borrower_name": " ".join(str(borrower_name).split()),
               "lender_names": sorted({" ".join(str(x).split()) for x in (lender_names or []) if str(x).strip()}),
               "region": str(region)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def cases_to_frame(cases: list[CaseInput]) -> pd.DataFrame:
    rows = []
    for c in cases:
        lenders = canonical_lenders(c.lender_names)
        clean = c.borrower_name_clean if c.borrower_name_clean is not None else legacy.normalize_name(c.borrower_name)[0]
        if c.lender_names_clean is not None:
            lenders_clean = [str(x) for x in c.lender_names_clean]
        else:
            lenders_clean = sorted({n for n in (legacy.normalize_name(l)[0] for l in lenders) if n})
        rows.append({"borrower_name_raw": c.borrower_name, "borrower_name_clean": clean,
                     "lender_names_raw": lenders, "lender_names_clean": lenders_clean})
    return pd.DataFrame(rows, columns=["borrower_name_raw", "borrower_name_clean", "lender_names_raw", "lender_names_clean"])


def predict_cases(cases: list[CaseInput], bundle: ReleaseBundle) -> list[Prediction]:
    """Shared by score-batch (Task 14) and the API (Plan C). Transforms once, scores, explains; the frozen-rules
    baseline comes from ucc_ml.legacy on the canonical lender set (contract K8)."""
    if not cases:
        return []
    frame = build_feature_frame(cases_to_frame(cases))
    X = transform_features(bundle.model.pipeline, frame)
    _raw, scores = bundle.model.scores_from_matrix(X)
    contribs = top_contributions_from_matrix(X, bundle.coef, bundle.feature_names, bundle.top_k)
    out: list[Prediction] = []
    for case, score, contrib in zip(cases, scores, contribs):
        lenders = canonical_lenders(case.lender_names)
        out.append(Prediction(
            case_id=case.case_id, input_hash=input_hash_for(case.borrower_name, case.lender_names, case.region),
            release_id=bundle.release_id, score=float(score), score_type=bundle.score_type,
            decision="suggest_relevant" if float(score) >= bundle.threshold else "review_needed",
            threshold=float(bundle.threshold), baseline_qualifies=legacy.baseline_qualifies(case.borrower_name, lenders),
            baseline_route=legacy.baseline_route(case.borrower_name, lenders),
            top_feature_contributions=[(str(n), float(w)) for n, w in contrib],
        ))
    return out


# ---------------------------------------------------------------------------
# Part 3 — score-batch: every candidate through predict_cases (contract K8: the only batch scorer)
# ---------------------------------------------------------------------------
import math  # noqa: E402
import re  # noqa: E402

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from ucc_ml.features import as_str_list  # noqa: E402

PREDICTIONS_SCHEMA = pa.schema([
    ("case_id", pa.string()), ("region", pa.string()), ("release_id", pa.string()), ("score", pa.float64()),
    ("score_type", pa.string()), ("decision", pa.string()), ("threshold", pa.float64()),
    ("baseline_qualifies", pa.bool_()), ("baseline_route", pa.string()), ("input_hash", pa.string()),
    ("top_feature_contributions", pa.string()), ("validation_error", pa.string()), ("scored_at", pa.string()),
])
RELEASE_DIR_NAME = re.compile(r"[0-9a-f]{12}")


def _clean_or_none(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def score_batch(bundle: ReleaseBundle, candidates_path: Path, out_dir: Path, *, chunk_rows: int = 20000) -> Path:
    """Write <out_dir>/<release_id>.parquet (PREDICTIONS_SCHEMA, one row per candidate) and <release_id>.summary.json.

    Chunked AT THE READ: iter_candidates pulls record batches of at most chunk_rows straight from the Parquet
    file, so neither the whole table nor a parent frame the chunks would be slices of is ever held.
    """
    from ucc_ml.dataset import iter_candidates    # Plan A's streaming reader (K7); duckdb stays out of the serving path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{bundle.release_id}.parquet"
    scored_at = utc_now_iso()
    counts: dict[tuple[str, str, bool], int] = {}
    n_rows = n_invalid = 0
    writer = pq.ParquetWriter(out_path, PREDICTIONS_SCHEMA, compression="zstd")
    try:
        for chunk in iter_candidates(candidates_path, chunk_rows=int(chunk_rows)):
            cases: list[CaseInput] = []
            rows: list[dict] = []
            for r in chunk.itertuples(index=False):
                lenders = as_str_list(r.lender_names_raw)
                try:
                    cases.append(CaseInput(borrower_name=str(r.borrower_name_raw), lender_names=lenders, region=str(r.region),
                                           case_id=str(r.case_id), borrower_name_clean=_clean_or_none(r.borrower_name_clean),
                                           lender_names_clean=as_str_list(r.lender_names_clean)))
                except ValidationError as err:
                    n_invalid += 1
                    first = err.errors()[0]
                    rows.append({"case_id": str(r.case_id), "region": str(r.region), "release_id": bundle.release_id,
                                 "score": None, "score_type": bundle.score_type, "decision": "review_needed",
                                 "threshold": float(bundle.threshold), "baseline_qualifies": bool(r.baseline_qualifies),
                                 "baseline_route": str(r.baseline_route),
                                 "input_hash": input_hash_for(str(r.borrower_name_raw), lenders, str(r.region)),
                                 "top_feature_contributions": "[]",
                                 "validation_error": f"{'.'.join(str(x) for x in first['loc'])}: {first['msg']}",
                                 "scored_at": scored_at})
            region_of = {c.case_id: c.region for c in cases}
            for p in predict_cases(cases, bundle):
                rows.append({"case_id": p.case_id, "region": region_of[p.case_id], "release_id": p.release_id,
                             "score": p.score, "score_type": p.score_type, "decision": p.decision, "threshold": p.threshold,
                             "baseline_qualifies": p.baseline_qualifies, "baseline_route": p.baseline_route,
                             "input_hash": p.input_hash, "top_feature_contributions": json.dumps(p.top_feature_contributions),
                             "validation_error": None, "scored_at": scored_at})
            for row in rows:
                key = (row["region"], row["decision"], bool(row["baseline_qualifies"]))
                counts[key] = counts.get(key, 0) + 1
            n_rows += len(rows)
            writer.write_table(pa.Table.from_pylist(rows, schema=PREDICTIONS_SCHEMA))
    finally:
        writer.close()
    summary = {"release_id": bundle.release_id, "scored_at": scored_at, "rows": int(n_rows), "invalid_rows": int(n_invalid),
               "by_region_decision_baseline": [{"region": k[0], "decision": k[1], "baseline_qualifies": k[2], "n": int(v)}
                                               for k, v in sorted(counts.items())],
               "review_queue_rules_rejected_suggested": int(sum(v for k, v in counts.items()
                                                                if k[1] == "suggest_relevant" and not k[2]))}
    write_json(out_dir / f"{bundle.release_id}.summary.json", summary)
    return out_path


def run_score_batch(config_path: Path, release_dir: Path | None = None) -> Path:
    """CLI `score-batch`: every candidate at artefact_paths(cfg).candidates_parquet scored with the release into
    artefact_paths(cfg).predictions_dir. release_dir None means the only release under artefact_paths(cfg).releases_dir."""
    from ucc_ml.config import artefact_paths, load_config

    cfg = load_config(config_path)
    paths = artefact_paths(cfg)
    rcfg = cfg.section("release")
    if release_dir is None:
        found = (sorted(p for p in paths.releases_dir.iterdir() if p.is_dir() and RELEASE_DIR_NAME.fullmatch(p.name))
                 if paths.releases_dir.is_dir() else [])
        if len(found) != 1:
            raise FileNotFoundError(f"{paths.releases_dir} holds {len(found)} release(s); pass the release directory explicitly")
        release_dir = found[0]
    release_dir = Path(release_dir)
    if not release_dir.is_dir():
        raise FileNotFoundError(f"release directory {release_dir} does not exist")
    bundle = load_release_bundle(release_dir, top_k=int(rcfg["top_k_contributions"]))
    return score_batch(bundle, paths.candidates_parquet, paths.predictions_dir, chunk_rows=int(rcfg["batch_chunk_rows"]))
