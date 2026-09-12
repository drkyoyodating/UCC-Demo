"""Synthetic UCC world for tests, CI, Dagster tests and the Lab preview (contract K10).

Nothing here opens ucc.duckdb or reads ml/data. The world is built with Plan A's own functions, so
every file it writes passes Plan A's readers (contract K7) and follows the real sampling design:

  candidates   CASE_COLUMNS rows; baseline columns from ucc_ml.legacy; region-scoped group_id from
               contracts.make_group_id; written by dataset.write_candidates
  pilot        sampling.make_pilot: PILOT_PER_STRATUM cases per stratum from the WHOLE population
               (inclusion_probability = n_h / N_h)
  splits       splitting.freeze_splits: every group holding a pilot case is in train (K12), the other
               groups fill test, validation and train
  main round   sampling.make_main_round inside every (split, stratum), from non-pilot cases
               (inclusion_probability = n_h / pool_h; validation and test have pool_h == N_h, K13)
  labels       the four K3 statuses only, labelling_round pilot_v1 / main_v1, 10% blind repeats,
               INSUFFICIENT_EVIDENCE in 5% of rules-accepted and 45% of rules-rejected cases
  manifest     labeling.labels_manifest with the K3 statistics

Rules-accepted strata are about 1/12 (CO) and 1/38 (CT) the size of the rules-rejected strata, as in
the real candidate table, so the design weights differ across strata the way they do in real data.
`groups_per_region` is the number of rules-accepted borrower groups in each region.

Names come from vocabularies checked against the frozen rules on 2026-09-12: RULE_POS words are
accepted in the borrower name; HIDDEN_POS names are relevant to a human screener but rejected by the
rules; NEG names positively state another activity. HEAVY_LENDERS pass the lender test, BANK_LENDERS
do not.
"""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ucc_ml import legacy
from ucc_ml.config import ArtefactPaths, artefact_paths, load_config
from ucc_ml.contracts import (
    CASE_COLUMNS,
    FOUNDER_REASON_CODE,
    LABEL_COLUMNS,
    LABELLER_AGREED,
    LABELLER_FOUNDER,
    LABELLER_PASS_A,
    LABELLER_PASS_B,
    LABELLING_ROUNDS,
    SPLITS,
    STRATA,
    UNRESOLVED_REASON_CODE,
    make_borrower_key,
    make_case_id,
    make_group_id,
)
from ucc_ml.provenance import sha256_bytes, write_json

POLICY_VERSION = "label_policy_v1"
DATASET_VERSION = "v1"

RULE_POS: tuple[str, ...] = ("EXCAVATING", "PAVING", "CONCRETE PUMPING", "GRADING", "DEMOLITION", "EARTHMOVING",
                             "TRENCHING", "DIRT WORK", "CRANE SERVICE")
HIDDEN_POS: tuple[str, ...] = ("SITE PREP", "CONSTRUCTION", "HEAVY HAUL", "CIVIL CONTRACTORS", "DIRT MOVERS")
NEG: tuple[str, ...] = ("DENTAL", "LAW OFFICE", "RESTAURANT", "FARMS", "BAPTIST CHURCH", "BAKERY", "REALTY",
                        "INSURANCE AGENCY", "AUTO SALES", "HAIR SALON")
HEAVY_LENDERS: tuple[str, ...] = ("CATERPILLAR FINANCIAL SERVICES CORPORATION", "KOMATSU FINANCIAL LIMITED PARTNERSHIP",
                                  "WAGNER EQUIPMENT CO", "VOLVO FINANCIAL SERVICES")
BANK_LENDERS: tuple[str, ...] = ("WELLS FARGO BANK NA", "US BANK NA", "FIRSTBANK", "PEOPLES UNITED BANK",
                                 "LIBERTY BANK", "DE LAGE LANDEN FINANCIAL SERVICES INC")
FIRST_NAMES: tuple[str, ...] = ("SMITH", "JONES", "GARCIA", "MILLER", "DAVIS", "LOPEZ", "WILSON", "ANDERSON",
                                "THOMAS", "TAYLOR", "MOORE", "MARTIN", "JACKSON", "WHITE", "HARRIS", "CLARK")
SECOND_NAMES: tuple[str, ...] = ("ALDEN", "BRIXTON", "CARVER", "DALTON", "ELMORE", "FENWICK", "GARLAND", "HOLLIS",
                                 "IRVING", "JARVIS", "KENDALL", "LANGLEY", "MERRICK", "NORTON", "OAKLEY", "PRESTON",
                                 "QUINCY", "RADLEY", "SUTTON", "THORNE")
PLACES: tuple[str, ...] = ("ALPINE", "SUMMIT", "VALLEY", "MESA", "CANYON", "PRAIRIE", "HARBOR", "LAKESIDE", "PINE",
                           "CEDAR")
SUFFIXES: tuple[str, ...] = ("LLC", "INC", "CO", "")
CITIES: dict[str, tuple[tuple[str, str], ...]] = {
    "CO": (("DENVER", "80202"), ("GRAND JUNCTION", "81501"), ("PUEBLO", "81003")),
    "CT": (("HARTFORD", "06103"), ("NEW HAVEN", "06510"), ("STAMFORD", "06901")),
}
KIND_VOCABULARY: dict[str, tuple[str, ...]] = {"rule_pos": RULE_POS, "hidden_pos": HIDDEN_POS, "neg": NEG}
REJECTED_GROUPS_PER_ACCEPTED_GROUP: dict[str, int] = {"CO": 12, "CT": 38}
INSUFFICIENT_SHARE: dict[str, float] = {"accepted": 0.05, "rejected": 0.45}
LABEL_NOISE = 0.04
REPEAT_FRACTION = 0.10
PILOT_PER_STRATUM = 50
MAIN_ROUND_PER_STRATUM: dict[str, int] = {"train": 60, "validation": 40, "test": 50}
SPLIT_RATIOS: dict[str, float] = {"train": 0.65, "validation": 0.15, "test": 0.20}
REASON_CODE: dict[str, str] = {"RELEVANT": "R_BORROWER_TRADE_WORD", "NOT_RELEVANT": "N_OTHER_INDUSTRY_EXPLICIT",
                               "INSUFFICIENT_EVIDENCE": "I_GENERIC_NAME"}
INCONSISTENT_REPEAT_LABEL: dict[str, str] = {"RELEVANT": "INSUFFICIENT_EVIDENCE", "NOT_RELEVANT": "INSUFFICIENT_EVIDENCE",
                                             "INSUFFICIENT_EVIDENCE": "NOT_RELEVANT"}
LABELLED_AT = "2026-09-12T10:00:00Z"
SYNTHETIC_LOCK = "# synthetic lock file written by ucc_ml.synthetic.write_world\nscikit-learn==1.9.1\nnumpy==2.5.3\n"
#: The only settings in which the synthetic config differs from ml/configs/v1.yaml (a fast test profile).
FAST_PROFILE: dict[str, dict] = {"training": {"cv_folds": 3}, "evaluation": {"bootstrap_resamples": 50},
                                 "budget": {"main_round": {s: 4 * n for s, n in MAIN_ROUND_PER_STRATUM.items()}}}
NULLABLE_TEXT_COLUMNS: tuple[str, ...] = ("lineage_id", "borrower_name_clean", "borrower_suffix", "borrower_city",
                                          "borrower_state", "borrower_zip", "source_filing_type", "source_status")
DESIGN_COLUMNS: tuple[str, ...] = ("case_id", "labelling_round", "split", "sampling_stratum", "N_h", "pool_h", "n_h",
                                   "inclusion_probability")


@dataclass
class World:
    candidates: pd.DataFrame        # CASE_COLUMNS
    splits: pd.DataFrame            # SPLIT_COLUMNS
    labels: pd.DataFrame            # LABEL_COLUMNS, is_repeat bool, inclusion_probability float
    labels_manifest: dict           # labeling.labels_manifest output; labels_sha256 = sha256 of labels_csv_bytes(labels)
    design: pd.DataFrame            # DESIGN_COLUMNS: one row per drawn case (pilot_v1 and main_v1)


def _group_names(rng: np.random.Generator, vocabulary: tuple[str, ...], count: int) -> list[str]:
    """`count` distinct '<first> <second> <place> <word>' names in a seeded order."""
    size = len(FIRST_NAMES) * len(SECOND_NAMES) * len(PLACES) * len(vocabulary)
    if count > size:
        raise ValueError(f"asked for {count} distinct names but the vocabulary has {size}")
    names = []
    for pick in rng.permutation(size)[:count].tolist():
        pick, word = divmod(pick, len(vocabulary))
        pick, place = divmod(pick, len(PLACES))
        first, second = divmod(pick, len(SECOND_NAMES))
        names.append(f"{FIRST_NAMES[first]} {SECOND_NAMES[second]} {PLACES[place]} {vocabulary[word]}")
    return names


def _group_plan(groups_per_region: int, region: str) -> list[tuple[str, str]]:
    """(kind, lender pool) per group: rules-accepted groups first, then the rules-rejected ones."""
    rule_pos = round(0.70 * groups_per_region)
    hidden_heavy = round(0.15 * groups_per_region)
    neg_heavy = groups_per_region - rule_pos - hidden_heavy
    rejected = REJECTED_GROUPS_PER_ACCEPTED_GROUP[region] * groups_per_region
    hidden_bank = round(0.30 * rejected)
    return ([("rule_pos", "any")] * rule_pos + [("hidden_pos", "heavy")] * hidden_heavy
            + [("neg", "heavy")] * neg_heavy + [("hidden_pos", "bank")] * hidden_bank
            + [("neg", "bank")] * (rejected - hidden_bank))


def _lenders(rng: np.random.Generator, pool: str) -> list[str]:
    if pool == "heavy":
        chosen = [HEAVY_LENDERS[int(i)] for i in rng.integers(0, len(HEAVY_LENDERS), size=int(rng.integers(1, 3)))]
    else:
        source = BANK_LENDERS if pool == "bank" or rng.random() < 0.5 else HEAVY_LENDERS
        chosen = [source[int(i)] for i in rng.integers(0, len(source), size=int(rng.integers(0, 3)))]
    return sorted(set(chosen))


def _candidates(rng: np.random.Generator, groups_per_region: int) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict] = []
    kind_of: dict[str, str] = {}
    for region in ("CO", "CT"):
        plan = _group_plan(groups_per_region, region)
        names = {kind: iter(_group_names(rng, KIND_VOCABULARY[kind], n))
                 for kind, n in Counter(kind for kind, _ in plan).items()}
        for kind, pool in plan:
            suffix = SUFFIXES[int(rng.integers(len(SUFFIXES)))]
            base = f"{next(names[kind])} {suffix}".strip()
            city, zip_code = CITIES[region][int(rng.integers(len(CITIES[region])))]
            for variant in range(int(rng.integers(1, 4))):
                raw = (base, base + ".", base.replace(" ", "  "))[variant]
                lenders = _lenders(rng, pool)
                name_clean, name_suffix = legacy.normalize_name(raw)
                file_id = f"{region}-F{len(rows):07d}"
                borrower_key = make_borrower_key(raw, f"{len(rows) % 900 + 1} MAIN ST", city, region, zip_code)
                case_id = make_case_id(region, file_id, borrower_key)
                kind_of[case_id] = kind
                rows.append({
                    "case_id": case_id, "dataset_version": DATASET_VERSION, "region": region, "file_id": file_id,
                    "lineage_id": f"M-{file_id}", "borrower_key": borrower_key, "borrower_name_raw": raw,
                    "borrower_name_clean": name_clean, "borrower_suffix": name_suffix,
                    "lender_names_raw": lenders,
                    "lender_names_clean": sorted({c for c in (legacy.normalize_name(l)[0] for l in lenders) if c}),
                    "earliest_observed_date": date(2015, 3, 1), "latest_observed_date": date(2015, 3, 1),
                    "borrower_city": city, "borrower_state": region, "borrower_zip": zip_code, "source_row_count": 1,
                    "source_filing_type": "ucc" if region == "CO" else "ORIG FIN STMT",
                    "source_status": "false" if region == "CO" else "Active",
                    "baseline_qualifies": legacy.baseline_qualifies(raw, lenders),
                    "baseline_route": legacy.baseline_route(raw, lenders),
                    "group_id": make_group_id(region, name_clean, raw),
                })
    candidates = pd.DataFrame(rows, columns=list(CASE_COLUMNS)).sort_values("case_id", kind="mergesort").reset_index(drop=True)
    for column in NULLABLE_TEXT_COLUMNS:   # pandas 3 stores None as NaN in its str dtype; Plan A keeps None (dataset.py)
        candidates[column] = pd.Series([v if isinstance(v, str) else None for v in candidates[column]],
                                       index=candidates.index, dtype=object)
    return candidates, kind_of


def _label_row(design_row, round_name: str, label: str, reason_code: str, labeller_id: str, status: str,
               is_repeat: bool, reason: str) -> dict:
    return {"case_id": design_row.case_id, "label": label, "reason_code": reason_code, "reason": reason,
            "labeller_id": labeller_id, "labelled_at": LABELLED_AT, "policy_version": POLICY_VERSION,
            "sampling_stratum": design_row.sampling_stratum,
            "inclusion_probability": float(design_row.inclusion_probability), "adjudication_status": status,
            "is_repeat": is_repeat, "labelling_round": round_name}


def _labels(rng: np.random.Generator, design: pd.DataFrame, kind_of: dict[str, str]) -> pd.DataFrame:
    rows: list[dict] = []
    # Only the rounds this world actually designed. LABELLING_ROUNDS is the registry of rounds that MAY
    # exist, never a claim that each one does; the real labels.csv carries two of them.
    for round_name in [r for r in LABELLING_ROUNDS if (design.labelling_round == r).any()]:
        originals: list[dict] = []
        for r in design[design.labelling_round == round_name].itertuples(index=False):
            kind = kind_of[r.case_id]
            if rng.random() < INSUFFICIENT_SHARE[r.sampling_stratum.split(":")[1]]:
                label = "INSUFFICIENT_EVIDENCE"
            else:
                label = "NOT_RELEVANT" if kind == "neg" else "RELEVANT"
                if rng.random() < LABEL_NOISE:
                    label = "RELEVANT" if label == "NOT_RELEVANT" else "NOT_RELEVANT"
            u = rng.random()
            if u < 0.78:
                status = "model_agreed"
            elif u < 0.88:
                status = "founder_confirmed"
            elif u < 0.97:
                status = "founder_adjudicated"
            else:
                # The two blind passes disagreed and nobody adjudicated it. The contract pins such a row
                # to INSUFFICIENT_EVIDENCE, the assembly-time code and the pass pair as labeller. It has
                # to occur here, or nothing downstream that must EXCLUDE this status is ever exercised.
                status = "blind_unresolved"
                label = "INSUFFICIENT_EVIDENCE"
            labeller = LABELLER_FOUNDER if status in ("founder_confirmed", "founder_adjudicated") else LABELLER_AGREED
            code = (FOUNDER_REASON_CODE if status == "founder_adjudicated"
                    else UNRESOLVED_REASON_CODE if status == "blind_unresolved"
                    else REASON_CODE[label])
            originals.append(_label_row(r, round_name, label, code, labeller, status, False, f"synthetic {kind} name"))
        rows.extend(originals)
        picks = sorted(rng.choice(len(originals), size=round(REPEAT_FRACTION * len(originals)), replace=False).tolist())
        for i, pick in enumerate(picks):
            original = originals[pick]
            label = original["label"] if rng.random() >= 0.05 else INCONSISTENT_REPEAT_LABEL[original["label"]]
            rows.append({**original, "label": label, "reason_code": REASON_CODE[label],
                         "labeller_id": LABELLER_PASS_A if i % 2 == 0 else LABELLER_PASS_B,
                         "adjudication_status": "blind_repeat", "is_repeat": True, "reason": "synthetic blind repeat"})
    labels = pd.DataFrame(rows, columns=list(LABEL_COLUMNS))
    labels["is_repeat"] = labels.is_repeat.astype(bool)
    labels["inclusion_probability"] = labels.inclusion_probability.astype(float)
    return labels


def labels_csv_bytes(labels: pd.DataFrame) -> bytes:
    """labels.csv exactly as Plan A's labeling.write_csv writes it (UTF-8, '\\n' line ends, minimal quoting)."""
    text = labels[list(LABEL_COLUMNS)].to_csv(index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    return text.encode("utf-8")


def _labels_manifest(labels: pd.DataFrame) -> dict:
    """The K3 manifest through Plan A's labeling.labels_manifest, with pass and founder statistics consistent with
    the rows: founder_adjudicated rows are decided disagreements, founder_confirmed rows are audited agreed rows."""
    from ucc_ml.labeling import labels_manifest

    agreements, founder = {}, {}
    # Only rounds with rows: labels_manifest derives counts_by_round from these keys, so an empty round
    # here publishes a manifest describing rounds the labels do not contain.
    for round_name in [r for r in LABELLING_ROUNDS if (labels.labelling_round == r).any()]:
        rows = labels[labels.labelling_round == round_name]
        originals, repeats = rows[~rows.is_repeat], rows[rows.is_repeat]
        decided = int((originals.adjudication_status == "founder_adjudicated").sum())
        confirmed = int((originals.adjudication_status == "founder_confirmed").sum())
        label_of = dict(zip(originals.case_id, originals.label))
        consistency = {}
        for key, labeller in (("pass_a", LABELLER_PASS_A), ("pass_b", LABELLER_PASS_B)):
            mine = repeats[repeats.labeller_id == labeller]
            same = int(sum(label_of[c] == label for c, label in zip(mine.case_id, mine.label)))
            consistency[key] = {"n": int(len(mine)), "consistent": same, "rate": same / len(mine) if len(mine) else None}
        n = int(len(originals))
        # A blind_unresolved row IS a pass disagreement that nobody settled, so it counts against
        # agreement and appears as undecided. A fixture that always reported 0 undecided would let a
        # later task pass against a world the real pipeline cannot produce: main_v1 reports 83.
        unresolved = int((originals.adjudication_status == "blind_unresolved").sum())
        agreed = n - decided - unresolved
        agreements[round_name] = {"pass_agreement": {"n": n, "agreed": agreed, "rate": agreed / n if n else None},
                                  "repeat_consistency": consistency}
        founder[round_name] = {
            "disagreements": {"n": decided + unresolved, "decided": decided, "undecided": unresolved},
            "audit": {"n_selected": confirmed, "n_audited": confirmed, "n_confirmed": confirmed, "n_overturned": 0,
                      "agreement_rate": 1.0 if confirmed else None},
        }
    return labels_manifest(labels, sha256_bytes(labels_csv_bytes(labels)), agreements, founder, POLICY_VERSION,
                           extra={"source": "synthetic world (ucc_ml.synthetic.make_world)"})


def make_world(seed: int = 7, groups_per_region: int = 150) -> World:
    from ucc_ml.sampling import make_main_round, make_pilot, split_stratum_populations
    from ucc_ml.splitting import freeze_splits

    rng = np.random.default_rng(seed)
    candidates, kind_of = _candidates(rng, groups_per_region)
    pilot = make_pilot(candidates, per_stratum=PILOT_PER_STRATUM, seed=seed)
    splits, _ = freeze_splits(candidates, pilot.case_id, SPLIT_RATIOS, seed, POLICY_VERSION, "synthetic")
    pools = split_stratum_populations(candidates, splits, exclude_case_ids=pilot.case_id)
    allocation = {split: {h: min(MAIN_ROUND_PER_STRATUM[split], pools[split][h]) for h in STRATA} for split in SPLITS}
    main = make_main_round(candidates, splits, pilot.case_id, allocation, seed)
    split_of = dict(zip(splits.case_id, splits.split))
    design = pd.concat([
        pilot.assign(labelling_round="pilot_v1", split=[split_of[c] for c in pilot.case_id]),
        main.assign(labelling_round="main_v1"),
    ], ignore_index=True)[list(DESIGN_COLUMNS)]
    labels = _labels(rng, design, kind_of)
    return World(candidates=candidates, splits=splits, labels=labels, labels_manifest=_labels_manifest(labels),
                 design=design)


def synthetic_config() -> dict:
    """The synthetic ml/configs/v1.yaml: every section of the real config with its values, plus FAST_PROFILE."""
    return {
        "version": {"dataset_version": DATASET_VERSION, "label_policy_version": POLICY_VERSION,
                    "baseline_version": "heavy_filter_v1"},
        "seed": 20260912,
        "paths": {"source_duckdb": "ucc.duckdb", "snapshot_dir": "ml/data/snapshots/v1",
                  "candidates_dir": "ml/data/candidates/v1", "pilot_dir": "ml/data/pilot/v1",
                  "main_round_dir": "ml/data/main/v1", "ablation_dir": "ml/data/ablation/v1",
                  "screening_dir": "ml/data/screening/v1", "splits_dir": "ml/data/splits/v1",
                  "labels_dir": "ml/data/labels/v1", "predictions_dir": "ml/data/predictions",
                  "artifacts_dir": "ml/artifacts", "mlflow_dir": "ml/mlflow", "reports_dir": "ml/reports",
                  "public_data_dir": "docs/data/ml", "specs_dir": "ml/specs"},
        "eligibility": {"year_min": "1990",
                        "junk_addresses": ["", "NONE", "NONE PROVIDED", "NA", "N/A", "UNKNOWN", "SAME", "COMPANY",
                                           "NOT PROVIDED", "TBD", "X", "XX", "NO ADDRESS", "ADDRESS UNKNOWN", "VARIOUS"]},
        "sampling": {"strata": list(STRATA), "pilot_per_stratum": PILOT_PER_STRATUM, "repeat_fraction": REPEAT_FRACTION},
        "splits": dict(SPLIT_RATIOS),
        # Mirrors the real labelling block exactly, including the per-round policy and disagreement
        # settings. The mirror is the point: a test compares the two, so the fixture cannot quietly
        # describe a different world from the one the pipeline actually runs.
        "labelling": {"labels": ["RELEVANT", "NOT_RELEVANT", "INSUFFICIENT_EVIDENCE"], "reason_max_chars": 300,
                      "disclosure": "model-labelled, founder-adjudicated", "chunk_size": 200,
                      "founder_audit_per_split_stratum": 10,
                      "policy_version_by_round": {"pilot_v1": "label_policy_v1",
                                                  "ablation_v1": "label_policy_v2",
                                                  "yield_probe_v1": "label_policy_v2",
                                                  "main_v1": "label_policy_v2",
                                                  "queue_v1": "label_policy_v2"},
                      "rounds_in_labels": ["pilot_v1", "main_v1"],
                      "disagreement_policy_by_round": {"pilot_v1": "founder", "main_v1": "unresolved"}},
        "budget": {"main_round": dict(FAST_PROFILE["budget"]["main_round"])},
        "mlflow": {"experiment": "ucc-ml-v1"},
        "features": {"policy_version": "features_v1", "lender_join": " | ",
                     "word": {"analyzer": "word", "ngram_range": [1, 2], "lowercase": True,
                              "token_pattern": r"(?u)\b\w+\b"},
                     "char": {"analyzer": "char_wb", "ngram_range": [3, 5], "lowercase": False}},
        "training": {"seed": 20260912, "cv_folds": FAST_PROFILE["training"]["cv_folds"], "c_grid": [0.1, 1.0, 10.0],
                     "class_weight": None, "weighting": "inverse_inclusion_probability_normalised",
                     "selection_metric": "weighted_average_precision", "candidate_variant": "borrower_lender",
                     "max_iter": 2000, "artifacts_dir": "ml/artifacts/train/v1"},
        "calibration": {"method": "auto", "isotonic_min_effective_positives": 200,
                        "max_weighted_ece_for_probability": 0.10, "n_bins": 10, "decision_region_min_score": 0.3,
                        "max_decision_region_ece": 0.15, "decision_region_min_cases": 20},
        "threshold": {"min_weighted_precision": 0.95, "min_predicted_positives": 30},
        "evaluation": {"bootstrap_resamples": FAST_PROFILE["evaluation"]["bootstrap_resamples"],
                       "bootstrap_seed": 20260912, "ci_level": 0.95},
        "release": {"top_k_contributions": 10, "batch_chunk_rows": 20000},
        "service": {"max_body_bytes": 16384, "max_name_chars": 300, "max_lenders": 20, "max_lender_chars": 300,
                    "allowed_origins": ["https://drkyoyodating.github.io", "http://127.0.0.1:8080",
                                        "http://localhost:8080"],
                    "rate_capacity": 30.0, "rate_refill_per_second": 0.5},
        "export": {"unlabelled_cap": 2000,
                   "curated_examples": {"per_cell_cap": 9, "error_cap_per_cell": 3, "minimum": 50, "maximum": 100},
                   "latency": {"warm_calls": 200, "batch_size": 1000}},
    }


def write_config(root: Path, **overrides) -> Path:
    """Write <root>/ml/configs/v1.yaml, loadable by ucc_ml.config.load_config. A dict override is merged into its
    section (write_config(root, threshold={"min_weighted_precision": 1.01})); any other value replaces the key."""
    cfg = synthetic_config()
    for key, value in overrides.items():
        cfg[key] = {**cfg[key], **value} if isinstance(value, dict) and isinstance(cfg.get(key), dict) else value
    out = Path(root) / "ml" / "configs" / "v1.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = "# Synthetic run config written by ucc_ml.synthetic.write_config (fast profile: cv_folds 3, 50 resamples)\n"
    out.write_text(header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def write_world(world: World, root: Path) -> ArtefactPaths:
    """Write candidates, splits, labels.csv, labels_manifest.json and the lock file at the K5 locations under root.
    ArtefactPaths come from a loaded RunConfig, so the default synthetic config is written first when none exists."""
    from ucc_ml.dataset import write_candidates, write_frame

    config_path = Path(root) / "ml" / "configs" / "v1.yaml"
    if not config_path.exists():
        write_config(root)
    paths = artefact_paths(load_config(config_path))
    write_candidates(world.candidates, paths.candidates_parquet)
    write_frame(world.splits, paths.splits_parquet)
    data = labels_csv_bytes(world.labels)
    if sha256_bytes(data) != world.labels_manifest["labels_sha256"]:
        raise ValueError("world.labels changed after make_world: labels_manifest.labels_sha256 no longer matches")
    paths.labels_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.labels_csv.write_bytes(data)
    write_json(paths.labels_manifest, world.labels_manifest)
    paths.lock_file.parent.mkdir(parents=True, exist_ok=True)
    paths.lock_file.write_text(SYNTHETIC_LOCK, encoding="utf-8")
    return paths


def build_synthetic_release(root: Path) -> Path:
    """Write make_world() and the synthetic config under root, then run the REAL train -> freeze-candidate ->
    evaluate-final -> build-release and return the release directory (<root>/ml/artifacts/releases/<release_id>)."""
    from ucc_ml.evaluation import run_evaluate_final
    from ucc_ml.inference import run_build_release
    from ucc_ml.training import run_freeze_candidate, run_train

    write_world(make_world(), root)
    config_path = write_config(root)
    run_train(config_path)
    run_freeze_candidate(config_path)
    run_evaluate_final(config_path)
    return run_build_release(config_path)
