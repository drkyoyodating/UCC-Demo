"""Typed access to ml/configs/<run>.yaml (contract K4) and the artefact locations every plan uses (K5).

The repository root is derived from the config's own location (``<root>/ml/configs/<file>.yaml``), so
every ``paths.*`` entry resolves to an absolute path without a second argument. Sections this module
does not validate (budget, mlflow, features, training, calibration, threshold, evaluation, release,
service, export) are kept verbatim in ``model_extra`` and read with ``RunConfig.section(name)``. A YAML
key that appears twice at any level is refused instead of silently overwritten, so no plan can shadow
another plan's setting. The release id is computed from the release bundle (Plan B), never configured.
"""
from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

DISCLOSURE = "model-labelled, founder-adjudicated"


class Versions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_version: str
    label_policy_version: str
    baseline_version: str


class Paths(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_duckdb: Path
    snapshot_dir: Path
    candidates_dir: Path
    pilot_dir: Path
    main_round_dir: Path = Path("ml/data/main/v1")
    ablation_dir: Path = Path("ml/data/ablation/v1")
    screening_dir: Path = Path("ml/data/screening/v1")
    splits_dir: Path
    labels_dir: Path
    predictions_dir: Path = Path("ml/data/predictions")
    artifacts_dir: Path
    mlflow_dir: Path
    reports_dir: Path
    public_data_dir: Path
    specs_dir: Path


class Eligibility(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    year_min: str
    junk_addresses: list[str]

    @field_validator("year_min")
    @classmethod
    def _four_digits(cls, v: str) -> str:
        if not re.fullmatch(r"\d{4}", v):
            raise ValueError("eligibility.year_min must be a four-digit string, e.g. \"1990\"")
        return v


class Sampling(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strata: list[str]
    pilot_per_stratum: int
    repeat_fraction: float

    @field_validator("repeat_fraction")
    @classmethod
    def _fraction(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError("sampling.repeat_fraction must be in [0, 1)")
        return v


class Splits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train: float
    validation: float
    test: float

    @model_validator(mode="after")
    def _sum_to_one(self) -> "Splits":
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"splits.train+validation+test must sum to 1, got {total}")
        return self


class Labelling(BaseModel):
    model_config = ConfigDict(extra="forbid")
    labels: list[str]
    reason_max_chars: int
    disclosure: str
    chunk_size: int = 200
    founder_audit_per_split_stratum: int = 10
    #: policy each round was labelled under; rounds absent here use version.label_policy_version.
    #: Rows of different policy versions are NEVER pooled in a reported statistic.
    policy_version_by_round: dict[str, str] = {}

    @field_validator("disclosure")
    @classmethod
    def _literal_disclosure(cls, v: str) -> str:
        if v != DISCLOSURE:
            raise ValueError(f"labelling.disclosure must be exactly {DISCLOSURE!r}")
        return v

    @field_validator("chunk_size")
    @classmethod
    def _positive_chunk(cls, v: int) -> int:
        if v < 1:
            raise ValueError("labelling.chunk_size must be >= 1")
        return v


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    version: Versions
    seed: int
    paths: Paths
    eligibility: Eligibility
    sampling: Sampling
    splits: Splits
    labelling: Labelling
    repo_root: Path
    config_path: Path
    config_sha256: str

    def path(self, key: str) -> Path:
        if key not in Paths.model_fields:
            raise KeyError(f"config has no path {key!r}")
        return (self.repo_root / getattr(self.paths, key)).resolve()

    def section(self, name: str) -> dict:
        extra = self.model_extra or {}
        if name not in extra:
            raise KeyError(f"config has no section {name!r}")
        return copy.deepcopy(extra[name])


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a mapping key appearing twice (PyYAML otherwise keeps the last one)."""


def _unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    loader.flatten_mapping(node)
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_config(path: Path | str) -> RunConfig:
    cfg_path = Path(path).resolve()
    if cfg_path.parent.name != "configs" or cfg_path.parent.parent.name != "ml":
        raise ValueError(f"{cfg_path}: a run config must live at <repo>/ml/configs/<name>.yaml")
    raw = cfg_path.read_bytes()
    data = yaml.load(raw, Loader=_UniqueKeyLoader)
    if not isinstance(data, dict):
        raise ValueError(f"{cfg_path}: top level must be a mapping")
    for computed in ("repo_root", "config_path", "config_sha256"):
        if computed in data:
            raise ValueError(f"{cfg_path}: {computed!r} is computed by load_config, not configured")
    data["repo_root"] = str(cfg_path.parents[2])      # <root>/ml/configs/v1.yaml
    data["config_path"] = str(cfg_path)
    data["config_sha256"] = hashlib.sha256(raw).hexdigest()
    return RunConfig.model_validate(data)


@dataclass(frozen=True)
class ArtefactPaths:
    repo_root: Path
    snapshot_manifest: Path
    candidates_parquet: Path
    candidates_manifest: Path
    reconciliation: Path
    pilot_cases: Path
    pilot_manifest: Path
    main_cases: Path
    main_manifest: Path
    ablation_cases: Path
    ablation_manifest: Path
    screen_cells: Path
    screening_manifest: Path
    splits_parquet: Path
    split_manifest: Path
    labels_csv: Path
    labels_manifest: Path
    mlflow_db: Path
    mlflow_artifacts: Path
    frozen_dir: Path
    final_eval_dir: Path
    releases_dir: Path
    predictions_dir: Path
    public_data_dir: Path
    lock_file: Path


def artefact_paths(cfg: RunConfig) -> ArtefactPaths:
    """The one place every artefact location is spelled; Plans B and C never build these paths."""
    artifacts = cfg.path("artifacts_dir")
    return ArtefactPaths(
        repo_root=cfg.repo_root,
        snapshot_manifest=cfg.path("snapshot_dir") / "manifest.json",
        candidates_parquet=cfg.path("candidates_dir") / "candidates.parquet",
        candidates_manifest=cfg.path("candidates_dir") / "candidates_manifest.json",
        reconciliation=cfg.path("candidates_dir") / "reconciliation.json",
        pilot_cases=cfg.path("pilot_dir") / "pilot_cases.parquet",
        pilot_manifest=cfg.path("pilot_dir") / "pilot_manifest.json",
        main_cases=cfg.path("main_round_dir") / "main_cases.parquet",
        main_manifest=cfg.path("main_round_dir") / "main_manifest.json",
        ablation_cases=cfg.path("ablation_dir") / "ablation_cases.parquet",
        ablation_manifest=cfg.path("ablation_dir") / "ablation_manifest.json",
        screen_cells=cfg.path("screening_dir") / "screen_cells.parquet",
        screening_manifest=cfg.path("screening_dir") / "screening_manifest.json",
        splits_parquet=cfg.path("splits_dir") / "splits.parquet",
        split_manifest=cfg.path("splits_dir") / "split_manifest.json",
        labels_csv=cfg.path("labels_dir") / "labels.csv",
        labels_manifest=cfg.path("labels_dir") / "labels_manifest.json",
        mlflow_db=cfg.path("mlflow_dir") / "mlflow.db",
        mlflow_artifacts=cfg.path("mlflow_dir") / "artifacts",
        frozen_dir=artifacts / "frozen" / "v1",
        final_eval_dir=artifacts / "final_eval" / "v1",
        releases_dir=artifacts / "releases",
        predictions_dir=cfg.path("predictions_dir"),
        public_data_dir=cfg.path("public_data_dir"),
        lock_file=(cfg.repo_root / "ml" / "requirements-ml.lock.txt").resolve(),
    )
