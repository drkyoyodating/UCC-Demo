"""Blind labelling for the UCC ML Lab (contracts K3, K13, K14).

Round-aware from the start: the pilot round (pilot_v1) and the main round (main_v1) run through the
same functions. The module grows over Tasks 13-19:
  Task 13  round paths, the chunked blind queue with hidden repeats, its private key, pre-registration
  Task 15  labeller briefs, structured-output validation, raw labeller CSVs, labelling status
  Task 16  import of both passes (vocabulary checks, hidden-repeat de-aliasing), pass digests
  Task 17  pass agreement, blind-repeat consistency, founder review selection and workbook
  Task 18  founder decision import and the K3 assembly of one round's labels
  Task 19  labels.csv, read_labels, labels_manifest.json and the round report

The blind queue (context pack §6) carries exactly case_id, borrower_name_raw, lender_names_raw, city,
state -- no baseline flag, no stratum, no score, no address, no date. A round's cases are ordered by a
seeded sha256 rank and cut into chunks of `chunk_size` (one labeller agent per chunk per pass, K14).
Inside every chunk round-half-up(repeat_fraction * chunk rows) cases are repeated under
repeat_alias(seed, round, case_id), which cannot be told apart from a real case_id, and the chunk is
re-ordered by rank. The private key maps every queue row back to its case, stratum and design weight.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ucc_ml.config import RunConfig, artefact_paths
from ucc_ml.contracts import LABEL_DISCLOSURE, LABELLING_ROUNDS, canonical_json, check_column, is_sha256_hex, sha256_hex
from ucc_ml.provenance import sha256_file
from ucc_ml.sampling import draw_rank

QUEUE_COLUMNS: tuple[str, ...] = ("case_id", "borrower_name_raw", "lender_names_raw", "city", "state")
KEY_COLUMNS: tuple[str, ...] = (
    "queue_case_id", "case_id", "is_repeat", "part", "sampling_stratum", "N_h", "pool_h", "n_h",
    "inclusion_probability", "baseline_qualifies", "queue_rank",
)
ROUND_CASE_COLUMNS: tuple[str, ...] = (
    "case_id", "borrower_name_raw", "lender_names_raw", "borrower_city", "borrower_state",
    "sampling_stratum", "N_h", "pool_h", "n_h", "inclusion_probability", "baseline_qualifies",
)
LENDER_JOIN = " | "
PASSES: tuple[str, ...] = ("a", "b")


@dataclass(frozen=True)
class RoundPaths:
    """Every file of one labelling round. Private files live under ml/data; digests under docs/data/ml."""
    round_name: str
    cases: Path
    manifest: Path
    queue_dir: Path
    key: Path
    report: Path
    labels_dir: Path
    raw_dir: Path
    structured_dir: Path
    briefs_dir: Path
    passes_dir: Path
    agreement: Path
    workbook: Path
    review_manifest: Path
    founder_decisions: Path
    preregistration: Path
    passes_digest: Path
    founder_digest: Path

    def queue_part(self, part: int) -> Path:
        return self.queue_dir / f"queue_{self.round_name}_part_{part:03d}.csv"

    def brief(self, part: int) -> Path:
        return self.briefs_dir / f"brief_{self.round_name}_part_{part:03d}.md"

    def structured_output(self, pass_letter: str, part: int) -> Path:
        return self.structured_dir / f"labeller_output_{self.round_name}_pass_{pass_letter}_part_{part:03d}.json"

    def raw_output(self, pass_letter: str, part: int) -> Path:
        return self.raw_dir / f"labeller_output_{self.round_name}_pass_{pass_letter}_part_{part:03d}.csv"

    def pass_file(self, pass_letter: str) -> Path:
        return self.passes_dir / f"pass_{pass_letter}_{self.round_name}.csv"


def round_paths(cfg: RunConfig, round_name: str) -> RoundPaths:
    if round_name not in LABELLING_ROUNDS:
        raise ValueError(f"unknown labelling round {round_name!r}; expected one of {LABELLING_ROUNDS}")
    paths = artefact_paths(cfg)
    if round_name == "pilot_v1":
        cases, manifest, queue_dir = paths.pilot_cases, paths.pilot_manifest, cfg.path("pilot_dir")
        report = queue_dir / "pilot_report.json"
    else:
        cases, manifest, queue_dir = paths.main_cases, paths.main_manifest, cfg.path("main_round_dir")
        report = queue_dir / "main_report.json"
    labels_dir = cfg.path("labels_dir")
    public = paths.public_data_dir
    return RoundPaths(
        round_name=round_name, cases=cases, manifest=manifest, queue_dir=queue_dir,
        key=queue_dir / f"queue_key_{round_name}.csv", report=report, labels_dir=labels_dir,
        raw_dir=labels_dir / "raw", structured_dir=labels_dir / "raw" / "structured",
        briefs_dir=labels_dir / "briefs", passes_dir=labels_dir / "passes",
        agreement=labels_dir / f"agreement_{round_name}.json",
        workbook=labels_dir / f"founder_review_{round_name}.xlsx",
        review_manifest=labels_dir / f"founder_review_{round_name}.json",
        founder_decisions=labels_dir / "raw" / f"founder_decisions_{round_name}.csv",
        preregistration=public / f"{round_name}.sha256",
        passes_digest=public / f"{round_name}_passes.sha256",
        founder_digest=public / f"{round_name}_founder_review.sha256",
    )


def repeat_alias(seed: int, round_name: str, case_id: str) -> str:
    return sha256_hex(canonical_json(["repeat", round_name, seed, case_id]))


def queue_rank(seed: int, round_name: str, queue_case_id: str) -> str:
    return sha256_hex(canonical_json(["queue_order", round_name, seed, queue_case_id]))


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _text(value) -> str:
    return value if isinstance(value, str) else ""


def build_blind_queue(round_cases: pd.DataFrame, round_name: str, repeat_fraction: float, seed: int,
                      chunk_size: int) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Return (chunks, key): one blind frame per part (QUEUE_COLUMNS) and the private key (KEY_COLUMNS)."""
    if round_name not in LABELLING_ROUNDS:
        raise ValueError(f"unknown labelling round {round_name!r}")
    missing = set(ROUND_CASE_COLUMNS) - set(round_cases.columns)
    if missing:
        raise ValueError(f"round cases lack columns {sorted(missing)}")
    if not round_cases.case_id.is_unique:
        raise ValueError("round cases repeat a case_id")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    design = round_cases.set_index("case_id")
    order = sorted(design.index, key=lambda cid: queue_rank(seed, round_name, cid))
    key_rows: list[dict] = []
    for start in range(0, len(order), chunk_size):
        part = start // chunk_size + 1
        members = order[start:start + chunk_size]
        n_repeats = _round_half_up(repeat_fraction * len(members))
        repeats = sorted(members, key=lambda cid: draw_rank(seed, f"{round_name}_repeat", cid))[:n_repeats]
        entries = [(cid, cid, False) for cid in members]
        entries += [(cid, repeat_alias(seed, round_name, cid), True) for cid in repeats]
        for case_id, queue_case_id, is_repeat in entries:
            row = design.loc[case_id]
            key_rows.append({
                "queue_case_id": queue_case_id, "case_id": case_id, "is_repeat": is_repeat, "part": part,
                "sampling_stratum": row.sampling_stratum, "N_h": int(row.N_h), "pool_h": int(row.pool_h),
                "n_h": int(row.n_h), "inclusion_probability": float(row.inclusion_probability),
                "baseline_qualifies": bool(row.baseline_qualifies),
                "queue_rank": queue_rank(seed, round_name, queue_case_id),
            })
    key = pd.DataFrame(key_rows, columns=list(KEY_COLUMNS))
    key = key.sort_values(["part", "queue_rank"], kind="mergesort").reset_index(drop=True)
    chunks = []
    for _, rows in key.groupby("part", sort=True):
        chunks.append(pd.DataFrame({
            "case_id": rows.queue_case_id.tolist(),
            "borrower_name_raw": [design.at[c, "borrower_name_raw"] for c in rows.case_id],
            "lender_names_raw": [LENDER_JOIN.join(design.at[c, "lender_names_raw"]) for c in rows.case_id],
            "city": [_text(design.at[c, "borrower_city"]) for c in rows.case_id],
            "state": [_text(design.at[c, "borrower_state"]) for c in rows.case_id],
        }, columns=list(QUEUE_COLUMNS)))
    return chunks, key


def write_csv(df: pd.DataFrame, path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    return sha256_file(path)


def read_exact_csv(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """Every value as str ('' for empty); refuses a header that is not exactly `columns`."""
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    if list(df.columns) != list(columns):
        raise ValueError(f"{path}: columns {list(df.columns)} != {list(columns)}")
    return df


_BOOL_TEXT = {"True": True, "False": False}


def parse_bool_column(path: Path, df: pd.DataFrame, column: str) -> None:
    check_column(path, df, column, lambda v: v in _BOOL_TEXT)
    df[column] = pd.Series([_BOOL_TEXT[v] for v in df[column]], index=df.index, dtype=bool)


def read_queue(path: Path) -> pd.DataFrame:
    df = read_exact_csv(path, QUEUE_COLUMNS)
    check_column(path, df, "case_id", is_sha256_hex)
    return df


def read_key(path: Path) -> pd.DataFrame:
    df = read_exact_csv(path, KEY_COLUMNS)
    for column in ("queue_case_id", "case_id"):
        check_column(path, df, column, is_sha256_hex)
    for column in ("part", "N_h", "pool_h", "n_h"):
        check_column(path, df, column, lambda v: v.isdigit())
        df[column] = df[column].astype(int)
    df["inclusion_probability"] = df.inclusion_probability.astype(float)
    for column in ("is_repeat", "baseline_qualifies"):
        parse_bool_column(path, df, column)
    return df


def labels_exist(rp: RoundPaths) -> bool:
    return any(rp.raw_dir.glob(f"labeller_output_{rp.round_name}_pass_*_part_*.csv"))


def write_digest_file(path: Path, files: list[Path], comments: list[str]) -> str:
    """`<sha256>  <file name>` for every file, then `# ` comment lines; returns the digest file's sha256."""
    path = Path(path)
    lines = [f"{sha256_file(f)}  {Path(f).name}" for f in files]
    lines += [f"# {c}" if c else "#" for c in comments]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256_file(path)


def write_preregistration(rp: RoundPaths, part_paths: list[Path]) -> str:
    comments = [
        f"Committed BEFORE any {rp.round_name} label exists (contract K13). The queue chunks are exactly what",
        "the blind labellers see: case_id, borrower name, lender names, city, state -- nothing else. The key",
        "holds the strata, N_h, inclusion probabilities and repeat aliases and stays private. Committing every",
        "hash first proves the sample, the design weights and the repeat structure were fixed in advance.",
        f"Labels for this round will be {LABEL_DISCLOSURE}.",
    ]
    return write_digest_file(rp.preregistration, [rp.cases, rp.manifest, rp.key, *part_paths], comments)
