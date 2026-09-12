"""Data contracts for the UCC ML Lab (contracts K2, K3).

These pydantic v2 models are the schema authority for candidates.parquet (``Case``),
splits.parquet (``Split``) and labels.csv (``Label``). They also own the deterministic identity
functions: every hash is SHA-256 over canonical JSON (compact separators, ensure_ascii=False,
list order as given), so there is no delimiter ambiguity and no dependence on pandas/numpy
versions. Nothing here reads a file.

Identity rules (context pack §3, §6):
  borrower_key = sha256(canonical_json([UPPER(TRIM(name)), UPPER(TRIM(address1)), UPPER(TRIM(city)),
                                        UPPER(TRIM(state)), UPPER(TRIM(zip))]))   -- nulls become ""
  case_id      = sha256(canonical_json([region, file_id, borrower_key]))
  group_id     = sha256(canonical_json([region, name_clean or "RAW:" + UPPER(TRIM(name))]))  -- region-scoped

Label vocabulary (contract K3). A labels.csv row carries one of four adjudication statuses:
  model_agreed         both blind Claude passes gave the same label; the founder did not review it
  founder_confirmed    the founder re-read an agreed row (audit sample) and kept the label
  founder_adjudicated  the founder decided a disagreement, or overturned an agreed row
  blind_repeat         one pass's answer on a hidden repeat; never fitted or evaluated
There is no "pending" status: validate-labels refuses to write labels.csv while a disagreement is undecided.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Region = Literal["CO", "CT"]
BaselineRoute = Literal["lender", "borrower", "both", "neither"]
SplitName = Literal["train", "validation", "test"]
LabelValue = Literal["RELEVANT", "NOT_RELEVANT", "INSUFFICIENT_EVIDENCE"]
AdjudicationStatus = Literal["model_agreed", "founder_confirmed", "founder_adjudicated", "blind_repeat"]
LabellingRound = Literal["pilot_v1", "ablation_v1", "yield_probe_v1", "main_v1", "queue_v1"]

REGIONS: tuple[str, ...] = ("CO", "CT")
BASELINE_ROUTES: tuple[str, ...] = ("lender", "borrower", "both", "neither")
LABELS: tuple[str, ...] = ("RELEVANT", "NOT_RELEVANT", "INSUFFICIENT_EVIDENCE")
SPLITS: tuple[str, ...] = ("train", "validation", "test")
STRATA: tuple[str, ...] = ("CO:accepted", "CO:rejected", "CT:accepted", "CT:rejected")
ADJUDICATION_STATUSES: tuple[str, ...] = ("model_agreed", "founder_confirmed", "founder_adjudicated", "blind_repeat")
RESOLVED_ADJUDICATION_STATUSES: tuple[str, ...] = ("model_agreed", "founder_confirmed", "founder_adjudicated")
LABELLING_ROUNDS: tuple[str, ...] = ("pilot_v1", "ablation_v1", "yield_probe_v1", "main_v1", "queue_v1")

LABELLER_PASS_A = "claude_blind_pass_a"
LABELLER_PASS_B = "claude_blind_pass_b"
LABELLER_AGREED = "claude_blind_pass_a+claude_blind_pass_b"
LABELLER_FOUNDER = "founder"
PASS_LABELLERS: dict[str, str] = {"a": LABELLER_PASS_A, "b": LABELLER_PASS_B}
LABEL_DISCLOSURE = "model-labelled, founder-adjudicated"

#: The labeller's reason codes, by label family. The text of each is in ml/specs/label_policy_v1.md
#: (Task 12) and a test there asserts the document lists every code below.
REASON_CODES: dict[str, tuple[str, ...]] = {
    "RELEVANT": (
        "R_LENDER_NAMED_MAKER",         # a lender is a named heavy-construction maker / captive / dealer
        "R_BORROWER_EQUIPMENT_WORD",    # an explicit equipment class in the borrower's own name
        "R_BORROWER_TRADE_WORD",        # an explicit trade word (demolition, earthwork, concrete family ...)
        "R_BORROWER_OTHER_EVIDENCE",    # the name positively states the market without a listed word
    ),
    "NOT_RELEVANT": (
        "N_OTHER_INDUSTRY_EXPLICIT",    # the name positively states a different activity
        "N_WORD_IS_DIFFERENT_SENSE",    # an equipment-like word used in another sense (car crushers, roller hockey)
    ),
    "INSUFFICIENT_EVIDENCE": (
        "I_GENERIC_NAME",               # holdings / enterprises / initials -- could be anything
        "I_PERSONAL_NAME",              # reads as an individual; no activity stated
        "I_AMBIGUOUS_WORD",             # CONSTRUCTION / TRUCKING / EQUIPMENT / SERVICES alone
        "I_UNREADABLE",                 # garbled, truncated or OCR-damaged beyond reading
    ),
}
FOUNDER_REASON_CODE = "ADJUDICATED"
ALL_REASON_CODES: tuple[str, ...] = tuple(
    code for label in LABELS for code in REASON_CODES[label]
) + (FOUNDER_REASON_CODE,)

_HEX64 = re.compile(r"[0-9a-f]{64}")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm_key(value: Any) -> str:
    """Key text: stripped and upper-cased; None / NaN / anything not a str is ''."""
    if not isinstance(value, str):
        return ""
    return value.strip().upper()


def is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def make_borrower_key(name_raw: Any, address1: Any, city: Any, state: Any, zip_code: Any) -> str:
    return sha256_hex(canonical_json([
        norm_key(name_raw), norm_key(address1), norm_key(city), norm_key(state), norm_key(zip_code),
    ]))


def make_case_id(region: str, file_id: str, borrower_key: str) -> str:
    return sha256_hex(canonical_json([region, file_id, borrower_key]))


def make_group_id(region: str, name_clean: str | None, name_raw: Any) -> str:
    key = name_clean if name_clean else "RAW:" + norm_key(name_raw)
    return sha256_hex(canonical_json([region, key]))


def stratum_name(region: str, baseline_qualifies: bool) -> str:
    return f"{region}:{'accepted' if baseline_qualifies else 'rejected'}"


def check_column(source: Any, frame: Any, column: str, ok: Callable[[Any], bool]) -> None:
    """Raise ValueError naming `source`, `column` and up to five offending values when any value fails `ok`."""
    if column not in frame.columns:
        raise ValueError(f"{source}: missing required column {column!r}")
    bad = [value for value in frame[column].tolist() if not ok(value)]
    if bad:
        shown = ", ".join(sorted({repr(v) for v in bad})[:5])
        raise ValueError(f"{source}: column {column!r} has {len(bad)} invalid value(s): {shown}")


def _require_hex64(v: str) -> str:
    if not is_sha256_hex(v):
        raise ValueError("must be a 64-character lowercase hex sha256")
    return v


def _require_sorted_unique_nonblank(v: list[str]) -> list[str]:
    for item in v:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("lender names must be non-blank strings")
    if v != sorted(set(v)):
        raise ValueError("lender names must be sorted and unique")
    return v


class Case(BaseModel):
    """One filing-borrower observation with the filing's complete sorted lender set."""
    model_config = ConfigDict(extra="forbid")

    case_id: str
    dataset_version: str
    region: Region
    file_id: str
    lineage_id: str | None = None
    borrower_key: str
    borrower_name_raw: str
    borrower_name_clean: str | None = None
    borrower_suffix: str | None = None
    lender_names_raw: list[str]
    lender_names_clean: list[str]
    earliest_observed_date: date | None = None
    latest_observed_date: date | None = None
    borrower_city: str | None = None
    borrower_state: str | None = None
    borrower_zip: str | None = None
    source_row_count: int = Field(ge=1)
    source_filing_type: str | None = None
    source_status: str | None = None
    baseline_qualifies: bool
    baseline_route: BaselineRoute
    group_id: str
    split: SplitName | None = None      # populated from splits.parquet, never stored in candidates

    _hex = field_validator("case_id", "borrower_key", "group_id")(_require_hex64)
    _lists = field_validator("lender_names_raw", "lender_names_clean")(_require_sorted_unique_nonblank)

    @field_validator("file_id", "borrower_name_raw")
    @classmethod
    def _nonblank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> "Case":
        if self.case_id != make_case_id(self.region, self.file_id, self.borrower_key):
            raise ValueError("case_id does not recompute from (region, file_id, borrower_key)")
        if self.baseline_qualifies != (self.baseline_route != "neither"):
            raise ValueError("baseline_qualifies must equal (baseline_route != 'neither')")
        if (self.earliest_observed_date and self.latest_observed_date
                and self.earliest_observed_date > self.latest_observed_date):
            raise ValueError("earliest_observed_date is after latest_observed_date")
        return self


CASE_COLUMNS: tuple[str, ...] = tuple(k for k in Case.model_fields if k != "split")


class Split(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    group_id: str
    split: SplitName

    _hex = field_validator("case_id", "group_id")(_require_hex64)


SPLIT_COLUMNS: tuple[str, ...] = tuple(Split.model_fields)


class Label(BaseModel):
    """One row of labels.csv under the K3 vocabulary."""
    model_config = ConfigDict(extra="forbid")

    case_id: str
    label: LabelValue
    reason_code: str
    reason: str
    labeller_id: str
    labelled_at: datetime
    policy_version: str
    sampling_stratum: str
    inclusion_probability: float = Field(gt=0.0, le=1.0)
    adjudication_status: AdjudicationStatus
    is_repeat: bool
    labelling_round: LabellingRound

    _hex = field_validator("case_id")(_require_hex64)

    @field_validator("sampling_stratum")
    @classmethod
    def _known_stratum(cls, v: str) -> str:
        if v not in STRATA:
            raise ValueError(f"sampling_stratum must be one of {STRATA}")
        return v

    @field_validator("reason")
    @classmethod
    def _reason_nonblank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reason must not be blank")
        return v

    @model_validator(mode="after")
    def _status_rules(self) -> "Label":
        status = self.adjudication_status
        if status == "blind_repeat":
            if not self.is_repeat:
                raise ValueError("a blind_repeat row must have is_repeat=True")
            if self.labeller_id not in (LABELLER_PASS_A, LABELLER_PASS_B):
                raise ValueError("a blind_repeat row is one blind pass's answer: labeller_id must be pass a or pass b")
        elif self.is_repeat:
            raise ValueError(f"is_repeat=True requires adjudication_status 'blind_repeat', not {status!r}")
        if status == "founder_adjudicated":
            if self.reason_code != FOUNDER_REASON_CODE:
                raise ValueError("founder_adjudicated rows carry reason_code ADJUDICATED")
            if self.labeller_id != LABELLER_FOUNDER:
                raise ValueError("founder_adjudicated rows carry labeller_id 'founder'")
            return self
        if self.reason_code not in REASON_CODES[self.label]:
            raise ValueError(f"reason_code {self.reason_code!r} is not valid for label {self.label}")
        if status == "model_agreed" and self.labeller_id != LABELLER_AGREED:
            raise ValueError(f"model_agreed rows carry labeller_id {LABELLER_AGREED!r}")
        if status == "founder_confirmed" and self.labeller_id != LABELLER_FOUNDER:
            raise ValueError("founder_confirmed rows carry labeller_id 'founder'")
        return self


LABEL_COLUMNS: tuple[str, ...] = tuple(Label.model_fields)
