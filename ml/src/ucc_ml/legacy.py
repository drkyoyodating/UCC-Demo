"""The frozen rules baseline, served from byte-verified vendored copies.

``ucc_ml`` never imports from the repository's ``src/`` tree. The copies under ``vendor/`` are the
comparison baseline ``heavy_filter_v1``; a later rule change creates ``heavy_filter_v2`` and a new
manifest, never an edit here. ``verify_vendor_hashes()`` runs at the start of ``build-candidates``
so a tampered baseline cannot silently score a dataset.

Baseline definition (context pack §3):
    baseline_qualifies = any(heavy_row(borrower, l) for l in lenders) or heavy_row(borrower, '')
    baseline_route     = both | lender | borrower | neither
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from ucc_ml.vendor import heavy_filter_v1 as _hf
from ucc_ml.vendor import normalize_v1 as _nz

BASELINE_VERSION = "heavy_filter_v1"
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
MANIFEST_PATH = VENDOR_DIR / "MANIFEST.json"


def vendor_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def verify_vendor_hashes() -> dict[str, str]:
    """Return {file: sha256} or raise RuntimeError naming the first file whose bytes moved."""
    manifest = vendor_manifest()
    verified: dict[str, str] = {}
    for name, entry in manifest["files"].items():
        actual = hashlib.sha256((VENDOR_DIR / name).read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise RuntimeError(
                f"vendored {name} sha256 {actual} != MANIFEST.json {entry['sha256']}: "
                "the frozen baseline was edited; create a new baseline version instead"
            )
        verified[name] = actual
    return verified


def is_heavy_lender(name) -> bool:
    return bool(_hf.is_heavy_lender(name))


def is_heavy_borrower(name) -> bool:
    return bool(_hf.is_heavy_borrower(name))


def heavy_row(borrower, lender) -> bool:
    return bool(_hf.heavy_row(borrower, lender))


def baseline_qualifies(borrower: str, lenders: Iterable[str]) -> bool:
    lenders = list(lenders)
    return any(heavy_row(borrower, lender) for lender in lenders) or heavy_row(borrower, "")


def baseline_route(borrower: str, lenders: Iterable[str]) -> str:
    borrower_hit = is_heavy_borrower(borrower)
    lender_hit = any(is_heavy_lender(lender) for lender in lenders)
    if borrower_hit and lender_hit:
        return "both"
    if lender_hit:
        return "lender"
    if borrower_hit:
        return "borrower"
    return "neither"


def normalize_name(raw) -> tuple[str | None, str | None]:
    return _nz.normalize_name(raw)
