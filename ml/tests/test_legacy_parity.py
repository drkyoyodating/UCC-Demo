"""The vendored baseline is byte-identical to its manifest and agrees with src/heavy_filter.py."""
import csv
import hashlib
import importlib
import sys
from pathlib import Path

import pytest

ML_ROOT = Path(__file__).resolve().parents[1]          # <repo>/ml
REPO_ROOT = ML_ROOT.parent
VENDOR = ML_ROOT / "src" / "ucc_ml" / "vendor"
FIXTURE = Path(__file__).parent / "fixtures" / "heavy_filter_parity.csv"


def _rows():
    with FIXTURE.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_fixture_has_thirty_rows_with_both_outcomes():
    rows = _rows()
    assert len(rows) == 30
    assert {r["expected_heavy_row"] for r in rows} == {"0", "1"}


def test_vendored_files_match_manifest():
    from ucc_ml import legacy

    manifest = legacy.vendor_manifest()
    assert manifest["baseline_version"] == legacy.BASELINE_VERSION == "heavy_filter_v1"
    assert set(manifest["files"]) == {"heavy_filter_v1.py", "normalize_v1.py"}
    for name, entry in manifest["files"].items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == entry["sha256"], name
        assert entry["origin"] in ("src/heavy_filter.py", "src/normalize.py")
        assert len(manifest["origin_git_head"]) == 40
    assert set(legacy.verify_vendor_hashes()) == set(manifest["files"])


def test_verify_vendor_hashes_detects_an_edit(tmp_path, monkeypatch):
    from ucc_ml import legacy

    edited = tmp_path / "vendor"
    edited.mkdir()
    for p in VENDOR.iterdir():
        if p.is_file():                      # skip __pycache__
            (edited / p.name).write_bytes(p.read_bytes())
    (edited / "heavy_filter_v1.py").write_bytes(b"# edited\n" + (VENDOR / "heavy_filter_v1.py").read_bytes())
    monkeypatch.setattr(legacy, "VENDOR_DIR", edited)
    monkeypatch.setattr(legacy, "MANIFEST_PATH", edited / "MANIFEST.json")
    with pytest.raises(RuntimeError, match="heavy_filter_v1.py"):
        legacy.verify_vendor_hashes()


def test_vendored_predicates_match_fixture():
    from ucc_ml import legacy

    for r in _rows():
        b, l = r["borrower"], r["lender"]
        assert legacy.heavy_row(b, l) is (r["expected_heavy_row"] == "1"), (b, l)
        assert legacy.is_heavy_lender(l) is (r["expected_lender"] == "1"), l
        assert legacy.is_heavy_borrower(b) is (r["expected_borrower"] == "1"), b


def test_vendored_agrees_with_original_source_tree():
    """The ONLY place src/ is imported: via sys.path, inside this test, to prove parity."""
    src = REPO_ROOT / "src"
    if not (src / "heavy_filter.py").exists() or not (src / "normalize.py").exists():
        pytest.skip("original src/ tree not present")
    sys.path.insert(0, str(src))
    try:
        original = importlib.import_module("heavy_filter")
        original_norm = importlib.import_module("normalize")
    finally:
        sys.path.remove(str(src))
    from ucc_ml import legacy

    for r in _rows():
        assert legacy.heavy_row(r["borrower"], r["lender"]) == bool(original.heavy_row(r["borrower"], r["lender"]))
    for raw in ["BOBS CRANES, LLC", "Acme L.L.C.", "COLORADO FOLIAGE INC", "LLC", "", None,
                "SMITH & SONS INC.", "  WESTERN   SLOPE EARTHMOVING  ",
                "WOODMEN JV LLC, A COLORADO LIMITED LIABILITY COMPANY"]:
        assert legacy.normalize_name(raw) == original_norm.normalize_name(raw)


def test_normalize_name_examples():
    from ucc_ml.legacy import normalize_name

    assert normalize_name("BOBS CRANES, LLC") == ("BOBS CRANES", "LLC")
    assert normalize_name("LLC") == (None, "LLC")
    assert normalize_name(None) == (None, None)
    assert normalize_name("SMITH & SONS INC.") == ("SMITH AND SONS", "INC")


def test_baseline_qualifies_and_route():
    from ucc_ml.legacy import baseline_qualifies, baseline_route

    assert baseline_qualifies("BOBS CRANES", []) is True
    assert baseline_route("BOBS CRANES", []) == "borrower"
    assert baseline_qualifies("BOBS COOKIES", ["WELLS FARGO BANK NA", "TEREX CORPORATION"]) is True
    assert baseline_route("BOBS COOKIES", ["WELLS FARGO BANK NA", "TEREX CORPORATION"]) == "lender"
    assert baseline_route("CARLOS EXCAVATION", ["CATERPILLAR FINANCIAL SERVICES CORP"]) == "both"
    assert baseline_qualifies("SMITH LAW OFFICES", ["WELLS FARGO BANK NA"]) is False
    assert baseline_route("SMITH LAW OFFICES", []) == "neither"
    assert baseline_qualifies("BOBS COOKIES", ["DEERE & COMPANY"]) is False
    assert baseline_qualifies("BOBS COOKIES", ["JOHN DEERE CONSTRUCTION & FORESTRY COMPANY"]) is True
    assert baseline_qualifies("", []) is False
