"""Contract K6: sha256 / canonical JSON / SHA256SUMS helpers, with known vectors and tamper detection."""
import hashlib
from pathlib import Path

import pytest

ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_known_vectors(tmp_path):
    from ucc_ml.provenance import sha256_bytes, sha256_file

    assert sha256_bytes(b"abc") == ABC_SHA256
    p = tmp_path / "abc.bin"
    p.write_bytes(b"abc")
    assert sha256_file(p) == ABC_SHA256


def test_canonical_json_bytes_is_exact_and_refuses_nan():
    from ucc_ml.provenance import canonical_json_bytes

    assert canonical_json_bytes({"z": 1, "a": [1, 2], "é": "ü"}) == (
        '{\n  "a": [\n    1,\n    2\n  ],\n  "z": 1,\n  "é": "ü"\n}\n'.encode("utf-8"))
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            canonical_json_bytes({"x": bad})


def test_write_json_returns_the_digest_of_the_bytes_written(tmp_path):
    from ucc_ml.provenance import read_json, sha256_file, write_json

    p = tmp_path / "a" / "b.json"
    digest = write_json(p, {"z": 1, "a": [1, 2]})
    assert p.read_text(encoding="utf-8") == '{\n  "a": [\n    1,\n    2\n  ],\n  "z": 1\n}\n'
    assert digest == sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()
    assert read_json(p) == {"z": 1, "a": [1, 2]}


def test_timestamp_git_head_and_finite_or_none(tmp_path):
    from ucc_ml.provenance import finite_or_none, git_head, utc_now_iso

    stamp = utc_now_iso()
    assert len(stamp) == 20 and stamp.endswith("Z") and stamp[10] == "T"
    assert git_head(tmp_path) is None                      # tmp_path is not a git repository
    assert finite_or_none(1) == 1.0 and finite_or_none(0.25) == 0.25
    assert finite_or_none(None) is None
    assert finite_or_none(float("nan")) is None and finite_or_none(float("inf")) is None


def _two_files(directory: Path) -> None:
    (directory / "a.txt").write_text("A\n", encoding="utf-8")
    (directory / "b.txt").write_text("B\n", encoding="utf-8")


def test_sha256sums_round_trip(tmp_path):
    from ucc_ml.provenance import SHA256SUMS, sha256_file, verify_sha256sums, write_sha256sums

    _two_files(tmp_path)
    out = write_sha256sums(tmp_path, ["b.txt", "a.txt", "a.txt"])
    assert out == tmp_path / SHA256SUMS
    assert out.read_text(encoding="utf-8") == (
        f"{sha256_file(tmp_path / 'a.txt')}  a.txt\n{sha256_file(tmp_path / 'b.txt')}  b.txt\n")
    assert verify_sha256sums(tmp_path, required=["a.txt"]) == {
        "a.txt": sha256_file(tmp_path / "a.txt"), "b.txt": sha256_file(tmp_path / "b.txt")}


def test_verify_sha256sums_failures_name_the_file(tmp_path):
    from ucc_ml.provenance import BundleIntegrityError, verify_sha256sums, write_sha256sums

    assert issubclass(BundleIntegrityError, ValueError)
    with pytest.raises(BundleIntegrityError, match="missing SHA256SUMS"):
        verify_sha256sums(tmp_path)
    _two_files(tmp_path)
    write_sha256sums(tmp_path, ["a.txt", "b.txt"])
    with pytest.raises(BundleIntegrityError, match="pipeline.joblib"):
        verify_sha256sums(tmp_path, required=["a.txt", "pipeline.joblib"])
    (tmp_path / "b.txt").write_text("B tampered\n", encoding="utf-8")
    with pytest.raises(BundleIntegrityError, match="b.txt"):
        verify_sha256sums(tmp_path)
    (tmp_path / "b.txt").unlink()
    with pytest.raises(BundleIntegrityError, match="b.txt"):
        verify_sha256sums(tmp_path)


@pytest.mark.parametrize("text", [
    "zz  a.txt\n",                                     # not a sha256
    ABC_SHA256 + " a.txt\n",                           # one space
    ABC_SHA256 + "  /etc/passwd\n",                    # absolute path
    ABC_SHA256 + "  ../escape.txt\n",                  # parent traversal
    ABC_SHA256 + "  sub/dir.txt\n",                    # "/" in a name
    ABC_SHA256 + "  ..\n",
    ABC_SHA256 + "  a.txt\n\n" + ABC_SHA256 + "  b.txt\n",   # blank line
    ABC_SHA256 + "  a.txt\n" + ABC_SHA256 + "  a.txt\n",     # duplicate name
])
def test_parse_sha256sums_rejects_malformed_text(text):
    from ucc_ml.provenance import BundleIntegrityError, parse_sha256sums

    with pytest.raises(BundleIntegrityError):
        parse_sha256sums(text)


def test_write_sha256sums_rejects_unsafe_names(tmp_path):
    from ucc_ml.provenance import BundleIntegrityError, parse_sha256sums, write_sha256sums

    with pytest.raises(BundleIntegrityError):
        write_sha256sums(tmp_path, ["../x.txt"])
    assert parse_sha256sums(ABC_SHA256 + "  model-card.md\n") == {"model-card.md": ABC_SHA256}
