"""Hashing, canonical JSON and bundle-integrity helpers shared by every manifest writer (contract K6).

`canonical_json_bytes` is the one JSON serialisation for manifests and reports: indent 2, sorted keys,
UTF-8, NaN refused, trailing newline. It is NOT `ucc_ml.contracts.canonical_json`, the compact form
used inside identity hashes. NaN never reaches JSON: producers pass floats through `finite_or_none`.
`SHA256SUMS` files use the coreutils format `<sha256>  <name>`; `verify_sha256sums` is called before
anything in a directory is trusted (Plan B calls it before any joblib.load).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SHA256SUMS = "SHA256SUMS"
_SUMS_LINE = re.compile(r"([0-9a-f]{64})  (.+)")


class BundleIntegrityError(ValueError):
    """A SHA256SUMS file is missing or malformed, or does not match the files beside it."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def write_json(path: Path, obj: Any) -> str:
    data = canonical_json_bytes(obj)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_bytes(data)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_head(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    head = out.stdout.strip()
    return head if out.returncode == 0 and len(head) == 40 else None


def finite_or_none(value: Any) -> float | None:
    """float(value) when it is finite; None for None, NaN and +/-inf (JSON has no NaN)."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _check_sums_name(name: str) -> str:
    if (not isinstance(name, str) or not name or name != name.strip() or "/" in name or "\\" in name
            or name in (".", "..") or "\n" in name):
        raise BundleIntegrityError(f"illegal file name for SHA256SUMS: {name!r}")
    return name


def write_sha256sums(directory: Path, filenames: Iterable[str]) -> Path:
    directory = Path(directory)
    names = sorted({_check_sums_name(n) for n in filenames})
    out = directory / SHA256SUMS
    out.write_text("".join(f"{sha256_file(directory / n)}  {n}\n" for n in names), encoding="utf-8")
    return out


def parse_sha256sums(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        match = _SUMS_LINE.fullmatch(line)
        if not match:
            raise BundleIntegrityError(f"SHA256SUMS line {number} is malformed: {line!r}")
        digest, name = match.groups()
        _check_sums_name(name)
        if name in entries:
            raise BundleIntegrityError(f"SHA256SUMS lists {name!r} twice")
        entries[name] = digest
    return entries


def verify_sha256sums(directory: Path, required: Iterable[str] = ()) -> dict[str, str]:
    directory = Path(directory)
    sums = directory / SHA256SUMS
    if not sums.is_file():
        raise BundleIntegrityError(f"{directory}: missing {SHA256SUMS}")
    entries = parse_sha256sums(sums.read_text(encoding="utf-8"))
    unlisted = sorted(set(required) - set(entries))
    if unlisted:
        raise BundleIntegrityError(f"{sums}: required file(s) not listed: {unlisted}")
    for name, expected in sorted(entries.items()):
        target = directory / name
        if not target.is_file():
            raise BundleIntegrityError(f"{sums} lists {name} but it is missing")
        actual = sha256_file(target)
        if actual != expected:
            raise BundleIntegrityError(f"{name}: sha256 {actual} does not match SHA256SUMS {expected}")
    return entries
