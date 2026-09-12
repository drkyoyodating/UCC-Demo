"""export_snapshot.py opens the source read-only, writes sorted Parquet, and its manifest is honest."""
import importlib.util
import json
from pathlib import Path

import duckdb
import pytest

from conftest import build_source_duckdb, co_debtor, co_filing, co_sp, ct_row, scope_row

TOOL = Path(__file__).resolve().parents[1] / "tools" / "export_snapshot.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("export_snapshot", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TABLES = {
    "co_filings": [co_filing("F1"), co_filing("F2", filingdate="1989-05-05T00:00:00.000")],
    "co_debtors": [co_debtor("F1", "ACME EXCAVATION LLC"), co_debtor("F2", "OLD CO")],
    "co_secured_parties": [co_sp("F1", "CATERPILLAR FINANCIAL")],
    "ct_filings": [ct_row("C1", "HARTFORD PAVING INC")],
    "scope_all": [scope_row("CO", "F1", "ACME EXCAVATION LLC", "CATERPILLAR FINANCIAL", route_a=True)],
}


def test_export_writes_five_parquet_files_and_manifest(tmp_path):
    tool = _load_tool()
    source = build_source_duckdb(tmp_path / "src.duckdb", TABLES)
    before = tool.sha256_file(source)
    out = tmp_path / "snap"
    manifest = tool.export(source, out)
    assert sorted(manifest["tables"]) == ["co_debtors", "co_filings", "co_secured_parties", "ct_filings", "scope_all"]
    assert manifest["tables"]["co_filings"]["row_count"] == 2
    assert manifest["tables"]["co_filings"]["source_table"] == "filings"
    assert manifest["tables"]["co_filings"]["columns"][-1] == "fileid"
    assert manifest["tables"]["scope_all"]["column_types"][-1] == "BOOLEAN"
    for name, entry in manifest["tables"].items():
        p = out / f"{name}.parquet"
        assert p.exists() and tool.sha256_file(p) == entry["sha256"]
        assert duckdb.connect().execute(f"SELECT count(*) FROM read_parquet('{p.as_posix()}')").fetchone()[0] == entry["row_count"]
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk == manifest
    assert manifest["source_sha256"] == before
    assert tool.sha256_file(source) == before  # the source was not modified


def test_export_is_deterministic(tmp_path):
    tool = _load_tool()
    source = build_source_duckdb(tmp_path / "src.duckdb", TABLES)
    a = tool.export(source, tmp_path / "a")
    b = tool.export(source, tmp_path / "b")
    assert {k: v["sha256"] for k, v in a["tables"].items()} == {k: v["sha256"] for k, v in b["tables"].items()}


def test_export_opens_source_read_only(tmp_path, monkeypatch):
    tool = _load_tool()
    source = build_source_duckdb(tmp_path / "src.duckdb", TABLES)
    seen = {}
    real_connect = duckdb.connect

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(tool.duckdb, "connect", spy)
    tool.export(source, tmp_path / "snap")
    assert seen.get("read_only") is True


def test_main_returns_zero(tmp_path, capsys):
    tool = _load_tool()
    source = build_source_duckdb(tmp_path / "src.duckdb", TABLES)
    assert tool.main(["--source", str(source), "--out", str(tmp_path / "snap")]) == 0
    assert "co_filings" in capsys.readouterr().out


def test_missing_source_is_an_error(tmp_path):
    tool = _load_tool()
    with pytest.raises(FileNotFoundError):
        tool.export(tmp_path / "nope.duckdb", tmp_path / "snap")
