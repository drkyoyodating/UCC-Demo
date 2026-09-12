"""Parity with scope_all is asserted row by row; every delta is counted and exemplified, never hidden."""
import json

from conftest import co_debtor, co_filing, co_sp, ct_row, scope_row

JUNK = ["", "NONE", "N/A"]

BASE = {
    "co_filings": [co_filing("F1"), co_filing("F2"), co_filing("F3")],
    "co_debtors": [co_debtor("F1", "ACME EXCAVATION LLC"), co_debtor("F2", "SMITH LAW OFFICES"),
                   co_debtor("F3", "ROCKY MOUNTAIN READY MIX")],
    "co_secured_parties": [co_sp("F1", "CATERPILLAR FINANCIAL"), co_sp("F2", "WELLS FARGO BANK"),
                           co_sp("F3", "WELLS FARGO BANK")],
    "ct_filings": [ct_row("C1", "HARTFORD PAVING INC", lender="FIRST BANK"),
                   ct_row("C2", "SMITH LAW OFFICES", lender="FIRST BANK")],
}
CLEAN_SCOPE = [
    scope_row("CO", "F1", "ACME EXCAVATION LLC", "CATERPILLAR FINANCIAL", route_a=True),
    scope_row("CO", "F3", "ROCKY MOUNTAIN READY MIX", "WELLS FARGO BANK"),
    scope_row("CT", "C1", "HARTFORD PAVING INC", "FIRST BANK", city="HARTFORD", state="CT", zipcode="06103"),
]


def _build(snapshot_factory, tmp_path, scope_rows):
    from ucc_ml.dataset import build_candidates

    snap = snapshot_factory({**BASE, "scope_all": scope_rows})
    return build_candidates(snapshot_dir=snap, out_dir=tmp_path / "cand", dataset_version="v1",
                            year_min="1990", junk=JUNK,
                            provenance={"config_sha256": "cfg", "config_path": "x.yaml", "git_head": None,
                                        "vendor": {"heavy_filter_v1.py": "h", "normalize_v1.py": "n"}})


def test_clean_parity(snapshot_factory, tmp_path):
    manifest = _build(snapshot_factory, tmp_path, CLEAN_SCOPE)
    rec = json.loads((tmp_path / "cand" / "reconciliation.json").read_text())
    assert manifest["parity_ok"] is True and rec["parity"]["ok"] is True
    assert rec["scope_all"]["rows"] == 3 and rec["scope_all"]["distinct_triples_normalised"] == 3
    assert rec["coverage"]["missing_case"]["count"] == 0
    assert rec["coverage"]["case_not_qualifying"]["count"] == 0
    assert rec["coverage"]["lender_not_in_set"]["count"] == 0
    assert rec["counts"]["qualifying_cases"] == 3 and rec["counts"]["delta_triples"] == 0
    assert rec["strata"] == {"CO:accepted": 2, "CO:rejected": 1, "CT:accepted": 1, "CT:rejected": 1}
    assert rec["routes"] == {"both": 1, "borrower": 2, "neither": 2}
    assert rec["source_filing_type"]["CO"] == {"ucc": 3} and rec["source_status"]["CT"] == {"Active": 2}
    assert manifest["candidates"]["rows"] == 5 and manifest["baseline_version"] == "heavy_filter_v1"
    assert manifest["snapshot"]["tables"]["co_filings"]["row_count"] == 3
    assert (tmp_path / "cand" / "candidates.parquet").exists()
    assert manifest["candidates"]["sha256"] == __import__("hashlib").sha256((tmp_path / "cand" / "candidates.parquet").read_bytes()).hexdigest()


def test_missing_and_non_qualifying_rows_fail_parity_with_examples(snapshot_factory, tmp_path):
    scope = CLEAN_SCOPE + [
        scope_row("CO", "F9", "GHOST LLC", "NOBODY"),                       # no such case
        scope_row("CO", "F2", "SMITH LAW OFFICES", "WELLS FARGO BANK"),      # case exists, baseline False
        scope_row("CT", "C1", "HARTFORD PAVING INC", "UNSEEN LENDER", city="HARTFORD", state="CT", zipcode="06103"),
    ]
    manifest = _build(snapshot_factory, tmp_path, scope)
    rec = json.loads((tmp_path / "cand" / "reconciliation.json").read_text())
    assert manifest["parity_ok"] is False
    assert rec["coverage"]["missing_case"]["count"] == 1
    assert rec["coverage"]["missing_case"]["examples"][0]["borrower"] == "GHOST LLC"
    assert rec["coverage"]["case_not_qualifying"]["count"] == 1
    assert rec["coverage"]["case_not_qualifying"]["examples"][0]["file_id"] == "F2"
    assert rec["coverage"]["lender_not_in_set"]["count"] == 1
    assert rec["coverage"]["lender_not_in_set"]["examples"][0]["lender"] == "UNSEEN LENDER"
    assert "address" not in json.dumps(rec["coverage"])


def test_qualifying_case_absent_from_scope_all_is_a_reported_delta(snapshot_factory, tmp_path):
    manifest = _build(snapshot_factory, tmp_path, CLEAN_SCOPE[:2])      # CT case qualifies but is not in scope_all
    rec = json.loads((tmp_path / "cand" / "reconciliation.json").read_text())
    assert manifest["parity_ok"] is True                                 # coverage direction still holds
    assert rec["counts"]["delta_triples"] == 1
    assert rec["counts"]["qualifying_case_ids_not_in_scope_all"]["count"] == 1
    assert rec["counts"]["qualifying_case_ids_not_in_scope_all"]["examples"][0]["borrower"] == "HARTFORD PAVING INC"


def test_cli_build_candidates(snapshot_factory, tmp_path, capsys):
    from ucc_ml.cli import main

    snap = snapshot_factory({**BASE, "scope_all": CLEAN_SCOPE})
    cfg = tmp_path / "ml" / "configs" / "v1.yaml"
    cfg.parent.mkdir(parents=True)
    from test_config import MINIMAL
    text = MINIMAL.replace("snapshot_dir: ml/data/snapshots/v1", f"snapshot_dir: {snap.relative_to(tmp_path)}")
    cfg.write_text(text)
    assert main(["build-candidates", "--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "parity: OK" in out and "cases=5" in out
    assert (tmp_path / "ml/data/candidates/v1/candidates_manifest.json").exists()
