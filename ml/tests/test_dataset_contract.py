"""Stage 4: every candidate row satisfies the Case contract; the Parquet round-trips and is stable."""
from datetime import date

import pandas as pd
import pytest

from conftest import co_debtor, co_filing, co_sp, ct_row

JUNK = ["", "NONE", "N/A"]

TABLES = {
    "co_filings": [co_filing("F1"), co_filing("F5", filingdate="2015-06-01T00:00:00.000"), co_filing("F6")],
    "co_debtors": [
        co_debtor("F1", "Acme Excavation, L.L.C."),
        co_debtor("F5", "ALPHA CRANES INC", debtorid="A"), co_debtor("F5", "BETA HOLDINGS", debtorid="B"),
        co_debtor("F6", "LLC"),                                   # name is pure legal form -> name_clean None
    ],
    "co_secured_parties": [
        co_sp("F1", "  Caterpillar Financial Services Corp "), co_sp("F1", "Caterpillar Financial Services Corp"),
        co_sp("F5", "ZETA BANK"), co_sp("F5", "TEREX FINANCIAL"), co_sp("F5", ""),
    ],
    "ct_filings": [
        ct_row("C1", "HARTFORD PAVING INC", lender="FIRST BANK"),
        ct_row("C1", "HARTFORD PAVING INC", lender="SECOND BANK", dt_accept="2003-04-05T00:00:00.000"),
        ct_row("C2", "SMITH LAW OFFICES", lender=None),
        ct_row("C3", "ACME EXCAVATION LLC", lender="ANY BANK"),
    ],
}


def _cases(snapshot_factory):
    from ucc_ml.dataset import connect_snapshot, extract_region, finalize_cases

    con = connect_snapshot(snapshot_factory(TABLES))
    co, _ = extract_region(con, "CO", "1990", JUNK)
    ct, _ = extract_region(con, "CT", "1990", JUNK)
    return finalize_cases(pd.concat([co, ct], ignore_index=True), "v1")


def test_helpers():
    from ucc_ml.dataset import canonical_lender_set, parse_iso_date

    assert canonical_lender_set([" B ", "A", "", None, float("nan"), "A"]) == ["A", "B"]
    assert canonical_lender_set([]) == [] and canonical_lender_set(None) == []
    assert parse_iso_date("2001-02-03T00:00:00.000") == date(2001, 2, 3)
    assert parse_iso_date(None) is None and parse_iso_date("bad") is None and parse_iso_date("2001-13-40T") is None


def test_every_row_validates_and_columns_match_contract(snapshot_factory):
    from ucc_ml.contracts import CASE_COLUMNS
    from ucc_ml.dataset import validate_cases

    df = _cases(snapshot_factory)
    assert list(df.columns) == list(CASE_COLUMNS)
    assert validate_cases(df) == 7
    assert df.case_id.is_unique and df.case_id.tolist() == sorted(df.case_id)
    assert (df.dataset_version == "v1").all()


def test_identities_normalisation_and_baseline(snapshot_factory):
    from ucc_ml.contracts import make_group_id
    from ucc_ml.dataset import validate_cases

    df = _cases(snapshot_factory).set_index("borrower_name_raw")
    acme = df.loc["Acme Excavation, L.L.C."]
    assert acme.borrower_name_clean == "ACME EXCAVATION" and acme.borrower_suffix == "LLC"
    assert acme.lender_names_raw == ["Caterpillar Financial Services Corp"]     # trimmed, deduped
    assert acme.lender_names_clean == ["CATERPILLAR FINANCIAL SERVICES"]
    assert acme.baseline_qualifies and acme.baseline_route == "both"
    assert acme.earliest_observed_date == date(2001, 2, 3) == acme.latest_observed_date
    assert acme.group_id == make_group_id("CO", "ACME EXCAVATION", "Acme Excavation, L.L.C.")

    ct_acme = df.loc["ACME EXCAVATION LLC"]
    assert ct_acme.region == "CT" and ct_acme.baseline_route == "borrower"
    assert ct_acme.group_id != acme.group_id                                     # region-scoped groups

    beta = df.loc["BETA HOLDINGS"]
    assert beta.lender_names_raw == ["TEREX FINANCIAL", "ZETA BANK"]             # blank dropped, sorted
    assert beta.baseline_qualifies and beta.baseline_route == "lender"
    alpha = df.loc["ALPHA CRANES INC"]
    assert alpha.baseline_route == "both"

    llc = df.loc["LLC"]
    assert llc.borrower_name_clean is None and llc.borrower_suffix == "LLC"
    assert llc.lender_names_raw == [] and llc.lender_names_clean == []
    assert not llc.baseline_qualifies and llc.baseline_route == "neither"
    assert llc.group_id == make_group_id("CO", None, "LLC")

    smith = df.loc["SMITH LAW OFFICES"]
    assert smith.lender_names_raw == [] and smith.baseline_route == "neither"
    hart = df.loc["HARTFORD PAVING INC"]
    assert hart.lender_names_raw == ["FIRST BANK", "SECOND BANK"]
    assert hart.latest_observed_date == date(2003, 4, 5) and hart.source_row_count == 2


def test_parquet_round_trip_is_deterministic(snapshot_factory, tmp_path):
    from ucc_ml.contracts import CASE_COLUMNS
    from ucc_ml.dataset import read_candidates, write_candidates

    df = _cases(snapshot_factory)
    sha1 = write_candidates(df, tmp_path / "a.parquet")
    sha2 = write_candidates(df, tmp_path / "b.parquet")
    assert sha1 == sha2
    back = read_candidates(tmp_path / "a.parquet")
    assert list(back.columns) == list(CASE_COLUMNS) and len(back) == 7
    assert isinstance(back.lender_names_raw.iloc[0], list)
    assert back.set_index("borrower_name_raw").loc["LLC"].lender_names_raw == []
    assert back.set_index("borrower_name_raw").loc["LLC"].borrower_name_clean is None
    assert back.set_index("borrower_name_raw").loc["Acme Excavation, L.L.C."].earliest_observed_date == date(2001, 2, 3)
    assert back.case_id.tolist() == df.case_id.tolist()


def test_validate_cases_reports_the_bad_row(snapshot_factory):
    from ucc_ml.dataset import validate_cases

    df = _cases(snapshot_factory)
    df.loc[3, "baseline_route"] = "neither"       # contradicts baseline_qualifies for a qualifying row
    if not df.loc[3, "baseline_qualifies"]:
        df.loc[3, "baseline_qualifies"] = True
    with pytest.raises(ValueError, match="row 3"):
        validate_cases(df)


@pytest.mark.parametrize("column, value", [
    ("region", "NY"),
    ("case_id", "not-a-sha256"),
    ("group_id", "G1"),
    ("baseline_route", "maybe"),
])
def test_read_candidates_names_file_column_and_value(snapshot_factory, tmp_path, column, value):
    from ucc_ml.dataset import read_candidates, write_candidates

    df = _cases(snapshot_factory)
    df.loc[0, column] = value
    write_candidates(df, tmp_path / "bad.parquet")
    with pytest.raises(ValueError, match=f"column '{column}'") as exc:
        read_candidates(tmp_path / "bad.parquet")
    assert "bad.parquet" in str(exc.value) and value in str(exc.value)


def test_read_candidates_rejects_flat_lenders_and_duplicate_ids(snapshot_factory, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from ucc_ml.contracts import CASE_COLUMNS
    from ucc_ml.dataset import read_candidates, write_candidates

    df = _cases(snapshot_factory)
    write_candidates(df, tmp_path / "good.parquet")
    table = pq.read_table(tmp_path / "good.parquet")
    flat = table.set_column(CASE_COLUMNS.index("lender_names_raw"), "lender_names_raw",
                            pa.array([" | ".join(v) for v in df.lender_names_raw], pa.string()))
    pq.write_table(flat, tmp_path / "flat.parquet")
    with pytest.raises(ValueError, match="lender_names_raw"):
        read_candidates(tmp_path / "flat.parquet")
    write_candidates(pd.concat([df, df.head(1)], ignore_index=True), tmp_path / "dup.parquet")
    with pytest.raises(ValueError, match="duplicate"):
        read_candidates(tmp_path / "dup.parquet")
