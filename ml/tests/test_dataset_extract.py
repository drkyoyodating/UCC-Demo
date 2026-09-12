"""Stages 1-3: eligibility exactly as build_scope, every exclusion counted, lender sets complete."""
import pytest

from conftest import co_debtor, co_filing, co_sp, ct_row

JUNK = ["", "NONE", "NONE PROVIDED", "NA", "N/A", "UNKNOWN", "SAME", "COMPANY", "NOT PROVIDED", "TBD",
        "X", "XX", "NO ADDRESS", "ADDRESS UNKNOWN", "VARIOUS"]

CO_TABLES = {
    "co_filings": [
        co_filing("F1", master="M1"), co_filing("F1", master="M1"),            # verbatim duplicate filing row
        co_filing("F2", filingdate="1989-12-31T00:00:00.000"),                  # year cut
        co_filing("F3"),                                                        # debtor has blank name
        co_filing("F4"),                                                        # debtor has placeholder address
        co_filing("F5", filingdate="2015-06-01T00:00:00.000"),                  # 2 debtors x 3 secured parties
        co_filing("F6", filingdate="2012-01-01T00:00:00.000"),                  # case-variant debtor rows
        co_filing("F7"),                                                        # filing with no debtor row
        co_filing("F8", filingdate=None),                                       # NULL date -> year cut
    ],
    "co_debtors": [
        co_debtor("F1", "ACME EXCAVATION LLC"),
        co_debtor("F2", "OLD TIMER INC"),
        co_debtor("F3", "   "),
        co_debtor("F4", "NOWHERE LLC", address="None Provided"),
        co_debtor("F5", "ALPHA CRANES INC", debtorid="D5A"),
        co_debtor("F5", "BETA HOLDINGS", debtorid="D5B"),
        co_debtor("F6", "Gamma Paving Co", debtorid="D6A"),
        co_debtor("F6", "GAMMA PAVING CO  ", debtorid="D6B"),
        co_debtor("F8", "NULL DATE CO"),
        co_debtor("F99", "ORPHAN LLC"),                                          # no filing row
    ],
    "co_secured_parties": [
        co_sp("F1", "CATERPILLAR FINANCIAL SERVICES"),
        co_sp("F5", "ZETA BANK"), co_sp("F5", "ALPHA LEASING"), co_sp("F5", ""),  # blank lender kept in _all
        co_sp("F6", None),
    ],
}

CT_TABLES = {
    "ct_filings": [
        ct_row("C1", "HARTFORD PAVING INC", lender="FIRST BANK"),
        ct_row("C1", "HARTFORD PAVING INC", lender="FIRST BANK"),                 # verbatim duplicate
        ct_row("C1", "HARTFORD PAVING INC", lender="SECOND BANK", dt_accept="2003-04-05T00:00:00.000",
               status="Released", ftype="AMENDMENT"),
        ct_row("C1", "OTHER DEBTOR LLC", lender="THIRD BANK"),                   # same filing, other borrower
        ct_row("C2", "TOO EARLY CO", dt_accept="1985-01-01T00:00:00.000"),
        ct_row("C3", None),
        ct_row("C4", "BAD ADDRESS CO", address="N/A"),
        ct_row(None, "NO ID CO"),
        ct_row("C5", "NULL LENDER CO", lender=None),
    ],
}


def _by_name(df):
    return {r.borrower_name_raw.strip().upper(): r for r in df.itertuples()}


def test_co_eligibility_exclusions_and_cases(snapshot_factory):
    from ucc_ml.dataset import connect_snapshot, extract_region

    con = connect_snapshot(snapshot_factory(CO_TABLES))
    obs, excl = extract_region(con, "CO", "1990", JUNK)
    assert excl["source"]["filings_rows"] == 9 and excl["source"]["filings_distinct_fileid"] == 8
    assert excl["source"]["filings_duplicate_rows"] == 1
    assert excl["source"]["filings_without_debtor_row"] == 1            # F7
    assert excl["source"]["debtors_rows"] == 10
    assert excl["source"]["debtors_blank_name_rows"] == 1               # F3 (whole table, incl. orphans)
    assert excl["source"]["debtors_without_filing_row"] == 1            # F99
    assert excl["source"]["secured_parties_blank_name_rows"] == 2       # '' and NULL
    w = excl["waterfall"]
    assert w["joined_rows"] == 9                                        # 10 debtors - 1 orphan
    assert w["excluded_year_before_1990_or_null"] == 2                  # F2, F8
    assert w["excluded_blank_name"] == 1                                # F3
    assert w["excluded_placeholder_address"] == 1                       # F4
    assert w["eligible_rows"] == 5                                      # F1, F5 x2, F6 x2
    assert sum(v for k, v in w.items() if k != "joined_rows") == w["joined_rows"]
    assert excl["standalone"] == {"year_fails": 2, "blank_name_fails": 1, "placeholder_address_fails": 1}
    assert excl["stage1_raw_groups"] == 5 and excl["cases"] == 4 and excl["case_insensitive_merges"] == 1

    rows = _by_name(obs)
    assert set(rows) == {"ACME EXCAVATION LLC", "ALPHA CRANES INC", "BETA HOLDINGS", "GAMMA PAVING CO"}
    assert rows["ACME EXCAVATION LLC"].lineage_id == "M1"
    assert rows["ACME EXCAVATION LLC"].source_row_count == 1
    assert rows["ACME EXCAVATION LLC"].source_filing_type == "ucc" and rows["ACME EXCAVATION LLC"].source_status == "false"
    # both F5 debtors receive the filing's full lender list (blank included at this stage)
    for name in ("ALPHA CRANES INC", "BETA HOLDINGS"):
        assert sorted(x for x in rows[name].lender_names_all if x) == ["ALPHA LEASING", "ZETA BANK"]
        assert rows[name].lender_row_count == 3
    g = rows["GAMMA PAVING CO"]
    assert g.source_row_count == 2 and g.stage1_rows == 2
    assert g.borrower_name_raw == "GAMMA PAVING CO  "                    # min() of the raw variants
    assert g.earliest_observed_date == "2012-01-01T00:00:00.000" == g.latest_observed_date
    assert obs.case_id.is_unique and (obs.region == "CO").all()
    assert obs.case_id.tolist() == sorted(obs.case_id)


def test_ct_grouping_and_exclusions(snapshot_factory):
    from ucc_ml.dataset import connect_snapshot, extract_region

    con = connect_snapshot(snapshot_factory(CT_TABLES))
    obs, excl = extract_region(con, "CT", "1990", JUNK)
    assert excl["source"]["ct_rows"] == 9
    assert excl["source"]["ct_verbatim_duplicate_rows"] == 1
    assert excl["source"]["ct_blank_name_rows"] == 1
    assert excl["source"]["ct_blank_lender_rows"] == 1
    w = excl["waterfall"]
    assert w["excluded_null_lien_id"] == 1 and w["excluded_year_before_1990_or_null"] == 1
    assert w["excluded_blank_name"] == 1 and w["excluded_placeholder_address"] == 1
    assert w["source_rows"] == 9 and w["excluded_verbatim_duplicate"] == 1 and w["joined_rows"] == 8
    assert w["eligible_rows"] == 4                                       # C1 x3 (verbatim duplicate dropped), C5
    assert excl["cases"] == 3
    rows = _by_name(obs)
    h = rows["HARTFORD PAVING INC"]
    assert h.source_row_count == 2
    assert sorted(h.lender_names_all) == ["FIRST BANK", "SECOND BANK", "THIRD BANK"]   # the filing's FULL lender set
    assert h.earliest_observed_date == "2001-02-03T00:00:00.000" and h.latest_observed_date == "2003-04-05T00:00:00.000"
    assert h.source_filing_type == "AMENDMENT|ORIG FIN STMT" and h.source_status == "Active|Released"
    assert h.lineage_id is None
    assert sorted(rows["OTHER DEBTOR LLC"].lender_names_all) == ["FIRST BANK", "SECOND BANK", "THIRD BANK"]   # same filing, same full set
    assert [x for x in rows["NULL LENDER CO"].lender_names_all if x] == []


def test_connect_snapshot_requires_every_table(tmp_path):
    from ucc_ml.dataset import connect_snapshot

    with pytest.raises(FileNotFoundError):
        connect_snapshot(tmp_path)


def test_junk_sql_quotes_and_escapes():
    from ucc_ml.dataset import junk_sql

    assert junk_sql(["", "N/A", "O'BRIEN"]) == "'','N/A','O''BRIEN'"
