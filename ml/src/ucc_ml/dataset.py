"""Candidate cases from the Parquet snapshot (context pack §3, Codex §4).

Stage 1 (SQL, DuckDB in .venv-ml, Parquet only): apply build_scope's eligibility per source row,
count every exclusion separately, and collapse rows that are identical on the raw borrower fields.
Stage 2 (Python): borrower_key / case_id from the raw fields with contracts.norm_key. Python is
the single normaliser -- DuckDB's upper() maps ß to ẞ where Python gives SS and its trim() strips
spaces only -- so no identity is ever computed in SQL.
Stage 3 (SQL): re-aggregate stage-1 rows that share a case_id (case/whitespace variants of one
borrower on one filing): union the lender lists, sum source_row_count, min/max the dates, join the
distinct filing types / statuses with '|'.
Stage 4 (Python, Task 7): name_clean / suffix, sorted unique lender sets, group_id, baseline.

Eligibility (copied from src/build_scope.py:27-28, 42-44, 57-59; the JUNK list arrives from
ml/configs/v1.yaml):
    substr(date, 1, 4) >= '1990'
    name IS NOT NULL AND trim(name) <> ''
    address1 IS NOT NULL AND upper(trim(address1)) NOT IN (JUNK)
No status or filing-type filter; heavy_row is never an eligibility test.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ucc_ml import legacy
from ucc_ml.contracts import (
    BASELINE_ROUTES,
    CASE_COLUMNS,
    REGIONS,
    Case,
    check_column,
    is_sha256_hex,
    make_borrower_key,
    make_case_id,
    make_group_id,
)
from ucc_ml.provenance import sha256_file

SNAPSHOT_TABLES: tuple[str, ...] = ("co_filings", "co_debtors", "co_secured_parties", "ct_filings", "scope_all")

OBSERVATION_COLUMNS: tuple[str, ...] = (
    "region", "file_id", "case_id", "borrower_key", "lineage_id", "borrower_name_raw",
    "borrower_city", "borrower_state", "borrower_zip", "earliest_observed_date", "latest_observed_date",
    "source_row_count", "source_filing_type", "source_status", "lender_names_all", "lender_row_count",
    "stage1_rows",
)


def connect_snapshot(snapshot_dir: Path) -> duckdb.DuckDBPyConnection:
    snapshot_dir = Path(snapshot_dir)
    # Default temp_directory is '.tmp' relative to cwd -- the founder's untracked .tmp/ when run from the
    # repo root -- and memory_limit defaults to ~12.7 GiB on this 16 GB machine. Keep both inside ml/data.
    con = duckdb.connect(config={"temp_directory": str(snapshot_dir.parent / ".duckdb_tmp"),
                                 "memory_limit": "8GB"})
    for table in SNAPSHOT_TABLES:
        path = snapshot_dir / f"{table}.parquet"
        if not path.exists():
            con.close()
            raise FileNotFoundError(path)
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path.as_posix()}')")
    return con


def junk_sql(junk: Sequence[str]) -> str:
    return ",".join("'" + j.replace("'", "''") + "'" for j in junk)


# ----------------------------------------------------------------------------- stage 1: CO

def _co_stage1(con: duckdb.DuckDBPyConnection, year_min: str, junk: Sequence[str]) -> dict:
    con.execute("""
        CREATE OR REPLACE TEMP TABLE co_filing_keys AS
        SELECT fileid AS file_id,
               min(masterdocumentid) AS lineage_id,
               min(filingdate)       AS filingdate,
               max(filingdate)       AS filingdate_max,
               min(filingtype)       AS filingtype,
               max(terminationflag)  AS terminationflag,
               count(*)              AS filing_rows
        FROM co_filings
        WHERE fileid IS NOT NULL
        GROUP BY fileid""")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE co_joined AS
        SELECT f.file_id, f.lineage_id, f.filingdate, f.filingdate_max, f.filingtype, f.terminationflag,
               d.organizationname AS borrower_name_raw, d.address1, d.city, d.state, d.zipcode,
               coalesce(substr(f.filingdate_max, 1, 4) >= '{year_min}', false)           AS ok_year,
               (d.organizationname IS NOT NULL AND trim(d.organizationname) <> '')         AS ok_name,
               (d.address1 IS NOT NULL AND upper(trim(d.address1)) NOT IN ({junk_sql(junk)})) AS ok_address
        FROM co_debtors d
        JOIN co_filing_keys f ON f.file_id = d.fileid""")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE co_lenders AS
        SELECT fileid AS file_id, list(organizationname) AS lender_names_all, count(*) AS lender_row_count
        FROM co_secured_parties
        WHERE fileid IS NOT NULL
        GROUP BY fileid""")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE stage1 AS
        SELECT row_number() OVER (ORDER BY j.file_id, j.borrower_name_raw, j.address1, j.city, j.state, j.zipcode) AS rid,
               'CO' AS region, j.file_id, j.lineage_id,
               j.borrower_name_raw, j.address1, j.city, j.state, j.zipcode,
               min(j.filingdate) AS earliest_observed_date, max(j.filingdate_max) AS latest_observed_date,
               count(*)          AS source_row_count,
               min(j.filingtype) AS source_filing_type, min(j.terminationflag) AS source_status,
               coalesce(any_value(l.lender_names_all), []::VARCHAR[]) AS lender_names_all,
               coalesce(any_value(l.lender_row_count), 0)             AS lender_row_count
        FROM co_joined j
        LEFT JOIN co_lenders l ON l.file_id = j.file_id
        WHERE j.ok_year AND j.ok_name AND j.ok_address
        GROUP BY j.file_id, j.lineage_id, j.borrower_name_raw, j.address1, j.city, j.state, j.zipcode""")
    q = lambda sql: con.execute(sql).fetchone()  # noqa: E731
    filings_rows, filings_distinct = q("SELECT count(*), count(DISTINCT fileid) FROM co_filings")
    (multi_date,) = q("SELECT count(*) FROM co_filing_keys WHERE filingdate IS DISTINCT FROM filingdate_max")
    (no_debtor,) = q("""SELECT count(*) FROM co_filing_keys f
                        LEFT JOIN (SELECT DISTINCT fileid FROM co_debtors) d ON d.fileid = f.file_id
                        WHERE d.fileid IS NULL""")
    debtors_rows, debtors_blank = q("""SELECT count(*), count(*) FILTER (WHERE organizationname IS NULL OR trim(organizationname) = '')
                                       FROM co_debtors""")
    (orphans,) = q("""SELECT count(*) FROM co_debtors d LEFT JOIN co_filing_keys f ON f.file_id = d.fileid
                      WHERE f.file_id IS NULL""")
    sp_rows, sp_blank = q("""SELECT count(*), count(*) FILTER (WHERE organizationname IS NULL OR trim(organizationname) = '')
                             FROM co_secured_parties""")
    w = q("""SELECT count(*),
                    count(*) FILTER (WHERE NOT ok_year),
                    count(*) FILTER (WHERE ok_year AND NOT ok_name),
                    count(*) FILTER (WHERE ok_year AND ok_name AND NOT ok_address),
                    count(*) FILTER (WHERE ok_year AND ok_name AND ok_address),
                    count(*) FILTER (WHERE NOT ok_year),
                    count(*) FILTER (WHERE NOT ok_name),
                    count(*) FILTER (WHERE NOT ok_address)
             FROM co_joined""")
    return {
        "source": {
            "filings_rows": int(filings_rows), "filings_distinct_fileid": int(filings_distinct),
            "filings_duplicate_rows": int(filings_rows - filings_distinct),
            "filings_with_conflicting_dates": int(multi_date), "filings_without_debtor_row": int(no_debtor),
            "debtors_rows": int(debtors_rows), "debtors_blank_name_rows": int(debtors_blank),
            "debtors_without_filing_row": int(orphans),
            "secured_parties_rows": int(sp_rows), "secured_parties_blank_name_rows": int(sp_blank),
        },
        "waterfall": {
            "joined_rows": int(w[0]), "excluded_year_before_1990_or_null": int(w[1]),
            "excluded_blank_name": int(w[2]), "excluded_placeholder_address": int(w[3]), "eligible_rows": int(w[4]),
        },
        "standalone": {"year_fails": int(w[5]), "blank_name_fails": int(w[6]), "placeholder_address_fails": int(w[7])},
    }


# ----------------------------------------------------------------------------- stage 1: CT

def _ct_stage1(con: duckdb.DuckDBPyConnection, year_min: str, junk: Sequence[str]) -> dict:
    # Verbatim duplicate rows are dropped before anything is counted (context pack §3: drop, count).
    con.execute("CREATE OR REPLACE TEMP TABLE ct_distinct AS SELECT DISTINCT * FROM ct_filings")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ct_rows AS
        SELECT id_lien_flng_nbr AS file_id, lien_status, cd_flng_type,
               debtor_nm_bus AS borrower_name_raw, debtor_ad_str1 AS address1,
               debtor_ad_city AS city, debtor_ad_state AS state, debtor_ad_zip AS zipcode,
               dt_accept,
               (id_lien_flng_nbr IS NOT NULL)                                              AS ok_id,
               coalesce(substr(dt_accept, 1, 4) >= '{year_min}', false)                    AS ok_year,
               (debtor_nm_bus IS NOT NULL AND trim(debtor_nm_bus) <> '')                    AS ok_name,
               (debtor_ad_str1 IS NOT NULL AND upper(trim(debtor_ad_str1)) NOT IN ({junk_sql(junk)})) AS ok_address
        FROM ct_distinct""")
    # A CT lien is a history of rows, each pairing ONE debtor with ONE secured party. The case carries
    # the filing's FULL lender set -- every secured party on ANY row of the lien, whatever that row's
    # debtor, address variant or date -- exactly as CO attaches every secured_parties row of the fileid.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE ct_lenders AS
        SELECT id_lien_flng_nbr AS file_id, list(sec_party_nm_bus) AS lender_names_all, count(*) AS lender_row_count
        FROM ct_distinct
        WHERE id_lien_flng_nbr IS NOT NULL
        GROUP BY id_lien_flng_nbr""")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE stage1 AS
        SELECT row_number() OVER (ORDER BY r.file_id, r.borrower_name_raw, r.address1, r.city, r.state, r.zipcode) AS rid,
               'CT' AS region, r.file_id, NULL::VARCHAR AS lineage_id,
               r.borrower_name_raw, r.address1, r.city, r.state, r.zipcode,
               min(r.dt_accept) AS earliest_observed_date, max(r.dt_accept) AS latest_observed_date,
               count(*)         AS source_row_count,
               string_agg(DISTINCT r.cd_flng_type, '|' ORDER BY r.cd_flng_type) AS source_filing_type,
               string_agg(DISTINCT r.lien_status,  '|' ORDER BY r.lien_status)  AS source_status,
               coalesce(any_value(l.lender_names_all), []::VARCHAR[]) AS lender_names_all,
               coalesce(any_value(l.lender_row_count), 0)             AS lender_row_count
        FROM ct_rows r
        LEFT JOIN ct_lenders l ON l.file_id = r.file_id
        WHERE r.ok_id AND r.ok_year AND r.ok_name AND r.ok_address
        GROUP BY r.file_id, r.borrower_name_raw, r.address1, r.city, r.state, r.zipcode""")
    q = lambda sql: con.execute(sql).fetchone()  # noqa: E731
    (rows,) = q("SELECT count(*) FROM ct_filings")
    (distinct_rows,) = q("SELECT count(*) FROM ct_distinct")
    blank_name, blank_lender = q("""SELECT count(*) FILTER (WHERE debtor_nm_bus IS NULL OR trim(debtor_nm_bus) = ''),
                                           count(*) FILTER (WHERE sec_party_nm_bus IS NULL OR trim(sec_party_nm_bus) = '')
                                    FROM ct_filings""")
    w = q("""SELECT count(*),
                    count(*) FILTER (WHERE NOT ok_id),
                    count(*) FILTER (WHERE ok_id AND NOT ok_year),
                    count(*) FILTER (WHERE ok_id AND ok_year AND NOT ok_name),
                    count(*) FILTER (WHERE ok_id AND ok_year AND ok_name AND NOT ok_address),
                    count(*) FILTER (WHERE ok_id AND ok_year AND ok_name AND ok_address),
                    count(*) FILTER (WHERE NOT ok_year),
                    count(*) FILTER (WHERE NOT ok_name),
                    count(*) FILTER (WHERE NOT ok_address)
             FROM ct_rows""")
    return {
        "source": {
            "ct_rows": int(rows), "ct_verbatim_duplicate_rows": int(rows - distinct_rows),
            "ct_blank_name_rows": int(blank_name), "ct_blank_lender_rows": int(blank_lender),
        },
        "waterfall": {
            "source_rows": int(rows), "excluded_verbatim_duplicate": int(rows - distinct_rows),
            "joined_rows": int(w[0]), "excluded_null_lien_id": int(w[1]),
            "excluded_year_before_1990_or_null": int(w[2]), "excluded_blank_name": int(w[3]),
            "excluded_placeholder_address": int(w[4]), "eligible_rows": int(w[5]),
        },
        "standalone": {"year_fails": int(w[6]), "blank_name_fails": int(w[7]), "placeholder_address_fails": int(w[8])},
    }


# ----------------------------------------------------------------------------- stages 2 + 3

def _stage2_keys(con: duckdb.DuckDBPyConnection) -> None:
    """Compute borrower_key / case_id in Python for every stage-1 row and register them by rid."""
    raw = con.execute("""SELECT rid, region, file_id, borrower_name_raw, address1, city, state, zipcode
                         FROM stage1 ORDER BY rid""").df()
    keys = [make_borrower_key(n, a, c, s, z)
            for n, a, c, s, z in zip(raw.borrower_name_raw, raw.address1, raw.city, raw.state, raw.zipcode)]
    case_ids = [make_case_id(r, f, k) for r, f, k in zip(raw.region, raw.file_id, keys)]
    con.register("stage2_keys", pd.DataFrame({"rid": raw.rid.astype("int64"), "borrower_key": keys, "case_id": case_ids}))


def _stage3(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("""
        SELECT s.region, s.file_id, k.case_id, k.borrower_key,
               min(s.lineage_id)         AS lineage_id,
               min(s.borrower_name_raw)  AS borrower_name_raw,
               min(s.city)               AS borrower_city,
               min(s.state)              AS borrower_state,
               min(s.zipcode)            AS borrower_zip,
               min(s.earliest_observed_date) AS earliest_observed_date,
               max(s.latest_observed_date)   AS latest_observed_date,
               sum(s.source_row_count)::BIGINT AS source_row_count,
               string_agg(DISTINCT s.source_filing_type, '|' ORDER BY s.source_filing_type) AS source_filing_type,
               string_agg(DISTINCT s.source_status,      '|' ORDER BY s.source_status)      AS source_status,
               list_distinct(flatten(list(s.lender_names_all))) AS lender_names_all,
               max(s.lender_row_count)::BIGINT AS lender_row_count,
               count(*)::BIGINT                AS stage1_rows
        FROM stage1 s
        JOIN stage2_keys k ON k.rid = s.rid
        GROUP BY s.region, s.file_id, k.case_id, k.borrower_key
        ORDER BY k.case_id""").df()
    con.unregister("stage2_keys")
    df["lender_names_all"] = [list(v) if v is not None else [] for v in df["lender_names_all"]]
    for col in ("source_row_count", "lender_row_count", "stage1_rows"):
        df[col] = df[col].astype("int64")
    df = df.reindex(columns=list(OBSERVATION_COLUMNS))
    df = df.astype({c: object for c in ("lineage_id", "source_filing_type", "source_status",
                                         "earliest_observed_date", "latest_observed_date",
                                         "borrower_city", "borrower_state", "borrower_zip")})
    df = df.where(pd.notna(df), None)
    return df.reset_index(drop=True)


def extract_region(con: duckdb.DuckDBPyConnection, region: str, year_min: str,
                   junk: Sequence[str]) -> tuple[pd.DataFrame, dict]:
    if region == "CO":
        exclusions = _co_stage1(con, year_min, junk)
    elif region == "CT":
        exclusions = _ct_stage1(con, year_min, junk)
    else:
        raise ValueError(f"unknown region {region!r}")
    (stage1_rows,) = con.execute("SELECT count(*) FROM stage1").fetchone()
    _stage2_keys(con)
    obs = _stage3(con)
    if not obs.case_id.is_unique:
        raise RuntimeError(f"{region}: duplicate case_id after stage 3")
    exclusions["stage1_raw_groups"] = int(stage1_rows)
    exclusions["cases"] = int(len(obs))
    exclusions["case_insensitive_merges"] = int(stage1_rows - len(obs))
    exclusions["join_expansion_rows"] = int((obs.source_row_count * obs.lender_row_count.clip(lower=1)).sum())
    return obs, exclusions
# ----------------------------------------------------------------------------- stage 4

def canonical_lender_set(values: Iterable | None) -> list[str]:
    """Sorted unique trimmed non-blank lender strings. None / NaN / '' are dropped (pack §3)."""
    out: set[str] = set()
    for v in values or []:
        if isinstance(v, str):
            s = v.strip()
            if s:
                out.add(s)
    return sorted(out)


def parse_iso_date(value) -> date | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


_NULLABLE_STRING_COLUMNS = ("lineage_id", "borrower_name_clean", "borrower_suffix", "borrower_city",
                            "borrower_state", "borrower_zip", "source_filing_type", "source_status")
_DATE_COLUMNS = ("earliest_observed_date", "latest_observed_date")


def _nullable_to_object(df: pd.DataFrame) -> pd.DataFrame:
    """pandas 3 stores None in its default `str` dtype as NaN; keep nullable columns as object with None."""
    for c in _NULLABLE_STRING_COLUMNS + _DATE_COLUMNS:
        df[c] = pd.Series([v if isinstance(v, (str, date)) else None for v in df[c]], index=df.index, dtype=object)
    return df


def finalize_cases(obs: pd.DataFrame, dataset_version: str) -> pd.DataFrame:
    """Stage 4: name normalisation, canonical lender sets, region-scoped group_id, frozen baseline."""
    clean_cache: dict[str, tuple[str | None, str | None]] = {}

    def clean(raw: str) -> tuple[str | None, str | None]:
        hit = clean_cache.get(raw)
        if hit is None:
            hit = legacy.normalize_name(raw)
            clean_cache[raw] = hit
        return hit

    rows: list[dict] = []
    for r in obs.itertuples(index=False):
        lenders = canonical_lender_set(r.lender_names_all)
        name_clean, suffix = clean(r.borrower_name_raw)
        lenders_clean = sorted({nc for nc in (clean(l)[0] for l in lenders) if nc})
        rows.append({
            "case_id": r.case_id,
            "dataset_version": dataset_version,
            "region": r.region,
            "file_id": r.file_id,
            "lineage_id": r.lineage_id if isinstance(r.lineage_id, str) else None,
            "borrower_key": r.borrower_key,
            "borrower_name_raw": r.borrower_name_raw,
            "borrower_name_clean": name_clean,
            "borrower_suffix": suffix,
            "lender_names_raw": lenders,
            "lender_names_clean": lenders_clean,
            "earliest_observed_date": parse_iso_date(r.earliest_observed_date),
            "latest_observed_date": parse_iso_date(r.latest_observed_date),
            "borrower_city": r.borrower_city if isinstance(r.borrower_city, str) else None,
            "borrower_state": r.borrower_state if isinstance(r.borrower_state, str) else None,
            "borrower_zip": r.borrower_zip if isinstance(r.borrower_zip, str) else None,
            "source_row_count": int(r.source_row_count),
            "source_filing_type": r.source_filing_type if isinstance(r.source_filing_type, str) else None,
            "source_status": r.source_status if isinstance(r.source_status, str) else None,
            "baseline_qualifies": legacy.baseline_qualifies(r.borrower_name_raw, lenders),
            "baseline_route": legacy.baseline_route(r.borrower_name_raw, lenders),
            "group_id": make_group_id(r.region, name_clean, r.borrower_name_raw),
        })
    df = _nullable_to_object(pd.DataFrame(rows, columns=list(CASE_COLUMNS)))
    return df.sort_values("case_id", kind="mergesort").reset_index(drop=True)


def validate_cases(df: pd.DataFrame) -> int:
    if list(df.columns) != list(CASE_COLUMNS):
        raise ValueError(f"columns {list(df.columns)} != CASE_COLUMNS")
    for i, row in enumerate(df.to_dict("records")):
        try:
            Case.model_validate(row)
        except Exception as exc:  # pydantic.ValidationError, re-raised with the row index
            raise ValueError(f"row {i}: {exc}") from exc
    return len(df)


CANDIDATE_SCHEMA = pa.schema([
    ("case_id", pa.string()), ("dataset_version", pa.string()), ("region", pa.string()),
    ("file_id", pa.string()), ("lineage_id", pa.string()), ("borrower_key", pa.string()),
    ("borrower_name_raw", pa.string()), ("borrower_name_clean", pa.string()), ("borrower_suffix", pa.string()),
    ("lender_names_raw", pa.list_(pa.string())), ("lender_names_clean", pa.list_(pa.string())),
    ("earliest_observed_date", pa.date32()), ("latest_observed_date", pa.date32()),
    ("borrower_city", pa.string()), ("borrower_state", pa.string()), ("borrower_zip", pa.string()),
    ("source_row_count", pa.int64()), ("source_filing_type", pa.string()), ("source_status", pa.string()),
    ("baseline_qualifies", pa.bool_()), ("baseline_route", pa.string()), ("group_id", pa.string()),
])
assert tuple(CANDIDATE_SCHEMA.names) == CASE_COLUMNS

_LIST_COLUMNS = ("lender_names_raw", "lender_names_clean")


def _column_values(df: pd.DataFrame, name: str) -> list:
    if name in _LIST_COLUMNS:
        return [list(v) for v in df[name]]
    if name in _DATE_COLUMNS:
        return [v if isinstance(v, date) else None for v in df[name]]
    if name == "source_row_count":
        return [int(v) for v in df[name]]
    if name == "baseline_qualifies":
        return [bool(v) for v in df[name]]
    return [v if isinstance(v, str) else None for v in df[name]]


def write_candidates(df: pd.DataFrame, path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict({c: _column_values(df, c) for c in CASE_COLUMNS}, schema=CANDIDATE_SCHEMA)
    pq.write_table(table, path, compression="zstd")
    return sha256_file(path)


def _is_string_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def read_candidates(path: Path) -> pd.DataFrame:
    """The only reader of candidates.parquet (contract K7).

    Checks the exact column order, that both lender columns are list<string> with no null lists, that
    case_id / borrower_key / group_id are sha256 hex, region is CO|CT, baseline_route is one of four
    values, baseline_qualifies is a real boolean and case_id is unique. Any failure raises ValueError
    naming the file, the column and the offending values.
    """
    path = Path(path)
    table = pq.read_table(path)
    if tuple(table.schema.names) != CASE_COLUMNS:
        raise ValueError(f"{path}: columns {table.schema.names} != CASE_COLUMNS")
    for name in _LIST_COLUMNS:
        if not pa.types.is_list(table.schema.field(name).type):
            raise ValueError(f"{path}: column {name!r} must be list<string>, got {table.schema.field(name).type}")
    df = pd.DataFrame({c: table.column(c).to_pylist() for c in CASE_COLUMNS}, columns=list(CASE_COLUMNS))
    for name in ("case_id", "borrower_key", "group_id"):
        check_column(path, df, name, is_sha256_hex)
    check_column(path, df, "region", lambda v: v in REGIONS)
    check_column(path, df, "baseline_route", lambda v: v in BASELINE_ROUTES)
    check_column(path, df, "baseline_qualifies", lambda v: isinstance(v, bool))
    for name in _LIST_COLUMNS:
        check_column(path, df, name, _is_string_list)
    if not df.case_id.is_unique:
        dupes = sorted(set(df.case_id[df.case_id.duplicated()]))
        raise ValueError(f"{path}: column 'case_id' has {len(dupes)} duplicate value(s): {dupes[:5]}")
    return _nullable_to_object(df)
