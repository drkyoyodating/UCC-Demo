#!/usr/bin/env python
"""Build the resolver corpus from the CURRENT heavy-construction scope (CO + CT).

ADDITIVE. Touches no frozen file: `resolve.py`, `corpus.py` and `heavy_filter.py`
are all imported or left alone, never edited.

WHY THIS EXISTS
---------------
`corpus.py` builds `corpus_debtors_eq` / `corpus_lenders_eq` from the Colorado
`debtors` / `secured_parties` tables filtered on `collateral = 'EQUIPMENT'`.
That is the OLD scope and it is stale twice over:

  1. **Colorado only.** Connecticut lives in `ct_filings` and was reinstated after
     Route B was found not to need a collateral column at all.
  2. **EQUIPMENT-collateral only.** Colorado stopped coding collateral after 2012
     (2011: 2,940 -> 2012: 998 -> 2013: 1), so that filter silently drops the
     entire post-2012 register while total filings kept rising.

The models trained on those tables have therefore never seen Connecticut and do
include Colorado records the current two-route filter excludes. Publishing views
over entities the model never resolved is not possible, which is why P5 must be
re-run on this corpus rather than the old one.

THE TWO REGISTERS ARE NOT COLUMN-COMPATIBLE
-------------------------------------------
This is the same incompatibility `normalize.COLUMN_MAP` records, one level up:

    Colorado (`debtors`)     debtorid | actiontype     | recordstatus
    Connecticut (`ct_filings`)  --    | cd_flng_type   | lien_status

Connecticut publishes **no party id at all**, so one is synthesised (see
`_CT_ID`). It is deterministic — the same input row always yields the same id —
because `resolve.py` requires a stable `unique_id` and the whole pipeline is
supposed to be reproducible across pulls.

`build_records()` in resolve.py re-derives its OWN record key by grouping on
(name_clean, suffix, address1, city, state, zipcode) and keeps `min(unique_id)`
only as `example_party_id`. So `unique_id` here needs to be stable and unique,
not meaningful.

SCOPE PREDICATE
---------------
Identical to `build_scope.py`: 1990+, borrower name present, borrower address
present and not a placeholder, and Route A or Route B per `heavy_filter`. It is
re-derived here from source rather than read off `scope_all`, because `scope_all`
is filing-level and carries no party id or record status.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heavy_filter import (BORROWER_SQL, LENDER_SQL,          # noqa: E402
                          is_heavy_borrower, is_heavy_lender)
from normalize import normalize_name                # noqa: E402
from splink_contract import SEED                    # noqa: E402

HOLDOUT_PCT = 10
OUT = "corpus_scope_all"

#: Placeholder addresses that count as MISSING. Kept byte-identical to
#: build_scope.py so the two cannot drift apart.
JUNK = ("'','NONE','NONE PROVIDED','NA','N/A','UNKNOWN','SAME','COMPANY',"
        "'NOT PROVIDED','TBD','X','XX','NO ADDRESS','ADDRESS UNKNOWN','VARIOUS'")

#: Colorado: a row is live unless the filing was deleted.
CO_DEAD_ACTIONS = ("delete only", "change and delete")
#: Connecticut: the analogous states. `lien_status` carries Active/Released/
#: Terminated; `cd_flng_type` carries the release flavours.
CT_DEAD_TYPES = ("RELEASE ORIG", "RELEASE", "PARTIAL RELEASE")


def in_holdout(party_id: str) -> bool:
    """Deterministic 10% assignment, identical in form to corpus.py's."""
    h = hashlib.md5(f"{SEED}:{party_id}".encode()).hexdigest()
    return int(h[:8], 16) % 100 < HOLDOUT_PCT


def _ct_id(fileid, name, addr) -> str:
    """Connecticut publishes no party id. Synthesise a stable one.

    Deterministic in the row's own content, so a re-pull of unchanged data
    yields the same id and the holdout split does not move under the model.
    """
    key = "|".join(str(x or "") for x in (fileid, name, addr))
    return "CT:" + hashlib.md5(key.encode()).hexdigest()[:16]


def build(con: duckdb.DuckDBPyConnection) -> dict:
    co = con.execute(f"""
        SELECT DISTINCT
               'CO:' || CAST(d.debtorid AS VARCHAR) AS unique_id,
               'CO'                    AS region,
               d.organizationname      AS name_raw,
               d.address1, d.city, d.state, d.zipcode,
               f.fileid,
               substr(f.filingdate, 1, 4) AS loan_year,
               d.actiontype            AS action_type,
               d.recordstatus          AS record_status,
               sp.organizationname     AS lender
        FROM filings f
        JOIN debtors d ON d.fileid = f.fileid
        LEFT JOIN secured_parties sp ON sp.fileid = f.fileid
        WHERE substr(f.filingdate, 1, 4) >= '1990'
          AND d.organizationname IS NOT NULL AND trim(d.organizationname) <> ''
          AND d.address1 IS NOT NULL AND upper(trim(d.address1)) NOT IN ({JUNK})
          AND (regexp_matches(upper(coalesce(sp.organizationname, '')), '{LENDER_SQL}')
            OR regexp_matches(upper(d.organizationname), '{BORROWER_SQL}'))
    """).df()

    ct = con.execute(f"""
        SELECT DISTINCT
               id_lien_flng_nbr        AS fileid,
               'CT'                    AS region,
               debtor_nm_bus           AS name_raw,
               debtor_ad_str1          AS address1,
               debtor_ad_city          AS city,
               debtor_ad_state         AS state,
               debtor_ad_zip           AS zipcode,
               substr(dt_accept, 1, 4) AS loan_year,
               cd_flng_type            AS action_type,
               lien_status             AS record_status,
               sec_party_nm_bus        AS lender
        FROM ct_filings
        WHERE substr(dt_accept, 1, 4) >= '1990'
          AND debtor_nm_bus IS NOT NULL AND trim(debtor_nm_bus) <> ''
          AND debtor_ad_str1 IS NOT NULL AND upper(trim(debtor_ad_str1)) NOT IN ({JUNK})
          AND (regexp_matches(upper(coalesce(sec_party_nm_bus, '')), '{LENDER_SQL}')
            OR regexp_matches(upper(debtor_nm_bus), '{BORROWER_SQL}'))
    """).df()

    ct["unique_id"] = [_ct_id(f, n, a)
                       for f, n, a in zip(ct.fileid, ct.name_raw, ct.address1)]

    # EXACT-PREDICATE PASS -- same reason as build_scope.py. The SQL regexes are a
    # fast superset that carries neither LENDER_DENY (banks / machine tools /
    # pure agriculture) nor the personal-name guard. Without this the corpus
    # would be WIDER than the scope it is supposed to mirror.
    for _d in (co, ct):
        _keep = [bool(is_heavy_lender(l) or is_heavy_borrower(b))
                 for b, l in zip(_d.name_raw, _d.lender)]
        _d.drop(_d.index[[not k for k in _keep]], inplace=True)

    import pandas as pd
    cols = ["unique_id", "region", "name_raw", "address1", "city", "state",
            "zipcode", "fileid", "loan_year", "action_type", "record_status", "lender"]
    df = pd.concat([co[cols], ct[cols]], ignore_index=True)

    # normalize_name is the P2 contract: never raises, never mutates, returns
    # (name_clean, suffix). A key-less row cannot be blocked, so it is dropped.
    norm = [normalize_name(n) for n in df["name_raw"]]
    df["name_clean"] = [a for a, _ in norm]
    df["suffix"] = [b for _, b in norm]

    is_co = df["region"].eq("CO")
    df["is_active"] = (
        (is_co & df["record_status"].eq("active")
              & ~df["action_type"].isin(CO_DEAD_ACTIONS))
        | (~is_co & df["record_status"].eq("Active")
                 & ~df["action_type"].isin(CT_DEAD_TYPES))
    )
    df["is_holdout"] = [in_holdout(str(u)) for u in df["unique_id"]]

    before = len(df)
    df = df[df["name_clean"].notna()]
    no_key = before - len(df)

    # One debtor on a filing with several secured parties comes back once per
    # secured party -- SELECT DISTINCT cannot collapse them because `lender`
    # differs. This corpus is party-RECORDS, so collapse to one row per party and
    # keep the first lender seen. Resolution blocks and scores on name/address
    # only, so lender multiplicity is not lost information here -- but it IS
    # dropped from this table, and the lender->borrower view must be built from
    # `scope_all`, never from this corpus.
    dup = int(df["unique_id"].duplicated().sum())
    if dup:
        df = df.drop_duplicates("unique_id")

    con.register("_scp", df)
    con.execute(f"CREATE OR REPLACE TABLE {OUT} AS SELECT * FROM _scp")
    con.unregister("_scp")
    (ROOT / "parquet").mkdir(exist_ok=True)
    con.execute(f"COPY {OUT} TO '{ROOT / 'parquet' / OUT}.parquet' (FORMAT parquet)")

    act = df[df.is_active]
    return {
        "rows": len(df),
        "dropped_no_key": no_key,
        "collapsed_multi_lender": dup,
        "co_rows": int(is_co.sum()),
        "ct_rows": int((~is_co).sum()),
        "active_rows": len(act),
        "distinct_raw": int(df.name_raw.nunique()),
        "distinct_key": int(df.name_clean.nunique()),
        "holdout_rows": int(df.is_holdout.sum()),
        "base_rows": int((~df.is_holdout).sum()),
        "with_suffix": int(df.suffix.notna().sum()),
    }


def main() -> int:
    con = duckdb.connect(str(ROOT / "ucc.duckdb"))
    con.execute("SET memory_limit='3GB'")
    s = build(con)
    print(f"=== {OUT} ===")
    w = max(len(k) for k in s)
    for k, v in s.items():
        print(f"  {k:<{w}}  {v:,}" if isinstance(v, int) else f"  {k:<{w}}  {v}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
