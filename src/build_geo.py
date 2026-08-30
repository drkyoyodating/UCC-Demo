#!/usr/bin/env python
"""Attach a map coordinate to every approved row, once, offline.

WHY ZIP CENTROIDS AND NOT STREET GEOCODING
------------------------------------------
The registers give a street address for 100% of approved rows, but 11.6% of those
are PO Boxes, which have no rooftop to find. More decisively: the published page
draws whole-state outlines. Colorado is ~380 miles wide; at ~500px that is
0.76 miles per pixel, and a populated ZIP spans 1-10 miles. A rooftop coordinate
and its ZIP centroid therefore land on the same pixel, or one apart. Street
geocoding would add a network dependency and a partial-failure mode to buy
sub-pixel accuracy nobody can see.

It is also cheap in a way worth stating: 107,353 approved rows resolve to only
**2,057 distinct ZIPs**, so this is two thousand lookups, not a hundred thousand.

SOURCE
------
US Census Bureau ZCTA Gazetteer (public domain, no key, no terms). Fetched once
into ref/ and cached; re-run with --refresh to pull it again. ZCTAs are the
Census tabulation approximation of USPS ZIP codes -- close enough that at this
zoom the difference is invisible, and unlike commercial ZIP tables it can be
redistributed.

OUTPUT
------
`zip_geo`   ZIP -> (lat, lon)             the lookup, ~33k rows
`scope_geo` every approved row + lat/lon  what the page ships

Rows whose ZIP does not resolve keep a NULL coordinate rather than being dropped
or guessed at: a dot in the wrong place is worse than no dot, and the row is
still real data that belongs in the corpus.
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "ref"
GAZ = REF / "zcta_gaz.zip"
URL = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
       "2023_Gazetteer/2023_Gaz_zcta_national.zip")


def load_gazetteer(refresh: bool = False) -> list[tuple[str, float, float]]:
    """ZIP -> (lat, lon) from the cached Census gazetteer, fetching if needed."""
    REF.mkdir(exist_ok=True)
    if refresh or not GAZ.exists():
        print(f"fetching {URL}")
        with urllib.request.urlopen(URL, timeout=120) as r:
            GAZ.write_bytes(r.read())
    z = zipfile.ZipFile(GAZ)
    raw = z.read(z.namelist()[0]).decode("utf-8", "ignore").splitlines()
    out = []
    for line in raw[1:]:
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        try:
            out.append((parts[0].strip().zfill(5),
                        float(parts[5].strip()), float(parts[6].strip())))
        except ValueError:
            continue
    return out


def main() -> int:
    refresh = "--refresh" in sys.argv
    geo = load_gazetteer(refresh)
    print(f"gazetteer ZIPs: {len(geo):,}")

    con = duckdb.connect(str(ROOT / "ucc.duckdb"))
    con.execute("SET memory_limit='2GB'")
    con.execute("CREATE OR REPLACE TABLE zip_geo (zip VARCHAR, lat DOUBLE, lon DOUBLE)")
    con.executemany("INSERT INTO zip_geo VALUES (?,?,?)", geo)

    # Normalise the register's ZIP the same way the labelling rules do: strip
    # non-digits, take the first five, and LEFT-PAD. Connecticut ZIPs lose their
    # leading zero to numeric coercion -- 06103 becomes 6103 -- and without the
    # pad every Connecticut dot would fail to resolve.
    con.execute("""
        CREATE OR REPLACE TABLE scope_geo AS
        SELECT s.*,
               lpad(regexp_replace(coalesce(s.borrower_zip, ''), '[^0-9]', '', 'g'), 5, '0') AS zip5,
               g.lat, g.lon
        FROM scope_all s
        LEFT JOIN zip_geo g
          ON g.zip = lpad(regexp_replace(coalesce(s.borrower_zip, ''), '[^0-9]', '', 'g'), 5, '0')
    """)

    n = con.execute("SELECT count(*) FROM scope_geo").fetchone()[0]
    hit = con.execute("SELECT count(*) FROM scope_geo WHERE lat IS NOT NULL").fetchone()[0]
    zips = con.execute("SELECT count(DISTINCT zip5) FROM scope_geo").fetchone()[0]
    plot = con.execute(
        "SELECT count(DISTINCT zip5) FROM scope_geo WHERE lat IS NOT NULL").fetchone()[0]
    print(f"\napproved rows          {n:,}")
    print(f"  with a coordinate    {hit:,}  ({100 * hit / n:.2f}%)")
    print(f"  no coordinate        {n - hit:,}")
    print(f"distinct ZIPs in pull  {zips:,}")
    print(f"  resolved to a point  {plot:,}")

    print("\nrows per region:")
    for r, c_, lo, la in con.execute("""
        SELECT region, count(*), round(min(lon), 2), round(min(lat), 2)
        FROM scope_geo WHERE lat IS NOT NULL GROUP BY 1 ORDER BY 1""").fetchall():
        print(f"  {r}  {c_:,}")

    print("\nbiggest dots (ZIP clusters):")
    for z, c_, la, lo in con.execute("""
        SELECT zip5, count(*) n, round(avg(lat), 3), round(avg(lon), 3)
        FROM scope_geo WHERE lat IS NOT NULL
        GROUP BY 1 ORDER BY n DESC LIMIT 8""").fetchall():
        print(f"  {z}  {c_:>6,} rows  ({la}, {lo})")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
