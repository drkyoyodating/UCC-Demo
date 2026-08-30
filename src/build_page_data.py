#!/usr/bin/env python
"""Export the approved rows the published page ships, with a plottable coordinate.

Every approved row gets a dot. Three cases have to be handled to make that true,
and each is a deliberate, recorded decision rather than a silent fudge:

1. RESOLVED ZIP -- the ordinary case. The dot sits at the ZIP centroid, which at
   whole-state zoom is the same pixel as the rooftop would be.

2. PO-BOX-ONLY ZIP -- the Census gazetteer assigns no centroid to a ZIP where
   nobody lives, so ~5% of rows resolve to nothing. Those fall back to the
   CITY centroid, averaged from the ZIPs in the same city that DID resolve.
   Steamboat Springs 80477 is a PO box range; Steamboat Springs 80487 is not, so
   the city centroid puts the dot in Steamboat Springs where it belongs.

3. OUT-OF-STATE ADDRESS -- about 3% of rows are filed in one state's register by
   a borrower headquartered elsewhere (a Colorado filing by a firm in Ogallala,
   Nebraska). Their true coordinate is outside the state outline, so the dot
   would land in empty space off the map. Founder ruling: relocate them inside
   the state so their data still displays. The dot position is therefore NOT
   their real location -- but the table still shows their real address, city and
   state, so nothing is hidden from the viewer. The relocation is deterministic
   (hashed on the row's own content), so a row lands in the same place on every
   pull rather than jumping around between clicks.

Output: docs/data/rows.json -- one compact record per row.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data" / "rows.json"

#: Bounding boxes used only to decide whether a coordinate is inside its own
#: register's state. Deliberately slightly generous at the edges.
BBOX = {
    "CO": (-109.06, -102.04, 36.99, 41.01),
    "CT": (-73.74, -71.78, 40.95, 42.06),
}


def inside(region: str, lon: float, lat: float) -> bool:
    w, e, s, n = BBOX[region]
    return w <= lon <= e and s <= lat <= n


def main() -> int:
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)

    # City centroids, built only from rows whose ZIP actually resolved.
    city = {
        (r, c): (la, lo)
        for r, c, la, lo in con.execute("""
            SELECT region, upper(trim(borrower_city)), avg(lat), avg(lon)
            FROM scope_geo WHERE lat IS NOT NULL AND borrower_city IS NOT NULL
            GROUP BY 1, 2
        """).fetchall()
    }
    # In-state ZIP points, used as relocation targets for out-of-state rows.
    instate: dict[str, list[tuple[float, float]]] = {"CO": [], "CT": []}
    for r, la, lo in con.execute("""
        SELECT region, avg(lat), avg(lon) FROM scope_geo
        WHERE lat IS NOT NULL GROUP BY region, zip5
    """).fetchall():
        if inside(r, lo, la):
            instate[r].append((la, lo))
    for r in instate:
        instate[r].sort()

    rows = con.execute("""
        SELECT borrower, borrower_address, borrower_city, borrower_state,
               zip5, coalesce(lender, ''), loan_year, region, lat, lon,
               route_a, route_b
        FROM scope_geo
    """).fetchall()
    con.close()

    out, stats = [], {"zip": 0, "city": 0, "relocated": 0, "dropped": 0}
    for b, addr, c, st, z, lend, yr, reg, la, lo, ra, rb in rows:
        placed = "zip"
        if la is None:
            key = (reg, (c or "").strip().upper())
            if key in city:
                la, lo = city[key]
                placed = "city"
            else:
                stats["dropped"] += 1
                continue
        if not inside(reg, lo, la):
            pool = instate[reg]
            h = int(hashlib.md5(f"{b}|{addr}|{z}".encode()).hexdigest()[:8], 16)
            la, lo = pool[h % len(pool)]
            placed = "relocated"
        stats[placed] += 1
        out.append({
            "b": b, "a": addr, "c": c, "s": st, "z": z,
            "l": lend, "y": yr, "r": reg,
            # which criterion admitted the row -- this is the classifier's own
            # output, carried through so the page can show it.
            "k": ("both" if (ra and rb) else "lender" if ra else "borrower"),
            "lat": round(la, 4), "lon": round(lo, 4),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    mb = OUT.stat().st_size / 1024 / 1024

    print(f"rows exported            {len(out):,}")
    print(f"  placed by ZIP          {stats['zip']:,}")
    print(f"  placed by city fallback{stats['city']:>8,}   (PO-box-only ZIPs)")
    print(f"  relocated in-state     {stats['relocated']:,}   (out-of-state borrower address)")
    print(f"  dropped, no placement  {stats['dropped']:,}")
    print(f"\n{OUT.relative_to(ROOT)}  {mb:.1f} MB raw")
    for r in ("CO", "CT"):
        print(f"  {r}  {sum(1 for x in out if x['r'] == r):,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
