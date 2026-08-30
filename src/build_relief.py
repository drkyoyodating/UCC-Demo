#!/usr/bin/env python
"""Real shaded relief for the two state panels, from public elevation data.

WHY REAL AND NOT INVENTED
-------------------------
The page needed the flat state fills to read as a surface. Simulated relief --
feTurbulence noise lit by a filter -- would have looked similar, but it would be
ornament on a thematic panel: relief implies elevation, and inventing it on a map
about lien filings is the kind of decoration a technical reader discounts.

Elevation is one of the most freely available datasets there is, so the honest
version costs the same. This fetches SRTM 90m samples from OpenTopoData (public,
free, no key), computes a standard hillshade, and embeds it as a greyscale PNG.
Colorado's relief is genuinely dramatic and Connecticut's genuinely is not --
which is true, visible, and costs nothing to be right about.

SPATIAL FREQUENCY IS THE DESIGN CONSTRAINT
------------------------------------------
The dots are r=3..9.5 in a 430-wide viewBox, i.e. roughly 1-2% of panel width.
Texture at that same scale would compete with them and destroy the data layer, so
the grid is deliberately COARSE (~28x18 for Colorado) and then smoothed. The
result is broad landform undulation an order of magnitude larger than a dot,
which reads as surface rather than as noise -- exactly the separation the
cartographic literature asks for when relief sits beneath a thematic layer.

Illumination is from the northwest at 315 degrees, the cartographic convention
(Imhof): lit from upper-left, because human perception resolves shape-from-shading
correctly under an assumed overhead-left light and inverts it under the opposite.

Output: docs/data/relief_co.png, relief_ct.png, plus relief.json carrying the
bounding boxes so the page can place each image exactly under its state path.
Cached -- once fetched, never refetched. Re-run with --refresh to pull again.
"""
from __future__ import annotations

import json
import struct
import sys
import time
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data"
CACHE = ROOT / "ref" / "elevation.json"
API = "https://api.opentopodata.org/v1/srtm90m"

#: Same bounding boxes the page projects with, so the image registers exactly.
BOX = {
    "co": {"w": -109.0448, "e": -102.0415, "s": 36.9930, "n": 41.0034, "nx": 28, "ny": 18},
    "ct": {"w": -73.7278, "e": -71.7870, "s": 40.9700, "n": 42.0500, "nx": 18, "ny": 12},
}
BATCH = 100          # OpenTopoData's per-request cap
AZIMUTH, ALTITUDE = 315.0, 45.0

#: Vertical exaggeration. At ~19 km grid spacing real slopes are far too gentle
#: to shade -- unexaggerated, Colorado's hillshade spans only 0.69-0.73 and is
#: invisible. Exaggeration is standard practice in relief depiction (Imhof).
#: 25 is chosen because it is the largest value that does not CLIP: Colorado
#: spans 0.23-0.94 and Connecticut 0.50-0.87. Crucially it is SHARED between the
#: two states, so the comparison stays truthful -- the Rockies read as dramatic
#: and Connecticut reads as gently rolling, which is the fact.
Z_FACTOR = 25.0


def fetch_grid(box: dict) -> np.ndarray:
    """Elevations on a lon/lat grid, north row first, in metres."""
    lats = np.linspace(box["n"], box["s"], box["ny"])
    lons = np.linspace(box["w"], box["e"], box["nx"])
    pts = [(la, lo) for la in lats for lo in lons]
    vals: list[float] = []
    for i in range(0, len(pts), BATCH):
        chunk = pts[i:i + BATCH]
        q = "|".join(f"{la:.4f},{lo:.4f}" for la, lo in chunk)
        url = API + "?" + urllib.parse.urlencode({"locations": q})
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
        vals += [(x["elevation"] if x["elevation"] is not None else 0.0)
                 for x in data["results"]]
        print(f"    {min(i + BATCH, len(pts))}/{len(pts)}")
        time.sleep(1.1)                     # be polite to a free public service
    return np.array(vals, dtype=float).reshape(box["ny"], box["nx"])


def smooth(a: np.ndarray, passes: int = 2) -> np.ndarray:
    """Box blur. Pushes the texture further from the dots' spatial frequency."""
    for _ in range(passes):
        p = np.pad(a, 1, mode="edge")
        a = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] + 4 * a) / 8.0
    return a


def hillshade(z: np.ndarray, box: dict) -> np.ndarray:
    """Standard hillshade, 0..1. Lambertian shading of the surface normal."""
    # Approximate ground spacing in metres so slope is not wildly exaggerated.
    mid = np.radians((box["n"] + box["s"]) / 2)
    dy = (box["n"] - box["s"]) * 111_320 / max(z.shape[0] - 1, 1)
    dx = (box["e"] - box["w"]) * 111_320 * np.cos(mid) / max(z.shape[1] - 1, 1)
    gy, gx = np.gradient(z * Z_FACTOR, dy, dx)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(360.0 - AZIMUTH + 90.0), np.radians(ALTITUDE)
    sh = (np.sin(alt) * np.cos(slope)
          + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(sh, 0, 1)


def png_gray(a: np.ndarray) -> bytes:
    """Minimal 8-bit greyscale PNG. Avoids a Pillow dependency."""
    h, w = a.shape
    raw = b"".join(b"\x00" + a[r].tobytes() for r in range(h))

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (struct.pack(">I", len(body)) + tag + body
                + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def main() -> int:
    refresh = "--refresh" in sys.argv
    CACHE.parent.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() and not refresh else {}

    meta = {}
    for k, box in BOX.items():
        if k in cache:
            z = np.array(cache[k], dtype=float)
            print(f"{k}: cached {z.shape[0]}x{z.shape[1]}")
        else:
            print(f"{k}: fetching {box['ny']}x{box['nx']} = {box['ny']*box['nx']} points")
            z = fetch_grid(box)
            cache[k] = z.tolist()
            CACHE.write_text(json.dumps(cache))
        sh = hillshade(smooth(z, 1), box)
        img = (sh * 255).astype(np.uint8)
        (OUT / f"relief_{k}.png").write_bytes(png_gray(img))
        meta[k] = {"w": box["w"], "e": box["e"], "s": box["s"], "n": box["n"]}
        print(f"   elevation {z.min():.0f}-{z.max():.0f} m   "
              f"shade {sh.min():.2f}-{sh.max():.2f}   "
              f"png {(OUT / f'relief_{k}.png').stat().st_size/1024:.1f} KB")

    (OUT / "relief.json").write_text(json.dumps(meta))
    print("\nwrote docs/data/relief_co.png, relief_ct.png, relief.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
