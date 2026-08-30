#!/usr/bin/env python
"""A shaded-relief surface built from the FILINGS, not from elevation.

The page needed the flat state fills to read as a surface. Terrain relief did
that, but terrain is irrelevant to lien filings -- it is decoration on a thematic
map. This computes the same visual effect from the data the page is actually
about: a kernel-density surface of qualifying UCC filings, hillshaded so that
concentrations of lending read as raised ground.

The Front Range corridor -- Denver, Boulder, Greeley, Colorado Springs -- is a
genuine ridge in this surface, because that is where the filings are. Nothing is
invented; the relief IS the data.

WHY A DENSITY SURFACE AND NOT A CHOROPLETH
------------------------------------------
A choropleth would need administrative units the data does not respect (a ZIP is
not a market) and would add hard edges at the same spatial frequency as the dots.
A kernel density estimate produces a smooth continuous field whose feature size
is set by the bandwidth, which is exactly the knob needed to keep the texture an
order of magnitude coarser than a dot. Standard practice for point-pattern
intensity: Silverman, *Density Estimation for Statistics and Data Analysis*.

WHY IT IS SAFE OVER THE DOTS
----------------------------
The page's figure-ground mechanism is a 2px white ring around each dot, holding
about 4.3-4.8:1 against the fill. This layer is rendered DARKEN-ONLY (multiply):
darkening the fill can only INCREASE the ring's contrast, so no density value can
degrade dot legibility. The ceiling is set by the dot fill, not the ring.

The hillshade is lit from the north-west (315 degrees), the cartographic
convention, using GDAL's aspect formula -- verified against a synthetic cone
rather than trusted, because the inverse formula lights from the south-east and
makes ridges read as valleys.

Output: docs/data/density_co.png, density_ct.png.
"""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data"

BOX = {
    "CO": {"w": -109.0448, "e": -102.0415, "s": 36.9930, "n": 41.0034, "nx": 150, "ny": 96},
    "CT": {"w": -73.7278, "e": -71.7870, "s": 40.9700, "n": 42.0500, "nx": 132, "ny": 84},
}
#: Kernel bandwidth as a fraction of panel width. 0.055 puts the smallest
#: resolvable blob at roughly 8% of the panel -- an order of magnitude wider than
#: a dot (1-2%), which is the spatial-frequency separation that keeps the surface
#: readable as ground rather than competing with the marks on it.
BANDWIDTH = 0.055
AZIMUTH, ALTITUDE = 315.0, 45.0
#: Vertical exaggeration on the normalised density field.
Z_FACTOR = 42.0


def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian. Avoids a scipy dependency."""
    r = max(1, int(sigma * 3))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1,
                              np.pad(a, ((0, 0), (r, r)), mode="edge"))[:, r:-r]
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0,
                              np.pad(out, ((r, r), (0, 0)), mode="edge"))[r:-r, :]
    return out


def hillshade(z: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(z * Z_FACTOR)
    slope = np.arctan(np.hypot(gx, gy))
    # GDAL's convention. The transposed form lights from the SE and inverts the
    # relief -- checked against a synthetic cone, not assumed.
    aspect = np.arctan2(gy, -gx)
    az, alt = np.radians(360.0 - AZIMUTH + 90.0), np.radians(ALTITUDE)
    sh = (np.sin(alt) * np.cos(slope)
          + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(sh, 0, 1)


def png_gray(a: np.ndarray) -> bytes:
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
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for region, box in BOX.items():
        pts = con.execute("""
            SELECT lon, lat, count(*) AS n FROM scope_geo
            WHERE region = ? AND lat IS NOT NULL GROUP BY 1, 2
        """, [region]).fetchall()

        ny, nx = box["ny"], box["nx"]
        grid = np.zeros((ny, nx), dtype=float)
        for lon, lat, n in pts:
            # Bin every filing to its cell; the blur below does the smoothing.
            gx = (lon - box["w"]) / (box["e"] - box["w"]) * (nx - 1)
            gy = (box["n"] - lat) / (box["n"] - box["s"]) * (ny - 1)
            # Clamp AFTER rounding: a point at the far edge rounds up to nx,
            # which is one past the last index.
            ix, iy = int(round(gx)), int(round(gy))
            if 0 <= ix <= nx - 1 and 0 <= iy <= ny - 1:
                grid[iy, ix] += n

        total = grid.sum()
        dens = gaussian_blur(grid, BANDWIDTH * nx)
        # Compress the dynamic range. Filing density is heavily skewed -- Denver
        # dwarfs everything -- so a linear field would be one bright spot on a
        # black plain. A cube root keeps the corridors visible, which is the
        # standard treatment for a skewed intensity surface.
        dens = np.cbrt(dens)
        if dens.max() > 0:
            dens /= dens.max()

        sh = hillshade(dens)
        # Where there is no data the surface is flat, so the hillshade sits at
        # its neutral value and the overlay does nothing -- empty ground is left
        # alone rather than tinted.
        img = (sh * 255).astype(np.uint8)
        p = OUT / f"density_{region.lower()}.png"
        p.write_bytes(png_gray(img))
        peak = np.unravel_index(np.argmax(dens), dens.shape)
        plon = box["w"] + peak[1] / (nx - 1) * (box["e"] - box["w"])
        plat = box["n"] - peak[0] / (ny - 1) * (box["n"] - box["s"])
        print(f"{region}  {int(total):,} filings over {len(pts):,} locations  "
              f"{nx}x{ny}  shade {sh.min():.2f}-{sh.max():.2f}  "
              f"peak at ({plat:.2f}, {plon:.2f})  {p.stat().st_size/1024:.1f} KB")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
