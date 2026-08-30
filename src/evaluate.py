#!/usr/bin/env python
"""P6 -- compute every published figure from the labels. Regenerates all of them.

Design notes that matter for reading the numbers:

STRATA ARE RE-BINNED ON ACTUAL WEIGHT. Batch 1's lender bands were coarse
([2,6),[6,8)) and batch 2 added [4,6),[6,7),[7,8) -- overlapping ranges. Pooling
overlapping strata would double count, so every labelled pair is instead assigned
to the FINE band its real weight falls in, and N_h for that fine band is counted
directly from the scored predictions. This is valid: a random draw from [2,6)
restricted to those landing in [4,6) is a random sample of [4,6), so it pools
correctly with a direct [4,6) draw -- just at a different sampling rate.

PRECISION IS STRATUM-WEIGHTED, NEVER POOLED RAW. An unweighted pool of a
stratified sample reports the SAMPLE's design weights, not the population's.
Band boundaries sit exactly on the published thresholds so each threshold is a
clean union of whole bands, with no partial-band arithmetic.

TARGETED STRATA ARE NEVER POOLED INTO PRECISION. They are not a random sample of
anything; each reports its own error rate.

UNSURE is excluded from both numerator and denominator, and counted.
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import duckdb, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EDGES = [2, 4, 6, 7, 8, 10, 999]
THRESHOLDS = [4, 6, 7, 8, 10]
SHIPPED = {"debtor": 6.0, "lender": 8.0}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def band_of(w):
    for lo, hi in zip(EDGES, EDGES[1:]):
        if lo <= w < hi:
            return (lo, hi)
    return None


def main() -> int:
    m = pd.read_csv(ROOT / "labels_joined.csv")
    m["corpus"] = m.stratum.str.split("_").str[0]
    m["band"] = m.weight.apply(band_of)
    d = duckdb.connect()

    print("=" * 74)
    print("P6 EVALUATION -- every figure below regenerates from labels + predictions")
    print("=" * 74)
    tot, uns = len(m), int((m.label == "UNSURE").sum())
    print(f"\nlabelled pairs: {tot}   UNSURE: {uns} ({100*uns/tot:.1f}%, excluded from all rates)")

    # ---------- intra-rater agreement (30 hidden repeats) ----------
    blank = pd.concat([pd.read_csv(ROOT / "docs" / "labels_blank.csv", dtype=str).fillna(""),
                       pd.read_csv(ROOT / "docs" / "labels_blank_batch2.csv", dtype=str).fillna("")],
                      ignore_index=True)
    fields = ["a_name", "a_address", "a_city", "a_state", "a_zip",
              "b_name", "b_address", "b_city", "b_state", "b_zip"]
    blank["sig"] = blank[fields].agg("|".join, axis=1)
    lab = dict(zip(m.pair_id, m.label))
    agree = dis = 0
    for _, g in blank.groupby("sig"):
        if len(g) != 2:
            continue
        a, b = [lab.get(x) for x in g.pair_id]
        if a and b:
            agree += (a == b); dis += (a != b)
    print(f"\nINTRA-RATER (30 hidden repeats): agreed {agree}/{agree+dis} "
          f"({100*agree/max(agree+dis,1):.1f}%)")

    for corpus in ("debtor", "lender"):
        c = m[(m.corpus == corpus) & (m.label != "UNSURE") & m.band.notna()]
        bands = c[c.stratum.str.contains("band")]
        kind = "debtors" if corpus == "debtor" else "lenders"
        pq = f"{ROOT}/parquet/predictions_{kind}.parquet"
        print(f"\n{'='*74}\n{corpus.upper()}S   (shipped threshold {SHIPPED[corpus]})\n{'='*74}")
        print(f"{'band':>12} {'labelled':>9} {'SAME':>6} {'precision':>10} {'95% CI':>16} {'N_h':>12}")
        rows = []
        for lo, hi in zip(EDGES, EDGES[1:]):
            g = bands[bands.band == (lo, hi)]
            n, k = len(g), int((g.label == "SAME").sum())
            N = d.execute(f"SELECT count(*) FROM '{pq}' WHERE match_weight>={lo} AND match_weight<{hi}").fetchone()[0]
            if n == 0:
                continue
            lo_ci, hi_ci = wilson(k, n)
            print(f"{f'[{lo},{hi})':>12} {n:>9} {k:>6} {k/n:>10.3f} "
                  f"{f'{lo_ci:.3f}-{hi_ci:.3f}':>16} {N:>12,}")
            rows.append(dict(lo=lo, hi=hi, n=n, k=k, N=N))

        print(f"\n  {'threshold':>9} {'weighted precision':>19} {'95% CI':>16} {'n':>6} "
              f"{'est. recall':>12}")
        # estimated true matches in the whole scored set, from band estimates
        tot_true = sum(r["N"] * r["k"] / r["n"] for r in rows)
        for T in THRESHOLDS:
            sel = [r for r in rows if r["lo"] >= T]
            if not sel:
                continue
            Nsum = sum(r["N"] for r in sel)
            p = sum(r["N"] * r["k"] / r["n"] for r in sel) / Nsum
            # stratified variance of the weighted mean
            var = sum((r["N"] / Nsum) ** 2 * (r["k"] / r["n"]) * (1 - r["k"] / r["n"]) / r["n"]
                      for r in sel)
            se = math.sqrt(var)
            n_eff = sum(r["n"] for r in sel)
            rec = sum(r["N"] * r["k"] / r["n"] for r in sel) / tot_true if tot_true else float("nan")
            star = "  <-- SHIPPED" if T == SHIPPED[corpus] else ""
            print(f"  {T:>9} {p:>19.3f} {f'{max(0,p-1.96*se):.3f}-{min(1,p+1.96*se):.3f}':>16} "
                  f"{n_eff:>6} {rec:>12.3f}{star}")

        print(f"\n  targeted strata (NEVER pooled into the precision above):")
        for s in sorted(m[(m.corpus == corpus)].stratum.unique()):
            if "band" in s:
                continue
            g = m[(m.stratum == s) & (m.label != "UNSURE")]
            if not len(g):
                continue
            same = int((g.label == "SAME").sum())
            N = int(g.N_h.iloc[0])
            print(f"    {s:34s} n={len(g):>3}  SAME {same:>3}  DIFFERENT {len(g)-same:>3}  "
                  f"({100*(len(g)-same)/len(g):>5.1f}% different)   N_h={N:,}")
    d.close()
    print("\n" + "=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
