#!/usr/bin/env python
"""P6 batch 2 -- 60 additional LENDER pairs. Purely ADDITIVE.

The P5 audit called the precision curve at 4/6/7/8/10 non-optional. Batch 1's
debtor bands honour those boundaries; its LENDER bands were hard-coded
[(2,6),(6,8),(8,10),(10,999)] -- no 4, no 7 -- so lender precision@7 rested on 11
sampled pairs and @4 on 6. That is a rumour, not a precision point.

Batch 1 is NOT regenerated. Re-drawing would void the key hash committed at
f43942d before any label existed, and that evidence chain is worth more than
tidiness -- the same precedent the decision rule set when it declined to re-draw
over a 0.67% independence issue. Instead the missing bands are drawn as a second
batch with its OWN pre-committed hash. Batch 1's 330 pairs and its hash remain
valid, untouched and still checkable; the chain is strengthened, not broken.

Records used in batch 1 are excluded by TEXT identity, not record id -- batch 1
de-duplicated on record id and two distinct source rows that print identically
slipped through (P122/P162). Fixed here.
"""
from __future__ import annotations
import hashlib, random, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from make_label_file import load, draw                  # noqa: E402
from splink_contract import SEED                        # noqa: E402


def main() -> int:
    rng = random.Random(SEED + 99)
    b1 = pd.read_csv(ROOT / "docs" / "labels_blank.csv", dtype=str).fillna("")
    seen = set()
    for side in ("a", "b"):
        for _, r in b1.iterrows():
            seen.add("|".join(r[f"{side}_{c}"] for c in
                              ("name", "address", "city", "state", "zip")))
    print(f"batch 1 record texts excluded: {len(seen):,}")

    d = load("lenders", "corpus_lenders_eq")
    rows, sizes, used = [], {}, set()
    for lo, hi in [(4, 6), (6, 7), (7, 8)]:
        got, N = draw(d, f"lender_band_{lo}_{hi}", f"w >= {lo} AND w < {hi}", 20, used, rng)
        got = [g for g in got
               if "|".join(str(g[f"a_{c}"] or "") for c in ("name","address","city","state","zip")) not in seen
               and "|".join(str(g[f"b_{c}"] or "") for c in ("name","address","city","state","zip")) not in seen]
        rows += got; sizes[f"lender_band_{lo}_{hi}"] = N
        print(f"  lender_band_{lo}_{hi}: drew {len(got)} of 20 (N_h={N:,})")
    d.close()

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=SEED + 100).reset_index(drop=True)
    df.insert(0, "pair_id", [f"B{i:03d}" for i in range(1, len(df) + 1)])
    blank = df[["pair_id", "a_name", "a_address", "a_city", "a_state", "a_zip",
                "b_name", "b_address", "b_city", "b_state", "b_zip"]].copy()
    blank["label"] = ""; blank["note"] = ""
    blank.to_csv(ROOT / "docs" / "labels_blank_batch2.csv", index=False)

    kp = ROOT / "labels_key_batch2.csv"
    df[["pair_id", "stratum", "N_h", "weight"]].to_csv(kp, index=False)
    h = hashlib.sha256(kp.read_bytes()).hexdigest()
    (ROOT / "docs" / "labels_key_batch2.sha256").write_text(
        f"{h}  labels_key_batch2.csv\n"
        f"# Committed BEFORE any label exists. Batch 1's hash (f43942d) is unaffected.\n")
    for c in ("weight", "stratum", "N_h"):
        assert c not in blank.columns
    print(f"\nbatch 2: {len(blank)} pairs   sha256 {h}\nBLIND-FILE CHECK: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
