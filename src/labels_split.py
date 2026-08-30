#!/usr/bin/env python
"""Stratified 65/35 train/test split of the 381 hand labels.

The TEST half is never touched by any fitting procedure. Every improved-model
number is reported on it and on it alone, so the comparison against the shipped
baseline is out-of-sample rather than a re-scoring of the data that trained it.
Split is seeded and reproducible.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from splink_contract import SEED  # noqa: E402

m = pd.read_csv(ROOT / "labels_joined.csv")
m = m[m.label != "UNSURE"].copy()
# DROP THE 30 HIDDEN REPEATS BEFORE SPLITTING. They are byte-identical
# re-presentations of other pairs; leaving them in put 20 in train and 10 in test,
# so 14 distinct pair-contents appeared on BOTH sides and the "held-out" set was
# not held out. Every number scored on the old split is contaminated.
if "is_repeat" in m.columns:
    m = m[~m.is_repeat.fillna(False).astype(bool)].copy()
tr, te = [], []
for s, g in m.groupby("stratum"):                      # stratified: preserves the design
    g = g.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    k = int(round(len(g) * 0.65))
    tr.append(g.iloc[:k]); te.append(g.iloc[k:])
train = pd.concat(tr, ignore_index=True); test = pd.concat(te, ignore_index=True)
train.to_csv(ROOT / "labels_train.csv", index=False)
test.to_csv(ROOT / "labels_test.csv", index=False)
print(f"train {len(train)}  test {len(test)}")
print("train:", train.label.value_counts().to_dict())
print("test :", test.label.value_counts().to_dict())
print("\nper-stratum test counts:")
for s, n in test.stratum.value_counts().sort_index().items():
    print(f"  {s:34s} {n}")
