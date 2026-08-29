#!/usr/bin/env python
"""P6 -- build the blind labelling file. 330 pairs: 300 + 30 hidden repeats.

Design answers the three questions a skeptical data engineer asks of any
published precision figure: is the number real, can I reproduce it, and did you
find your own defects or did someone have to.

TWO KINDS OF STRATUM, deliberately kept separate:

  WEIGHT BANDS -- tile the match-weight space on FIXED ABSOLUTE boundaries
    (2,4,6,7,8,10). NEVER relative to the shipped threshold: the P5 audit found
    the debtor threshold was chosen partly to give P6 a denser borderline band,
    so a threshold-relative stratum would make measured precision a function of
    the very choice under audit. The boundaries are placed at 4/6/7/8/10 because
    those are the thresholds the precision curve must be published at, so each is
    a clean union of whole bands -- no partial-band arithmetic.
    These, and only these, feed the weighted precision estimate.

  TARGETED STRATA -- the known defect classes. They are NOT a random sample of
    anything, so they are NEVER pooled into the precision estimate; each is
    reported as its own error rate. They exist so the published numbers price the
    defects the audit found instead of averaging over them.

Population size N_h is recorded for EVERY stratum before a single pair is drawn,
so the pooled estimate can be stratum-weighted (an unweighted pool of a
stratified sample reports the sample's design weights, not the population's, and
was measured in the pre-flight review to overstate precision by ~0.23).

No record appears in more than one pair, so pairs are independent.

The founder sees: names and addresses. Never a weight, a prediction, a cluster
id, or a stratum. Order is shuffled under a fixed seed.

ANTI-TAMPER: the answer key is NOT published now. Its SHA-256 is committed
before labelling begins, and the key itself afterwards. That proves the strata,
weights and repeat structure were fixed in advance without exposing them to the
labeller -- a stronger guarantee than "I promise I didn't look".
"""
from __future__ import annotations
import hashlib, random, sys
from pathlib import Path
import duckdb, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from resolve import build_records                      # noqa: E402
from splink_contract import SEED                       # noqa: E402

BANDS = [(2, 4), (4, 6), (6, 7), (7, 8), (8, 10), (10, 999)]


def load(kind: str, corpus: str):
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    df = build_records(con, corpus); con.close()
    d = duckdb.connect()
    d.register("r", df[["unique_id", "name_clean", "address1", "city", "state", "zipcode"]])
    d.execute(f"CREATE TABLE p AS SELECT * FROM '{ROOT}/parquet/predictions_{kind}.parquet'")
    d.execute("""CREATE TABLE pr AS
        SELECT p.unique_id_l l, p.unique_id_r r, p.match_weight w,
               a.name_clean an, a.address1 aa, a.city ac, a.state ast, a.zipcode az,
               b.name_clean bn, b.address1 ba, b.city bc, b.state bst, b.zipcode bz
        FROM p JOIN r a ON a.unique_id=p.unique_id_l JOIN r b ON b.unique_id=p.unique_id_r""")
    return d


def draw(d, name, where, n, used, rng):
    """Draw n pairs matching `where`, skipping any record already used."""
    N = d.execute(f"SELECT count(*) FROM pr WHERE {where}").fetchone()[0]
    rows = d.execute(f"SELECT * FROM pr WHERE {where}").df()
    rows = rows.sample(frac=1.0, random_state=rng.randint(0, 2**31)).reset_index(drop=True)
    out = []
    for _, x in rows.iterrows():
        if len(out) >= n:
            break
        if x.l in used or x.r in used:
            continue
        used.add(x.l); used.add(x.r)
        out.append(dict(stratum=name, N_h=int(N), weight=float(x.w),
                        a_name=x.an, a_address=x.aa, a_city=x.ac, a_state=x.ast, a_zip=x.az,
                        b_name=x.bn, b_address=x.ba, b_city=x.bc, b_state=x.bst, b_zip=x.bz))
    return out, N


def main() -> int:
    rng = random.Random(SEED)
    rows, sizes = [], {}
    dd = load("debtors", "corpus_debtors_eq")
    used = set()
    for lo, hi in BANDS:                                        # 120 debtor band pairs
        got, N = draw(dd, f"debtor_band_{lo}_{hi}", f"w >= {lo} AND w < {hi}", 20, used, rng)
        rows += got; sizes[f"debtor_band_{lo}_{hi}"] = N
    got, N = draw(dd, "debtor_same_addr_diff_name",              # the 4,217 defect class
                  "w >= 6.0 AND ac = bc AND az = bz AND aa = ba "
                  "AND jaro_winkler_similarity(an, bn) < 0.7", 40, used, rng)
    rows += got; sizes["debtor_same_addr_diff_name"] = N
    got, N = draw(dd, "debtor_recall_probe",                     # under-merge side
                  "an = bn AND az = bz AND w >= 4 AND w < 8", 40, used, rng)
    rows += got; sizes["debtor_recall_probe"] = N
    dd.close()

    dl = load("lenders", "corpus_lenders_eq")
    used = set()
    for lo, hi in [(2, 6), (6, 8), (8, 10), (10, 999)]:          # 80 lender band pairs
        got, N = draw(dl, f"lender_band_{lo}_{hi}", f"w >= {lo} AND w < {hi}", 20, used, rng)
        rows += got; sizes[f"lender_band_{lo}_{hi}"] = N
    got, N = draw(dl, "lender_recall_probe",                     # under-merge side
                  "an = bn AND w >= 2 AND w < 8", 20, used, rng)
    rows += got; sizes["lender_recall_probe"] = N
    dl.close()

    df = pd.DataFrame(rows)
    reps = df.sample(n=30, random_state=SEED).copy()             # 30 hidden repeats
    reps["is_repeat"] = True
    df["is_repeat"] = False
    full = pd.concat([df, reps], ignore_index=True)
    full = full.sample(frac=1.0, random_state=SEED + 1).reset_index(drop=True)
    full.insert(0, "pair_id", [f"P{i:03d}" for i in range(1, len(full) + 1)])

    blank = full[["pair_id", "a_name", "a_address", "a_city", "a_state", "a_zip",
                  "b_name", "b_address", "b_city", "b_state", "b_zip"]].copy()
    blank["label"] = ""
    blank["note"] = ""
    blank.to_csv(ROOT / "docs" / "labels_blank.csv", index=False)

    key = full[["pair_id", "stratum", "N_h", "weight", "is_repeat"]]
    kp = ROOT / "labels_key.csv"
    key.to_csv(kp, index=False)
    h = hashlib.sha256(kp.read_bytes()).hexdigest()
    (ROOT / "docs" / "labels_key.sha256").write_text(
        f"{h}  labels_key.csv\n"
        f"# Committed BEFORE labelling. The key itself is published after the labels are in.\n"
        f"# This proves the strata, weights and repeat structure were fixed in advance.\n")

    print(f"pairs: {len(full)} ({len(df)} unique + 30 hidden repeats)")
    print(f"key sha256: {h}\n")
    print(f"{'stratum':34s} {'drawn':>6} {'N_h (population)':>18}")
    for s in sorted(sizes):
        print(f"  {s:32s} {int((df.stratum==s).sum()):>6} {sizes[s]:>18,}")
    assert not blank.drop(columns=['label','note']).isin([None]).all().any()
    for c in ("weight", "stratum", "N_h", "is_repeat"):
        assert c not in blank.columns, f"{c} leaked into the blind file"
    print("\nBLIND-FILE CHECK: no weight, stratum, N_h or repeat flag present. PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
