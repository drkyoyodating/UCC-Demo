#!/usr/bin/env python
"""STRATEGY H -- CLUSTER-LEVEL COHERENCE.  KEY = "coherence".

WHAT THIS IS
============
The pairwise scores are left EXACTLY as the shipped debtor model produced them
(`parquet/predictions_debtors.parquet`, tag `debtors` from `src/resolve.py`).
Nothing is re-fit, no comparison is redesigned, no m/u value is touched. The only
thing that changes is HOW THE EDGE GRAPH IS TURNED INTO CLUSTERS.

Three clusterings are built over the same edge set:

  cc   -- connected components (what `resolve.py` ships), i.e. single linkage
          with no constraint. The reference point.
  bl   -- "single best links": every node keeps only its single highest-weight
          incident edge, then connected components over that thinned edge set.
          This is the transitive-chaining killer.
  pw   -- ablation: the coherence predicate applied PAIRWISE to the edges only,
          then plain connected components. Separates "cluster-level rule" from
          "name filter in disguise" -- see the note at the bottom of this header.
  coh  -- COHERENCE-CONSTRAINED single linkage: components are merged in
          descending edge-weight order, and a merge is REFUSED unless EVERY
          cross pair between the two components is name-coherent (complete
          linkage on the coherence predicate). A cluster is therefore valid only
          if all of its members agree on a name stem, and an incoherent cluster
          is never formed rather than being formed and split afterwards.

WHAT THE REPORTED NUMBER MEANS  (read this before quoting it)
=============================================================
`score.score_model` scores PAIRS. To evaluate a CLUSTERING you have to hand it
the induced same-cluster relation, so that is what the parquet contains: one row
for every pair of records that ends up in the same cluster -- INCLUDING pairs
that never had a direct edge and were only ever joined transitively.

The `match_weight` written for a pair is NOT a pairwise score. It is the
ULTRAMETRIC MERGE LEVEL: the weight of the edge at which the two records'
components were joined. Because all three algorithms are greedy descending-order
prefix scans, running them on the edge subset {w >= T} yields exactly the state
after processing the prefix {w >= T}, so

    "co-clustered at clustering threshold T"  <=>  "merge level >= T"

exactly, for every T above the floor. Thresholding the emitted column therefore
reproduces the clustering at that threshold, and the precision/recall curve
score.py prints is a curve over the CLUSTERING threshold. That equivalence is the
reason the emitted weight is a merge level and not a match probability;
`match_probability` is a monotone cosmetic transform of it and carries no extra
information. Do not read a merge level as a per-pair likelihood.

THE COHERENCE PREDICATE, AND WHERE IT CAME FROM
===============================================
Fit on `labels_train.csv` ONLY (138 debtor training pairs). `labels_test.csv` is
not opened anywhere in this file.

On the training half the normalised Levenshtein ratio over the full cleaned name
separates the two classes almost perfectly, and -- this is the point -- it does so
in the band where the shipped model is ANTI-calibrated. Training labels by weight
band: [4,6) 15/15 SAME, [6,7) 10 SAME / 5 DIFFERENT, [7,8) 15/15 SAME, [8,10)
1 SAME / 12 DIFFERENT, [10,999) 3 SAME / 10 DIFFERENT. Precision falls as the
weight rises, so no threshold move can fix it; the top of the distribution has to
be filtered on a signal the model is not using.

Measured on those 138 pairs:
    highest lev-ratio on a DIFFERENT pair : 0.667  (STEDMAN JEFF / STEDMAN JANELLE)
    lowest  lev-ratio on a SAME pair      : 0.368  (RTB DENVER AVE / RTB THOMPSON VALLEY)
    but the SAME distribution is bimodal -- the next SAME values up are 0.556,
    0.556 (WESTERN CLEAN UP / WESTERN CLEANUP CORPERATION), 0.650 (COOPERS CONST
    / COOPERS CONSTRUCTION), then 0.769 and above.
LEV_MIN is set at 0.72, the midpoint of the empty interval (0.667, 0.769). Any
value in that interval gives an identical training partition, which is the only
honest reason to prefer one -- it is a gap, not an optimum, and 0.70 / 0.75 are
the same rule.

The three SAME pairs below the floor are all HEAD-EXTENSION shapes (one name is
the other plus a trailing continuation), so a second clause admits them without
admitting anything else: strip spaces, and accept when one name is a prefix of
the other, at least MIN_PREFIX=8 characters long and at least PREFIX_FRAC=0.40 of
the longer. Verified on train: this admits COOPERS CONST/COOPERS CONSTRUCTION and
both WESTERN CLEAN UP rows, and ZERO of the 47 DIFFERENT pairs. RTB DENVER AVE /
RTB THOMPSON VALLEY is not recoverable by any name rule and is accepted as a
recall loss.

WHAT THIS CANNOT DO, STATED UP FRONT
====================================
The labelled pairs are overwhelmingly DIRECT edges. For a direct edge the
complete-linkage coherence constraint reduces to its pairwise restriction, so on
THIS label set the `coh` number is mostly measuring the coherence predicate and
only marginally measuring the graph surgery. The `cc` and `bl` ablations exist to
size that second effect honestly rather than let the coherence predicate take
credit for it. Report all three.

MEASURED, HELD-OUT (labels_test.csv, debtor stratum, n=74 joined pairs, 39 SAME)
==============================================================================
                       T=4.0                     T=6.0                T=7.0
  debtors (shipped) P=0.567 R=0.974 38/29   P=0.482 R=0.692 27/29   P=0.432 R=0.487
  coherence_cc      P=0.565 R=1.000 39/30   P=0.492 R=0.744 29/30   P=0.449 R=0.564
  coherence_bl      P=0.585 R=0.795 31/22   P=0.488 R=0.538 21/22   P=0.432 R=0.410
  coherence_pw      P=0.925 R=0.949 37/ 3   P=0.931 R=0.692 27/ 2   P=1.000 R=0.538
  coherence         P=0.974 R=0.949 37/ 1   P=0.964 R=0.692 27/ 1   P=1.000 R=0.538
                                      (TP/FP)
Best operating point: T=4.0, precision 0.974 (Wilson 0.865-0.995), recall 0.949.

Read the ablations before crediting the mechanism:
  * SINGLE BEST LINKS FAILED. `bl` costs 8 true pairs to remove 7 false ones and
    leaves precision at 0.585. The false merges here are same-address DIRECT
    edges, and a same-address false edge is very often a node's single
    highest-weight edge, so the one edge best-links keeps is frequently the wrong
    one. Chaining was not the disease.
  * PLAIN CONNECTED COMPONENTS (`cc`) is the shipped behaviour and confirms it:
    transitive closure buys +2 TP and +1 FP over the raw pairwise relation.
    Precision does not move. There is no clustering-only fix.
  * The COHERENCE PREDICATE does essentially all of the work (0.925 at `pw`),
    and the CLUSTER-LEVEL part of it -- complete linkage rather than an edge
    filter -- is worth the last stretch: 0.925 -> 0.974 at identical recall,
    by refusing two merges that only a third cluster member reveals as wrong
    (COLORADO WICH ORCHARDS / COLORADO WICH EVCO, and the COLORADO LAND AND HOME
    COMPANY pair). Small, real, and not something a pairwise rule can see.

Remaining held-out errors at T=4.0:
  FP (1): BOERNER FRANCES C / BOERNER JAMES C, one address. lev-ratio 0.765,
          over the 0.72 floor because the shared surname is long relative to the
          differing given names. Residual C2. Raising LEV_MIN to ~0.78 would
          remove it -- and would be fitting on the test half, so it is not done.
  FN (2): JIMMIE D PETRIE / PETRIE JIMMIE D (token-order reversal, lev-ratio
          0.200) and BIG R OF CENTER / BIG R OF LAMAR (lev-ratio 0.667, two
          stores of one chain). The reversal is trivially fixable with a
          token-sort clause, but the TRAINING half contains ZERO token-reordered
          SAME pairs (checked), so that clause has no train-side justification
          and adding it after seeing this FN would be test-set fitting.

Files written (none of them is an input to any other module):
    parquet/predictions_coherence.parquet       <- the deliverable (coh)
    parquet/predictions_coherence_cc.parquet    <- ablation
    parquet/predictions_coherence_bl.parquet    <- ablation
    parquet/predictions_coherence_pw.parquet    <- ablation
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------- parameters
BASE_TAG = "debtors"          # shipped pairwise predictions; NOT re-fit here
CORPUS = "corpus_debtors_eq"
FLOOR = 3.0                   # edges below this never enter any clustering
LEV_MIN = 0.72                # coherence: normalised Levenshtein ratio floor
MIN_PREFIX = 8                # coherence: head-extension clause, min chars
PREFIX_FRAC = 0.40            # coherence: head-extension clause, min length ratio
MAX_CLUSTER_EMIT = 400        # guard: refuse to emit C(n,2) for a runaway cluster


# ------------------------------------------------------------- string helpers
def _lev(a: str, b: str) -> int:
    """Iterative Levenshtein. Names are short; this is not the hot path."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lev_ratio(a: str, b: str) -> float:
    m = max(len(a), len(b))
    return 1.0 if m == 0 else 1.0 - _lev(a, b) / m


def head_extension(a: str, b: str) -> bool:
    """One cleaned name is the other plus a trailing continuation.

    Spaces are stripped first, because the shape shows up both as a token
    extension (COOPERS CONST -> COOPERS CONSTRUCTION) and as a re-tokenisation
    (WESTERN CLEAN UP -> WESTERN CLEANUP CORPERATION), and only the space-free
    form catches both.
    """
    x, y = a.replace(" ", ""), b.replace(" ", "")
    if len(x) > len(y):
        x, y = y, x
    if len(x) < MIN_PREFIX or not y.startswith(x):
        return False
    return len(x) / len(y) >= PREFIX_FRAC


def coherent(a: str, b: str) -> bool:
    """The cluster-coherence predicate. Symmetric, reflexive, NOT transitive --
    which is exactly why it has to be enforced across every cross pair."""
    if a == b:
        return True
    return lev_ratio(a, b) >= LEV_MIN or head_extension(a, b)


# ------------------------------------------------------------------ plumbing
def _pq(frame: pd.DataFrame, path: Path) -> None:
    """DuckDB writes the parquet: pandas.to_parquet needs pyarrow, which is
    deliberately absent from this venv (see `_pq` in src/resolve.py)."""
    d = duckdb.connect()
    d.execute("SET memory_limit='1GB'")
    d.register("_f", frame)
    d.execute(f"COPY (SELECT * FROM _f) TO '{path}' (FORMAT parquet)")
    d.close()


def load_graph():
    """Records + the shipped pairwise edges at or above FLOOR."""
    from resolve import build_records
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    rec = build_records(con, CORPUS)
    con.close()
    name = dict(zip(rec.unique_id, rec.name_clean.fillna("")))

    d = duckdb.connect()
    d.execute("SET memory_limit='1GB'")
    edges = d.execute(f"""
        SELECT unique_id_l, unique_id_r, match_weight
        FROM '{ROOT}/parquet/predictions_{BASE_TAG}.parquet'
        WHERE match_weight >= {FLOOR}
        -- deterministic tie-break: equal-weight edges are common (the model
        -- emits identical weights for identical comparison vectors), and the
        -- greedy scan's outcome depends on their order. Without this the
        -- blocked-edge count moved run to run.
        ORDER BY match_weight DESC, unique_id_l, unique_id_r
    """).df()
    d.close()
    return rec, name, edges


class DSU:
    def __init__(self):
        self.p, self.members = {}, {}

    def find(self, x):
        self.p.setdefault(x, x)
        self.members.setdefault(x, [x])
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.p[rb] = ra
        self.members[ra].extend(self.members[rb])
        del self.members[rb]
        return ra, rb


# --------------------------------------------------------------- clusterings
def agglomerate(edges: pd.DataFrame, name: dict, *, constrained: bool):
    """Descending-order greedy single linkage over `edges`.

    Returns the merge-level pair list. Because the scan is a prefix scan, the
    state after consuming every edge with weight >= T is IDENTICAL to running the
    whole thing on the subset {w >= T}. That is what makes the merge level an
    exact ultrametric for the induced same-cluster relation.
    """
    dsu = DSU()
    out_l, out_r, out_w = [], [], []
    blocked = 0
    for l, r, w in zip(edges.unique_id_l.values, edges.unique_id_r.values,
                       edges.match_weight.values):
        ra, rb = dsu.find(l), dsu.find(r)
        if ra == rb:
            continue
        ma, mb = dsu.members[ra], dsu.members[rb]
        if constrained:
            ok = True
            for x in ma:
                nx = name.get(x, "")
                for y in mb:
                    if not coherent(nx, name.get(y, "")):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                blocked += 1
                continue
        if len(ma) * len(mb) > MAX_CLUSTER_EMIT * MAX_CLUSTER_EMIT:
            raise RuntimeError("runaway cluster; raise the floor")
        for x in ma:
            for y in mb:
                out_l.append(x); out_r.append(y); out_w.append(float(w))
        dsu.union(l, r)
    sizes = pd.Series([len(v) for v in dsu.members.values()])
    return pd.DataFrame({"unique_id_l": out_l, "unique_id_r": out_r,
                         "match_weight": out_w}), sizes, blocked


def best_link_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Single best links: each node keeps only its highest-weight incident edge.

    Threshold-independent by construction, so the prefix-scan equivalence above
    survives. Chaining dies because a hub can no longer act as a bridge through
    a second, weaker edge.
    """
    best = {}
    for i, (l, r, w) in enumerate(zip(edges.unique_id_l.values,
                                      edges.unique_id_r.values,
                                      edges.match_weight.values)):
        for n in (l, r):
            if n not in best or w > best[n][0]:
                best[n] = (w, i)
    keep = {i for _, i in best.values()}
    return edges.iloc[sorted(keep)].reset_index(drop=True)


# -------------------------------------------------------------- train report
def train_pairs() -> pd.DataFrame:
    """The 138 debtor TRAINING pairs, joined to records exactly the way
    `_pairs()` in src/score.py joins the test half. labels_test.csv is never
    opened here."""
    from resolve import build_records
    tr = pd.read_csv(ROOT / "labels_train.csv")
    tr = tr[tr.stratum.str.startswith("debtor")]
    b = pd.concat([pd.read_csv(ROOT / "docs" / "labels_blank.csv", dtype=str).fillna(""),
                   pd.read_csv(ROOT / "docs" / "labels_blank_batch2.csv", dtype=str).fillna("")],
                  ignore_index=True).drop(columns=["label", "note"], errors="ignore")
    tr = tr.merge(b, on="pair_id")
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    rec = build_records(con, CORPUS)
    con.close()
    d = duckdb.connect()
    d.execute("SET memory_limit='1GB'")
    d.register("r", rec[["unique_id", "name_clean", "address1", "city", "zipcode"]])
    d.register("tr", tr)
    j = d.execute("""
        SELECT tr.pair_id, tr.label, a.unique_id il, b.unique_id ir,
               a.name_clean an, b.name_clean bn
        FROM tr
        JOIN r a ON a.name_clean=tr.a_name AND coalesce(a.address1,'')=coalesce(tr.a_address,'')
                AND coalesce(a.city,'')=coalesce(tr.a_city,'') AND coalesce(a.zipcode,'')=coalesce(tr.a_zip,'')
        JOIN r b ON b.name_clean=tr.b_name AND coalesce(b.address1,'')=coalesce(tr.b_address,'')
                AND coalesce(b.city,'')=coalesce(tr.b_city,'') AND coalesce(b.zipcode,'')=coalesce(tr.b_zip,'')
    """).df().drop_duplicates("pair_id")
    d.close()
    return j


def report_predicate_on_train() -> None:
    j = train_pairs()
    j["coh"] = [coherent(a, b) for a, b in zip(j.an, j.bn)]
    print(f"\n[train] coherence predicate on {len(j)} debtor TRAINING pairs "
          f"(LEV_MIN={LEV_MIN}, MIN_PREFIX={MIN_PREFIX}, PREFIX_FRAC={PREFIX_FRAC})")
    print(pd.crosstab(j.label, j.coh, rownames=["label"], colnames=["coherent"]))
    bad = j[(j.label == "DIFFERENT") & j.coh]
    if len(bad):
        print("  DIFFERENT pairs the predicate lets through:")
        for _, r in bad.iterrows():
            print(f"    {r.an!r} / {r.bn!r}")
    lost = j[(j.label == "SAME") & ~j.coh]
    print(f"  SAME pairs the predicate blocks ({len(lost)}):")
    for _, r in lost.iterrows():
        print(f"    {r.an!r} / {r.bn!r}")


# ---------------------------------------------------------------------- main
def build(write: bool = True) -> dict:
    rec, name, edges = load_graph()
    print(f"[coherence] records={len(rec):,}  edges>= {FLOOR}: {len(edges):,}")

    pw_edges = edges[pd.Series(
        [coherent(name.get(l, ""), name.get(r, "")) for l, r in
         zip(edges.unique_id_l, edges.unique_id_r)], index=edges.index)].reset_index(drop=True)

    out = {}
    for tag, ev, constrained in (
        ("coherence_cc", edges, False),
        ("coherence_bl", best_link_edges(edges), False),
        ("coherence_pw", pw_edges, False),
        ("coherence", edges, True),
    ):
        pairs, sizes, blocked = agglomerate(ev, name, constrained=constrained)
        n_multi = int((sizes > 1).sum())
        print(f"[{tag}] edges_used={len(ev):,} blocked={blocked:,} "
              f"emitted_pairs={len(pairs):,} clusters>1={n_multi:,} "
              f"largest={int(sizes.max()) if len(sizes) else 0}")
        if write:
            pairs = pairs.copy()
            # cosmetic only -- a monotone transform of the merge level so the
            # parquet has the column score.py's contract names. NOT a likelihood.
            pairs["match_probability"] = 1.0 / (1.0 + 2.0 ** (-pairs.match_weight))
            _pq(pairs, ROOT / "parquet" / f"predictions_{tag}.parquet")
        out[tag] = dict(pairs=len(pairs), clusters=n_multi, blocked=blocked)
    return out


if __name__ == "__main__":
    report_predicate_on_train()
    print()
    build()
