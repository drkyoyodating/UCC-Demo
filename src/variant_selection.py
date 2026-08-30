#!/usr/bin/env python
"""VARIANT `selection` -- WAVE-2 SELECTION LEAD: consensus over independent variants.

WHY A CONSENSUS AND NOT ANOTHER MODEL
=====================================
By the time wave 2 finished, the debtor label set was SATURATED: on the held-out
half, `person`, `name_floor`, `coherence`, `combo_pf`, `comparison`, `ensemble`,
`two_stage` and `addr_blind` ALL score zero false positives. Six of them are
within one labelled pair of each other on recall. The labels can no longer tell
these models apart -- n=70 (46 of them uncontaminated) simply has no resolving
power left at precision 1.000.

But the models DISAGREE at corpus scale, by thousands of merges. That disagreement
is where the residual false-positive risk actually lives, and it is invisible to
the label set. A consensus is the one construction that turns disagreement into a
measurable, tunable quantity: each variant votes, and `match_weight` IS THE VOTE
COUNT (1..8), so sweeping the threshold sweeps k in "k-of-8 must agree".

Each voter attacks the shipped model's defect from a DIFFERENT direction, which is
what makes the votes worth counting rather than merely correlated:
    person      post-hoc person/org gate on the shipped score   (Strategy D)
    name_floor  deterministic name floor on the shipped score   (Strategy B)
    comparison  redesigned Splink comparison set                (Strategy C)
    ensemble    logistic regression on explicit features        (evidence stacking)
    coherence   cluster-level complete-linkage coherence        (Strategy H)
    blocking    name-only blocking keys                         (Strategy G)
    two_stage   deterministic name rules + locality veto        (Strategy E)
    addr_blind  Splink refit with address deleted               (Strategy F)

NOTHING IS FITTED HERE. Every voter enters at the operating point its own author
selected on labels_train.csv. The only free parameter is k, and k is chosen on
labels_train.csv by a stated rule (see `pick_k`). labels_test.csv is never opened
by this module -- src/score.py is the only reader.

VOTE = the pair is present in that variant's parquet AND its weight >= that
variant's operating threshold. Absent from the parquet means "not proposed" and
therefore NOT a vote (this is how `blocking`, 52,613 rows, and `coherence`,
6,108 rows, participate honestly alongside the 2.49M-row full-candidate models).

Pair keys are normalised with least()/greatest() before the union: the eight
parquets do not agree on which record of a pair is `_l`.

Run:  ./.venv/bin/python src/variant_selection.py          # build + train report
      ./.venv/bin/python src/variant_selection.py --curve  # held-out curve via score.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import duckdb, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
P = ROOT / "parquet"
TAG = "selection"

#: voter -> operating threshold, each set by that variant's own author on TRAIN.
VOTERS = {
    "person":     4.0,
    "name_floor": 4.0,
    "comparison": -9.0,
    "ensemble":    0.0,
    "coherence":   4.0,
    "blocking":    4.0,
    "two_stage":  14.0,
    "addr_blind": -24.0,
}
#: `person`'s given-name gazetteer contains tokens that occur in exactly one
#: labelled pair each, three of them CURRENT-TEST pairs (JERI/P282, MERIDIAN/P248,
#: GLENDA/P283). It is reported both ways for that reason.
CONTAMINATED = ["person"]


def _con():
    d = duckdb.connect()
    d.execute("SET memory_limit='1GB'")          # up to nine Splink runs share the box
    return d


def build(voters: dict[str, float], tag: str) -> pd.DataFrame:
    d = _con()
    parts = [
        f"""SELECT least(unique_id_l,unique_id_r) l, greatest(unique_id_l,unique_id_r) r,
                   '{m}' AS voter
            FROM '{P}/predictions_{m}.parquet' WHERE match_weight >= {t}"""
        for m, t in voters.items()
    ]
    d.execute(f"CREATE OR REPLACE VIEW v AS {' UNION ALL '.join(parts)}")
    d.execute(f"""CREATE OR REPLACE VIEW agg AS
        SELECT l, r, count(DISTINCT voter) AS votes,
               string_agg(DISTINCT voter, '+' ORDER BY voter) AS who
        FROM v GROUP BY l, r""")
    n = len(voters)
    d.execute(f"""COPY (SELECT l AS unique_id_l, r AS unique_id_r,
                          CAST(votes AS DOUBLE) AS match_weight,
                          CAST(votes AS DOUBLE)/{n} AS match_probability
                   FROM agg) TO '{P}/predictions_{tag}.parquet' (FORMAT parquet)""")
    dist = d.execute("SELECT votes, count(*) n FROM agg GROUP BY 1 ORDER BY 1").df()
    d.close()
    return dist


# --------------------------------------------------------------------------
# TRAIN-ONLY scoring (labels_test.csv is NOT read here)
# --------------------------------------------------------------------------
def _train_pairs():
    tr = pd.read_csv(ROOT / "labels_train.csv")
    tr = tr[tr.stratum.str.startswith("debtor")]
    b = pd.concat([pd.read_csv(ROOT / "docs" / "labels_blank.csv", dtype=str).fillna(""),
                   pd.read_csv(ROOT / "docs" / "labels_blank_batch2.csv", dtype=str).fillna("")],
                  ignore_index=True).drop(columns=["label", "note"], errors="ignore")
    return tr.merge(b, on="pair_id")


def train_curve(tag: str, kmax: int) -> pd.DataFrame:
    """Same join as src/score.py, but against labels_TRAIN and with the
    pair->record ambiguity resolved deterministically (min over combinations,
    the conservative direction) instead of score.py's arbitrary keep-first."""
    from resolve import build_records
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    rec = build_records(con, "corpus_debtors_eq"); con.close()
    tr = _train_pairs()
    d = _con()
    d.register("r", rec[["unique_id", "name_clean", "address1", "city", "zipcode"]])
    d.register("tr", tr)
    j = d.execute(f"""
        SELECT tr.pair_id, any_value(tr.label) AS "label",
               min(coalesce(p.match_weight,-1e9)) w
        FROM tr
        JOIN r a ON a.name_clean=tr.a_name AND coalesce(a.address1,'')=coalesce(tr.a_address,'')
                AND coalesce(a.city,'')=coalesce(tr.a_city,'') AND coalesce(a.zipcode,'')=coalesce(tr.a_zip,'')
        JOIN r b ON b.name_clean=tr.b_name AND coalesce(b.address1,'')=coalesce(tr.b_address,'')
                AND coalesce(b.city,'')=coalesce(tr.b_city,'') AND coalesce(b.zipcode,'')=coalesce(tr.b_zip,'')
        LEFT JOIN '{P}/predictions_{tag}.parquet' p
               ON (p.unique_id_l=a.unique_id AND p.unique_id_r=b.unique_id)
               OR (p.unique_id_l=b.unique_id AND p.unique_id_r=a.unique_id)
        GROUP BY tr.pair_id""").df()
    d.close()
    ns = int((j.label == "SAME").sum())
    rows = []
    for k in range(1, kmax + 1):
        m = j[j.w >= k]
        tp = int((m.label == "SAME").sum()); fp = int((m.label == "DIFFERENT").sum())
        rows.append(dict(k=k, tp=tp, fp=fp, merged=tp + fp,
                         precision=tp / (tp + fp) if tp + fp else float("nan"),
                         recall=tp / ns if ns else float("nan")))
    return pd.DataFrame(rows)


def pick_k(curve: pd.DataFrame) -> int:
    """STATED RULE, fixed before the held-out set was scored: the SMALLEST k whose
    train precision is 1.000 over at least 20 merged train pairs. Smallest,
    because precision is the gated number and recall is bought back by taking the
    loosest k that still shows no train error -- raising k further cannot raise a
    precision already at 1.000 and can only cost recall."""
    ok = curve[(curve.precision >= 1.0) & (curve.merged >= 20)]
    return int(ok.k.min()) if len(ok) else int(curve.k.max())


def clusters(tag: str, k: int):
    """Transitive closure at k votes; the pre-registered bar is largest <=1%."""
    from resolve import build_records
    from collections import Counter
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    nrec = len(build_records(con, "corpus_debtors_eq")); con.close()
    d = _con()
    e = d.execute(f"""SELECT unique_id_l l, unique_id_r r
                      FROM '{P}/predictions_{tag}.parquet' WHERE match_weight>={k}""").df()
    d.close()
    par: dict[str, str] = {}
    def find(x):
        while par.get(x, x) != x:
            par[x] = par.get(par[x], par[x]); x = par[x]
        return x
    for l, r in zip(e.l, e.r):
        a, b = find(l), find(r)
        if a != b: par[a] = b
    sizes = Counter()
    for x in par: sizes[find(x)] += 1
    big = max(sizes.values()) if sizes else 0
    return dict(merges=len(e), clustered=sum(sizes.values()), clusters=len(sizes),
                largest=big, largest_pct=100 * big / nrec, records=nrec)


def main(argv):
    for tag, voters in (("selection", VOTERS),
                        ("selection_nc", {k: v for k, v in VOTERS.items() if k not in CONTAMINATED})):
        n = len(voters)
        print(f"\n=== {tag}: {n} voters -> {', '.join(voters)}")
        dist = build(voters, tag)
        print("  corpus vote distribution:",
              ", ".join(f"{int(r.votes)}:{int(r.n):,}" for _, r in dist.iterrows()))
        c = train_curve(tag, n)
        print("  TRAIN (labels_train.csv only):")
        for _, r in c.iterrows():
            print(f"    k={int(r.k)}  merged={int(r.merged):>3} TP={int(r.tp):>3} FP={int(r.fp):>3} "
                  f"prec={r.precision:.3f} rec={r.recall:.3f}")
        k = pick_k(c)
        print(f"  PICKED k={k} by the stated rule")
        cl = clusters(tag, k)
        print(f"  corpus @k={k}: merges={cl['merges']:,} clusters={cl['clusters']:,} "
              f"largest={cl['largest']} ({cl['largest_pct']:.3f}% -> "
              f"{'PASS' if cl['largest_pct']<=1.0 else 'FAIL'} the <=1% bar)")


if __name__ == "__main__":
    main(sys.argv[1:])
