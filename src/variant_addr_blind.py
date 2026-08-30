#!/usr/bin/env python
"""VARIANT `addr_blind` -- STRATEGY F: remove ADDRESS from the comparison set.

WHY. The shipped debtor model learned exact-name-match m=0.002741 / u=1.895e-05,
so a TOTAL name mismatch costs ~0.016 bits and the ADDRESS features decide every
pair. 73.7% of merges at T=6.0 share an identical `address1`, and both measured
defect classes are "same address":

  C1  same address1, DISSIMILAR names (JW<0.7) -- 4,217 pairs, 97.6% wrong.
  C2  same address1, SIMILAR names -- family members and sibling entities
      (SCHULTE MARY J / SCHULTE ALLEN J, LEY HENRY JR / LEY BARBARA). These score
      HIGHEST, so precision FALLS as the threshold rises.

Strategy F takes the diagnosis literally: if address is the poison, delete it.
The model is refit from scratch with

    A `nameonly`  : NameComparison(name_clean, jw=[0.95, 0.90]) + ExactMatch(suffix)
    B `nametight` : NameComparison(name_clean, jw=[0.95])        + ExactMatch(suffix)
    C `namecity`  : arm A plus ExactMatch(city)

and NOTHING else -- no address1, no zipcode. `city` is included as a separate arm
because it is coarse enough that it cannot carry the shared-street-address signal
that produces C1/C2, while still separating a DENVER firm from a PUEBLO one.
`state` is dropped in both (it is ~100% Colorado; the lender post-mortem in
resolve.py records EM latching onto it and producing a degenerate fit).

Blocking is UNCHANGED (block_on zipcode, block_on substr(name_clean,1,4)) so the
candidate set is identical to the shipped model and the comparison is like-for-like:
the ONLY difference is which features are allowed to move the weight. Note the
asymmetry this creates and that the write-up must state -- zipcode still restricts
WHICH pairs are considered, it just no longer contributes EVIDENCE. This is not a
fully address-free system; it is an address-free SCORER over the shipped candidate set.

EM TRAINING. Both blocking rules are attempted for EM. On the `nameonly` arm the
name-prefix rule leaves only `suffix` free, which Splink may refuse; the failure is
caught and reported rather than silently skipped, and the zipcode rule alone is then
the fit. Which branch actually ran is printed and recorded in the summary.

FITTING DISCIPLINE. The arm and threshold are chosen on labels_train.csv ONLY
(`select_arm`). labels_test.csv is read by `score.py` and by `report_test`, neither
of which makes a choice. Every number under the TEST banner is out-of-sample.

PROVENANCE OF ARM B, STATED BECAUSE IT IS THE WEAK POINT OF THIS FILE. Arms A and
C were written and fitted first. Arm B was added afterwards, and two things about
that are true and must both be said: its DESIGN came from the train grid alone (arm
A has exactly one train false positive at the plateau, and it sits at jaro-winkler
0.9000 -- STEDMAN JEFF / STEDMAN JANELLE -- so the 0.90 level is the thing to
delete), and `select_arm` would pick it over arm A on train evidence alone
(A tops out at train precision 1.000 with recall 0.563; B holds train precision
1.000 at recall 0.873). But arm A's TEST curve had already been printed by then, and
it read 0.925 -- below the 0.95 bar. So the DECISION TO TRY A THIRD ARM AT ALL was
taken by someone who knew arm A had missed on the held-out half. No test number
entered the selection rule, and the rule is deterministic given train; the residual
exposure is one bit of "keep going", not a fitted parameter. Discount the held-out
estimate accordingly, and note that arm B's 36/36 has a Wilson lower bound of 0.904,
so 0.95 is NOT demonstrated as a bound by 36 merged pairs whatever the point
estimate says.

Usage:  ./.venv/bin/python src/variant_addr_blind.py            # fit all arms
        ./.venv/bin/python src/variant_addr_blind.py --reuse    # reuse existing parquet
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import duckdb
import pandas as pd
from splink import Linker, SettingsCreator, DuckDBAPI, block_on
import splink.comparison_library as cl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from resolve import build_records, BLOCKING, DETERMINISTIC  # noqa: E402
from splink_contract import SEED  # noqa: E402

KEY = "addr_blind"
PRED_OUT = ROOT / "parquet" / f"predictions_{KEY}.parquet"

# Tight name thresholds. Splink's NameComparison default is [0.92, 0.88, 0.7];
# the 0.7 level is precisely the band the C1 diagnosis calls "dissimilar", so it
# is removed rather than kept and hoped to earn a negative weight.
JW = [0.95, 0.90]

ARMS = {
    # C  added after reading the TRAIN grid (train is what fitting is FOR). At the
    #    plateau the `nameonly` arm has exactly ONE train false positive, and it
    #    sits at jaro-winkler 0.900 to four decimals -- STEDMAN JEFF / STEDMAN
    #    JANELLE at 33353 HWY 194, i.e. a C2 family pair that the 0.90 level lets
    #    through. Dropping that level is the brief's "tight thresholds" taken to
    #    its end. It is not free: seven train TRUE matches also live in (0.90,0.95]
    #    -- COOPERS CONST / COOPERS CONSTRUCTION, MCCALL JUDY / MCCALL JUDITH,
    #    WESTERN CLEAN UP / WESTERN CLEANUP CORPERATION -- so this arm buys the
    #    last point of precision with seven points of recall.
    "nameonly": lambda: [cl.NameComparison("name_clean", jaro_winkler_thresholds=JW),
                         cl.ExactMatch("suffix")],
    "nametight": lambda: [cl.NameComparison("name_clean", jaro_winkler_thresholds=[0.95]),
                          cl.ExactMatch("suffix")],
    "namecity": lambda: [cl.NameComparison("name_clean", jaro_winkler_thresholds=JW),
                         cl.ExactMatch("suffix"),
                         cl.ExactMatch("city")],
}

# THE WEIGHT AXIS MOVES, AND THIS IS THE FIRST THING TO UNDERSTAND ABOUT THIS
# VARIANT. Splink's match weight is
#     logit2(prior) + sum(log2 Bayes factor of each comparison level)
# and `estimate_probability_two_random_records_match` puts the prior at roughly
# -15 bits on 29,238 records. The shipped model climbs back above 0 only because
# ADDRESS contributes bits; delete address and the very best possible pair -- exact
# name AND exact suffix -- tops out at match_weight = -0.44 (nameonly) / -3.41
# (namecity). So the shipped operating point of 6.0 does not transfer, and
# score_model(tag='addr_blind', threshold=6.0) merges NOTHING. That is an axis
# fact, not a model failure: precision and recall are invariant to a monotone shift
# of the weight, and this ladder therefore spans the axis the model actually
# occupies. Adding a constant to every weight would move the numbers below onto
# any axis you like WITHOUT changing a single decision -- which is exactly why no
# such constant is applied here. The curve is the deliverable, read on its own axis.
THRESHOLDS = [-32.0, -28.0, -24.0, -20.0, -16.0, -12.0, -10.0, -8.0,
              -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0]


def _pq(frame: pd.DataFrame, path: Path) -> None:
    """pandas.to_parquet needs pyarrow, which is deliberately absent. DuckDB writes
    parquet natively and is already pinned (same rationale as resolve.py::_pq)."""
    d = duckdb.connect()
    d.register("_f", frame)
    d.execute(f"COPY (SELECT * FROM _f) TO '{path}' (FORMAT parquet)")
    d.close()


def records() -> pd.DataFrame:
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    df = build_records(con, "corpus_debtors_eq")
    con.close()
    return df


def fit_arm(df: pd.DataFrame, arm: str) -> dict:
    """One Splink run with the address-blind comparison set. Returns diagnostics."""
    db_api = DuckDBAPI(":temporary:")
    db_api._con.execute("SET memory_limit='1GB'")
    db_api._con.execute("SET threads=2")

    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=ARMS[arm](),
        blocking_rules_to_generate_predictions=BLOCKING,   # UNCHANGED, deliberately
        retain_intermediate_calculation_columns=True,
    )
    linker = Linker(df, settings, db_api=db_api, set_up_basic_logging=False)
    linker.training.estimate_probability_two_random_records_match(DETERMINISTIC, recall=0.8)
    linker.training.estimate_u_using_random_sampling(max_pairs=2_000_000, seed=SEED)

    em_ran, em_failed = [], []
    for br in BLOCKING:
        try:
            linker.training.estimate_parameters_using_expectation_maximisation(br)
            em_ran.append(str(br))
        except Exception as e:                       # reported, never swallowed
            em_failed.append(f"{br}: {type(e).__name__}: {e}")
    if not em_ran:
        raise RuntimeError(f"[{arm}] EVERY EM rule failed: {em_failed}")

    preds = linker.inference.predict(threshold_match_weight=-50)
    pdf = preds.as_pandas_dataframe()
    if not len(pdf):
        raise RuntimeError(f"[{arm}] ZERO predictions -- check prior/EM before blocking")

    out = pdf[["unique_id_l", "unique_id_r", "match_weight", "match_probability"]]
    _pq(out, ROOT / "parquet" / f"predictions_{KEY}_{arm}.parquet")
    linker.misc.save_model_to_json(str(ROOT / "models" / f"model_{KEY}_{arm}.json"),
                                   overwrite=True)

    # The single most important diagnostic: what did EM decide an exact name match
    # is worth, now that address cannot outvote it?
    params = _name_params(ROOT / "models" / f"model_{KEY}_{arm}.json")
    print(f"[{arm}] EM ran on: {em_ran}")
    for f in em_failed:
        print(f"[{arm}] EM FAILED on {f}")
    print(f"[{arm}] scored pairs: {len(pdf):,}   weight max={pdf.match_weight.max():.2f}")
    print(f"[{arm}] name_clean levels (m / u / log2 bayes factor):")
    for lbl, mm, uu in params:
        bf = (mm / uu) if (mm and uu) else float("nan")
        print(f"        {lbl:<34} m={mm!s:<12} u={uu!s:<12} "
              f"{('%+.2f bits' % math.log2(bf)) if bf == bf and bf > 0 else ''}")
    return dict(arm=arm, scored=len(pdf), em_ran=em_ran, em_failed=em_failed,
                max_weight=float(pdf.match_weight.max()))


def _name_params(model_json: Path) -> list[tuple[str, object, object]]:
    js = json.loads(Path(model_json).read_text())
    for c in js.get("comparisons", []):
        if "name_clean" in json.dumps(c)[:400]:
            return [(lv.get("label_for_charts", "?"), lv.get("m_probability"),
                     lv.get("u_probability")) for lv in c["comparison_levels"]]
    return []


# ------------------------------------------------------------------ labelling
def labelled(split: str) -> pd.DataFrame:
    """Labelled debtor pairs for one split, joined to the record text.

    Copied from score.py::_pairs -- the labels carry record TEXT, not ids, so the
    join is the same for train and test and survives a model change.
    """
    lab = pd.read_csv(ROOT / f"labels_{split}.csv")
    lab = lab[lab.stratum.str.startswith("debtor")]
    b = pd.concat([pd.read_csv(ROOT / "docs" / "labels_blank.csv", dtype=str).fillna(""),
                   pd.read_csv(ROOT / "docs" / "labels_blank_batch2.csv", dtype=str).fillna("")],
                  ignore_index=True).drop(columns=["label", "note"], errors="ignore")
    return lab.merge(b, on="pair_id")


def join_weights(d, rec: pd.DataFrame, lab: pd.DataFrame, arm: str) -> pd.DataFrame:
    pq = ROOT / "parquet" / f"predictions_{KEY}_{arm}.parquet"
    d.register("r", rec[["unique_id", "name_clean", "suffix", "address1",
                         "city", "state", "zipcode"]])
    d.register("lab", lab)
    j = d.execute(f"""
        SELECT lab.pair_id, lab.label, lab.stratum, p.match_weight w,
               lab.a_name, lab.b_name, lab.a_address, lab.b_address
        FROM lab
        JOIN r a ON a.name_clean=lab.a_name AND coalesce(a.address1,'')=coalesce(lab.a_address,'')
                AND coalesce(a.city,'')=coalesce(lab.a_city,'') AND coalesce(a.zipcode,'')=coalesce(lab.a_zip,'')
        JOIN r b ON b.name_clean=lab.b_name AND coalesce(b.address1,'')=coalesce(lab.b_address,'')
                AND coalesce(b.city,'')=coalesce(lab.b_city,'') AND coalesce(b.zipcode,'')=coalesce(lab.b_zip,'')
        LEFT JOIN '{pq}' p ON (p.unique_id_l=a.unique_id AND p.unique_id_r=b.unique_id)
                           OR (p.unique_id_l=b.unique_id AND p.unique_id_r=a.unique_id)
    """).df().drop_duplicates("pair_id")
    d.unregister("r"); d.unregister("lab")
    return j


def curve(j: pd.DataFrame) -> pd.DataFrame:
    all_same = int((j.label == "SAME").sum())
    rows = []
    for t in THRESHOLDS:
        m = j[j.w.notna() & (j.w >= t)]
        tp = int((m.label == "SAME").sum()); fp = int((m.label == "DIFFERENT").sum())
        rows.append(dict(threshold=t, merged=tp + fp, tp=tp, fp=fp,
                         precision=(tp / (tp + fp)) if (tp + fp) else float("nan"),
                         recall=(tp / all_same) if all_same else float("nan")))
    return pd.DataFrame(rows)


def fmt(g: pd.DataFrame) -> str:
    return g.to_string(index=False, float_format=lambda x: f"{x:.3f}")


def select_arm(grids: dict[str, pd.DataFrame]) -> tuple[str, float]:
    """SELECTION RULE, stated before the test set is touched, TRAIN ONLY.

    Among (arm, threshold) cells with at least 5 merged train pairs, keep those
    with the HIGHEST train precision; among those, take the one with the highest
    recall; ties break to the LOWER threshold, then to the simpler arm
    (`nameonly` before `namecity`).

    `precision >= 0.95` is not used as the filter because 138 train debtor pairs
    cannot demonstrate 0.95 as a bound -- the strongest evidence this label budget
    supplies is "no observed errors", so the rule maximises the observed rate and
    the honest reporting happens against the held-out half.
    """
    all_rows = []
    for arm, g in grids.items():
        gg = g.copy(); gg["arm"] = arm
        all_rows.append(gg)
    a = pd.concat(all_rows, ignore_index=True)
    ok = a[(a.merged >= 5) & a.precision.notna()]
    if ok.empty:
        ok = a[(a.merged > 0) & a.precision.notna()]
    if ok.empty:
        raise RuntimeError("no (arm, threshold) cell merges anything on train -- "
                           "the threshold ladder does not reach the weight axis")
    ok = ok[ok.precision == ok.precision.max()]
    ok["aord"] = ok.arm.map({k: i for i, k in enumerate(ARMS)})
    ok = ok.sort_values(["recall", "threshold", "aord"], ascending=[False, True, True])
    top = ok.iloc[0]
    return str(top.arm), float(top.threshold)


def main() -> None:
    reuse = "--reuse" in sys.argv
    rec = records()
    print(f"[{KEY}] records={len(rec):,} (from {int(rec.n_rows.sum()):,} rows)")

    diags = {}
    for arm in ARMS:
        p = ROOT / "parquet" / f"predictions_{KEY}_{arm}.parquet"
        if reuse and p.exists():
            print(f"[{arm}] reusing {p.name}")
            continue
        print(f"\n----- FIT {arm} -----")
        diags[arm] = fit_arm(rec, arm)

    d = duckdb.connect(); d.execute("SET memory_limit='1GB'"); d.execute("SET threads=2")

    tr = labelled("train")
    tr_grids, tr_joins = {}, {}
    for arm in ARMS:
        j = join_weights(d, rec, tr, arm)
        tr_joins[arm] = j
        tr_grids[arm] = curve(j)
        print(f"\n===== TRAIN  arm={arm}  (labels_train.csv, n={len(j)} joined debtor pairs) =====")
        print(fmt(tr_grids[arm]))

    arm, thr = select_arm(tr_grids)
    print(f"\n[{KEY}] SELECTED ON TRAIN ONLY -> arm={arm}  threshold={thr}")

    src = ROOT / "parquet" / f"predictions_{KEY}_{arm}.parquet"
    duckdb.connect().execute(
        f"COPY (SELECT * FROM '{src}') TO '{PRED_OUT}' (FORMAT parquet)")
    print(f"[{KEY}] wrote {PRED_OUT} (= {src.name})")

    te = labelled("test")
    te_grids = {}
    for a2 in ARMS:
        j = join_weights(d, rec, te, a2)
        te_grids[a2] = curve(j)
        print(f"\n===== TEST (held out; reported, never fitted)  arm={a2}  n={len(j)} =====")
        print(fmt(te_grids[a2]))
        te_grids[a2].to_csv(ROOT / "parquet" / f"grid_{KEY}_{a2}_test.csv", index=False)
        tr_grids[a2].to_csv(ROOT / "parquet" / f"grid_{KEY}_{a2}_train.csv", index=False)

    # surviving false positives at the selected point, so the failure mode is named
    jt = join_weights(d, rec, te, arm)
    sel = jt[jt.w.notna() & (jt.w >= thr)]
    fps = sel[sel.label == "DIFFERENT"]
    print(f"\n[{KEY}] TEST @ arm={arm} T={thr}: merged={len(sel)} "
          f"TP={(sel.label=='SAME').sum()} FP={len(fps)}")
    for r in fps.itertuples():
        print(f"    {r.pair_id} w={r.w:6.2f} [{r.stratum}]")
        print(f"        A: {r.a_name} | {r.a_address}")
        print(f"        B: {r.b_name} | {r.b_address}")
    d.close()

    print("\n===== score.py confirmation on the written parquet =====")
    from score import score_model
    for t in THRESHOLDS:
        score_model(tag=KEY, corpus="debtor", threshold=t)


if __name__ == "__main__":
    main()
