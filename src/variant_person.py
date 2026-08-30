#!/usr/bin/env python
"""VARIANT `person` -- Strategy D: person-name awareness for debtor resolution.

WHY
===
The shipped debtor model decides on ADDRESS, not name (exact-name m/u Bayes factor
0.989 -> a total name mismatch costs 0.016 bits). Every high-weight false positive
is therefore "two records at one address". Two defect classes:

  class 1  same address, DISSIMILAR names  -- registered agents, shared suites,
           franchise HQs, landlord/tenant.  97.6% DIFFERENT.
  class 2  same address, SIMILAR names     -- FAMILY MEMBERS and related-but-
           distinct firms.  SCHULTE MARY J / SCHULTE ALLEN J.

Class 2 is the one address can never fix, because a near-identical name at an
identical address is exactly what a TRUE match looks like. The only thing that
separates GUENZI KENNETH J from GUENZI EVA M is that the SURNAME agrees and the
GIVEN NAME does not -- a fact you can only use if you know the string is a person.

WHAT THIS MODULE DOES
=====================
1. Harvests a GIVEN-NAME gazetteer from the corpus itself (no external list), by
   exploiting the middle-initial pattern: in `<A> <B> <single letter>` and
   `<A> <single letter> <B>` the token adjacent to the initial is a given name.
2. Classifies each record as PERSON / ORG with that gazetteer plus a business-word
   veto and a token-count bound.
3. Scores the corpus with the SAME Splink model as src/resolve.py (identical
   blocking, comparisons, seed) so match_weight stays comparable to the baseline,
   then applies a post-hoc NAME-AGREEMENT GATE:
       PERSON x PERSON  -> every non-initial token must agree (JW >= 0.80).
                           This is the given-name requirement: the surname already
                           agrees, so an unmatched token IS the given name.
       PERSON x ORG     -> veto. A natural person and a company are not one entity.
       ORG x ORG        -> class-1 guard: despaced Jaro-Winkler >= 0.90, or one
                           despaced name is a prefix of the other (>= 8 chars).
   Vetoed pairs get match_weight = -99.0 (kept in the parquet rather than dropped,
   so the curve is honest at every threshold).

The gate is fitted on labels_train.csv ONLY. labels_test.csv is never read here.

Run:  ./.venv/bin/python src/variant_person.py            # full: fit + predict + score
      ./.venv/bin/python src/variant_person.py calibrate  # detector/gate report on TRAIN only
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolve import BLOCKING, DETERMINISTIC, build_records, comparisons_for  # noqa: E402
from splink_contract import SEED  # noqa: E402

TAG = "person"
MEMORY_LIMIT = "2GB"          # four agents share a 16GB box

# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

#: Tokens that make a string an ORGANISATION no matter what else is in it.
#: Calibrated by reading the 120 most frequent tokens of the debtor corpus and
#: the 8,538 two-token / 9,901 three-token names, NOT guessed from English.
BUSINESS_TOKENS = set("""
INC INCORPORATED CORP CORPORATION CO COMPANY LLC LLLP LLP LP PLLC PC PA LTD LIMITED
LIABILITY PARTNERSHIP PARTNERS PARTNERSH GP JV TRUST TRUSTEE ESTATE FOUNDATION
FUND FCU CU BANK BANCORP CREDIT UNION HOLDINGS HOLDING GROUP ENTERPRISE ENTERPRISES
VENTURES VENTURE ASSOCIATES ASSOCIATION ASSOC ASSN SOCIETY CLUB CHURCH SCHOOL DISTRICT
COUNTY CITY TOWN STATE MUNICIPAL AUTHORITY DEPARTMENT DEPT BUREAU AGENCY OFFICE
CONSTRUCTION CONSTRUCTORS CONTRACTING CONTRACTORS CONTRACTOR BUILDERS BUILDING BUILT
EXCAVATING EXCAVATION EXCAVATORS PAVING ASPHALT CONCRETE ROOFING PLUMBING HEATING
COOLING ELECTRIC ELECTRICAL MECHANICAL WELDING FABRICATION MACHINE MACHINERY
EQUIPMENT SUPPLY SUPPLIES PRODUCTS PRODUCTIONS MANUFACTURING MFG INDUSTRIES INDUSTRIAL
SERVICE SERVICES SOLUTIONS SYSTEMS TECHNOLOGIES TECHNOLOGY TECH CONSULTING CONSULTANTS
MANAGEMENT MARKETING ADVERTISING DESIGN DESIGNS DEVELOPMENT PROPERTIES PROPERTY REALTY
REAL INVESTMENT INVESTMENTS CAPITAL LEASING RENTAL RENTALS SALES RETAIL WHOLESALE
STORE STORES SHOP SHOPPE MART MARKET MARKETS CENTER CENTRE CENTERS PLAZA PARK
RESTAURANT CAFE GRILL BAR TAVERN PIZZA BAKERY DELI LIQUOR LIQUORS BREWING
FARMS FARM RANCH RANCHES CATTLE DAIRY FEEDLOT FEED GRAIN SEED SEEDING AGRI AG
LAND LANDSCAPING NURSERY GREENHOUSE ORCHARD VINEYARD
TRUCKING TRANSPORT TRANSPORTATION HAULING FREIGHT LOGISTICS DELIVERY COURIER
AUTO AUTOMOTIVE MOTORS MOTOR TIRE TIRES BODY COLLISION GARAGE TOWING
OIL GAS PETROLEUM ENERGY DRILLING MINING MINERALS RESOURCES WATER SEWER UTILITIES
MEDICAL DENTAL DDS DMD MD CLINIC HOSPITAL PHARMACY HEALTH HEALTHCARE CARE THERAPY
SALON SPA FITNESS GYM SPORTS RECREATION GOLF
CLEANING CLEANERS CLEAN LAUNDRY CARWASH WASH JANITORIAL MAINTENANCE REPAIR
PRINTING PRINTERS PRESS PUBLISHING MEDIA BROADCASTING COMMUNICATIONS TELECOM
INSURANCE AGENCY FINANCIAL FINANCE MORTGAGE TITLE ESCROW ACCOUNTING TAX LAW LEGAL
HOMES HOME HOUSING APARTMENTS APARTMENT CONDOMINIUM MOTEL HOTEL LODGE INN RESORT
BROTHERS BROS SONS SON DAUGHTERS FAMILY AND OF THE DBA FBO ETAL ETUX AKA
STORAGE WAREHOUSE DISTRIBUTION DISTRIBUTORS DISTRIBUTORSHIP IMPORTS EXPORTS
INTERNATIONAL NATIONAL AMERICAN AMERICA USA WESTERN EASTERN NORTHERN SOUTHERN
COLORADO DENVER ROCKY MOUNTAIN VALLEY CREEK SPRINGS RIVER MESA CANYON SUMMIT
""".split())

#: Generational / honorific tails that are part of a PERSON name, not an org marker.
GENERATIONAL = {"JR", "SR", "II", "III", "IV", "V", "VI"}

_SEED_GIVEN = set("""
JAMES JOHN ROBERT MICHAEL WILLIAM DAVID RICHARD CHARLES JOSEPH THOMAS
CHRISTOPHER DANIEL PAUL MARK DONALD GEORGE KENNETH STEVEN EDWARD BRIAN
RONALD ANTHONY KEVIN JASON MATTHEW GARY TIMOTHY JOSE LARRY JEFFREY FRANK
SCOTT ERIC STEPHEN ANDREW RAYMOND GREGORY JOSHUA JERRY DENNIS WALTER PATRICK
PETER HAROLD DOUGLAS HENRY CARL ARTHUR RYAN ROGER JOE JUAN JACK ALBERT
JONATHAN JUSTIN TERRY GERALD KEITH SAMUEL WILLIE RALPH LAWRENCE NICHOLAS ROY
BENJAMIN BRUCE BRANDON ADAM HARRY FRED WAYNE BILLY STEVE LOUIS JEREMY AARON
RANDY HOWARD EUGENE CARLOS RUSSELL BOBBY VICTOR MARTIN ERNEST PHILLIP PHILIP
TODD JESSE CRAIG ALAN SHAWN CLARENCE SEAN LEONARD DALE ALLEN CHRIS MARVIN
VERNON GLEN GLENN LEROY CLARK CURTIS TROY NORMAN EARL WESLEY RODNEY
MARY PATRICIA LINDA BARBARA ELIZABETH JENNIFER MARIA SUSAN MARGARET DOROTHY
LISA NANCY KAREN BETTY HELEN SANDRA DONNA CAROL RUTH SHARON MICHELLE LAURA
SARAH KIMBERLY DEBORAH JESSICA SHIRLEY CYNTHIA ANGELA MELISSA BRENDA AMY ANNA
REBECCA VIRGINIA KATHLEEN PAMELA MARTHA DEBRA AMANDA STEPHANIE CAROLYN CHRISTINE
JANET CATHERINE FRANCES ANN JOYCE DIANE ALICE JULIE HEATHER TERESA DORIS GLORIA
EVELYN JEAN CHERYL MILDRED KATHERINE JOAN ASHLEY JUDITH ROSE JANICE KELLY NICOLE
JUDY CHRISTINA KATHY THERESA BEVERLY DENISE TAMMY IRENE JANE LORI RACHEL MARILYN
ANDREA KATHRYN LOUISE SARA ANNE JACQUELINE WANDA BONNIE JULIA RUBY LOIS TINA
PHYLLIS NORMA PAULA DIANA ANNIE LILLIAN PEGGY CRYSTAL GRACE CONNIE EDNA
FLORENCE TRACY EDITH TANYA JOANN SHELLY SHELLEY JANELLE ADRIENNE JOLEEN JERI
ARDITH FELISA ERIN MERIDIAN MARLENE GLENDA MARJORIE VELMA BECKY DEAN AUDRA
""".split())


def _harvest(names: list[str], min_count: int = 2) -> tuple[set[str], set[str]]:
    """Bootstrap a given-name gazetteer FROM THE CORPUS.

    Pattern exploited: a lone single-letter token in a 3-token personal name is a
    MIDDLE INITIAL, and the token beside it is the given name.
        "PHYTHIAN DONALD L" / "MARTINEZ LEROY O" / "ARAGON ROBERTA J"  -> tok[1]
        "PAUL R NEILSON"    / "ROBERT D MARTIN"  / "CAROL K WERNER"    -> tok[0]
    No external name list is needed and the result is calibrated to THIS corpus
    (1990s-2000s Colorado UCC filings), which a modern census list would not be.
    """
    given, sur = collections.Counter(), collections.Counter()
    for nm in names:
        t = [x for x in nm.split() if x not in GENERATIONAL]
        if len(t) != 3 or any(x in BUSINESS_TOKENS for x in t):
            continue
        if len(t[2]) == 1 and len(t[1]) > 1 and len(t[0]) > 1:
            given[t[1]] += 1
            sur[t[0]] += 1                      # "PHYTHIAN DONALD L"
        elif len(t[1]) == 1 and len(t[0]) > 1 and len(t[2]) > 1:
            given[t[0]] += 1
            sur[t[2]] += 1                      # "PAUL R NEILSON"

    def keep(c):
        return {w for w, n in c.items() if n >= min_count and len(w) >= 3
                and w not in BUSINESS_TOKENS and not w.isdigit()}
    g, sn = keep(given), keep(sur)
    return g, sn - g          # a token that is BOTH is left to the given list only


# --------------------------------------------------------------------------
# Person detector
# --------------------------------------------------------------------------

class PersonDetector:
    def __init__(self, given: set[str], surnames: set[str] | None = None):
        self.given = given
        self.surnames = surnames or set()

    def parse(self, name: str | None):
        """-> None for an ORG, else (surname, tuple(given_tokens), tuple(initials)).

        A record is PERSON-LIKE when, after dropping generational tails:
          * it has 2-4 tokens (a Colorado personal filing is SURNAME GIVEN [MI]
            or GIVEN [MI] SURNAME -- never longer),
          * no token is in BUSINESS_TOKENS,
          * at least one multi-letter token is in the given-name gazetteer,
          * and at least one multi-letter token is NOT (that is the surname; a
            string of nothing but given names is left to the org path).
        """
        if not name:
            return None
        toks = [t for t in name.split() if t]
        while toks and toks[-1] in GENERATIONAL:
            toks.pop()
        if not (2 <= len(toks) <= 4):
            return None
        if any(t in BUSINESS_TOKENS for t in toks):
            return None
        if any(t.isdigit() for t in toks):
            return None
        words = [t for t in toks if len(t) > 1]
        inits = tuple(sorted(t for t in toks if len(t) == 1))
        if len(words) < 2:
            return None
        hits = [t for t in words if t in self.given]
        misses = [t for t in words if t not in self.given]
        if not hits or not misses:
            # STRUCTURAL fallback, no gazetteer needed: exactly two multi-letter
            # tokens with a lone single-letter token beside one of them is the
            # middle-initial shape -- "HANSEN DONAL D", "SMITH RUSSELL V".
            # A rare or misspelt given name ("DONAL") is missed by any gazetteer,
            # and the initial is the structure that gives it away.
            # It is deliberately narrow: TWO initials ("L T LITHO", "SPINNERS K B")
            # do NOT qualify, because that is an org's initialism, not a person.
            if len(words) == 2 and len(inits) == 1 and len(toks) == 3:
                return (words[0], (words[1],), inits)
            # SURNAME fallback. A gazetteer of given names cannot cover LONNY,
            # STAN, LANNY or SUNG, and a MISSED person is the dangerous error:
            # it drops the pair onto the lenient organisation rule, where
            # HITCHCOCK STAN / HITCHCOCK SUE scores despaced JW 0.92. The surname
            # list is harvested from the same middle-initial pattern, so a
            # two-token name whose head is a known SURNAME and whose tail is no
            # business word is read as a person.
            if self.surnames and len(words) == 2 and not inits:
                if words[0] in self.surnames or words[1] in self.surnames:
                    return (words[0], (words[1],), inits)
            return None
        # Surname = the non-given token. Order (surname-first vs given-first) does
        # not have to be resolved: the gate below is symmetric over token sets.
        surname = misses[0] if len(misses) == 1 else " ".join(misses)
        return (surname, tuple(sorted(hits)), inits)

    def is_person(self, name):
        return self.parse(name) is not None


# --------------------------------------------------------------------------
# String similarity (pure python -- no extra dependency)
# --------------------------------------------------------------------------

def _jaro(s: str, t: str) -> float:
    if s == t:
        return 1.0
    ls, lt = len(s), len(t)
    if not ls or not lt:
        return 0.0
    win = max(ls, lt) // 2 - 1
    if win < 0:
        win = 0
    sf = [False] * ls
    tf = [False] * lt
    m = 0
    for i, ch in enumerate(s):
        lo, hi = max(0, i - win), min(i + win + 1, lt)
        for j in range(lo, hi):
            if not tf[j] and t[j] == ch:
                sf[i] = tf[j] = True
                m += 1
                break
    if not m:
        return 0.0
    k = trans = 0
    for i in range(ls):
        if sf[i]:
            while not tf[k]:
                k += 1
            if s[i] != t[k]:
                trans += 1
            k += 1
    trans //= 2
    return (m / ls + m / lt + (m - trans) / m) / 3.0


def jw(s: str, t: str, p: float = 0.1) -> float:
    j = _jaro(s, t)
    l = 0
    for a, b in zip(s[:4], t[:4]):
        if a != b:
            break
        l += 1
    return j + l * p * (1 - j)


# --------------------------------------------------------------------------
# The name-agreement gate
# --------------------------------------------------------------------------

TOK_JW = 0.80        # given-name / token agreement bar   (person x person)
ORG_TOK_JW = 0.85    # per-token bar                      (org x org)
#: 0.94, not 0.90. Despaced Jaro-Winkler gives an UNEARNED prefix bonus to two
#: names that share a leading surname, which is precisely the shape a MISSED
#: person pair has: HITCHCOCK STAN / HITCHCOCK SUE = 0.9205, SMITH JOHN /
#: SMITH JANE = 0.9111, BRADY BROTHERS / BRADY DENOYER = 0.8314. All three are
#: DIFFERENT and all three clear 0.90. Every genuine organisation match in the
#: TRAIN labels that sits between 0.90 and 0.94 is recovered by the prefix or
#: token path below, so the tightening costs no measured recall.
ORG_JW = 0.94
ORG_PREFIX_MIN = 8   # min despaced length for the prefix escape hatch


def _tokens_agree(a: list[str], b: list[str], bar: float = TOK_JW) -> bool:
    """Every multi-letter token on BOTH sides has a partner on the other side."""
    def covered(xs, ys):
        return all(any(jw(x, y) >= bar for y in ys) for x in xs)
    return covered(a, b) and covered(b, a)


def gate(pa, pb, na: str, nb: str, det: PersonDetector) -> tuple[bool, str]:
    """True == the pair may keep its Splink weight."""
    if pa and pb:
        wa = [t for t in na.split() if len(t) > 1 and t not in GENERATIONAL]
        wb = [t for t in nb.split() if len(t) > 1 and t not in GENERATIONAL]
        if _tokens_agree(wa, wb):
            return True, "person_ok"
        return False, "person_given_mismatch"
    if bool(pa) != bool(pb):
        return False, "person_vs_org"
    da, db = na.replace(" ", ""), nb.replace(" ", "")
    if jw(da, db) >= ORG_JW:
        return True, "org_ok_jw"
    lo, hi = (da, db) if len(da) <= len(db) else (db, da)
    if len(lo) >= ORG_PREFIX_MIN and hi.startswith(lo):
        return True, "org_ok_prefix"
    ta = [t for t in na.split() if len(t) > 1]
    tb = [t for t in nb.split() if len(t) > 1]
    if ta and tb and _tokens_agree(ta, tb, ORG_TOK_JW):
        return True, "org_ok_tokens"
    return False, "org_name_mismatch"


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------

def corpus_records() -> pd.DataFrame:
    con = duckdb.connect(str(ROOT / "ucc.duckdb"), read_only=True)
    df = build_records(con, "corpus_debtors_eq")
    con.close()
    return df


def train_pairs() -> pd.DataFrame:
    """labels_train.csv joined to the record text.  NEVER touches labels_test.csv."""
    tr = pd.read_csv(ROOT / "labels_train.csv")
    b = pd.concat([pd.read_csv(ROOT / "docs" / "labels_blank.csv", dtype=str).fillna(""),
                   pd.read_csv(ROOT / "docs" / "labels_blank_batch2.csv", dtype=str).fillna("")],
                  ignore_index=True).drop(columns=["label", "note"], errors="ignore")
    j = tr.merge(b, on="pair_id")
    return j[j.stratum.str.startswith("debtor")].reset_index(drop=True)


def build_detector(recs: pd.DataFrame) -> PersonDetector:
    g, sn = _harvest(list(recs.name_clean.dropna()))
    return PersonDetector(g | _SEED_GIVEN, sn)


# --------------------------------------------------------------------------
# Calibration report -- TRAIN ONLY
# --------------------------------------------------------------------------

def calibrate():
    recs = corpus_records()
    det = build_detector(recs)
    print(f"gazetteer: {len(det.given):,} given names "
          f"({len(det.given - _SEED_GIVEN):,} harvested from the corpus, "
          f"{len(_SEED_GIVEN):,} seeded)")

    flags = [det.is_person(n) for n in recs.name_clean]
    n_p = sum(flags)
    print(f"corpus: {len(recs):,} debtor records, PERSON-like {n_p:,} "
          f"({100*n_p/len(recs):.1f}%), ORG-like {len(recs)-n_p:,}")
    import random
    random.seed(0)
    pers = [n for n, f in zip(recs.name_clean, flags) if f]
    orgs = [n for n, f in zip(recs.name_clean, flags) if not f]
    print("  PERSON sample:", random.sample(pers, 12))
    print("  ORG    sample:", random.sample(orgs, 12))

    tp = train_pairs()
    rows = []
    for _, r in tp.iterrows():
        pa, pb = det.parse(r.a_name), det.parse(r.b_name)
        ok, why = gate(pa, pb, r.a_name, r.b_name, det)
        rows.append(dict(pair_id=r.pair_id, label=r.label, stratum=r.stratum,
                         a=r.a_name, b=r.b_name, kind=("PP" if pa and pb else
                         "OO" if not pa and not pb else "PO"), keep=ok, why=why))
    g = pd.DataFrame(rows)
    print("\n--- TRAIN gate behaviour (debtor, n=%d) ---" % len(g))
    print(pd.crosstab([g.kind, g.keep], g.label))
    kept = g[g.keep]
    print(f"\nkept {len(kept)}: SAME={sum(kept.label=='SAME')} DIFFERENT={sum(kept.label=='DIFFERENT')}"
          f"  -> gate-only precision {sum(kept.label=='SAME')/max(1,len(kept)):.3f}")
    print(f"SAME pairs killed by the gate (recall cost): "
          f"{sum((~g.keep)&(g.label=='SAME'))} of {sum(g.label=='SAME')}")
    print("\nFalse ACCEPTS still standing (DIFFERENT but kept):")
    print(kept[kept.label == "DIFFERENT"][["pair_id", "kind", "why", "a", "b"]].to_string())
    print("\nTrue matches KILLED (SAME but vetoed):")
    print(g[(~g.keep) & (g.label == "SAME")][["pair_id", "kind", "why", "a", "b"]].to_string())

    # class-2 audit: same-address + similar-name pairs in the TRAIN labels
    c2 = g[[jw(a.replace(" ", ""), b.replace(" ", "")) >= 0.70
            for a, b in zip(g.a, g.b)]]
    print(f"\nclass-2-like (despaced JW>=0.70) train pairs: {len(c2)}; "
          f"DIFFERENT among them {sum(c2.label=='DIFFERENT')}, "
          f"of which vetoed {sum((c2.label=='DIFFERENT') & (~c2.keep))}")
    return g


# --------------------------------------------------------------------------
# Full run
# --------------------------------------------------------------------------

def _pq(frame, path):
    d = duckdb.connect()
    d.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    d.register("_f", frame)
    d.execute(f"COPY (SELECT * FROM _f) TO '{path}' (FORMAT parquet)")
    d.close()


def run(write: bool = True) -> pd.DataFrame:
    from splink import DuckDBAPI, Linker, SettingsCreator

    recs = corpus_records()
    det = build_detector(recs)
    print(f"[{TAG}] records={len(recs):,}  gazetteer={len(det.given):,}")

    db_api = DuckDBAPI(":temporary:")
    db_api._con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=comparisons_for("debtors"),
        blocking_rules_to_generate_predictions=BLOCKING,
        retain_intermediate_calculation_columns=False,
    )
    linker = Linker(recs, settings, db_api=db_api, set_up_basic_logging=False)
    linker.training.estimate_probability_two_random_records_match(DETERMINISTIC, recall=0.8)
    linker.training.estimate_u_using_random_sampling(max_pairs=2_000_000, seed=SEED)
    for br in BLOCKING:
        linker.training.estimate_parameters_using_expectation_maximisation(br)

    preds = linker.inference.predict(threshold_match_weight=-50)
    pdf = preds.as_pandas_dataframe()[
        ["unique_id_l", "unique_id_r", "match_weight", "match_probability"]]
    print(f"[{TAG}] scored pairs: {len(pdf):,}")

    name = dict(zip(recs.unique_id, recs.name_clean))
    cache: dict[str, object] = {}

    def pinfo(uid):
        if uid not in cache:
            cache[uid] = det.parse(name.get(uid))
        return cache[uid]

    # The gate is only consulted for pairs that could ever be merged. Every
    # threshold on the reported curve is >= 2.0, so a pair scoring below 1.0 is
    # left untouched: gating it would change no number and the gate is pure
    # python over ~2.5M pairs. GATE_FLOOR is a compute bound, not a tuning knob.
    GATE_FLOOR = 1.0
    memo: dict[tuple, tuple] = {}
    keep = []
    why = []
    for l, r, w_ in zip(pdf.unique_id_l, pdf.unique_id_r, pdf.match_weight):
        if w_ < GATE_FLOOR:
            keep.append(True)
            why.append("below_gate_floor")
            continue
        na, nb = name.get(l) or "", name.get(r) or ""
        k = (na, nb) if na <= nb else (nb, na)
        hit = memo.get(k)
        if hit is None:
            hit = memo[k] = gate(pinfo(l), pinfo(r), na, nb, det)
        ok, w = hit
        keep.append(ok)
        why.append(w)
    pdf["_keep"] = keep
    pdf["_why"] = why
    n_veto = int((~pdf._keep).sum())
    print(f"[{TAG}] gate vetoed {n_veto:,} of {len(pdf):,} scored pairs "
          f"({100*n_veto/len(pdf):.1f}%)")
    print(pdf._why.value_counts().to_string())
    hi = pdf[pdf.match_weight >= 6.0]
    print(f"[{TAG}] at weight>=6.0: {len(hi):,} pairs, vetoed {int((~hi._keep).sum()):,} "
          f"({100*(~hi._keep).mean():.1f}%)")

    out = pdf.copy()
    out.loc[~out._keep, "match_weight"] = -99.0
    out.loc[~out._keep, "match_probability"] = 0.0
    out = out[["unique_id_l", "unique_id_r", "match_weight", "match_probability"]]
    if write:
        _pq(out, ROOT / "parquet" / f"predictions_{TAG}.parquet")
        print(f"[{TAG}] wrote parquet/predictions_{TAG}.parquet")
    return pdf


def score_curve(thresholds=(2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0)):
    from score import score_model
    rows = []
    for t in thresholds:
        rows.append(score_model(tag=TAG, corpus="debtor", threshold=t))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "calibrate":
        calibrate()
    elif cmd == "score":
        print(score_curve().to_string())
    else:
        calibrate()
        run()
        print()
        print(score_curve().to_string())
