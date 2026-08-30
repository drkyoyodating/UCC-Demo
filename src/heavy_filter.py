#!/usr/bin/env python
"""The heavy-construction filter. Two independent routes into scope, per the founder.

A filing qualifies if EITHER:
  ROUTE A -- the LENDER is a heavy-construction-equipment manufacturer or its
             captive finance arm / franchised dealer. The machine class is then
             evidenced by who financed it.
  ROUTE B -- the BORROWER'S OWN BUSINESS NAME contains a heavy-construction
             equipment or trade word, matched 1-to-1 on a whole word, with plurals
             and possessives handled: "Jims Cranes", "Carlos Excavation",
             "Shifts Excavators", "John's Demolition".

Route B exists because a firm called BOBS CRANES is in the market whoever lent to
it. Route A exists because CATERPILLAR FINANCIAL does not finance bathroom
remodels. Neither route alone is sufficient coverage; together they are the
market.

NOT included by design: brand model designations (CAT 9, BOBCAT S70). Those are
product SKUs, not evidence, and no free register carries them (Appendix B).
"""
from __future__ import annotations
import re

# --- ROUTE A: heavy construction equipment makers, captives, and major dealers ---
MANUFACTURERS = [
    "CATERPILLAR", "CAT FINANCIAL", "CAPTERPILLAR", "CETERPILLAR", "CATERPILLR", "KOMATSU", "VOLVO",
    # OCR corruptions of maker names, from the same distance-1 sweep. Only
    # non-words: DEER, VOGEL, MERLE, METRO, AZTEC, ALTEC, GENE and GROVER are
    # real words or real firms at distance 1 and are deliberately NOT listed.
    # DERE and DEEE were considered and REJECTED: both are short enough to be
    # real standalone tokens ("DERE VALLEY LLC"), and a whole-word match on them
    # admits unrelated firms. Only unambiguous long corruptions are listed.
    
    "HITACHI CONSTRUCTION", "LIEBHERR", "DOOSAN", "DEVELON", "HYUNDAI CONSTRUCTION", "KOBELCO",
    "CASE CONSTRUCTION", "JCB", "TEREX",
    "GENIE INDUSTRIES", "JLG", "MANITOWOC", "GROVE U.S", "LINK-BELT", "LINK BELT", "SANY", "XCMG",
    "ZOOMLION", "TAKEUCHI", "YANMAR", "BOBCAT", "WACKER NEUSON",
    "VERMEER", "DITCH WITCH", "ASTEC", "GRADALL", "GEHL", "MANITOU", "MERLO",
    "SKYJACK", "HAULOTTE", "BOMAG", "DYNAPAC", "WIRTGEN", "VOGELE", "HAMM",
    "SAKAI", "AMMANN", "ATLAS COPCO", "EPIROC", "SANDVIK", "METSO", "POWERSCREEN",
    "MOROOKA", "PRINOTH", "MUSTANG MANUFACTURING", "ALLIED CONSTRUCTION",
    "WAGNER EQUIPMENT", "FARIS MACHINERY", "POWER EQUIPMENT COMPANY", "4 RIVERS EQUIPMENT",
    "HONNEN EQUIPMENT", "COLORADO MACHINERY", "RMS RENTALS", "WESTERN STATES EQUIPMENT",
    # --- REMOVED 2026-08-30 by founder ruling: "don't allow the 1ST SOURCE
    # BANK, CONSTRUCTION EQUIPMENT DIVISION in, too vague with the data we have
    # access to -- we work with what we have, not what we want."
    # CONSTRUCTION EQUIPMENT / HEAVY EQUIPMENT / HEAVY CONSTRUCTION EQUIPMENT
    # were the last three DESCRIPTIVE phrases among otherwise-named criteria.
    # Measured: of 767 rows qualifying on them alone, 762 were 1st Source Bank's
    # equipment-finance division. The phrase was carrying a bank, not a maker.
    # The five genuine rows are named explicitly instead -- the same treatment
    # MACHINERY got when it was replaced by its real dealers.
    "FLAGLER CONSTRUCTION EQUIPMENT", "CONSTRUCTION EQUIPMENT COMPANY",
    # --- NAMED heavy-equipment dealers that used to ride the bare word MACHINERY.
    # Founder ruling 2026-08-30: Route A is a NAMED manufacturer or dealer. The
    # bare token MACHINERY is gone (a machine can be a laundry machine or a key
    # machine), so every real dealer it used to carry is now listed explicitly.
    # These are franchised Caterpillar / Komatsu / Deere / Case dealers:
    "WYOMING MACHINERY", "CARTER MACHINERY", "NEBRASKA MACHINERY",
    "BUTLER MACHINERY", "WARREN POWER", "LINDER INDUSTRIAL",
    "ROAD MACHINERY", "TITAN MACHINERY", "PAPE MACHINERY",
    "BLANCHARD MACHINERY", "TRACTOR AND EQUIPMENT CO", "BLAW-KNOX", "BLAW KNOX",
    "DENVER EAST MACHINERY",
    # --- Named dealers and OEMs found MISSING by the P5 audit 2026-08-30. These
    # are recall losses, not precision ones: 1,268 party-sides from a 20-name
    # hand probe were failing BOTH criteria only because the dealer was absent
    # from this list. POWER MOTIVE is the Denver Volvo/Case/Doosan dealer, in the
    # same class as WAGNER; NORTH CENTRAL RENTAL is Butler Machinery's own
    # leasing arm and carried 363 rows while BUTLER MACHINERY itself carried 3.
    # --- THE 12 NYSE-LISTED HEAVY CONSTRUCTION EQUIPMENT MAKERS, added by
    # founder direction 2026-08-30, with their operating brands. Corporate
    # parents AND brands are both listed because the register files under either.
    # Short/ambiguous tokens are spelled out: ALAMO alone is a rental car and a
    # cinema chain, YALE alone is a university and a lock, WABASH alone is a
    # river and a valley, OSHKOSH alone is a childrenswear label.
    "OSHKOSH CORPORATION", "OSHKOSH TRUCK", "MCNEILUS",     # Oshkosh (OSK)
    "ALAMO GROUP",                                          # Alamo (ALG)
    "VOLVO CONSTRUCTION",                                   # AB Volvo (VLVLY)
    "GORMAN-RUPP", "GORMAN RUPP",                           # Gorman-Rupp (GRC)
    "FEDERAL SIGNAL",                                       # Federal Signal (FSS)
    "HYSTER-YALE", "YALE MATERIALS HANDLING",               # Hyster-Yale (HY)
    "WABASH NATIONAL",                                      # Wabash National (WNC)
    "POWER MOTIVE", "RDO EQUIPMENT", "POTESTIO BROTHERS", "H & E EQUIPMENT",
    "H&E EQUIPMENT", "NORTH CENTRAL RENTAL", "COLORADO EQUIPMENT",
    "CLARK EQUIPMENT", "UNITED RENTALS", "TADANO",
    "NATIONAL CRANE", "ELLIOTT EQUIPMENT", "PETTIBONE", "CLEVELAND BROTHERS",
    "SHAWMUT EQUIPMENT", "CROWN EQUIPMENT", "MONROE TRACTOR", "HYSTER",
    "INGERSOLL RAND",
    # --- PURE AGRICULTURE IS OUT. Founder ruling 2026-08-30, after weighing it:
    # farm tractors are heavy iron, but pure agriculture is a DIFFERENT MARKET,
    # not construction-adjacent -- separate dealers, separate buyers, and Deere
    # itself splits Ag from Construction & Forestry. The pitch is heavy
    # CONSTRUCTION equipment finance, and admitting hay-tool and combine dealers
    # dilutes exactly that claim. Consistent with LABELLING_CRITERIA.md s1.4
    # ("AGCO (pure agriculture)", "pure crop and livestock agriculture").
    # Excluded by name: KUHN FARM MACHINERY, STOTZ EQUIPMENT, AGPRO, AGCO,
    # LIVINGSTON MACHINERY, BINGHAM FARM MACHINERY, ARIZONA MACHINERY.
    # DUAL-LINE dealers are KEPT above (TITAN, BUTLER): they sell construction
    # equipment too, which qualifies them regardless of also selling ag.
]

# REMOVED 2026-08-30 after auditing the v4 label pull against real lender strings:
#   EQUIPMENT FINANCE / EQUIPMENT LEASING
#       Route A is defined as "the LENDER is a heavy-construction manufacturer,
#       captive finance arm, or dealer". A BANK's general equipment-leasing
#       division is none of the three, but these two bare phrases matched every
#       one of them. In the 1300-row v4 sample they admitted 698 of 2,600 party
#       sides (26.85%) ON THEIR OWN -- U.S. Bank Equipment Finance (231),
#       Key Equipment Finance / KeyBank (88), Wells Fargo Equipment Finance (60),
#       Stearns (39), TD (29), PNC, TCF, People's United, Western.
#       Those lessors finance anything, which is how a dairy (COLORADO COW LLC),
#       a tree service, a tox lab (ROCKY MOUNTAIN TOX LLC), solar SPEs
#       (CEC SOLAR #1062 LLC), COLORADO DOG ACADEMY and COLORADO EYE SURGEONS
#       entered a heavy-construction corpus.
#       Captive arms are unaffected: CATERPILLAR FINANCIAL still matches on
#       CATERPILLAR, KUBOTA CREDIT on KUBOTA, JOHN DEERE CONSTRUCTION on
#       JOHN DEERE. Named dealers keep their own entries.
#       MACHINERY / CONSTRUCTION EQUIPMENT / HEAVY EQUIPMENT are KEPT: they
#       admitted only 25 sides (0.96%) on their own and those are real dealers
#       (TITAN MACHINERY, FARIS MACHINERY).

# --- ROUTE B: equipment classes and trade words that appear in a firm's own name ---
#: Machine classes a job site actually runs. Categories, never model numbers.
EQUIPMENT_WORDS = [
    "EXCAVATOR", "EXCAVATION", "EXCAVATING", "BACKHOE", "BULLDOZER", "DOZER",
    "LOADER", "SKIDSTEER", "SKID STEER", "GRADER", "SCRAPER", "TRENCHER",
    "CRANE", "RIGGING", "HOIST", "TELEHANDLER", "FORKLIFT",
    "BOOMLIFT", "BOOM LIFT", "MANLIFT", "SCISSOR LIFT", "AERIAL LIFT",
    "COMPACTOR", "PAVING", "CRUSHING",
    "DRILL", "DRILLING", "BORING",
    "PILE DRIVER", "PILING", "SHORING", "DREDGE", "DREDGING",
    # --- Product categories named in the founder's NYSE-manufacturer brief,
    # 2026-08-30. Each is a machine a job site actually runs, and each is a
    # multi-word phrase or an unambiguous noun -- no bare surname risk.
    "WHEEL LOADER", "CRAWLER DOZER", "CRAWLER CRANE", "TOWER CRANE",
    "ROUGH TERRAIN CRANE", "MOBILE CRANE", "HYDRAULIC EXCAVATOR",
    "ARTICULATED DUMP TRUCK", "ARTICULATED HAULER", "MINING TRUCK",
    "AERIAL WORK PLATFORM", "CONCRETE MIXER", "MIXER TRUCK",
    "MATERIALS PROCESSING", "VACUUM EXCAVATION", "STREET SWEEPER",
    "DEWATERING",
]
# REMOVED after auditing every word against real Colorado borrower names:
#   FOUNDATION  4,647 hits, almost all charities -- FOUNDATION FOR SENIOR CITIZENS,
#               Saint Joseph Hospital Foundation. By far the largest polluter.
#   DERRICK     a first name -- Derrick LeRoy Tadlock, DERRICK DESEAN WILLIAMS.
#   AUGER       a surname -- Cameron Auger DDS, JEAN-CLAUDE ALAIN AUGER TRUST.
#   ROLLER      roller hockey, Roller & Associates.
#   WRECKING    auto salvage, not demolition -- ACTION AUTO WRECKING, JAPA IMPORT AUTO.
#               NOTE: this word was documented as cut in three places but was still
#               LIVE in TRADE_WORDS until 2026-08-30; it was pulling 105 route-B-only
#               rows, every one of them an auto-salvage yard (ELM CITY AUTO WRECKING,
#               FOWLER'S AUTO WRECKING, JAPA IMPORT AUTO WRECKING). Actually removed now.
#   CRUSHER     car crushers -- BONNIE'S CAR CRUSHERS, COLORADO CAR CRUSHER.
#               (CRUSHING kept: Colorado Crushing, JMB Crushing Systems are aggregate.)
#   SCREENING   AMERICAN PRE-EMPLOYMENT SCREENING.
#   CONVEYOR    industrial belt suppliers, not site equipment.
#   HEAVY HAUL / HEAVY HAULING
#               Removed 2026-08-30. Founder ruling: "we aren't looking for
#               freight trucks, we are looking for equipment used on JOB SITES."
#               Heavy haul is over-the-road transport of oversize loads -- a
#               trucking business, not a firm that owns and runs site iron.
#               TRUCKING itself was never a qualifier and stays unlisted.
#   PAVER       BRICK / HARDSCAPE retailers, not asphalt pavers -- REIS BRICKS AND
#               PAVERS, SYSTEM PAVERS OF COLORADO, THE STONE & PAVER COMPANY,
#               PIONEER PAVERS. Removed 2026-08-30. PAVING is KEPT and still
#               catches every real road-paving firm.
#   MILLING     FLOUR/GRAIN milling -- PANHANDLE MILLING, PRIME MILLING. Removed
#               2026-08-30. Real road-milling firms carry PAVING or CONCRETE too
#               (MILLING PAVING & CONCRETE), so they are not lost.
#   HOIST/RIGGING/RECLAMATION kept -- audited clean on real names.
#: Trades that own and run that equipment.
TRADE_WORDS = [
    "DEMOLITION", "DEMO", "EARTHWORK", "EARTHMOVING", "EARTH MOVING",
    "SITEWORK", "SITE WORK", "DIRT WORK", "GRADING", "TRENCHING", "ASPHALT",
    "CONCRETE", "AGGREGATE", "QUARRY", "GRAVEL", "SAND AND GRAVEL", "READY MIX",
    # CONCRETE FAMILY -- founder's ruling: a concrete outfit runs mixers, pumpers and
    # boom trucks, which ARE heavy construction equipment by definition. So these
    # business types qualify on the NAME alone (Route B), without needing a
    # manufacturer lender attached.
    "CEMENT", "MIXER", "SHOTCRETE", "PRECAST", "FLATWORK", "CURB AND GUTTER",
    "CONCRETE PUMPING", "TILT UP", "REBAR", "POST TENSION",
    "PIPELINE", "UNDERGROUND UTILITIES", "UNDERGROUND UTILITY",
    "MINING", "RECLAMATION", "LAND CLEARING",
    "SEPTIC", "WELL DRILLING", "UTILITY CONTRACTOR",
]

_WORDS = EQUIPMENT_WORDS + TRADE_WORDS


def _variants(w: str) -> list[str]:
    """Whole-word forms to accept: the word, its plural, and its possessive plural.
    "CRANE" also matches "CRANES" and "CRANE'S"; the apostrophe is already stripped
    by normalisation, so "BOBS CRANES" and "JIMS CRANE" both hit.
    """
    out = {w}
    if not w.endswith("S"):
        out.add(w + "S")
        if w.endswith(("CH", "SH", "X", "Z", "S")):
            out.add(w + "ES")
    return sorted(out)


def _alt(words):
    parts = []
    for w in words:
        for v in _variants(w):
            parts.append(re.escape(v).replace(r"\ ", r"\s+"))
    return r"(?:" + "|".join(sorted(set(parts), key=len, reverse=True)) + r")"


#: Whole-word anchored -- \b stops CRANE matching CRANEBROOK and DEMO matching DEMOGRAPHIC.
BORROWER_RE = re.compile(r"\b" + _alt(_WORDS) + r"\b")
#: WHOLE-WORD anchored, exactly like BORROWER_RE. Without \b a short entry
#: matches inside an unrelated word: DERE hits "DEREK SMITH TRUCKING" and
#: "ANDERE HOLDINGS", GENIE hits "GENIENE CORP". That admitted private
#: individuals as heavy-equipment lenders.
LENDER_RE = re.compile(r"\b(?:" + "|".join(re.escape(m).replace(r"\ ", r"\s+")
                                          for m in MANUFACTURERS) + r")\b")

#: DuckDB takes the pattern verbatim; escaping backslashes for SQL breaks \b and \s.
BORROWER_SQL = BORROWER_RE.pattern
LENDER_SQL = LENDER_RE.pattern


#: Dealer entities that split their business into a construction LLC and a
#: separate AGRICULTURE LLC, and file under whichever one sold the machine. The
#: dealer name alone therefore does not establish criterion 1 -- "4 Rivers
#: Equipment - Ag LLC" lending to JW Farms is a farm sale, not a job-site one.
#: This is a refinement of the CRITERION, not a reject list: the criterion is
#: "a named dealer of heavy CONSTRUCTION equipment", and the ag arm is not that.
_AG_ARM = re.compile(r"-\s*AG\b|\bAG\s+LLC\b|\bAGRICULTURE\b|\bFARM\s+EQUIPMENT\b")


def is_heavy_lender(name):
    """Route A criterion: is the LENDER a named maker or dealer of heavy
    construction equipment?

    THERE IS NO REJECT LIST. A row is never "excluded" -- it simply fails to meet
    a criterion. Every string in the register is pulled and assessed; only an
    explicit match is added.

    That is why MANUFACTURERS carries construction-SPECIFIC strings and no bare
    parent-company names: "DEERE & COMPANY" does not say which division filed, so
    it does not qualify, while "JOHN DEERE CONSTRUCTION & FORESTRY" does.

    Measured: an earlier reject list of banks, machine tools and agriculture
    killed 23,714 distinct lender strings, of which a construction-specific
    criterion list would have admitted only 22 anyway -- it was compensating for
    loose criteria. Worse, it deleted 406 real party rows whose maker name sits
    beside a bank ("Ditch Witch Financial Services, a program of Bank of the
    West"). Tight criteria need no reject list and cannot fail that way.
    """
    if not name:
        return False
    u = str(name).upper()
    if _AG_ARM.search(u):
        return False
    return bool(LENDER_RE.search(u))


#: Equipment words whose SINGULAR form is a plausible SURNAME. Founder ruling
#: 2026-08-30: "a guy can be named Jim Crane but not Jim Cranes -- nobody is
#: named a plural." So these qualify only in a form no person is ever called, or
#: when something else in the string proves it is a business.
#:
#: EXCAVATION / EXCAVATING / DEMOLITION / CONCRETE / PAVING are deliberately NOT
#: here: "nobody in the history of the world has the last name Excavation", so
#: they qualify in every form, singular included. This is why DERRICK and AUGER
#: had to be deleted outright -- they are surnames whose PLURAL never appears.
_SURNAME_RISK = {"CRANE", "LOADER", "QUARRY", "MIXER", "DOZER", "GRADER"}

#: Proof the string is a business rather than a person.
_BIZ_PROOF = re.compile(
    r"\b(?:INC|LLC|LLLP|LLP|LP|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLLC|PC|"
    r"TRUST|PARTNERSHIP|ENTERPRISES|INDUSTRIES|GROUP|HOLDINGS|SERVICES|SYSTEMS|"
    r"SOLUTIONS|ASSOCIATES|PARTNERS|VENTURES|PROPERTIES|SALES|RENTAL|RENTALS)\b|\d")


#: The classic family-firm frame. "CRANE & SON, INC." is the Crane family, not a
#: crane company -- the corporate marker proves it is a FIRM but not that it is a
#: heavy-construction firm, so it must not rescue a singular surname-risk word.
#: The plural, possessive and second-equipment-word clauses are untouched, which
#: is what still correctly admits DUFFY CRANE INC and COLORADO SUMMIT CRANES.
_FAMILY_FRAME = re.compile(r"&\s*SONS?\b|\bAND\s+SONS?\b|\bBROTHERS\b|\bBROS\b|"
                           r"&\s*ASSOCIATES\b|\bFAMILY\b|&\s*DAUGHTERS?\b")


def _surname_risk_ok(u: str) -> bool:
    """Decide a Route B hit that rests ONLY on a surname-risk word.

    Clauses are ordered so the strongest evidence wins first.
      1. the equipment word is PLURAL -- BOBS CRANES, COLORADO SUMMIT CRANES.
         A surname is never plural, so this settles it.
      2. a SECOND, non-surname-risk equipment word -- CRANE & RIGGING. This is
         also what rescues a genuine "Crane & Sons Rigging" from clause 3.
      3. a FAMILY-FIRM frame -- CRANE & SON, CRANE BROTHERS. The corporate marker
         proves it is a firm but not a heavy-construction firm, so it stops here.
      4. a corporate marker or digit, but only when the risk word does not LEAD --
         DUFFY CRANE INC qualifies; LOADER EMPLOYEE LEASING COMPANY does not.
      5. another token is possessive -- JIMS CRANE, the 's' sitting on the owner.
    Otherwise it reads as a person: CRANE, ROBERT GALE / LOADER DINAH P.
    """
    toks = [t.strip(",.") for t in u.replace(",", " ").split() if t.strip(",.")]

    def _sing(t):
        return t[:-2] if t.endswith("ES") else t[:-1] if t.endswith("S") else None

    # 1. the equipment word itself, pluralised
    for t in toks:
        sg = _sing(t)
        if sg and sg in _SURNAME_RISK:
            return True
    # 2. a second equipment word that is not itself surname-risk
    for t in toks:
        if BORROWER_RE.fullmatch(t) and t not in _SURNAME_RISK:
            return True
    # 3. family-firm frame -- a firm, but not evidence of heavy construction
    if _FAMILY_FRAME.search(u):
        return False
    # 4. corporate marker, only when the risk word is not in the surname slot
    if _BIZ_PROOF.search(u) and not (toks and toks[0] in _SURNAME_RISK):
        return True
    # 5. a possessive owner token
    for t in toks:
        sg = _sing(t)
        if (t.endswith("S") and t not in _SURNAME_RISK and sg
                and sg not in _SURNAME_RISK and not BORROWER_RE.fullmatch(t)):
            return True
    return False


def is_heavy_borrower(name):
    """Route B. The borrower's own name must state a heavy-equipment category.

    Founder ruling 2026-08-30: the category must be EXPLICIT. Vague words do not
    qualify -- CONSTRUCTION is deliberately absent from both word lists, so
    "BOBS CONSTRUCTION" is out while "BOBS EXCAVATION" and "JOHNS CRANES" are in.
    """
    if not name:
        return False
    u = str(name).upper()
    m = BORROWER_RE.findall(u)
    if not m:
        return False
    # If EVERY match is a surname-risk word, the string must prove it is a firm.
    if all(x in _SURNAME_RISK for x in m):
        return _surname_risk_ok(u)
    return True


if __name__ == "__main__":
    for t in ["BOBS CRANES", "JIMS CRANE", "CARLOS EXCAVATION", "JOHNS DEMOLITION",
              "SHIFTS EXCAVATORS", "HERNANDEZ CONCRETE", "ACME BATHROOM REMODELING",
              "SMITH LAW OFFICES", "CRANEBROOK APARTMENTS", "DEMOGRAPHIC RESEARCH INC",
              "WESTERN SLOPE EARTHMOVING", "PEAK AGGREGATE LLC"]:
        print(f"  borrower {t:34s} -> {'IN ' if is_heavy_borrower(t) else 'out'}")
    print()
    for t in ["CATERPILLAR FINANCIAL SERVICES", "JOHN DEERE CONSTRUCTION & FORESTRY",
              "WAGNER EQUIPMENT CO", "WELLS FARGO BANK NA", "KUBOTA CREDIT CORPORATION"]:
        print(f"  lender   {t:34s} -> {'IN ' if is_heavy_lender(t) else 'out'}")


def heavy_row(borrower, lender) -> bool:
    """A row qualifies if EITHER criterion is met. Nothing rejects a row.

    Founder's model, 2026-08-30: *"we don't have a reject database, we have
    criteria that must be met to qualify."* Everything is pulled; a row is added
    only when the lender is a named heavy-construction maker or dealer, or the
    borrower's own name states an explicit job-site equipment category.

    The two criteria are independent. A named manufacturer makes the borrower
    name irrelevant -- Terex sells cranes, so "BOBS COOKIES <- TEREX" qualifies.
    An equipment category in the borrower name makes the lender irrelevant --
    "BOBS EXCAVATION" qualifies with no lender at all. Neither met, not added.
    """
    return is_heavy_lender(lender) or is_heavy_borrower(borrower)
