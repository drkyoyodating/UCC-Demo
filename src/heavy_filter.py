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
    "CATERPILLAR", "CAT FINANCIAL", "JOHN DEERE", "DEERE", "KOMATSU", "VOLVO",
    "HITACHI", "LIEBHERR", "DOOSAN", "DEVELON", "HYUNDAI CONSTRUCTION", "KOBELCO",
    "CASE CONSTRUCTION", "CASE CREDIT", "CNH", "NEW HOLLAND", "JCB", "TEREX",
    "GENIE", "JLG", "MANITOWOC", "GROVE", "LINK-BELT", "LINK BELT", "SANY", "XCMG",
    "ZOOMLION", "TAKEUCHI", "KUBOTA", "YANMAR", "BOBCAT", "WACKER NEUSON",
    "VERMEER", "DITCH WITCH", "ASTEC", "GRADALL", "GEHL", "MANITOU", "MERLO",
    "SKYJACK", "HAULOTTE", "BOMAG", "DYNAPAC", "WIRTGEN", "VOGELE", "HAMM",
    "SAKAI", "AMMANN", "ATLAS COPCO", "EPIROC", "SANDVIK", "METSO", "POWERSCREEN",
    "MOROOKA", "PRINOTH", "MUSTANG MANUFACTURING", "ALLIED CONSTRUCTION",
    "WAGNER EQUIPMENT", "FARIS MACHINERY", "POWER EQUIPMENT", "4 RIVERS EQUIPMENT",
    "HONNEN EQUIPMENT", "COLORADO MACHINERY", "RMS RENTALS", "WESTERN STATES EQUIPMENT",
    "CONSTRUCTION EQUIPMENT", "HEAVY EQUIPMENT", "EQUIPMENT FINANCE",
    "EQUIPMENT LEASING", "MACHINERY",
]

# --- ROUTE B: equipment classes and trade words that appear in a firm's own name ---
#: Machine classes a job site actually runs. Categories, never model numbers.
EQUIPMENT_WORDS = [
    "EXCAVATOR", "EXCAVATION", "EXCAVATING", "BACKHOE", "BULLDOZER", "DOZER",
    "LOADER", "SKIDSTEER", "SKID STEER", "GRADER", "SCRAPER", "TRENCHER",
    "CRANE", "RIGGING", "HOIST", "TELEHANDLER", "FORKLIFT",
    "BOOMLIFT", "BOOM LIFT", "MANLIFT", "SCISSOR LIFT", "AERIAL LIFT",
    "COMPACTOR", "PAVER", "PAVING", "MILLING", "CRUSHING",
    "DRILL", "DRILLING", "BORING",
    "PILE DRIVER", "PILING", "SHORING", "DREDGE", "DREDGING",
]
# REMOVED after auditing every word against real Colorado borrower names:
#   FOUNDATION  4,647 hits, almost all charities -- FOUNDATION FOR SENIOR CITIZENS,
#               Saint Joseph Hospital Foundation. By far the largest polluter.
#   DERRICK     a first name -- Derrick LeRoy Tadlock, DERRICK DESEAN WILLIAMS.
#   AUGER       a surname -- Cameron Auger DDS, JEAN-CLAUDE ALAIN AUGER TRUST.
#   ROLLER      roller hockey, Roller & Associates.
#   WRECKING    auto salvage, not demolition -- ACTION AUTO WRECKING, JAPA IMPORT AUTO.
#   CRUSHER     car crushers -- BONNIE'S CAR CRUSHERS, COLORADO CAR CRUSHER.
#               (CRUSHING kept: Colorado Crushing, JMB Crushing Systems are aggregate.)
#   SCREENING   AMERICAN PRE-EMPLOYMENT SCREENING.
#   CONVEYOR    industrial belt suppliers, not site equipment.
#   HOIST/RIGGING/RECLAMATION kept -- audited clean on real names.
#: Trades that own and run that equipment.
TRADE_WORDS = [
    "DEMOLITION", "DEMO", "WRECKING", "EARTHWORK", "EARTHMOVING", "EARTH MOVING",
    "SITEWORK", "SITE WORK", "DIRT WORK", "GRADING", "TRENCHING", "ASPHALT",
    "CONCRETE", "AGGREGATE", "QUARRY", "GRAVEL", "SAND AND GRAVEL", "READY MIX",
    # CONCRETE FAMILY -- founder's ruling: a concrete outfit runs mixers, pumpers and
    # boom trucks, which ARE heavy construction equipment by definition. So these
    # business types qualify on the NAME alone (Route B), without needing a
    # manufacturer lender attached.
    "CEMENT", "MIXER", "SHOTCRETE", "PRECAST", "FLATWORK", "CURB AND GUTTER",
    "CONCRETE PUMPING", "TILT UP", "REBAR", "POST TENSION",
    "PIPELINE", "UNDERGROUND UTILITIES", "UNDERGROUND UTILITY",
    "HEAVY HAUL", "HEAVY HAULING", "MINING", "RECLAMATION", "LAND CLEARING",
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
LENDER_RE = re.compile(r"(?:" + "|".join(re.escape(m).replace(r"\ ", r"\s+")
                                         for m in MANUFACTURERS) + r")")

#: DuckDB takes the pattern verbatim; escaping backslashes for SQL breaks \b and \s.
BORROWER_SQL = BORROWER_RE.pattern
LENDER_SQL = LENDER_RE.pattern


def is_heavy_lender(name): return bool(name) and bool(LENDER_RE.search(str(name).upper()))
def is_heavy_borrower(name): return bool(name) and bool(BORROWER_RE.search(str(name).upper()))


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
