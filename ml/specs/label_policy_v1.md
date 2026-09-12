# Label policy v1 — heavy-construction-equipment relevance of a UCC borrower

policy_version: label_policy_v1
status: FROZEN
labelling method disclosure: **model-labelled, founder-adjudicated** — labels are produced by a blind
Claude labeller under this policy, in two independent passes; the founder reviews every
disagreement plus an audit sample of agreed rows; blind repeats (10% of every queue chunk) are
measured. This sentence appears in every report that uses these labels.

## 1. What is being labelled

One **case** = one borrower on one UCC filing, with the filing's complete set of lender names.
Population: Colorado and Connecticut UCC filings with `loan_year` ≥ 1990, a named borrower and a
real (non-placeholder) address — the same eligibility as the published `scope_all`, **without**
the rules applied. The rules baseline (`src/heavy_filter.py`, vendored as `heavy_filter_v1`) is
the frozen comparison; this policy is what a **human screener** would decide from the same names.

Scope, restated from the founder's settled rulings (never re-opened here):
- Subject: **heavy construction equipment finance**, CO + CT, `loan_year` ≥ 1990.
- A borrower is in the market by **Route A** (a lender is a named heavy-construction maker,
  captive finance arm or franchised dealer) or **Route B** (an explicit equipment or trade word
  appears in the borrower's OWN name). `src/heavy_filter.py` is the authority on the word lists.
- **Precision over recall** is the product position (99.4% published). The ML model is a
  **review queue** over rules-rejected cases plus a second opinion on accepted ones; it **never
  replaces the rules** and never changes `scope_all`, the headline counts or the existing map.
- A borrower **never crosses CO ↔ CT**: the two registers are separate; nothing in one informs the other.

## 2. The labels

| label | meaning |
|---|---|
| `RELEVANT` | a human screener, reading only this borrower name, its lender names and its city/state, would say this borrower is in the heavy-construction-equipment-finance market |
| `NOT_RELEVANT` | the names **positively indicate a different activity** (a hospital, a law office, a farm, a restaurant, a software firm, a charity, an auto-salvage yard …). Absence of an equipment word is **not** evidence of non-relevance |
| `INSUFFICIENT_EVIDENCE` | the names could be anything: a holding company, initials, a personal name, a generic word (`CONSTRUCTION`, `SERVICES`, `EQUIPMENT`, `TRUCKING`) that does not settle it |

Every row gets exactly one label and exactly one `reason_code` from §5, plus a one-line `reason`.

## 3. The no-lookup rule

**Do not look anything up.** No Google, no Secretary of State search, no maps, no memory of a
particular firm's website. Outside research makes the labels unreproducible and is not
available at scale, so a reviewer discounts the whole number (`docs/LABELLING_CRITERIA.md`
Part 9). Labels are decided from the borrower name, the lender names and the city/state shown —
nothing else. City/state are for **disambiguation only** (a name that is also a place, a lender
whose name is a state) and are never a reason on their own.

The labeller sees: `case_id`, `borrower_name_raw`, `lender_names_raw` (sorted, joined with
` | `), `city`, `state`. The labeller does **not** see: the rules baseline flag, the route, the
sampling stratum, any model score, any address, any date. The labeller has no tools: this policy
and one queue chunk arrive inline in its prompt and it answers through structured output. Some
rows are repeated under a
different `case_id` as a consistency check: label every row independently and do not search for
duplicates. Do not revisit an earlier row after seeing a later one.

## 4. Decision procedure (apply in order; the first step that decides, decides)

**Step 1 — the lenders (Route A).** If **any** lender is a *named* heavy-construction maker,
captive or dealer → `RELEVANT`, `R_LENDER_NAMED_MAKER`, whatever the borrower is called
("Terex sells cranes, so `BOBS COOKIES` ← `TEREX` is in scope").
Named makers/captives/dealers (the `MANUFACTURERS` list in `heavy_filter_v1`): CATERPILLAR /
CAT FINANCIAL (and OCR variants CAPTERPILLAR, CETERPILLAR, CATERPILLR), `JOHN DEERE CONSTRUCTION`,
`DEERE CONSTRUCTION`, KOMATSU, VOLVO, VOLVO CONSTRUCTION, HITACHI CONSTRUCTION, LIEBHERR, DOOSAN,
DEVELON, HYUNDAI CONSTRUCTION, KOBELCO, CASE CONSTRUCTION, JCB, TEREX, GENIE INDUSTRIES, JLG,
MANITOWOC, GROVE U.S, LINK-BELT, SANY, XCMG, ZOOMLION, TAKEUCHI, YANMAR, BOBCAT, WACKER NEUSON,
VERMEER, DITCH WITCH, ASTEC, GRADALL, GEHL, MANITOU, MERLO, SKYJACK, HAULOTTE, BOMAG, DYNAPAC,
WIRTGEN, VOGELE, HAMM, SAKAI, AMMANN, ATLAS COPCO, EPIROC, SANDVIK, METSO, POWERSCREEN, MOROOKA,
PRINOTH, MUSTANG MANUFACTURING, ALLIED CONSTRUCTION, FLAGLER CONSTRUCTION EQUIPMENT, OSHKOSH
CORPORATION, OSHKOSH TRUCK, MCNEILUS, ALAMO GROUP, GORMAN-RUPP, FEDERAL SIGNAL, HYSTER-YALE, YALE
MATERIALS HANDLING, HYSTER, WABASH NATIONAL, CLARK EQUIPMENT, UNITED RENTALS, TADANO, NATIONAL
CRANE, ELLIOTT EQUIPMENT, PETTIBONE, CROWN EQUIPMENT, INGERSOLL RAND; and the named dealers WAGNER
EQUIPMENT, FARIS MACHINERY, POWER EQUIPMENT COMPANY, 4 RIVERS EQUIPMENT, HONNEN EQUIPMENT,
COLORADO MACHINERY, RMS RENTALS, WESTERN STATES EQUIPMENT, WYOMING MACHINERY, CARTER MACHINERY,
NEBRASKA MACHINERY, BUTLER MACHINERY, WARREN POWER, LINDER INDUSTRIAL, ROAD MACHINERY, TITAN
MACHINERY, PAPE MACHINERY, BLANCHARD MACHINERY, TRACTOR AND EQUIPMENT CO, BLAW-KNOX, DENVER EAST
MACHINERY, POWER MOTIVE, RDO EQUIPMENT, POTESTIO BROTHERS, H & E EQUIPMENT, NORTH CENTRAL RENTAL,
COLORADO EQUIPMENT, CLEVELAND BROTHERS, SHAWMUT EQUIPMENT, MONROE TRACTOR.

Lenders that are **not evidence either way** (continue to Step 2): banks, credit unions, general
equipment-finance / leasing arms of banks (`U.S. BANK EQUIPMENT FINANCE`, `KEY EQUIPMENT
FINANCE`, `WELLS FARGO EQUIPMENT FINANCE`), a `BANK … CONSTRUCTION EQUIPMENT DIVISION` (a bank),
dual-line or agricultural makers — **Bare `DEERE`, `DEERE & COMPANY`, `DEERE CREDIT`,
`JOHN DEERE FINANCIAL`, `JOHN DEERE COMPANY` do NOT qualify** (only the construction entity
strings above do; `docs/LABELLING_CRITERIA.md` lines 37-38 are stale on this point), KUBOTA,
CNH, NEW HOLLAND, AGCO, a dealer's ag arm (`… - AG LLC`, `… AGRICULTURE`, `… FARM EQUIPMENT`),
DE LAGE LANDEN, SNAP-ON, an individual, a blank lender. A lender that positively names another
industry (`MEDICAL LEASING`, `RESTAURANT FINANCE`) is weak evidence **against** and may support
`NOT_RELEVANT` together with the borrower name, never alone.

**Step 2 — the borrower's own name (Route B).** An explicit equipment class or trade word in
the borrower's name → `RELEVANT`. Whole words only (`CRANE` does not match `CRANEBROOK`; `DEMO`
does not match `DEMOGRAPHIC`); singular, plural, possessive and spaced/joined forms all count.
- Equipment classes → `R_BORROWER_EQUIPMENT_WORD`: excavator, excavation, excavating, backhoe,
  bulldozer, dozer, loader, skidsteer / skid steer, grader, scraper, trencher, crane, rigging,
  hoist, telehandler, forklift, boomlift / boom lift, manlift, scissor lift, aerial lift,
  compactor, paving, crushing, drill, drilling, boring, pile driver, piling, shoring, dredge,
  dredging, wheel loader, crawler dozer, crawler crane, tower crane, rough terrain crane, mobile
  crane, hydraulic excavator, articulated dump truck, articulated hauler, mining truck, aerial
  work platform, concrete mixer, mixer truck, materials processing, vacuum excavation, street
  sweeper, dewatering.
- Trade words → `R_BORROWER_TRADE_WORD`: demolition, demo, earthwork, earthmoving / earth moving,
  sitework / site work, dirt work, grading, trenching, asphalt, aggregate, quarry, gravel, sand
  and gravel, ready mix, pipeline, underground utilities / utility, mining, reclamation, land
  clearing, septic, well drilling, utility contractor — and the **CONCRETE family, which
  qualifies on the name alone**: concrete, cement, mixer, shotcrete, precast, flatwork, rebar,
  post-tension, curb-and-gutter, concrete pumping, tilt up.
- **Surname-risk words** (`CRANE`, `LOADER`, `GRADER`, `DOZER`, `QUARRY`, `MIXER`) qualify only
  when the string proves a firm: the word is plural (`BOBS CRANES`), another equipment word is
  present (`CRANE & RIGGING`), another token is possessive (`JIMS CRANE`), or a corporate marker
  or digit is present and the risk word does not lead (`DUFFY CRANE INC`). `CRANE, ROBERT GALE`
  and `LOADER DINAH P` are people; `CRANE & SON, INC.` is the Crane family (a firm, not a crane
  firm). `DRILL` immediately followed by TEAM / SQUAD / SERGEANT / INSTRUCTOR / CORPS / PRESS /
  BIT(S) is not a machine.
- **Words that are NOT evidence, by ruling — never treat them as Route B words:** the eight cut
  words FOUNDATION, DERRICK, AUGER, ROLLER, WRECKING, CRUSHER, SCREENING, CONVEYOR; also PAVER
  (brick retailers), MILLING (flour), HEAVY HAUL (freight), TRUCKING, MOWING, TRAILER, FLATBED;
  and **`CONSTRUCTION` alone** ("`BOBS CONSTRUCTION`" is out; `BOBS EXCAVATION` is in). A name
  whose only signal is one of these is decided by Steps 3–5, not by the word.

**Step 3 — other positive evidence in the borrower's name** → `RELEVANT`,
`R_BORROWER_OTHER_EVIDENCE`. The name states, in words not on the lists, that the firm owns and
runs job-site iron: `DIRT MOVERS OF DENVER`, `TRACKHOE SERVICES LLC`, `HEAVY EQUIPMENT RENTALS
INC`, `CAT RENTAL STORE OF PUEBLO`, `EARTH WORKS UNLIMITED`. This is the step where the human
adds to the rules; it requires a **statement of activity**, not a guess from a surname, a place or
a generic word. When in doubt, it is Step 5.

**Step 4 — the name positively states a different activity** → `NOT_RELEVANT`.
- `N_OTHER_INDUSTRY_EXPLICIT`: hospital / clinic / dental / DDS / MD / law / attorney / CPA /
  restaurant / cafe / bakery (`BOBS COOKIES` with a bank lender) / church / school / daycare /
  salon / apartments / realty / property management / software / consulting / insurance / auto
  sales / auto body / car wash / farm / ranch / dairy / cattle / feeders / orchard / vineyard /
  nursery / landscaping / lawn care / trucking / freight / logistics / retail / furniture /
  jewelry / printing / staffing / a charity. An equipment or trade word from Step 2 **beats**
  this step (`SMITH FARMS EXCAVATING` is RELEVANT).
- `N_WORD_IS_DIFFERENT_SENSE`: an equipment-like word used in another sense — `BONNIES CAR
  CRUSHERS`, `ROLLER & ASSOCIATES`, `FOUNDATION FOR SENIOR CITIZENS`, `AMERICAN PRE-EMPLOYMENT
  SCREENING`, `PANHANDLE MILLING`, `REIS BRICKS AND PAVERS`, `ACTION AUTO WRECKING`, `MOD SQUAD
  DRILL TEAM`, `CRANEBROOK APARTMENTS`.

**Step 5 — otherwise** → `INSUFFICIENT_EVIDENCE` with the closest code:
`I_GENERIC_NAME` (`ABC HOLDINGS LLC`, `JMK ENTERPRISES`, `4 M INC`), `I_PERSONAL_NAME` (`CRANE,
ROBERT GALE`, `JOHN SMITH`, `SMITH JOHN A` — an individual may well finance a machine; the name
does not say), `I_AMBIGUOUS_WORD` (`BOBS CONSTRUCTION`, `WESTERN EQUIPMENT CO`, `ACE SERVICES`,
`ROCKY MOUNTAIN TRUCKING`, `SUMMIT CONTRACTORS`), `I_UNREADABLE` (`XXXX`, `########`, a string
that is not a name).

Precedence: Step 1 > Step 2 > Step 3 > Step 4 > Step 5, except that `N_WORD_IS_DIFFERENT_SENSE`
applies to the very word that would otherwise have matched in Step 2.

## 5. Reason codes

| label | reason_code | use when |
|---|---|---|
| `RELEVANT` | `R_LENDER_NAMED_MAKER` | a lender is on the named maker / captive / dealer list |
| `RELEVANT` | `R_BORROWER_EQUIPMENT_WORD` | an explicit equipment class in the borrower's name |
| `RELEVANT` | `R_BORROWER_TRADE_WORD` | an explicit trade word, including the concrete family |
| `RELEVANT` | `R_BORROWER_OTHER_EVIDENCE` | the name states the market in unlisted words (Step 3) |
| `NOT_RELEVANT` | `N_OTHER_INDUSTRY_EXPLICIT` | the name positively states a different activity |
| `NOT_RELEVANT` | `N_WORD_IS_DIFFERENT_SENSE` | an equipment-like word used in another sense |
| `INSUFFICIENT_EVIDENCE` | `I_GENERIC_NAME` | holdings / enterprises / initials — could be anything |
| `INSUFFICIENT_EVIDENCE` | `I_PERSONAL_NAME` | reads as an individual, no activity stated |
| `INSUFFICIENT_EVIDENCE` | `I_AMBIGUOUS_WORD` | CONSTRUCTION / EQUIPMENT / SERVICES / TRUCKING alone |
| `INSUFFICIENT_EVIDENCE` | `I_UNREADABLE` | garbled, truncated or OCR-damaged beyond reading |
| (founder only) | `ADJUDICATED` | the founder's review decision; `reason` carries the founder's note |

`reason` is one sentence, at most 300 characters, quoting the deciding word(s). It must not
contain a URL or refer to anything outside the row.

## 6. Worked examples

| borrower | lenders | label | reason_code |
|---|---|---|---|
| BOBS CRANES | WELLS FARGO BANK NA | RELEVANT | R_BORROWER_EQUIPMENT_WORD |
| BOBS COOKIES | TEREX CORPORATION | RELEVANT | R_LENDER_NAMED_MAKER |
| BOBS COOKIES | DEERE & COMPANY | NOT_RELEVANT | N_OTHER_INDUSTRY_EXPLICIT |
| JW FARMS | KUBOTA CREDIT CORPORATION | NOT_RELEVANT | N_OTHER_INDUSTRY_EXPLICIT |
| HERNANDEZ CONCRETE | (none) | RELEVANT | R_BORROWER_TRADE_WORD |
| BOBS CONSTRUCTION | FIRST NATIONAL BANK | INSUFFICIENT_EVIDENCE | I_AMBIGUOUS_WORD |
| CRANE, ROBERT GALE | (none) | INSUFFICIENT_EVIDENCE | I_PERSONAL_NAME |
| DIRT MOVERS OF DENVER LLC | U.S. BANK EQUIPMENT FINANCE | RELEVANT | R_BORROWER_OTHER_EVIDENCE |
| ROLLER & ASSOCIATES | KEY EQUIPMENT FINANCE | NOT_RELEVANT | N_WORD_IS_DIFFERENT_SENSE |
| ABC HOLDINGS LLC | CATERPILLAR FINANCIAL SERVICES | RELEVANT | R_LENDER_NAMED_MAKER |
| SMITH LAW OFFICES PC | WELLS FARGO BANK | NOT_RELEVANT | N_OTHER_INDUSTRY_EXPLICIT |
| JMK ENTERPRISES | 1ST SOURCE BANK, CONSTRUCTION EQUIPMENT DIVISION | INSUFFICIENT_EVIDENCE | I_GENERIC_NAME |

## 7. How the labels are used

Two independent blind Claude passes label every queue chunk, first in the pilot round (`pilot_v1`)
and then in the main round (`main_v1`). Each final row of `labels.csv` carries one of four statuses:

- `model_agreed` — both passes gave the same label and the founder did not review the row; the row
  keeps pass A's `reason_code` and `reason`.
- `founder_confirmed` — an agreed row the founder re-read in the audit sample and kept.
- `founder_adjudicated` — a disagreement the founder decided, or an agreed row the founder
  overturned; `reason_code = ADJUDICATED` and `reason` is the founder's note.
- `blind_repeat` — one pass's answer on a hidden repeat; never fitted or evaluated.

There is no pending status: `labels.csv` is not written while any disagreement lacks a founder
decision. Only `RELEVANT` / `NOT_RELEVANT` rows are resolved to a binary target for training and
evaluation; `INSUFFICIENT_EVIDENCE` is never relabelled as negative and its prevalence is reported by
region and stratum. The 200-case pilot is development data: every pilot group is assigned to train
and never enters validation or test.

## 8. Founder sign-off

- [x] The rulings in §1–§4 match the founder's intent (nothing re-opened, nothing added).
- [x] The reason-code set in §5 is complete for the pilot.
- [x] Founder audit sample: `founder_audit_per_split_stratum: 10` in `ml/configs/v1.yaml` (10 agreed rows
      re-read per split x stratum, plus every disagreement) — **confirmed, kept at 10**.
- [x] Signed off by: William Kerr  date: 2026-09-12  (`status: FROZEN`)

Answers recorded at the gate: the policy was approved as presented. No ruling was changed, no reason
code was added, and the audit sample stays at ten agreed rows per split x stratum plus every
disagreement, so `ml/configs/v1.yaml` is unchanged.
