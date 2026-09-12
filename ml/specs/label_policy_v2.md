# Label policy v2 — heavy-construction-equipment relevance of a UCC borrower

policy_version: label_policy_v2
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
  appears in the borrower's OWN name). The frozen rules implement those two routes with word lists;
  **this policy deliberately does not show you them.** You are a screener applying a criterion, and
  your agreement or disagreement with the rules is the measurement.
- **Precision over recall** is the product position (99.4% published). The ML model is a
  **review queue** over rules-rejected cases plus a second opinion on accepted ones; it **never
  replaces the rules** and never changes `scope_all`, the headline counts or the existing map.
- A borrower **never crosses CO ↔ CT**: the two registers are separate; nothing in one informs the other.
- **Each filing is judged on its own evidence** (founder ruling, 2026-09-12). The unit is the case:
  one borrower on one filing with that filing's lenders. The same firm may be `RELEVANT` on a filing
  financed by a named maker and `INSUFFICIENT_EVIDENCE` on another financed by a bank. That is
  correct, not a contradiction: you are screening a filing, not adjudicating a company. Never carry
  evidence from one row to another, and never search the chunk for other filings of the same firm.

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
Part 9).

**Ordinary world knowledge is allowed and expected; knowledge of THIS borrower is not.**
You may use general knowledge of what an industry or a well-known brand does — that some brands
build construction machines, that a trencher digs trenches, that an aerial work platform is
a machine. You may NOT use knowledge about the specific firm in the row: who owns it, what its
website says, whether it still trades, which dealership it is. That is a lookup performed from
memory, it is not reproducible, and the no-lookup rule above already forbids it.

When a world-knowledge claim is what decides a row, state it in your `reason`
("this brand builds excavators"). When you do not recognise any lender, say so in the `reason`
rather than guessing.

**The founder's rulings override your knowledge.** Only `JOHN DEERE CONSTRUCTION` and
`DEERE CONSTRUCTION` qualify — bare `DEERE`, `DEERE & COMPANY`, `DEERE CREDIT`,
`JOHN DEERE FINANCIAL` and `JOHN DEERE COMPANY` do not. Pure agriculture is a different market and
does not qualify. A bank's general equipment-finance or leasing division is a bank, not a maker,
including `1ST SOURCE BANK, CONSTRUCTION EQUIPMENT DIVISION`. Rulings above knowledge, knowledge
above nothing.
 Labels are decided from the borrower name, the lender names and the city/state shown —
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

**You are a screener, not a rulebook.** You are not being asked to reproduce any list, and there is
no list to reproduce. If your `reason` amounts to *"this word is on the list"*, you have not applied
this policy. The frozen rules are a separate instrument; where you and they differ, that difference
is the measurement this round exists to make.

**Step 1 — the lenders (Route A).** If **any** lender is a *named* heavy-construction maker,
captive or dealer → `RELEVANT`, `R_LENDER_NAMED_MAKER`, whatever the borrower is called
(a lender you recognise decides the row whatever the borrower is called: `BOBS COOKIES`
financed by a construction-equipment maker is in scope).
A lender qualifies when you recognise it as a manufacturer, a captive finance arm, or a franchised
dealer of heavy CONSTRUCTION equipment — machines that move earth, lift, pave, drill, crush or
handle material on a job site. **You are not given a list and there is no list to reproduce.**
Decide from the lender string and ordinary general knowledge of what firms make and sell. If you do
not recognise the lender, it is not evidence — say so in your `reason` and continue to Step 2.

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
- `R_BORROWER_EQUIPMENT_WORD` — the name states an explicit **machine class a job site runs**.
  You are deciding whether the word names a machine, not whether it appears on a list.
- `R_BORROWER_TRADE_WORD` — the name states an explicit **trade that owns and runs that machinery**
  on a job site — and the **CONCRETE family, which qualifies on the name alone**: concrete, cement,
  mixer, shotcrete, precast, flatwork, rebar, post-tension, curb-and-gutter, concrete pumping,
  tilt up. (This family is a founder ruling that a whole business class qualifies, not a detection
  list, and it stays.)
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
INC`, `CAT RENTAL STORE OF PUEBLO`, `EARTH WORKS UNLIMITED`. **This is the step where you add what
a word list cannot, and you must consider it on EVERY row before falling to Step 5. A row that
reaches Step 5 without Step 3 having been considered is an error.** It requires a **statement of
activity**, not a guess from a surname, a place or a generic word: a generic trade noun with no
activity (`SUMMIT CONTRACTORS`, `SENERGY BUILDERS`, `BOBS CONSTRUCTION`) is Step 5, not Step 3.
When in doubt, it is Step 5.

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
| BOBS COOKIES | a lender you recognise as a construction-equipment maker | RELEVANT | R_LENDER_NAMED_MAKER |
| BOBS COOKIES | DEERE & COMPANY | NOT_RELEVANT | N_OTHER_INDUSTRY_EXPLICIT |
| JW FARMS | KUBOTA CREDIT CORPORATION | NOT_RELEVANT | N_OTHER_INDUSTRY_EXPLICIT |
| HERNANDEZ CONCRETE | (none) | RELEVANT | R_BORROWER_TRADE_WORD |
| BOBS CONSTRUCTION | FIRST NATIONAL BANK | INSUFFICIENT_EVIDENCE | I_AMBIGUOUS_WORD |
| CRANE, ROBERT GALE | (none) | INSUFFICIENT_EVIDENCE | I_PERSONAL_NAME |
| DIRT MOVERS OF DENVER LLC | U.S. BANK EQUIPMENT FINANCE | RELEVANT | R_BORROWER_OTHER_EVIDENCE |
| ROLLER & ASSOCIATES | KEY EQUIPMENT FINANCE | NOT_RELEVANT | N_WORD_IS_DIFFERENT_SENSE |
| ABC HOLDINGS LLC | that maker's captive finance arm | RELEVANT | R_LENDER_NAMED_MAKER |
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
- [x] The reason-code set in §5 is complete.
- [x] Founder audit sample: `founder_audit_per_split_stratum: 10` — confirmed, unchanged from v1.
- [x] **World knowledge (founder ruling, 2026-09-12): ALLOWED, bounded.** General knowledge of what an
      industry or well-known brand does is permitted and expected; knowledge of the specific firm in
      the row remains forbidden by §3's no-lookup rule.
- [x] **Each filing judged on its own (founder ruling, 2026-09-12).** Recorded in §1.
- [x] Signed off by: William Kerr  date: 2026-09-12  (`status: FROZEN`)

Why v2 exists: under v1 the labeller was handed the frozen rules' own manufacturer roster and word
lists, so it executed those rules by hand. Measured on the 200-case pilot: labels matched
`heavy_row` on 100% of cases, and `R_BORROWER_OTHER_EVIDENCE` — the one code by which a screener can
add what a word list cannot — fired 0 times in 240 rows. v1 stays committed and FROZEN as the policy
those 240 rows were produced under, and as the control arm of the ablation that tests whether v2
changes anything. Rows of different policy versions are never pooled.
