# Pair decision rule — "are these two records the same firm?"

**Status: SIGNED OFF by the founder 2026-08-30, R4 confirmed SAME.**

**Precise pre-registration chain, because the distinction matters and an earlier version of this line
overstated it.** The DRAFT — carrying R1, R2, R3 and R5–R10 in final form — was committed at `a725496`,
six minutes BEFORE the labelling file and key hash at `f43942d6`. That ordering is genuine and checkable.
**R4 is the exception:** it stood in the draft as "recommended, not locked", and the founder's
confirmation was written into this file in the SAME commit as the labelling file. R4's lock therefore does
NOT strictly predate the sample. It does not bias the sample — strata are selected on match weight and
record patterns, never on the decision rule — but the git-evidence chain for R4 alone is weaker than for
the other nine rules, and saying so is cheaper than being caught not saying it.
**R11–R13 were added later still**, after the pre-label audit found the gaps and BEFORE any label existed
(commit below). Same reasoning: they cannot affect which pairs were drawn.
This commit's timestamp is the pre-registration evidence for Ship Gate 2. Once signed, the rule is
followed **even when it feels wrong**; deviations are noted in a comment column, never improvised.

You will see two records side by side with: name, address, city, state, ZIP. You will **not** see match
weights, model predictions, cluster ids, or which stratum a pair came from. Label each pair
**SAME / DIFFERENT / UNSURE**. `UNSURE` is a real answer — use it rather than guessing; it is reported.

---

## The question you are answering
> *Do these two records refer to the same legal or commercial entity — the firm a lender would treat as
> one borrower?*

Not "are these related", not "do these share an owner". **One borrower.**

---

## Rules, in priority order

**R1 — Suffix-only difference at the same address → SAME.**
`ACME EXCAVATING LLC` vs `ACME EXCAVATING INC` at one address is **SAME**. Entities convert between legal
forms routinely and keep filing. *(Founder-locked, question 8a.)*

**R2 — Distinct legal entities of one corporate family → DIFFERENT.**
`WELLS FARGO BANK NA` vs `WELLS FARGO EQUIPMENT FINANCE INC` is **DIFFERENT** — separately chartered
entities. Any commercial roll-up is presentation, shown separately on the league table, never in the
ground truth. Same for `X, A DIVISION OF Y` vs `Y`: **DIFFERENT**. *(Founder-locked, question 8b.)*

**R3 — Same name, different Colorado cities, no shared address → DIFFERENT.**
`JOHNSON CONSTRUCTION` in Denver and in Pueblo is **DIFFERENT** unless something else ties them.
*(Founder-locked, question 8c.)*

**R4 — Same name, same city, different street address → SAME.**
A firm with a yard and an office, or one that moved. `ACME CONTRACTING LLC` at two Lakewood addresses is
**SAME**. *(FOUNDER-CONFIRMED at sign-off, 2026-08-30: SAME.)*

**R5 — DIFFERENT NAMES AT THE SAME ADDRESS → DIFFERENT, unless the names themselves say otherwise.**
**This is the most important rule in the document.** A shared address is *not* evidence of one firm:
registered agents, franchise headquarters, medical groups, law offices and rural family properties all
put unrelated filers at one address. `ARROWHEAD TRAVEL PLAZA` and `ERNST BROTHERS` at one address are
**DIFFERENT**. Numbered outlets of one chain — `COUNTRY HARVEST BUFFET 103` vs `COUNTRY HARVEST BUFFET
500` — are **SAME** (the name says so, not the address).
*The model is known to over-merge on exactly this pattern; the rule exists to measure that, so apply it
strictly.*

**R6 — Individuals.** Two person-names at one address (spouses, parent and child, e.g.
`LESTER HASART` and `DIXIE HASART`) are **DIFFERENT** — different borrowers. A person and their farm
(`LESTER HASART` vs `TOP END FARMS`) are **DIFFERENT** unless the names overlap.

**R7 — Obvious typo / OCR variant of the same name → SAME.**
`AMEERICAN NATIONAL BANK`, `AMERICAN NATINAL BANK`, `AMERICAN NATIONAL BANJ` are all **SAME** as
`AMERICAN NATIONAL BANK`. Truncations too: `COLORADO BANK AND TRUST CO OF LA JUN` = `... LA JUNTA`.

**R8 — Abbreviation of the same name → SAME.** `COLO NATIONAL BANK` = `COLORADO NATIONAL BANK`;
`U S BANK NATIONAL ASSOCIATION` = `US BANK NATIONAL ASSOCIATION`.

**R9 — Successor / renamed entity → DIFFERENT**, unless one name visibly contains the other.
`NORWEST BANK COLORADO` and `WELLS FARGO BANK` are **DIFFERENT** here even though one became the other:
we are labelling records as filed, not corporate history.

**R11 — Address FORMATTING is not an address difference.** `WARS RD` / `WARD RD`, `EAST` / `E`,
`CO RD` / `COUNTY RD`, `HAMPSEN` / `HAMPDEN`, a ZIP+4 written `80110-2109` or `801102109` — these are one
address written two ways, not two addresses. Resolve them to "same address" FIRST, then apply R1 or R5.
R4 is for a genuinely different *location*, not for a typo.

**R12 — A divergent state code on an otherwise-Colorado record is a data error. Ignore it.**
If one record says `CT` while its city, ZIP and street are plainly Colorado, treat the state field as
noise and decide on everything else. Do not let it trigger R3.

**R13 — Different PO Box numbers are a genuine address difference**, so R4 governs: same name, same city,
different box → **SAME**. A PO Box on one record and a street address on the other, same name and city,
is also **SAME** under R4. Different name at either → R5 applies and the answer is **DIFFERENT**.

**R10 — When two rules collide, the lower number wins.** When genuinely undecidable after applying all of
them, answer **UNSURE**.

---

## What you must NOT do
- Do not look anything up. No Google, no Secretary of State search. The rule is the rule; outside
  research would make the labels unreproducible and would not be available at scale.
- Do not skip ahead, re-order, or revisit an earlier answer after seeing a later pair.
- Do not try to infer what the model thinks. Nothing in the file tells you, by design.
- Do not stop mid-file and resume days later if avoidable — drift between sittings is a real effect.

## Two known imperfections in this file, found by the pre-label audit and disclosed rather than patched
- **The 30 hidden repeats are byte-identical to their originals, in the same A/B order.** A labeller who
  recognises one may recall their earlier answer instead of re-deriving it, which would inflate the
  intra-rater agreement figure. The number is published WITH this caveat rather than presented as a clean
  reliability estimate.
- **One record appears in two different pairs** (`NORWEST BANK COLORADO / 1740 BROADWAY / DENVER 80274`,
  in P122 and P162). The de-duplication keyed on record id, and two distinct source rows printed
  identically. 2 pairs of 300 (0.67%); it does not bias either label, only introduces slight
  non-independence between those two. Not regenerated: re-drawing would invalidate the pre-committed key
  hash — a real loss of evidential strength — to fix a 0.67% independence issue. Disclosed instead.

## Known limitation, stated in the README
Single rater. There is no inter-rater agreement figure. As partial mitigation the file contains
**30 hidden repeated pairs**, so an intra-rater agreement number can be reported: *"re-labelled 30 pairs
blind and agreed with myself on N of 30."*
