# Pair decision rule — "are these two records the same firm?"

**Status: DRAFT awaiting founder sign-off. Committed BEFORE the labelling file is generated.**
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
**SAME**. *(Recommended default — this case is not yet founder-locked. Confirm or reverse it at sign-off.)*

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

**R10 — When two rules collide, the lower number wins.** When genuinely undecidable after applying all of
them, answer **UNSURE**.

---

## What you must NOT do
- Do not look anything up. No Google, no Secretary of State search. The rule is the rule; outside
  research would make the labels unreproducible and would not be available at scale.
- Do not skip ahead, re-order, or revisit an earlier answer after seeing a later pair.
- Do not try to infer what the model thinks. Nothing in the file tells you, by design.
- Do not stop mid-file and resume days later if avoidable — drift between sittings is a real effect.

## Known limitation, stated in the README
Single rater. There is no inter-rater agreement figure. As partial mitigation the file contains
**30 hidden repeated pairs**, so an intra-rater agreement number can be reported: *"re-labelled 30 pairs
blind and agreed with myself on N of 30."*
