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
**The rules were RESTRUCTURED after the pre-label and cross-phase audits**, with ZERO labels in existence
(verified: 330 rows, 0 non-blank). The first draft is `a725496`; the R11-R13 additions `54b076c`; this
restructure the commit below. None of it can affect which pairs were drawn — strata are selected on match
weight and record patterns, never on the rule. Same reasoning: they cannot affect which pairs were drawn.
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

## How to work a pair: two steps

### STEP 1 — Clean up the fields before you compare anything
These are *pre-processing*, not tie-breakers. Apply them first, always, before any rule below.

- **P1 — Address formatting is not an address difference.** `WARS RD`/`WARD RD`, `EAST`/`E`,
  `CO RD`/`COUNTY RD`, `HAMPSEN`/`HAMPDEN`, `P.O. BOX`/`PO BOX`, `1200 S. TOWNSEND`/`1200 S TOWNSEND`,
  a ZIP+4 written `80110-2109` or `801102109` — one address written two ways. Treat as the SAME address.
- **P2 — A ZIP that disagrees while the name and street agree is a data error.** Ignore it and decide on
  name and street. (A ZIP+4 vs its 5-digit form is not a disagreement at all — see P1.)
- **P3 — A wrong state code is a data error.** `CT` on a record whose city, street and ZIP are plainly
  Colorado: ignore the state field.
- **P4 — A blank field is not evidence either way.** Decide on the fields that are present. Never read a
  missing address as "a different address".

### STEP 2 — Then apply the rules, in this priority order

**R0 — Identical name AND identical address → SAME.** The trivial case. Stated so it is never in doubt.

**R1 — Plainly different firms with no shared address → DIFFERENT.**
`NORWEST BANK COLORADO / 129 South 3rd St` vs `FIRST SECURITY BANK / 201 South Third St` → **DIFFERENT**.
This is the commonest case in the file. It needs no further thought.

**R2 — Is one name a damaged, shortened or abbreviated version of the other? → SAME.**
`AMEERICAN NATIONAL BANK`, `AMERICAN NATINAL BANK`, `AMERICAN NATIONAL BANJ` are all the same firm as
`AMERICAN NATIONAL BANK`. `COOPERS CONST` = `COOPERS CONSTRUCTION`. `COLO NATIONAL BANK` = `COLORADO
NATIONAL BANK`. `U S BANK` = `US BANK`. Truncations too: `COLORADO BANK AND TRUST CO OF LA JUN` = `... LA
JUNTA`. **Test this BEFORE R5** — a spelling variant of one name is not a different name.

**R3 — Suffix-only difference → SAME.** `ACME EXCAVATING LLC` vs `ACME EXCAVATING INC`. Entities convert
legal form and keep filing. *(Founder-locked, question 8a.)*

**R4 — Same name, same city, different address → SAME.** A yard and an office, or a firm that moved.
Different PO Box numbers count as a different address, so this rule applies to them too.
*(Founder-confirmed at sign-off.)*

**R5 — Same name, different Colorado cities, no shared address → DIFFERENT.**
`JOHNSON CONSTRUCTION` in Denver and in Pueblo. *(Founder-locked, question 8c.)*

**R6 — DIFFERENT NAMES AT THE SAME ADDRESS → DIFFERENT.**
**The most important rule in the document.** A shared address is not evidence of one firm: registered
agents, franchise headquarters, medical groups, law offices and rural family properties all put unrelated
filers at one address. `ARROWHEAD TRAVEL PLAZA` and `ERNST BROTHERS` at one address are **DIFFERENT**.
`INTERNATIONAL KINGS TABLE 105` and `GILCHRIST FOOD GROUP` at one address are **DIFFERENT**.
*Exception, and it is the only one:* numbered outlets of one chain — `COUNTRY HARVEST BUFFET 103` vs
`COUNTRY HARVEST BUFFET 500` — are **SAME**, because the **name** says so, not the address.
*R2 is tested first: `AMERICAN NATIONAL BANJ` at the same address as `AMERICAN NATIONAL BANK` is a typo,
not a different name, and is SAME.*
*The model is known to over-merge on exactly this pattern. Apply R6 strictly — measuring that is the point.*

**R7 — Distinct legal entities of one corporate family → DIFFERENT.**
`WELLS FARGO BANK NA` vs `WELLS FARGO EQUIPMENT FINANCE INC` — separately chartered. `X, A DIVISION OF Y`
vs `Y` → **DIFFERENT**. *(Founder-locked, question 8b.)*

**R8 — TWO DIFFERENT person-names at one address → DIFFERENT.** Spouses, parent and child
(`LESTER HASART` / `DIXIE HASART`) are different borrowers. A person and their farm
(`LESTER HASART` vs `TOP END FARMS`) → **DIFFERENT** unless the names overlap.
**The SAME person at one address is SAME** — `HASART JEROLD G` vs `HASART JEROLD G` is one person, and a
spelling difference in the address is handled by P1.

**R9 — Successor / renamed entity → DIFFERENT**, unless one name visibly contains the other.
`NORWEST BANK COLORADO` and `WELLS FARGO BANK` are **DIFFERENT** here: we label records as filed, not
corporate history.

**R10 — Precedence.** Step 1 always runs first. Then, among the rules above, **the lower number wins** —
with the ordering as written, which already puts R2 (typos and abbreviations) ahead of R6 (same address).
When genuinely undecidable after all of it → **UNSURE**, and move on without agonising.

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
