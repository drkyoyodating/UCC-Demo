# Labelling brief — read once, then work from the workbook

**File:** `UCC_labelling.xlsx` · **381 pairs** · **~110 minutes** · Tab 1 = the rule, Tab 2 = the pairs.

---

## THE MISSION

For each row you answer **one question**:

> **Do these two records refer to the same firm — the entity a lender would treat as one borrower?**

You are building the **ground truth**. The model already guessed on every one of these pairs; you never
see its guess. Where you and it disagree, that disagreement *is* the error rate.

Your labels are the measuring instrument. Which means **consistency matters more than being right about
the real world.** A rule applied mechanically 381 times produces a meaningful number. Good instincts
applied inconsistently produce noise.

---

## THE THREE LABELS

| Label | Means |
|---|---|
| **`SAME`** | One firm, two records. Allowing typos, truncations, abbreviations, and legal-form changes. |
| **`DIFFERENT`** | Two firms. **Including when they share an identical address.** The label people under-use, and the one that matters most here. |
| **`UNSURE`** | **The rule does not determine an answer.** Not "I'm not confident". Not "I don't know if these are really related in reality". |

### The single most important sentence in this document

**If the rule gives you an answer, write it — even if it feels wrong.**

The rule *is* the definition of truth for this exercise. Doubting reality is fine. Overriding the rule is
not, because then the published method is not the method that was used, and the pre-registration becomes
a description of something that did not happen.

Each `UNSURE` costs a data point. **Aim for under ~5%** (≈19 rows). Above that the estimate weakens.

---

## THE PROCEDURE — names first, always

This is the speed unlock. **~90% of pairs are decidable from the two names alone, in about 3 seconds.**
Do not read the addresses first. They will mislead you.

**Step 1 — Read the two names. Is one a damaged version of the other?**
Typos (`AMERICAN NATIONAL BANJ`), truncations (`...CO OF LA JUN`), abbreviations (`COLO` = `COLORADO`,
`CONST` = `CONSTRUCTION`, `U S` = `US`), or just a suffix change (`LLC` vs `INC`).
→ **`SAME`**. Stop. Do not look at the address.

**Step 2 — Are they plainly different businesses?**
`NORWEST BANK COLORADO` vs `FIRST SECURITY BANK`. `ARROWHEAD TRAVEL PLAZA` vs `ERNST BROTHERS`.
→ **`DIFFERENT`**. Stop. **Do not look at the address. It does not matter.**

**Step 3 — Only if the names are identical or near-identical, look at the address.**

| Situation | Answer |
|---|---|
| Same city, genuinely different street | **`SAME`** — moved, or a yard and an office |
| Different Colorado cities, nothing shared | **`DIFFERENT`** |
| Address differs only in spelling or formatting | not a difference → **`SAME`** |
| ZIP disagrees but name and street agree | data error, ignore the ZIP → **`SAME`** |
| A field is blank | not evidence either way — decide on the fields present |

**Step 4 — Still undetermined? → `UNSURE`.** Move on. Do not agonise.

---

## THE FIVE TRAPS IN THIS DATA

**1. The identical address is bait.** About a quarter of your pairs share an address exactly. Registered
agents, franchise HQs, medical groups and family farms all park unrelated filers at one address.
**Different names + same address = `DIFFERENT`.** The model gets this wrong 4,217 times, and your labels
are what prove it.

**2. One exception to that.** Numbered outlets of a chain — `COUNTRY HARVEST BUFFET 103` vs
`COUNTRY HARVEST BUFFET 500` — are **`SAME`**, because *the name* says so, not the address.

**3. Corporate families are `DIFFERENT`.** `WELLS FARGO BANK NA` vs `WELLS FARGO EQUIPMENT FINANCE INC`
→ **`DIFFERENT`**. `X, A DIVISION OF Y` vs `Y` → **`DIFFERENT`**. Separately chartered, separate borrowers.

**4. Successors are `DIFFERENT`.** `NORWEST BANK COLORADO` vs `WELLS FARGO BANK` → **`DIFFERENT`**, even
though one became the other. You are labelling records as filed, not corporate history.

**5. People are individuals.** Two **different** people at one address (spouses, parent and child)
→ **`DIFFERENT`**. A person vs their farm → **`DIFFERENT`** unless the names overlap. But the **same**
person twice → **`SAME`**.

---

## WORKING EFFICIENTLY

- **Type `S` / `D` / `U` + Enter.** The dropdown autocompletes. Do not reach for the mouse.
- **Target ~15 seconds average.** Many are 3 seconds. If one takes more than 30, it is an `UNSURE` —
  mark it and move.
- **Never go back and revise.** Later pairs must not change earlier answers; that is drift and it
  corrupts the sample.
- **Never look anything up.** No Google, no Secretary of State search. It makes the labels
  unreproducible and is not available at scale — a reviewer would discount the entire number.
- **Use `note` sparingly** — only when a pair genuinely surprised you. Not required.
- **One sitting if you can.** Drift between sessions is real. If you must break, break at a round number
  and note where.
- **Yellow = unlabelled.** When nothing is yellow, you are done.

---

## WHEN YOU FINISH

**File → Download → Microsoft Excel (.xlsx)** or **Comma-separated values (.csv)** — either works.
Send it back and I compute:

- precision per weight band, weighted back up by the population sizes recorded before the draw
- the precision curve at thresholds 4 / 6 / 7 / 8 / 10
- error rates for the two known defect classes, reported separately and never pooled
- intra-rater agreement from the 30 hidden repeated pairs
- recall, conditional on blocking and on the candidate set

---

**You will hit some genuinely hard ones. That is deliberate** — about a quarter are the same-address cases
where the model is weakest, and they are the highest-value rows in the file. Hesitating there means the
sample is doing its job.
