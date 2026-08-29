# P4 — Exploratory data analysis

Every number below is followed by the P5 design decision it changes. Nothing here is reported for its own sake.

## DEBTORS (P5) — `corpus_debtors_eq`

- **37,587 rows → 26,926 distinct raw names → 24,488 distinct `name_clean`.** Normalisation alone collapsed 2,438 variants (9.1%) before any model runs.
- Suffix distribution (top 8 of 43): `(none)` 15,552, `INC` 11,464, `LLC` 6,219, `COMPANY` 718, `CO` 613, `PC` 540, `CORPORATION` 514, `COMPANY INC` 352
  → *P5: suffix is a comparison feature, not part of the blocking key. A suffix-only difference gets a mild discount, never a penalty.*
- **5,146 keys already have ≥2 rows by exact `name_clean` match** — that is the floor Splink must beat, not its result. Largest: `COLLINS CONSTRUCTION` ×177 (3 ZIPs), `CREEKSTONE DEVELOPMENT` ×158 (1 ZIPs), `SEMA CONSTRUCTION` ×139 (1 ZIPs), `S AND S HOMES` ×126 (1 ZIPs), `TRINITY LEASING CAPITAL` ×124 (1 ZIPs)
- Completeness — address1 99.4%, city 99.4%, zipcode 99.2%.
  → *P5: high enough that address belongs in the comparison set and ZIP is a viable blocking key. If ZIP had been sparse, blocking on it would silently drop the missing rows from every comparison.*

### Blocking cardinality — `corpus_debtors_eq`

| rule | blocks | comparisons | largest block | pairs in largest |
|---|---:|---:|---:|---:|
| `zipcode` | 986 | 3,565,664 | 706 | 248,865 |
| `substr(name_clean,1,4)` | 7,362 | 827,699 | 556 | 154,290 |
| *(no blocking — for scale)* | 1 | 706,372,491 | 37,587 | 706,372,491 |

- **Prefix-rule escape: 4,712 distinct-key pairs differ inside the first four characters yet agree once spaces are collapsed** (the `ACM EXCAVATION` / `ACME EXCAVATING` shape). The prefix rule cannot propose them; 124 of those are rescued by the ZIP rule, leaving **4,588 reachable by neither**.
  → *P5: this is the measured lower bound on blocking loss — recall is reported as conditional on the union of the two rules, and this number is what a third rule would have to be worth.*

## LENDERS (P5b) — `corpus_lenders_eq`

- **44,919 rows → 3,848 distinct raw names → 3,174 distinct `name_clean`.** Normalisation alone collapsed 674 variants (17.5%) before any model runs.
- Suffix distribution (top 8 of 26): `(none)` 29,856, `CO` 6,230, `NA` 3,280, `INC` 2,292, `COMPANY` 1,333, `CORPORATION` 749, `LLC` 522, `CORP` 477
  → *P5: suffix is a comparison feature, not part of the blocking key. A suffix-only difference gets a mild discount, never a penalty.*
- **1,485 keys already have ≥2 rows by exact `name_clean` match** — that is the floor Splink must beat, not its result. Largest: `WAGNER EQUIPMENT` ×6334 (2 ZIPs), `VECTRA BANK COLORADO NATIONAL ASSOCIATION` ×1739 (60 ZIPs), `AMERICAN NATIONAL BANK` ×1423 (34 ZIPs), `BANK OF COLORADO` ×1315 (34 ZIPs), `COLORADO EAST BANK AND TRUST` ×973 (8 ZIPs)
- Completeness — address1 91.1%, city 91.5%, zipcode 91.2%.
  → *P5: high enough that address belongs in the comparison set and ZIP is a viable blocking key. If ZIP had been sparse, blocking on it would silently drop the missing rows from every comparison.*

### Blocking cardinality — `corpus_lenders_eq`

| rule | blocks | comparisons | largest block | pairs in largest |
|---|---:|---:|---:|---:|
| `zipcode` | 1,412 | 24,128,705 | 5,743 | 16,488,153 |
| `substr(name_clean,1,4)` | 904 | 49,590,742 | 6,680 | 22,307,860 |
| *(no blocking — for scale)* | 1 | 1,008,835,821 | 44,919 | 1,008,835,821 |

- **Prefix-rule escape: 619 distinct-key pairs differ inside the first four characters yet agree once spaces are collapsed** (the `ACM EXCAVATION` / `ACME EXCAVATING` shape). The prefix rule cannot propose them; 219 of those are rescued by the ZIP rule, leaving **400 reachable by neither**.
  → *P5: this is the measured lower bound on blocking loss — recall is reported as conditional on the union of the two rules, and this number is what a third rule would have to be worth.*

- ⚠ **Comparison space warning.** This corpus has 44,919 rows but only 8,922 distinct (name, address, city, zip) RECORDS — a 5× row-level redundancy, because one lender files thousands of times (`WAGNER EQUIPMENT` ×6,334). Blocking on raw rows costs ~74M comparisons against the debtor corpus's ~4.4M.
  → *P5b: resolve DISTINCT PARTY RECORDS, not rows, then map canonical ids back to rows for the league table. Same answer, ~25× less work, and it removes the duplicate-row mass that would otherwise dominate the match-weight histogram and flatter the high-weight labelling stratum.*

## Context that shapes the write-up

- EQUIPMENT filings by `filingtype`: `ucc` 35,281, `efs` 6,111. → *The register is not UCC-only; the write-up must say 'lien register', and the five-year lapse premise governs only the `ucc` rows.*
- Non-ASCII names in both corpora combined: **0**. → *Confirms the decision not to build a Unicode confusable map: a stage that provably fires ~never is padding. This is the measured number, not an assumption.*
- Stability holdout reserved: 3,688 debtor rows (seeded, reproducible). → *P6 run 1 uses base only; run 2 uses all rows.*
