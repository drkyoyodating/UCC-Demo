# Capability Evidence — UCC Entity Resolution Demo

**Prepared for résumé drafting. Every figure below is measured, not estimated, and
reproducible from the repository.**

Repository: https://github.com/drkyoyodating/UCC-Demo
Live demo: https://drkyoyodating.github.io/UCC-Demo/
Built: 2026-08-30, approximately nine working hours.

---

## 0. What the project is, in one paragraph

US states publish UCC-1 financing statements — a public record created every time a
lender takes a security interest in a borrower's equipment. The registers are large,
uncategorised and mostly irrelevant to any given question: Colorado alone carries
2.6 million filings covering everything from office copiers to cattle. This project
ingests 3.4 million filings across five jurisdictions, reduces them to the
**80,971 filings by 22,774 borrowers** who verifiably finance heavy construction
equipment, resolves duplicate name variants into roughly **21,000 distinct firms**,
and publishes the result as an interactive map that re-runs the filter live in the
browser. The qualifying criteria were measured as a classifier and iterated from
**72.19% to 99.4% precision** across three adjudicated rounds.

**The pitch in one line:** not "I ingested 8 million rows" but *"I can tell you which
3% matter, and I can show you how I know."*

---

## 1. Skill-by-skill evidence

> **Note for the drafter:** the original skills list was supplied in a table whose
> rows 1, 3 and 5 lost their labels in transit. Those three are reconstructed from
> context and marked **[inferred]** — confirm the exact wording before publishing.

### 1 · Data ingestion and acquisition **[inferred label]** — DEMONSTRATED

| | |
|---|---:|
| source filings ingested | **3,432,167** |
| jurisdictions evaluated | **5** |
| jurisdictions retained | 2 (Colorado, Connecticut) |

Ingested from Socrata APIs into a local DuckDB snapshot. **Three jurisdictions were
ingested and then cut, each for a stated structural reason** — this is the part worth
emphasising, because negative results that are *reasoned* are stronger evidence than
a large row count:

- **New York City** — all **11,035,386** party rows ingested specifically to find out.
  Result: 0.17% machinery, and the collateral decodes to fixture filings and
  co-operative apartment liens. The cause is structural, not incidental: after
  Article 9's 2001 revision, **UCC §9-501(a)(1)** leaves county recorders holding
  fixture and real-property filings, not equipment finance. No amount of filtering
  recovers a market that is not in the register.
- **Oregon** — 220,515 rows ingested. No borrower column exists at all. A lender with
  no identifiable borrower cannot be contacted or verified, so the source is unusable
  regardless of volume.
- **Philadelphia** — 8 machinery records.

**Résumé angle:** demonstrates evaluating a data source on evidence and discarding it
with a documented reason, rather than using whatever is easiest to obtain.

---

### 2 · Record linkage / fuzzy matching / entity resolution — DEMONSTRATED
*(named verbatim in the job posting's table-stakes list)*

Splink 4.0.16 over DuckDB, probabilistic linkage with EM parameter estimation.

| | |
|---|---:|
| party records resolved | **47,114** |
| distinct entities produced | **~21,000** |
| scored pairs | **5.1 million** |
| largest cluster | **0.62%** of corpus — passes the pre-registered <1% non-degeneracy bar |

The substantive point is not that Splink was run, but that its **failure modes were
diagnosed**. The shipped baseline model was found to be *an address matcher wearing a
name matcher's costume*: exact-name agreement carried m=0.002741, meaning a total
name mismatch cost only 0.016 bits, and **73.7% of merges shared an identical
address**. Two defect classes were isolated and quantified — 4,217 same-address /
dissimilar-name pairs (97.6% wrong) and 4,625 same-address / similar-name family
members.

**A custom normalisation layer** (`src/normalize.py`) handles the name-string problem
underneath: legal-suffix stripping into a separate comparison feature, dotted-variant
collapse (`L.L.C.` → `LLC`), and a surname rule where entity type is treated as
identity (`ACME EXCAVATING LLC` ≠ `ACME EXCAVATING INC`). It has a 50-case fixture
whose expected values were **hand-derived by three independent passes before the
implementation existed**, agreeing 50/50 — so the test is an external check rather
than a transcript of the code.

**Résumé angle:** probabilistic record linkage at scale, plus the ability to audit a
model that reports good numbers for the wrong reason.

---

### 3 · Evaluation methodology and labelling **[inferred label]** — DEMONSTRATED

This is the skill that most separates the project from "ran a linkage library over a
weekend."

- **A signed decision rule written before labelling**, with rules R0–R10 in strict
  priority order, plus explicit address-expansion, name-normalisation and
  person-vs-company tables (`docs/LABELLING_CRITERIA.md`, 22 KB).
- **1,300 labelled pairs** across two question types (entity identity; premises
  co-location), stratified by region and by match difficulty.
- **Fixture expectations derived independently of the implementation** — three blind
  passes, unanimous on all 50 normaliser cases.
- **A measured instrument defect, found and fixed.** The scoring script was
  non-deterministic: a labelled pair matched records by *text*, so several record
  combinations shared one `pair_id` and `drop_duplicates` kept whichever the parallel
  join emitted first. Baseline recall flipped between **0.703 and 0.730 across
  identical runs**. Fixed by aggregating deterministically; verified byte-identical
  across three runs.
- **A blind spot in the label set itself, identified and closed.** An earlier
  381-label round exercised rule R5 *zero times*, so five independent agents measured
  1.000 precision on three models that were each violating a signed rule 2,000 times.
  The lesson — *a model's score is only as good as the rules the label set can see* —
  drove the v4 design, which carries 94 R5-shape rows.

**Résumé angle:** designs evaluation that can detect its own blind spots, and treats a
measurement instrument as something to be validated before its output is trusted.

---

### 4 · SQL / relational analytics — DEMONSTRATED
*("solid SQL/relational databases" in the posting)*

SQL is the backbone, not an afterthought:

- The two-route qualifying criteria run as **regex-matched SQL across 2.6M filings
  joined to 2.0M debtors and 2.1M secured parties**.
- `src/build_scope_corpus.py` unions **two structurally incompatible register
  schemas** — Colorado exposes `debtorid` / `actiontype` / `recordstatus`; Connecticut
  publishes **no party identifier at all** and uses `cd_flng_type` / `lien_status`.
  A deterministic surrogate key is synthesised for Connecticut so re-pulls are
  reproducible.
- Named analytical views ship with the product and **the page prints each query above
  its result** — filings by qualifying criterion, top lenders, top borrowers, filings
  by decade, cross-register matches.

**Résumé angle:** relational modelling across heterogeneous public schemas, not just
querying a clean warehouse.

---

### 5 · Search / retrieval surface **[inferred label]** — DEMONSTRATED

The published page is the retrieval surface: select a jurisdiction, select a location,
read the filings at it, and pivot from any lender to every borrower it financed. The
filter **executes live in the browser on each interaction** rather than replaying a
precomputed result.

---

### 6 · Graph construction over resolved entities — DEMONSTRATED
*("the product shape itself — a graph, not a report")*

The lender→borrower bipartite graph is built over **resolved entities**, not raw
strings, and is rendered as interaction rather than as a node diagram: clicking any
lender highlights every dot it financed across the map and dims the rest. **3,421
distinct lenders** across **22,774 borrowers**.

A structural subtlety worth stating in an interview: the party-record corpus collapses
rows where one borrower faced several secured parties, so **the graph must be built
from the filing-level table, not the corpus** — a detail recorded in the module
docstring so a future reader cannot get it wrong.

---

### 7 · Classification — DEMONSTRATED
*(previously the one listed skill with zero coverage)*

The qualifying criteria **are** a binary classifier over party rows, and they are
measured, iterated and reported like one.

| round | precision | what changed |
|---|---:|---|
| initial | **72.19%** | criteria as originally written |
| after round 1 | **98.0%** | bank leasing arms, machine tools, agriculture, auto salvage removed |
| after round 2 | **99.4%** | dual-line makers, substring defects, surname frames closed |

**Recall was measured separately and honestly: ~86%, 95% CI 53–97%** — the interval is
wide because the pool-wide rate rests on a single miss in a 120-row random stratum,
and the README says so rather than quoting the point estimate alone.

Every precision gain came from **using the failures as the blueprint**. Representative
defects found and closed:

- Generic phrases `EQUIPMENT FINANCE` / `EQUIPMENT LEASING` admitted **698 of 2,600
  audited party sides (26.85%)** — U.S. Bank, KeyBank, Wells Fargo. That is how a
  dairy, a tree service and an eye-surgery practice entered a heavy-construction
  corpus.
- `WRECKING` was documented as removed in three places and was **still live in the
  code**, pulling 105 auto-salvage yards.
- A **surname rule** derived from the domain: a surname can be singular but never
  plural, so `CRANE, ROBERT GALE` is a person while `BOBS CRANES` is a firm.
- A dealer name re-admitted an excluded manufacturer as a **substring**
  (`CONSTRUCTION EQUIPMENT COMPANY` matching inside `JOHN DEERE CONSTRUCTION EQUIPMENT
  COMPANY`).

**An architectural correction worth citing:** the filter was initially built with
reject lists. Measurement showed the denylist killed **23,714 distinct lender strings
of which a tightened criteria list would have admitted only 22** — it was compensating
for loose criteria — and worse, it had a failure mode tight criteria cannot have:
checked before the whitelist, `BANK` beat every named captive whose funding is
bank-administered, **silently deleting 406 real rows** such as *"Ditch Witch Financial
Services, a program of Bank of the West."* The model was rebuilt as pure inclusion
criteria with no reject list at all.

**Résumé angle:** builds a classifier, measures both precision *and* recall, reports
the uncertainty honestly, and re-architects when measurement says the design is wrong.

---

### 8 · Cross-source linkage — DEMONSTRATED, INCLUDING THE RESTRAINT

The pipeline generalises across two independently-structured sources: Connecticut was
ingested, filtered and resolved through the same criteria as Colorado despite
publishing an incompatible schema and no party key.

**The stronger result is a negative one.** Fifteen borrower names appear in *both*
registers — `ACEVES CONCRETE LLC` files in Aurora, Colorado *and* Meriden,
Connecticut; `ELITE EXCAVATION & CONSTRUCTION LLC` in both. **They are deliberately
not merged**, and the demo says so on the page with the query beside it: a shared name
across two states is not evidence of a shared firm, so the pipeline never generates a
cross-jurisdiction pair.

**Résumé angle:** anyone can join two registers on a name string. Finding the linkage,
declining it, and publishing the reason is the judgement a data buyer is actually
paying for.

---

## 2. Cross-cutting evidence

**Reproducibility.** Two independent pulls over the same snapshot produce
**byte-identical output**, verified by checksum. Two separate non-determinism bugs
were found and fixed — both the same defect class, a `drop_duplicates` retaining
whichever row a parallel join emitted first.

**Correctness under measurement, repeatedly.** The habit that recurs throughout: when
something is checkable, check it rather than reason about it. The colour palette was
run through a contrast validator rather than judged by eye (and the muted state
colours failed four of six checks — both rendered as grey). A hillshade was verified
against a synthetic cone and found to be **lit from the wrong direction**, inverting
hills into hollows — invisible on screen, caught by test.

**Scope discipline.** Eight qualifying words were cut after auditing each against real
borrower names: `FOUNDATION` (4,647 hits, almost all charities), `DERRICK` and `AUGER`
(personal names), `ROLLER`, `WRECKING`, `CRUSHER`, `SCREENING`, `CONVEYOR`.

**Honest caveats published, not buried.** The README carries the recall interval and
why it is wide, the ZIP-centroid placement approximation, the ~3% of rows relocated
in-state for plotting, and the deliberate refusal to link across registers.

---

## 3. Suggested résumé lines

Pick per the target role; each is supported above.

- Ingested and evaluated **3.4M public lien filings across five US jurisdictions**,
  cutting three on documented structural grounds including a statutory finding
  (**UCC §9-501(a)(1)**) that removed an 11M-row source from consideration.
- Built a two-criteria classifier over public registers and **raised precision from
  72% to 99.4%** across three adjudicated measurement rounds, tracing every defect to
  the exact term that admitted it.
- Measured **recall (~86%)** as well as precision and published the confidence
  interval and its limitation rather than the point estimate alone.
- Resolved **47,114 party records into ~21,000 canonical firms** with Splink,
  diagnosing a shipped baseline that was matching on address while appearing to match
  on name (**73.7% of merges shared an identical address**).
- Found and fixed a **non-deterministic evaluation harness** whose recall flipped
  between 0.703 and 0.730 across identical runs, then proved the corrected pipeline
  byte-identical across independent runs.
- Identified a **blind spot in a ground-truth label set** that had allowed three
  models to score 1.000 precision while violating a signed rule 2,000 times each; the
  replacement label set exercises that rule 94 times.
- Re-architected a filter from **exclusion lists to pure inclusion criteria** after
  measurement showed the exclusions were 99.9% redundant and were silently deleting
  406 valid records.
- Unified **two incompatible state register schemas**, one of which publishes no party
  identifier, behind a single deterministic pipeline.
- Shipped an **interactive published demo** that re-executes the qualifying criteria
  live in the browser against a 3.4M-row snapshot.
