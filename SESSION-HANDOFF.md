# SESSION HANDOFF — UCC Entity Resolution Demo
**Complete state as of 2026-08-30. Read this FIRST. It supersedes all earlier handoffs.**

Repo: https://github.com/drkyoyodating/UCC-Demo (public, `main`) · Dir `/Users/user/Downloads/UCC-Demo`
Python: `./.venv/bin/python` (3.14.7, splink 4.0.16, duckdb 1.4.5, pandas 2.3.3 — pins are load-bearing)
Runbook: `UCC_DEMO_RUNBOOK.md` (**gitignored — never publish**, it carries interview framing and a draft email)

---
## 1. WHAT THIS IS AND WHO IT IS FOR
A public portfolio demo proving entity-resolution capability to **Tex**, a lender-intelligence data
company whose spec reads *"entity resolution that turned 48M raw names into 9.1M canonical firms"* across
310+ sources. Founder: Will Kerr. The job posting explicitly wants someone **adept with AI-first tools
(Claude, Cursor) and pays for the subscriptions** — so AI-in-the-loop is the demonstrated skill, not
something to apologise for. Document the methodology proudly and accurately.

**Terminal budget: no token reset, no second attempt.** Founder is present continuously until done.

**Founder's working model, stated by him:** the agent generates, he reviews audit reports for variance and
makes the executive call. He has been right on every call this session — the premises model, entity type
in the dedup key, the name-string matching rule, killing placeholder rows. **Bring him findings and a
recommendation, not questions you can answer yourself.**

---
## 2. THE SCOPE — settled after many rounds. Do not re-litigate.
**Heavy construction equipment finance only. Colorado + Connecticut. 1990 onward.**

Two independent routes qualify a filing (`src/heavy_filter.py`):
- **ROUTE A** — the LENDER is a heavy-construction manufacturer, captive or dealer. ~60 brands:
  Caterpillar, John Deere Construction & Forestry, Komatsu, Kubota, CNH, Case, New Holland, Volvo,
  Hitachi, Liebherr, Doosan, Kobelco, JCB, Terex, Genie, JLG, Manitowoc, Grove, Link-Belt, Sany,
  Takeuchi, Bobcat, Wacker Neuson, Vermeer, Ditch Witch, Astec, Gradall, Gehl, Manitou, Bomag, Wirtgen,
  Atlas Copco, Epiroc, Sandvik — plus Colorado dealers **Wagner Equipment** (Cat's CO dealer), Faris
  Machinery, Honnen, 4 Rivers, Power Equipment.
- **ROUTE B** — a heavy-construction equipment or trade word appears in the BORROWER'S OWN NAME,
  whole-word anchored, **plurals and possessives accepted** (`BOBS CRANES`, `SHIFTS EXCAVATORS`,
  `JIMS CRANE`). Includes the **CONCRETE family** by founder's ruling — a concrete outfit runs mixers,
  pumpers and boom trucks, so `concrete / cement / mixer / shotcrete / precast / flatwork / rebar /
  post-tension / curb-and-gutter / concrete pumping` qualify on the name alone.

**EIGHT WORDS WERE CUT after auditing every one against real borrower names — do not re-add:**
`FOUNDATION` (4,647 hits, almost all charities — *Saint Joseph Hospital Foundation*), `DERRICK` and
`AUGER` (personal names — *Derrick LeRoy Tadlock*, *Cameron Auger DDS*), `ROLLER` (roller hockey),
`WRECKING` (auto salvage), `CRUSHER` (car crushers; `CRUSHING` kept — that one is aggregate),
`SCREENING` (pre-employment), `CONVEYOR`.

### Completeness — hard line, enforced in `src/build_scope.py`
Required: **(person OR business name) + address + loan year + route A or B.**
**Lender MAY be blank** — banks will not disclose and cannot be cold-called; the borrower can.
Instant discard: no address · placeholder address · **both** person and business absent · not heavy construction.

### Jurisdictions cut, and why (each is a line in the write-up)
| cut | reason |
|---|---|
| **NYC ACRIS** | ingested all 11,035,386 party rows to find out: **0.17% machinery**, and it is laundry/knitting equipment. Collateral decodes to FIXTURE FILING 1.74M + COOPERATIVE 2.2M (co-op apartments). Structural, per **§9-501(a)(1)** — after Article 9's 2001 revision county recorders hold fixture and real-property filings, not equipment finance |
| **Oregon** | ingested 220,515 rows: **no borrower column at all.** A lender with no borrower cannot be contacted or verified |
| **Philadelphia** | 8 machinery records |
| **Connecticut** | **CUT THEN REINSTATED.** Cut for having no collateral field — reinstated because **Route B never needed one**: it reads the borrower's name, and CT publishes both party names |

---
## 3. THE CORPUS AS BUILT
| region | rows | filings | borrowers | route A | route B | 2013+ | span |
|---|---:|---:|---:|---:|---:|---:|---|
| CO | 88,162 | 85,612 | 26,436 | 71,033 | 29,801 | 58,621 | 1990–2026 |
| CT | 17,173 | 10,989 | 5,120 | 11,820 | 6,702 | 16,205 | 1990–2026 |
| **total** | **105,335** | **96,601** | **31,538** | | | | |

**⚠ REBUILT 2026-08-30 after the filter was tightened.** The previous figures (137,236 rows /
44,478 borrowers) included bank leasing arms, machine-tool dealers, pure-agriculture dealers,
auto-salvage yards, hardscape paver retailers, flour mills and freight haulers. Strict pull
precision went **72.19% → 99.99%**. Any figure quoting 137,236 or 44,478 is stale.

**→ 49,959 entities at 2.75 loans each** (CO 42,123 · CT 7,836). Tables: `scope_co`, `scope_ct`,
`scope_all`, `co_entities`, `co_machinery_loans` in `ucc.duckdb` (gitignored).

**The funnel that IS the pitch:** 8,384,455 Colorado rows in → **238,374** machinery party records →
**97.16% of the register discarded.** Not "I ingested 8M rows" — *"I can tell you which 3% matter."*

---
## 4. THE FIVE FACTS THAT WOULD BE EXPENSIVE TO REDISCOVER
1. **Colorado stopped coding collateral after 2012.** `EQUIPMENT` runs 1990–2012 (2011: 2,940 · 2012: 998
   · **2013: 1**) while total filings ROSE to 134,391 in 2025. Post-2013 only agricultural categories
   survive, because **EFS** filings are legally required to carry collateral and UCC-1s are not.
2. **The inference that recovers it, VERIFIED:** 3,848 lenders wrote categorised EQUIPMENT liens; **747
   still file after 2013, accounting for 136,416 filings — 3.3× the entire historical EQUIPMENT
   population.** *Who lent* is the proxy for *what the collateral is*. This is the demo's best idea and it
   is why post-2012 data is KEPT (founder confirmed his earlier "cut post-2012" is superseded).
3. **The debtor model was an address matcher wearing a name matcher's costume.** Exact-name `m=0.002741`
   → a total name mismatch cost 0.016 bits → 73.7% of merges shared an identical address. Two defect
   classes: **C1** 4,217 same-address/dissimilar-name (97.6% wrong) and **C2** 4,625 same-address/
   similar-name family members (100% of the [10,999) band).
4. **PREMISES GROUPS — founder's model, and it is right.** `WOOD DONNA L` / `WOOD DONALD J` at one PO box
   are **two legal borrowers, one operation** (married, co-signed). Do NOT merge, do NOT discard. Third
   layer: entity (identity) · **premises (co-location)** · lender→borrower. Converts the model's hardest
   failures into a feature and sidesteps an ambiguity no algorithm can resolve.
5. **No phone numbers exist** in any of the four registers. Name + address only.

---
## 5. MODEL STATUS — read this before changing anything
| model | precision | recall |
|---|---|---|
| shipped baseline `resolve.py` @6.0 | 0.500–0.509 | 0.703–0.730 |
| **`src/variant_combo.py` → `combo_pf` @4.0 (RECOMMENDED)** | **1.000** (Wilson 0.901–1.000) | **0.946** |

`combo_pf` = Strategy B's name floor (`jw≥0.92` OR shared 4-char token set) **AND** Strategy D's
person/org gate, applied as a **veto** on the shipped score. **R5 violations 0 · C1 defect class 0 ·
largest cluster 0.072%.**

### ⚠ THE FINDING THAT MATTERS MOST — five agents reported 1.000 on broken models
Wave 2's selection lead verified instead of trusting and found that **`addr_blind`, `comparison` and
`ensemble` each merge 2,000 pairs that violate R5** (identical name, different city, nothing shared) —
`HYDE PARK OF LAS VEGAS` in Aspen merging with the one in Las Vegas. **Not one of the 381 labels
exercises R5**, so every agent measured 1.000 on a model breaking a signed rule 2,000 times.
**Lesson: a model's score is only as good as the rules the label set can see.**
Also: wave 1's `person` gazetteer is **partly label-fitted** (`MERIDIAN` is not a given name). Prefer
`combo_pf`; `name_floor` @4.0 is the clean-provenance fallback with identical measured numbers.

### Known instrument defects — FIX BEFORE PUBLISHING ANY NUMBER
- **`src/score.py` is non-deterministic**: 7 of 70 test pairs map to several record combinations and
  `drop_duplicates("pair_id")` keeps whichever the parallel join emitted first. Baseline recall flips
  between 0.703 and 0.730 across identical runs. Join on `unique_id`, not text.
- `score.py` claims `evaluate.py`'s estimator and actually computes raw `tp/(tp+fp)`; `EDGES` is defined
  and unused; it pools targeted strata that `evaluate.py` forbids pooling.
- The 30 hidden repeats leaked across the train/test split until 10:12 — **fixed** in `labels_split.py`,
  but every variant fitted before then was scored on a contaminated split (24 of 70 test pairs).

---
## 6. LABELLING
**`docs/UCC_labelling_v4.xlsx` — 1300 rows, CO 650 / CT 650, labels BLANK, with the founder now.**
Carries loan year, loan count and lender per side. Two question types:
- **ENTITY** "same firm?" → `SAME` / `DIFFERENT` / `UNSURE`
- **PREMISES** "one operation?" → `ONE-OP` / `SEPARATE` / `UNSURE` (green rows)

Rules live in `docs/decision_rule.md` (signed) and `docs/LABELLING_BRIEF.md`. **Identity matching per the
founder:** name is one string split on spaces, **middle initials ignored, forwards or backwards is the
same party** — `HOWARD JOHN F` = `JOHN HOWARD`. Company not person: dash between surnames
(`Stutsman-Gerbaz`), any trade word (`Hernandez Excavating`), possessive-plural (`Cohen's`, `Spencers`);
`O'Brian` is a person. **Entity type is identity: LLC ≠ INC.** Addresses expand fully
(`RD`=`ROAD`, `HWY`=`HIGHWAY`, `S`=`SOUTH`); same address if street number matches and the abbreviation
expands to the same string. **Never generate cross-jurisdiction pairs** — *"there are hundreds of
businesses called YoYo in different states."*

**Prior round (381 labels, superseded scope but methodologically valid):** 30/30 intra-rater,
98.3% three-way agreement, κ=0.973. The variance auditor's verdict was **"consistency-validated, not
accuracy-validated"** — two model passes agreeing proves determinism, not correctness. A human pass is the
only thing that breaks that correlation. Founder is running v4 through a second model outside this shell
plus manual review, which is the right answer.

---
## 7. PHASE STATE
| phase | state |
|---|---|
| P0 setup · P1 ingest · P2 normalizer · P3 corpora · P4 EDA · P5/P5b resolution | ✅ complete, audited |
| P6 evidence | ✅ 381 labels done; **v4 1300-row round in progress with the founder** |
| P7 publish | ⏳ **NOT BUILT.** `docs/index.html` has 0 charts, 0 tables, 0 scripts |
| P8 stretch · P9 skill closure | not started |

**P7 as scoped** (founder-directed): views = active equipment financiers (inference-recovered) · lender
league table · entity timeline · refi window (**must group by `masterdocumentid`** — continuation and
termination are separate amendment rows, NOT flags) · **lender→borrower bipartite graph** · **premises
groups**. Plus an **interactive page** the founder asked for: a draggable match-weight threshold that
live-updates precision, recall, cluster count and the league table. Single self-contained HTML, D3 from
cdnjs, no server. **He dictates the final display — do not build it unilaterally.**

### Publication blockers still open (from the 43-agent foundation sweep)
- `README.md` is **31 bytes** while three documents cite it as containing caveats it does not.
- `docs/eda.md` record-level figures were computed on a 4-column record key that P5 replaced with 6.
- `labels_key.csv` was never committed despite a commit titled "keys published".
- Retracted claims still live in `DECISIONS.md` §319-322 and in `resolve.py` docstrings.
- `SESSION-HANDOFF.md` previously published `1.000/0.974` computed on the leaky split — corrected here.

---
## 8. TRAPS THAT HAVE ALREADY COST TIME
- `predict(threshold_match_weight=X)` **FILTERS**, does not merely retain. `-10` silently dropped 595,382 pairs.
- **`pandas.to_parquet` fails** — pyarrow is deliberately absent. Write parquet via DuckDB.
- **DuckDB `USING SAMPLE` binds to the table scan, not the filtered result.** Push filters into a subquery
  or it silently returns almost nothing. This bit twice.
- **Do not double-escape regex for DuckDB** — pass the pattern verbatim or `\b` and `\s` break silently.
- DuckDB UDFs returning NULL need `null_handling="special"`. `role` is a reserved word.
- macOS `sed` has no `\b`. Use Python for word-boundary edits.
- `git add -A` sweeps up another session's in-flight files. Commit by explicit path.
- **I overstated pre-registration strength FOUR times** (fallback clause, "lowest threshold", "degeneracy
  fixed", R4 commit ordering) — all retracted in `DECISIONS.md`. **Watch for this reflex.**

---
## 9. FILE MAP
`src/`: `ingest.py` · `ingest_multi.py` (CT+PHL) · `ingest_nyc_or.py` · `normalize.py` · `corpus.py` ·
`eda.py` · `resolve.py` (frozen shipped model) · `heavy_filter.py` (**the scope filter**) ·
`build_scope.py` (**builds `scope_*`**) · `build_entities.py` (entity→loans) · `build_sheet_v4.py` ·
`make_workbook_v4.py` · `evaluate.py` · `score.py` (⚠ defective, see §5) · `labels_split.py` ·
`variant_combo.py` (**the recommended model**) · `variant_*.py` (9 experiments)
`docs/`: `index.html` (placeholder) · `decision_rule.md` · `LABELLING_BRIEF.md` · `eda.md` ·
`UCC_labelling_v4.xlsx` (**live, with founder**) · `UCC_labelled_v3.xlsx` (592, prior scope)
Root: `DECISIONS.md` (**the running log — read P5/P6**) · `STATUS.md` · `ucc.duckdb` (gitignored)

---
## 10. NEXT ACTIONS IN ORDER
1. Founder returns `UCC_labelling_v4.xlsx` (1300 labels).
2. **Fix `score.py` first** (§5) — it is non-deterministic and every number depends on it.
3. Re-score `combo_pf`, `name_floor` and the baseline on the new labels. Report precision/recall with
   intervals and denominators, and **check R5 violations explicitly** — the label set must now be able to
   see them.
4. Close the publication blockers (§7), README first.
5. Build P7 views + the interactive page **to the founder's direction**.
6. P8, then P9 skill closure (classification is the one listed skill with zero coverage).
7. Draft the email — **the founder sends it, never the agent.**

---
# 11. REMAINING WORK — corrected 2026-08-30, read this before planning anything

**The founder's assumption was "P7 and P8 are all that's left." That is wrong, and the correction
matters: P5/P5b MUST BE RE-RUN, and it gates everything downstream.**

## Why the models are stale
The shipped models were trained on `corpus_debtors_eq` (37,587 rows) and `corpus_lenders_eq` (44,919) —
**the OLD Colorado-only, EQUIPMENT-collateral-only corpus.** Every cluster, prediction and precision
figure in `parquet/` and `models/` rests on that.

The scope is now `scope_all` — **137,236 rows across CO + CT, 49,959 entities** — built on the two-route
heavy-construction filter, with **Connecticut reinstated** and **post-2012 kept**.

**These do not overlap cleanly.** The trained models have never seen Connecticut, and they include
Colorado records the current filter excludes. **You cannot publish views over entities the model never
resolved.**

## The accurate remaining list
| | work | state |
|---|---|---|
| **P5 / P5b** | Re-run resolution on `scope_all` (CO + CT) | ✅ **DONE 2026-08-30.** `src/build_scope_corpus.py` (new, additive) builds `corpus_scope_all`; `run_p5_scope_all.py` resolves it. 46,540 records from 98,571 rows · 5.1M scored pairs · 20,766 clusters · 14,487 singletons · largest 0.625% → non-degeneracy **PASS**. `combo_pf` still needs its component parquets rebuilt on this corpus |
| **P6** | Re-evaluate against the 1300 labels. **Check R5 violations explicitly** | `score.py` determinism ✅ **FIXED** (was `drop_duplicates("pair_id")` keeping whichever row the parallel join emitted first; now `GROUP BY pair_id` with `max(match_weight)`, verified byte-identical across 3 runs, recall stable at 0.730). Scoring outstanding |
| **P7** | The views + the interactive page. `docs/index.html` is still a placeholder: 0 charts, 0 tables, 0 scripts | **not started** |
| **P8** | Stretch — largely overtaken. Its items were the full-state lender pass and "P5b if it slipped"; both are moot now | mostly moot |
| **P9** | Skill closure. **Classification is the only listed skill with zero coverage** | not started, genuinely optional |
| **—** | **Publication blockers** from the 43-agent sweep: `README.md` is **31 bytes** while three documents cite it as carrying caveats; `docs/eda.md` record-level figures were computed on a 4-column record key P5 replaced with 6; retracted claims still live in `resolve.py` and `DECISIONS.md` §319-322 | **BLOCKS PUBLICATION** |

## Dependency order — do not reorder
```
fix score.py  ->  P5/P5b re-run on scope_all  ->  P6 re-score  ->  P7 views  ->  publish
```
**The re-run is the gate. Nothing downstream is real until the models match the corpus.**

## Effort
P5/P5b re-run ~1h · P6 re-score ~30 min once labels land · P7 ~3–4h · publication blockers ~1h.
P8 mostly moot. P9 optional — it only matters if *"none of the listed skills is missing"* has to be
defensible in full.

## On the visual layer — division of labour
**The agent builds it; the founder directs it.** He has been the one catching what is wrong all session —
the premises model, entity type in the dedup key, the name-string matching rule, killing the placeholder
rows, the concrete ruling, spotting that Route B brings Connecticut back. What is needed from him at P7 is
the same: **which views earn their place, what the page leads with, and what gets cut.** Do not build the
display unilaterally.
