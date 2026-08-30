# SESSION HANDOFF — UCC Entity Resolution Demo
**Written 2026-08-30 mid-session. Read this FIRST, then `UCC_DEMO_RUNBOOK.md` §1.0 (locked decisions).**
Repo: https://github.com/drkyoyodating/UCC-Demo (public, `main`). Working dir `/Users/user/Downloads/UCC-Demo`.
Python: `./.venv/bin/python` (3.14.7). Never `pip install` outside `requirements*.txt`.

---
## 1. WHAT THIS IS
A portfolio demo proving entity-resolution capability to **Tex**, a lender-intelligence data company
(their spec: *"entity resolution that turned 48M raw names into 9.1M canonical firms"*, 310+ sources).
Founder is Will Kerr. The demo must be **live, public, honest, and machinery-relevant**.

**Terminal budget: there is no token reset and no second attempt.** Finish or it does not ship.

---
## 2. STATUS
| Phase | State |
|---|---|
| P0 setup · P1 ingest · P2 normalizer · P3 corpora · P4 EDA · P5/P5b resolution | ✅ complete, audited |
| P6 evidence | ✅ 381 labels in, evaluated. **Model improvement done — see §4** |
| P7 publish | ⏳ scope rewritten (§5). NOT built. `docs/index.html` is still a placeholder |
| P8 stretch · P9 skill closure | not started |

**Live but unbuilt:** the published page has **0 charts, 0 tables, 0 scripts**.

---
## 3. THE FIVE THINGS THAT MATTER MOST
1. **Colorado stopped coding collateral after 2012.** `EQUIPMENT` runs 1990–2012 (2011: 2,940 filings ·
   2012: 998 · **2013: 1**), while total filings ROSE to 134,391 in 2025. Post-2013 the only categorised
   collateral is agricultural, because EFS filings are legally required to carry it and UCC-1s are not.
2. **The inference that fixes it, and it is VERIFIED.** 3,848 lenders wrote categorised EQUIPMENT liens;
   **747 still file after 2013, accounting for 136,416 filings — 3.3× the entire historical EQUIPMENT
   population.** They are Tex's exact universe: Caterpillar Financial, John Deere Construction & Forestry,
   Kubota Credit, Komatsu Financial, CNH Industrial, **Wagner Equipment Co** (Cat's Colorado dealer).
   *Who lent* is the proxy for *what the collateral is*. This is the demo's best idea.
3. **The debtor model was an address matcher wearing a name matcher's costume.** Exact-name `m=0.002741`
   meant a total name mismatch cost 0.016 bits, so 73.7% of merges shared an identical address.
   Two defect classes: 4,217 same-address/dissimilar-name (97.6% wrong) and 4,625 same-address/
   similar-name family members (100% of the [10,999) weight band).
4. **FIXED — see §4.** Precision 0.482 → **1.000**, recall 0.692 → **0.974** on held-out labels.
5. **PREMISES GROUPS — founder's idea, 2026-08-30, and it is the right model.** `WOOD DONNA L` /
   `WOOD DONALD J` at one PO box are **two distinct legal borrowers, one operation.** Do NOT merge them
   and do NOT discard them. Add a **third layer**: entity (identity) · **premises group (co-location)** ·
   lender→borrower. This converts the model's hardest failures into a feature and sidesteps an ambiguity
   no algorithm can resolve. **Must appear in both the workbook and the P7 views.**

---
## 4. THE MODEL RESULT — `variant_person` @ threshold 4.0
| model | precision | recall |
|---|---:|---:|
| shipped baseline @6.0 | 0.482 | 0.692 |
| **`src/variant_person.py` @4.0** | **1.000** | **0.974** |

Strict dominance on both axes. Strategy: detect person-like names, require the **given name** to agree,
not just the surname. Fit on `labels_train.csv` (246), scored on held-out `labels_test.csv` (132).
- **0.95 is reached as a POINT ESTIMATE** on held-out labels (1.000) and on an independent 180-pair audit
  of real output (0.967, 0.929–0.985). Pooled 212/218 = 0.973.
- **It is NOT certified at 95% confidence** — 74 held-out labels bound it at 0.908. Certifying needs
  ~337 audited merges drawn from what THIS model merges. One labelling sitting. **This is the cheapest
  remaining action.**
- **Rejected:** `variant_comparison` looked good (0.974) but **half its merges land where no label exists**;
  it merged bare surname `THURSTON` across five towns. Do not ship it.
- **Do NOT implement an "enumeration veto"** — `docs/decision_rule.md` R6 makes numbered chain outlets
  (`COUNTRY HARVEST BUFFET 103/500`) correctly SAME.

---
## 5. P7 SCOPE (rewritten v2.6, founder-directed)
**Machinery and construction ONLY.** Dropped: IRS liens (157,558), hospital liens (32,847), consumer goods,
Snap-on (hand tools), AGCO (pure agriculture), De Lage Landen (office/medical leasing).
**Kept lender set** (construction/heavy-machinery OEM captives + dealers): Caterpillar, Komatsu, Bobcat,
Terex, JLG, Vermeer, Ditch Witch, Hitachi, Liebherr, Doosan, Takeuchi, Manitowoc, JCB, Kubota, CNH,
Case, New Holland, Volvo, John Deere Construction & Forestry, Wagner Equipment, plus name patterns
MACHINERY / CONSTRUCTION EQUIP / HEAVY EQUIP / CRANE.
Volumes: **CO 697 construction lenders → 17,350 debtors; 23,776 construction-named debtors.
CT 127 lenders → 3,388 debtors; 10,190 construction-named debtors.**

**Views to build (none exist yet):**
1. Active equipment financiers, recovered by inference (**the headline**)
2. Lender league table · 3. Entity timeline · 4. Refi window (must group by `masterdocumentid` —
   continuation/termination are SEPARATE amendment rows, not flags)
5. Lender→borrower bipartite graph · 6. **Premises groups (§3.5)**
**Interactive page**, founder-requested: draggable match-weight threshold that live-updates precision,
recall, cluster count and the league table. Single self-contained HTML, D3 from cdnjs, no server.

---
## 6. JURISDICTIONS (4 — Philadelphia EXCLUDED, only 8 machinery records)
| | endpoint | status |
|---|---|---|
| Colorado | `data.colorado.gov` wffy-3uut / 8upq-58vz / ap62-sav4 / 4am6-w6u4 | ✅ 8,384,455 rows |
| Connecticut | `data.ct.gov` xfev-8smz — both parties + addresses + `dt_lapse` on one row | ✅ 844,675 |
| NYC ACRIS | `data.cityofnewyork.us` nbbg-wtuz (11.0M parties) + sv7x-dduq | ⏳ ingesting, slow |
| Oregon | `data.oregon.gov` 2kf7-i54h — **secured parties ONLY, no debtor field** | ⏳ |

**NEVER generate cross-jurisdiction pairs.** Founder: *"there are hundreds of businesses called YoYo in
different states"* — a shared name across registers is not evidence of a shared firm. One sheet per
jurisdiction, sampled and labelled independently.

**Only 2 of 50 states publish this free and complete** (CO, CT). *"The generalisation problem is
commercial, not technical: the ingest code ports in a day; the access does not."*

---
## 7. KEY FILES
`src/`: `ingest.py` · `ingest_multi.py` (CT+PHL) · `ingest_nyc_or.py` · `normalize.py` · `corpus.py` ·
`eda.py` · `resolve.py` (frozen shipped model) · **`variant_person.py` (the winner)** · `evaluate.py` ·
`score.py` (held-out scorer) · `labels_split.py` · `make_label_file*.py` · `make_workbook.py` ·
`build_machinery_areas.py`
`docs/`: `index.html` (placeholder) · `decision_rule.md` (13 rules, signed) · `LABELLING_BRIEF.md` ·
`eda.md` · `UCC_labelling_CO-complete.xlsx` (**381 labels, DONE**) · `labels_key*.sha256`
Root: `DECISIONS.md` (the running log — **read the P5/P6 sections**) · `STATUS.md` · `ucc.duckdb` (gitignored)

---
## 8. TRAPS THAT HAVE ALREADY COST TIME
- `predict(threshold_match_weight=X)` **FILTERS**, it does not merely retain. Passing -10 silently dropped
  595,382 pairs and made blocking look broken.
- `pandas.to_parquet` fails — **pyarrow is deliberately not installed.** Write parquet via DuckDB.
- DuckDB `USING SAMPLE` binds to the **table scan, not the filtered result**. Push the filter into a subquery.
- `git add -A` will sweep up another session's in-flight files. Commit by explicit path.
- The runbook is **gitignored** (it carries interview framing and a drafted email). Never publish it.
- **I overstated pre-registration strength FOUR times** (fallback clause, "lowest threshold", "degeneracy
  fixed", R4 commit ordering). All retracted in `DECISIONS.md`. Watch for this reflex.

---
## 9. NEXT ACTIONS, IN ORDER
1. Finish NYC + Oregon ingest.
2. Build the per-jurisdiction workbook (CO / CT / NYC / OR), construction-only, **plus premises-group pairs**.
3. Founder labels it.
4. Build P7 views + the interactive page.
5. Re-audit, publish, draft the email (**founder sends it, not the agent**).
