# STATUS — UCC Entity Resolution Demo

Repo: https://github.com/drkyoyodating/UCC-Demo · Branch `main`
**Budget doctrine: NO RESET, NO SECOND ATTEMPT.** The remaining allowance is the whole project (runbook §1.12).

## Meters
| Meter | Reading |
|---|---|
| BUDGET (planned hours earned at DoD) | **14.5 / 24** |
| Modelling freeze | hour 16 |
| FOUNDER minutes spent | 0 / 240 (90 reserved for labelling) |

## Phase ledger
| Phase | Planned | Status | Gate | Audit | Commit | Deviations |
|---|---|---|---|---|---|---|
| P0 Setup | 0–0.5 | ✅ **COMPLETE** | n/a (§1.0 pre-answered) | 1 combined **PASS** | `67ebc57` | venv is `.venv/` on py3.14.7, not `ucc-venv/`; optional deps split out |
| P1 Ingest | 0.5–2 | ✅ **COMPLETE** | auto-proceed | 1 combined **PASS-w-findings** | `e6bb419` | acceptance test corrected mid-phase (see DECISIONS); pages landed as CSV.gz not NDJSON |
| P2 Normalizer + skeleton | 2–4 | ✅ **COMPLETE** | auto-proceed | 1 combined **PASS** | `2c32c40` | step 5 (boilerplate strip) added post-audit, founder-approved; walking skeleton deferred to P3 |
| P3 Corpora | 4–6 | ✅ **COMPLETE** | ✅ corpus A confirmed | 1 combined **PASS-w-findings** | `b1a1b3a` | NULL-key guard relabelled as inert after audit |
| P4 EDA | 6–7.5 | ✅ **COMPLETE** | auto-proceed | 1 **PASS-w-findings** | `e2cb8a0`+ | blocking-loss stat wrong twice, replaced with a decidable measure; record-level blocking adopted for P5b |
| P5 Splink debtors | 7.5–15 | ✅ **COMPLETE (artifacts)** | ✅ threshold 6.0 founder-decided | 3-panel: correctness+scope PASS, **honesty NO SIGN-OFF** | `03fa75a` | rule overridden; 4,217 same-address over-merges disclosed |
| P5b Splink lenders | 13.5–14.5 | ✅ **COMPLETE (artifacts)** | auto-proceed | (same panel) | `03fa75a` | threshold 8.0 = lowest INTEGER passing the bar |
| ⛔ FREEZE | 16 | — | | | | |
| P6 Evidence | 16–20 | — | **HARD STOP** (labelling) | **4** | | |
| P7 Publish | 20–24 | — | **HARD STOP** | 3 | | |

## Founder actions outstanding
- [ ] Enable GitHub Pages: Settings → Pages → branch `main` → `/docs` (2 min, needed by P7)
- [ ] Open a SECOND SESSION at the P0→P1 boundary for Tandem A
- [ ] Book the labelling window (70–100 min, must land in hours 16–20)
