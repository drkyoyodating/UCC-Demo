# Evaluation protocol v1 — pre-registered before any model is trained

Status: FROZEN on commit. Changing anything below after `freeze-candidate` has run
requires a new protocol version and a fresh, untouched test set. Labels are
model-labelled, founder-adjudicated, and every report says so.

## 1. What is being measured
A screener over UCC borrower/lender names for heavy-construction-equipment
relevance (pack §1 label meaning). The product position is precision over recall;
the model is a review queue over rules-REJECTED cases plus a second opinion on
rules-accepted ones. It never replaces the frozen rules.

## 2. Populations, strata, weights
- Prediction unit: one case = (region, file_id, borrower_key) with its full sorted
  lender set (Codex §4). Region-scoped `group_id` from Plan A controls leakage.
- Sampling strata: region × frozen-baseline accepted/rejected, written `CO:accepted`,
  `CO:rejected`, `CT:accepted`, `CT:rejected` (contract K2). Each labelled case carries
  `sampling_stratum`, `inclusion_probability` and `labelling_round` (`pilot_v1`, `main_v1`).
- Evaluation population for a split S ∈ {validation, test}: every candidate whose
  group is assigned to S. For stratum h in S: `N_h` = candidates in S with stratum h;
  `n_h` = ALL non-repeat labelled cases drawn in S with stratum h, INSUFFICIENT_EVIDENCE
  included; `w_h = N_h / n_h`, which equals 1/inclusion_probability because validation and
  test contain no pilot case (K12) and main_v1 draws within (split, stratum) (K13).
  Rates use the resolved cases with these weights, so they estimate performance on the
  RESOLVABLE population of S (cases a screener could label RELEVANT or NOT_RELEVANT).
  The design-weighted INSUFFICIENT_EVIDENCE share of S and of the model's predicted
  positives is reported next to every rate.
- The run stops if `N_h / n_h` disagrees with 1/inclusion_probability for any validation or
  test row, or if a stratum with candidates in S has no labelled case.
- Training weight per labelled TRAIN case: `1 / inclusion_probability`, normalised to mean 1 over fitted rows. Each row carries the rate of the cell it was actually drawn from, so a pilot row and a main row in the same stratum keep their own rates. `N_train_h / n_h` is wrong here and only looked right while every stratum had a single cell at a single rate. The validation and test guard that `w_h = N_h / n_h` equals `1/inclusion_probability` is applied to those two splits ONLY: it is false in TRAIN by construction, and silently so.

## 3. Label resolution
Only rows whose `adjudication_status` is `model_agreed`, `founder_confirmed` or
`founder_adjudicated` count (contract K3). `blind_repeat` rows (`is_repeat = True`) are
removed before anything is fitted or measured, and any other status stops the run.
RELEVANT → 1, NOT_RELEVANT → 0. INSUFFICIENT_EVIDENCE is excluded from fitting and from
every rate but stays in `n_h`; its design-weighted share is reported by region, stratum and
in total.

## 4. Metrics
Weighted precision = Σ w·TP / Σ w·(TP+FP); weighted recall = Σ w·TP / Σ w·(TP+FN);
F1 is the harmonic mean of the two. Raw TP/FP/FN/TN counts and sample sizes are always
printed beside the weighted figures. No unweighted pooling of the enriched sample is
ever presented as population performance; the total unresolved share is N_h-weighted.

## 5. Uncertainty
Stratified cluster Bayesian bootstrap with Jeffreys pseudo-counts: within each stratum h,
every (stratum, group_id) cluster gets a Gamma(1,1) weight and each structurally possible
(label, model decision, rules decision) cell plus the INSUFFICIENT_EVIDENCE cell gets
Gamma(0.5,1) pseudo-mass; cell shares are scaled to N_h; P/R/F1, the model − rules
deltas, the review-queue rates and the unresolved share are recomputed from the same
2,000 draws; percentile 95% interval around the design-weighted point estimate. The
pseudo-mass keeps a zero false-positive cell in a heavily weighted stratum uncertain.
Validated by Monte Carlo (300 samples) on a population with rare, heavily weighted false
positives: coverage >= 0.90 for precision and recall is a test (measured 0.937 / 0.953).

## 6. Model selection and threshold — decided BEFORE test access
- Hyperparameters: C ∈ {0.1, 1, 10}, `StratifiedGroupKFold(5, shuffle, seed)` inside
  TRAIN, selected by mean out-of-fold weighted average precision (ties → smaller C). Sample
  weights are normalised to mean 1 within each fit, because LogisticRegression multiplies
  C by the scale of the weights. A best C at either end of the grid is flagged at the
  founder gate; widening the grid is a founder decision.
- Calibration: isotonic when the Kish effective number of out-of-fold positives,
  (Σ w)² / Σ w², is ≥ 200, otherwise sigmoid, fitted on group-separated out-of-fold scores
  within TRAIN. Scores are called probabilities only if the validation weighted ECE ≤ 0.10 AND the weighted ECE among scores ≥ 0.3 (≥ 20 cases) ≤ 0.15; otherwise
  `score_type = raw_score`.
- Deployment candidate: the borrower+lender variant (Codex §7). The borrower-only
  variant and the frozen rules are comparison rows.
- Threshold objective on VALIDATION: maximise weighted recall subject to weighted
  precision ≥ 0.95 with ≥ 30 labelled validation cases predicted positive, counted as the Kish effective number (sum w)^2 / sum w^2 of predicted positives, and the one-sided 95% lower bound of weighted precision (section 5 bootstrap) ≥ 0.95. The threshold is selected with the unweighted count; the Kish count
  and the lower bound are then checked at the selected threshold, and a failure makes the
  candidate experimental (it does not start a second search). If no threshold qualifies
  the candidate is `experimental` (threshold = best weighted F1 with ≥ 30 predicted
  positives) and the rules remain authoritative. The validation figures at the chosen
  threshold are selection-biased upward and are labelled so; only TEST is unbiased.
- Primary reported quantity: design-weighted precision of the review queue (model-suggested cases among rules-rejected strata) with its 95% interval; overall P/R and the model − rules deltas are secondary.
- The Kish count and the lower bound tighten context pack §6; the founder confirms them at
  the gate before TEST (Plan B Task 16).

## 7. The single test run
`evaluate-final` scores TEST exactly once with the frozen candidate. It writes a guard
file and refuses to run again without `--force-i-know`; a forced re-run keeps the
first result on disk and logs the override. The frozen candidate's digest is committed
to `docs/data/ml/frozen_candidate_v1.sha256` BEFORE the run. Any later model change is
a new release evaluated on a fresh benchmark, never on this test set again.
Every `freeze-candidate` run adds one to `ml/artifacts/frozen/VALIDATION_LOOKS.json`.
From the 3rd look on, the validation precision is not quoted as an estimate and a fresh
validation draw is required before quoting it.

## 8. What is reported
TP/FP/FN/TN, weighted P/R/F1 with intervals, rules vs model on identical cases, the
paired deltas with intervals and a direction for each (higher / lower / not
distinguishable), the review-queue block (the primary quantity), per-region and
per-stratum tables, the design-weighted unresolved share overall and among model
suggestions, calibration diagnostics (overall and decision-region ECE), the validation
threshold's one-sided lower bound and Kish support, the number of validation looks,
sample sizes and denominators. The two model − rules comparisons are secondary and are
not multiplicity-adjusted, and the report says so.
