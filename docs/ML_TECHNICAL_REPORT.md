# UCC relevance screener — technical report, release `80b952a3e9a8`

A machine-learning screener was built for the UCC heavy-construction-equipment demo, evaluated once
on a held-out test set under a protocol registered before the test was opened, and published. This
report states what was built, what it measured, and why the answer is what it is.

**What it is.** A classifier that reads *only the borrower and lender names* on a UCC filing — no
collateral text, no documents — and decides whether that filing is heavy-construction equipment
finance.

**What it did.** On a held-out test set it never saw during training, it made 171 suggestions and 170
were correct: design-weighted precision **0.998** [0.872, 0.995]. On filings the written rules already
accept it was right every time — precision **1.000** in both Colorado and Connecticut. It then scored
the entire population, **1,226,257** filings, refusing 5 as out of bounds.

**What it shows.** A names-only linear model reproduces a hand-built rules screen to within
statistical noise: its precision is **not distinguishable** from the rules' own 0.991 (paired
difference +0.007 [−0.111, +0.023]). That is the substantive finding — the signal needed to make this
call is carried in the names themselves. The labelling underneath it holds up independently: two blind
passes agreed 96.5% and 97.5%, hidden repeated cases came back identical 520/520, and the design
weights reconstruct every held-out population exactly.

**What is not established.** The model was built to do one thing beyond matching the rules — recover
relevant filings the rules *reject*. On that extension the test is **inconclusive, not unfavourable**:
review-queue precision reads 0.000 with an interval from 0.028 to 0.948 over 216 cases, which cannot
separate a useless queue from an excellent one, because that slice held only 8 positives to find. The
model is also lower on recall than the rules (−0.173 [−0.216, −0.046]), which is a real difference and
the reason the rules stay authoritative. The candidate is published as `experimental` — the verdict it
was given on validation, before the test set was opened. Nothing about the demo's headline counts, its
map or its `scope_all` figure changes.

## 1. What the model is for

The existing screen is a written rules filter over UCC filing text. It is precise but it is a filter:
anything it rejects is gone. The question worth asking is whether a model can find the relevant cases
the rules throw away, cheaply enough to be worth a human's time. So the model was never framed as a
replacement. It is **a review queue over rules-rejected cases, plus a second opinion on rules-accepted
ones**, and the pre-registered primary quantity is the design-weighted precision of that queue.

## 2. Data and sampling

1,226,257 filing-borrower cases (CO 905,197; CT 321,060). Splits are by **borrower group within
region**, so no borrower's filings appear on both sides of a split: 801,607 train / 183,670 validation
/ 240,980 test rows.

Labelling was drawn, not convenience-sampled. The population was divided into four estimation strata
(region × whether the rules accepted the case) and, inside those, six **screen cells** used only to
steer the draw toward cases likely to be informative. The unscreened remainder cell is always drawn at
a floor, which is what keeps the estimates valid even if the screen is poor. Each row carries the
inclusion probability of the cell it was actually drawn from.

Every estimate is design-weighted by `1 / inclusion_probability` **per row**. The stratum average
`N_h / n_h` is reported per stratum and applied to no row — that distinction matters here, because the
boundary screen puts several sampling rates inside a single stratum, so the two disagree. A guard
requires the design weights of each held-out stratum to reconstruct its population exactly; they do.

| stratum | population `N_h` | labelled `n_h` | average weight |
|---|---|---|---|
| CO:accepted | 12,703 | 120 | 105.9 |
| CO:rejected | 164,929 | 314 | 525.3 |
| CT:accepted | 1,292 | 120 | 10.8 |
| CT:rejected | 62,056 | 246 | 252.3 |

## 3. Labels

3,120 labelled rows across two rounds, and **the two rounds were produced under different
arrangements**. This is stated per round everywhere rather than pooled into one sentence, because
naming either arrangement alone would describe rows it does not cover.

- `pilot_v1`, 240 rows, policy `label_policy_v1`: model-labelled, founder-adjudicated. All 5 pass
  disagreements were decided by the founder.
- `main_v1`, 2,880 rows, policy `label_policy_v2`: model-labelled by **two independent blind passes**,
  with disagreements retained as unresolved rather than decided. All 83 disagreements are retained as
  `INSUFFICIENT_EVIDENCE` and counted in the denominator; none was adjudicated.

Label statistics: `model_agreed` 2,512; `blind_repeat` 520; `blind_unresolved` 83;
`founder_adjudicated` 5; `founder_confirmed` 0. Blind-pass agreement was 0.9654 on `main_v1`
(2,317/2,400) and 0.975 on `pilot_v1` (195/200). Hidden repeated cases came back identical on both
passes (260/260 each). **0 of the 160 rows selected for founder audit were read** — that is a gap in
the evidence, and it is published as one rather than left implicit.

Labelling used the written policy only. Nothing was looked up.

## 4. Model and threshold

TF-IDF over borrower and lender **names only** (word 1–2-grams on the normalised name, character
3–5-grams on the raw name, separate borrower and lender blocks) into logistic regression. 20,873
features, C = 10.0, seed 20260912, `StratifiedGroupKFold(5)` inside train, selected by weighted
average precision.

C = 10 sits at the top of the grid `{0.1, 1, 10}`, so the release publishes
`best_C_at_grid_boundary: true`. That flag normally means a search was cut short. Here it was
measured and it does not: extending the grid on train only, the selection metric for the shipping
variant climbs to an asymptote and never turns over — 0.9281 at C=10, 0.9414 at C=1,000, 0.9428 at
C=100,000 — so no finite grid clears the flag. The gain from C=10 to C=1,000 is 0.28 of one fold's
standard deviation, the best-to-runner-up margin is 4% of a fold's standard deviation, and out-of-fold
log-loss is *minimised* at C=300 and degrades past it. The grid was deliberately left alone and the
curve published as the answer to the flag.

Calibration is `sigmoid`, not isotonic: the Kish effective number of out-of-fold positives is 135.4
against a floor of 200 required for isotonic. Weighted ECE 0.032, decision-region ECE 0.057, so the
score is published as a calibrated probability.

The threshold was chosen on validation, **before test was opened**: 0.8612124616682861. It did not
meet the production gate, which requires weighted precision ≥ 0.95 *and* a one-sided 95% lower bound
≥ 0.95 *and* ≥ 30 Kish-effective predicted positives. Validation weighted precision was 1.000 and
Kish support 30.1, but the **lower bound was 0.816**. The candidate was therefore frozen as
`experimental` with fallback `precision_lower_bound_or_effective_support_below_floor`, and that
verdict was committed to `docs/data/ml/frozen_candidate_v1.sha256` before any test row was scored.

## 5. Evaluation protocol

Pre-registered in `ml/specs/evaluation_protocol_v1.md`. Test was scored **exactly once**: a guard file
records the run and the command refuses a second evaluation. Intervals are a stratified cluster
Bayesian bootstrap over 709 borrower clusters (8 straddling groups), 2,000 resamples, seed 20260912,
with Jeffreys pseudo-mass at prior 0.5 so a cell with no observed error stays honestly uncertain.

Because the point estimate is the unsmoothed design-weighted plug-in while the interval carries that
prior, the two disagree near the boundary. Rather than reconcile them, any such pair is marked
`(estimate outside interval)` wherever it is published.

## 6. Results

Test: 800 labelled rows, **453 resolved** (243 positive). 347 rows — 47.7% — could not be labelled
either way by two independent passes and are retained as unresolved, counted in the denominator.

### The primary, pre-registered quantity

Review-queue weighted precision over rules-rejected cases: **0.000 [0.028, 0.948]** over 216 resolved
cases *(estimate outside interval)*.

In plain terms: across 216 rules-rejected test cases, the model made **one** suggestion, and that
suggestion was wrong. The queue contained 8 true positives and the model found none of them.

### Overall, model versus the frozen rules

| | weighted precision | weighted recall | weighted F1 | TP/FP/FN/TN |
|---|---|---|---|---|
| model | 0.998 [0.872, 0.995] *(outside)* | 0.795 [0.689, 0.842] | 0.885 [0.795, 0.902] | 170/1/73/209 |
| frozen rules | 0.991 [0.952, 0.996] | 0.969 [0.821, 0.968] *(outside)* | 0.980 [0.892, 0.978] *(outside)* | 235/2/8/208 |

Paired differences (model − rules): precision +0.007 [−0.111, +0.023] — **not distinguishable**;
recall −0.173 [−0.216, −0.046] — **lower**. Two secondary comparisons, not multiplicity-adjusted.

By region: CO n=248, model wP 0.998 [0.875, 0.998], wR 0.817 [0.712, 0.871]. CT n=205, model wP 1.000
[0.523, 0.998] *(outside)*, wR 0.584 [0.391, 0.728].

By stratum, which is where the result is explained:

| stratum | resolved | positives | model weighted precision |
|---|---|---|---|
| CO:accepted | 118 | 117 | 1.000 [0.973, 1.000] *(outside)* |
| CO:rejected | 130 | 6 | 0.000 [0.001, 0.975] *(outside)* |
| CT:accepted | 119 | 118 | 1.000 [0.957, 1.000] *(outside)* |
| CT:rejected | 86 | 2 | n/a — no predicted positive at all |

## 7. Where the evidence runs out

The review queue's entire evidential base is **8 positive cases**: 6 in CO:rejected and 2 in
CT:rejected. In CT:rejected the model predicted nothing positive, so its precision there is undefined
rather than bad. An interval running from 0.028 to 0.948 is not a measurement of a model; it is a
measurement of a sample too small to distinguish a useless queue from an excellent one.

This is a denominator fact, not a modelling failure, and it was foreseeable: the same arithmetic put
the validation lower bound at 0.816 and forced the `experimental` verdict before the test was opened.

Two further constraints are structural. Roughly half the rules-rejected cases (52.9% in CO, 44.1% in
CT) could not be resolved by two independent passes reading the written policy — the filing text
frequently does not say enough to decide. And the model sees **names only**: CO's collateral field is
a 124-value category list and CT has none, so there is no collateral text to learn from.

## 8. What would change the result

A test sample drawn after freezing, stratified on the frozen decision — roughly 100 cases per region ×
rules-outcome × model-decision — would narrow the review-queue interval from about 0.59 wide to about
0.09. That is the measurement worth making, and it requires labels, not a better classifier. It was
deliberately not taken here: it is a change to the sampling contract, and opening the test set first
forecloses it.

## 9. Reproduction

Release `80b952a3e9a8`, source commit `6c204059b43634aef35a127c1c368a150fe5ef45`, clean tree. The
release id **is** the content digest of its own four manifests:

```
cat model-manifest.json threshold.json dataset-manifest.json split-manifest.json | shasum -a 256 | cut -c1-12
```

Digests of every input are committed under `docs/data/ml/`: the labels and their manifest
(`labels_v1.sha256`), the frozen candidate as pre-registered before the test run
(`frozen_candidate_v1.sha256`), and the release with its measured result (`release_v1.sha256`).
The full model card ships inside the release bundle.

Batch scoring over all 1,226,257 candidates produced a proposed review queue of 5,101 rules-rejected
cases. Given the measurement above, that queue's precision is **not established**; it is published as
a candidate list, not as a result.
