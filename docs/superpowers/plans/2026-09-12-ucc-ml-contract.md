<!-- Published copy of the binding interface contract. Absolute local paths are rewritten to placeholders. -->

# UCC ML Lab — binding interface contract v1 (2026-09-12)

**Precedence: this contract > `context_pack.md` > the Codex plan.** It resolves every conflict found
between the three draft sub-plans (`plan_a.md`, `plan_b.md`, `plan_c.md`). Each K-item names the
owner (the ONLY plan that defines the symbol) and what the other plans must delete or change.
"A" = data+pilot+labels, "B" = model+evaluation+release, "C" = export+API+page+CI+deploy.

Execution order of the plans: A code tasks → A pilot labelling → A main labelling round (in
parallel with B code tasks 1–15 and C code tasks 1–9, which need no real labels) → B real-data
run → C real export, page, deploy.

---

## K1. Plan file-change format (mechanically applicable — the build-check depends on it)
Every step that changes a file uses EXACTLY one of these three forms, and nothing else:

1. `Create \`path\`:` on its own line, followed immediately by ONE fenced block holding the
   complete file.
2. `Replace in \`path\`:` on its own line, followed by TWO fenced blocks: the first is text that
   occurs EXACTLY ONCE in the file at that point of execution; the second is its replacement.
3. `Append to \`path\`:` on its own line, followed by ONE fenced block.

No "add after the imports", "insert inside build_parser", "similar to", or prose-described edits.
Verbatim vendored copies are produced by a `Run:` step (`cp src/heavy_filter.py …`), which is
allowed because the build-check executes `cp`/`mkdir` commands. Commands sit in fenced `bash`
blocks after `Run:` with an `Expected:` line or block. Every task ends with a commit step:
`git add <exact paths>` (never `-A`, never `.`) and `git commit -F - <<'MSG' … MSG` with a detailed
body, ending with exactly these two trailer lines:

```
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01T2qeexjws3qbeDg1WLsLab
```
Replace any other trailer (e.g. the older "Claude Fable 5.1" one) with these two lines.

## K2. Strata — owner A (`ucc_ml.contracts`)
`STRATA = ("CO:accepted", "CO:rejected", "CT:accepted", "CT:rejected")`,
`stratum_name(region: str, baseline_qualifies: bool) -> str` → `"CO:accepted"` form (colon).
B deletes `stratum_key` and `STRATUM` (pipe form `"CO|accepted"`) and imports A's. Every table,
bootstrap stratum, weight table and public document uses the colon form.

**Screen cells (added 2026-09-12, owner `ucc_ml.screening`).** A round MAY sub-divide a stratum into
CELLS for SAMPLING only. `SCREEN_CELLS = ("B1_maker_group", "B2_score_top", "B3_declared_words",
"B4_remainder")` partition the rules-rejected pool in that priority order;
`ACCEPTED_CELLS = ("A1_lender_only", "A2_name_decidable")` partition the accepted pool. A cell is
recorded as the column `screen_cell` in the round's cases file and manifest and NEVER in
`labels.csv`: its entire effect on estimation is carried by the per-row `inclusion_probability`, so
`Label.sampling_stratum` keeps its four values and the 240 committed pilot rows stay valid. A cell is
a sampling artefact and must never become a model feature. `B4_remainder` is sampled at a floor of
`screening.b4_floor_per_split` in EVERY split, as a refusal condition in code: that floor is the only
reason the estimates are design-unbiased whatever the screen is worth.

## K3. Label vocabulary — owner A (`ucc_ml.contracts`)
```python
AdjudicationStatus = Literal["model_agreed", "founder_confirmed", "founder_adjudicated", "blind_repeat"]
RESOLVED_ADJUDICATION_STATUSES: tuple[str, ...] = ("model_agreed", "founder_confirmed", "founder_adjudicated")
LabellingRound = Literal["pilot_v1", "ablation_v1", "yield_probe_v1", "main_v1", "queue_v1"]
LABELLER_PASS_A = "claude_blind_pass_a"; LABELLER_PASS_B = "claude_blind_pass_b"
LABELLER_AGREED = "claude_blind_pass_a+claude_blind_pass_b"; LABELLER_FOUNDER = "founder"
```
- `model_agreed`: both blind passes gave the same `label`; the founder did not review it. Row carries
  pass A's `reason_code`/`reason`, `labeller_id = LABELLER_AGREED`.
- `founder_confirmed`: the founder reviewed an agreed row (audit sample) and kept the label. Row keeps
  pass A's `reason_code`/`reason`; `labeller_id = LABELLER_FOUNDER`.
- `founder_adjudicated`: the founder decided a disagreement, or overturned an agreed row.
  `reason_code = "ADJUDICATED"`, `labeller_id = LABELLER_FOUNDER`.
- `blind_repeat`: a hidden repeat row; `is_repeat = True`; never fitted or evaluated.
- There is NO `pending` status. `validate-labels` refuses to write `labels.csv` while any disagreement
  lacks a founder decision.
- `Label` gains the field `labelling_round: LabellingRound`. `LABEL_COLUMNS` (derived from `Label`) =
  `case_id, label, reason_code, reason, labeller_id, labelled_at, policy_version, sampling_stratum,
  inclusion_probability, adjudication_status, is_repeat, labelling_round`.
- B deletes `ADJUDICATED_STATUS = "adjudicated"`; `resolve_binary_labels` keeps non-repeat rows whose
  status is in `RESOLVED_ADJUDICATION_STATUSES` and raises `ValueError` on any status outside the four.
  B's synthetic world uses only these four statuses; its "pending rows are dropped" test becomes
  "blind_repeat rows are dropped" plus "an unknown status raises".
- `labels_manifest.json` (A writes; B reads; C publishes a whitelist) has at least:
  `policy_version, disclosure ("model-labelled, founder-adjudicated"), labels_sha256, rows,
  counts_by_status {status: n}, counts_by_round {round: n}, pass_agreement {round: {n, agreed, rate}},
  founder_audit {round: {n_audited, n_confirmed, n_overturned, agreement_rate}},
  repeat_consistency {pass_a: {n, consistent, rate}, pass_b: {…}}`.
  B's `labels_summary` passes `disclosure, counts_by_status, counts_by_round, pass_agreement,
  founder_audit, repeat_consistency` into the metrics `labels` block; C whitelists them publicly.
- **`policy_version` is stamped PER ROUND, never from one config value** (shipped 7f6a144).
  `labelling.policy_version_by_round` maps each round to the policy it was LABELLED under;
  `labeling.policy_version_for_round(cfg, round)` resolves it and falls back to
  `version.label_policy_version`. Stamping one global value would put `label_policy_v2` on the 240
  rows the v1 brief produced, which is a provenance lie in a published field. `labels_manifest.json`
  records the whole mapping, and rows of different policy versions are NEVER pooled in a reported
  statistic. `ml/specs/label_policy_v1.md` is frozen and never edited: it is the pilot's policy and
  the ablation's control arm, pinned by a sha256 test.

## K4. Config — owner A (`ucc_ml.config`, `ml/configs/v1.yaml`)
- A owns the WHOLE `ml/configs/v1.yaml`. It contains A's sections plus every section B and C read,
  with B's and C's exact values: `mlflow, features, training, calibration, threshold, evaluation,
  release, service, export, budget`. B and C never write or modify `v1.yaml`; B Task 1 only adds a
  test asserting the sections it needs exist. No YAML key appears twice.
- `load_config(path: Path | str) -> RunConfig` is the only config loader. `RunConfig` keeps
  `model_config = ConfigDict(extra="allow")` and gains `def section(self, name: str) -> dict` returning
  a copy of `self.model_extra[name]` and raising `KeyError(f"config has no section {name!r}")` if
  absent. `repo_root` = `config_path.parents[2]` (config must live at `<root>/ml/configs/v1.yaml`).
- `Versions.release_id` is deleted (the release id is computed, K8).
- `Paths` gains `predictions_dir` (`ml/data/predictions`), `main_round_dir` (`ml/data/main/v1`),
  `ablation_dir` (`ml/data/ablation/v1`) and `screening_dir` (`ml/data/screening/v1`). A may add
  further path fields; B and C never build these paths themselves.
- `Labelling` gains `policy_version_by_round: dict[str, str]` (K3). A `screening:` section carries
  `fit_split`, `positive_route`, `score_quantile`, `b4_floor_per_split` and `seed`.
- B deletes `training.load_config`, `resolve_path`, `DataPaths`; C deletes `exporting.load_config`,
  `repo_root_from`, `ExportPaths`. Both use `load_config` + `artefact_paths` + `cfg.section(...)`.

## K5. Artefact paths — owner A (`ucc_ml.config`)
```python
@dataclass(frozen=True)
class ArtefactPaths:
    repo_root: Path; snapshot_manifest: Path
    candidates_parquet: Path; candidates_manifest: Path; reconciliation: Path
    pilot_cases: Path; pilot_manifest: Path
    main_cases: Path; main_manifest: Path
    ablation_cases: Path; ablation_manifest: Path
    screen_cells: Path; screening_manifest: Path
    splits_parquet: Path; split_manifest: Path
    labels_csv: Path; labels_manifest: Path
    mlflow_db: Path; mlflow_artifacts: Path
    frozen_dir: Path; final_eval_dir: Path; releases_dir: Path; predictions_dir: Path
    public_data_dir: Path; lock_file: Path

def artefact_paths(cfg: RunConfig) -> ArtefactPaths
```
Values (all absolute via `cfg.path(key)`): `snapshot_dir/manifest.json`,
`candidates_dir/{candidates.parquet,candidates_manifest.json,reconciliation.json}`,
`pilot_dir/{pilot_cases.parquet,pilot_manifest.json}`,
`main_round_dir/{main_cases.parquet,main_manifest.json}`,
`splits_dir/{splits.parquet,split_manifest.json}`, `labels_dir/{labels.csv,labels_manifest.json}`,
`mlflow_dir/{mlflow.db,artifacts}`, `artifacts_dir/frozen/v1`, `artifacts_dir/final_eval/v1`,
`artifacts_dir/releases`, `predictions_dir`, `public_data_dir` (= `docs/data/ml`),
`repo_root/ml/requirements-ml.lock.txt`. B's config keys `threshold.frozen_dir` and
`evaluation.final_eval_dir` are deleted in favour of these.

## K6. Hashing and JSON I/O — owner A (`ucc_ml.provenance`, created in A Task 2)
```python
def sha256_file(path: Path) -> str
def sha256_bytes(data: bytes) -> str
def canonical_json_bytes(obj: Any) -> bytes      # json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", utf-8
def write_json(path: Path, obj: Any) -> str      # mkdir parents, write canonical_json_bytes, return its sha256
def read_json(path: Path) -> Any
def utc_now_iso() -> str                         # "%Y-%m-%dT%H:%M:%SZ"
def git_head(repo_root: Path) -> str | None
class BundleIntegrityError(ValueError)
def write_sha256sums(directory: Path, filenames: Iterable[str]) -> Path          # "<sha256>  <name>\n", sorted by name
def parse_sha256sums(text: str) -> dict[str, str]  # rejects malformed lines, absolute paths, "..", "/" in names
def verify_sha256sums(directory: Path, required: Iterable[str] = ()) -> dict[str, str]
    # raises BundleIntegrityError: missing SHA256SUMS, a listed file missing, a digest mismatch, or a
    # `required` name not listed. Returns {name: digest}. Called before any joblib.load.
```
`ucc_ml.contracts.canonical_json(obj) -> str` (compact, identity hashing — known vectors pinned) is a
different function and stays unchanged. B does NOT create `ucc_ml/hashing.py`; C defines none of
these in `exporting.py`. Both import from `ucc_ml.provenance`. NaN never reaches JSON: producers
convert with a `finite_or_none` helper before writing.

## K7. Artefact readers — owner A
`ucc_ml.dataset.read_candidates(path: Path) -> pd.DataFrame`,
`ucc_ml.splitting.read_splits(path: Path) -> pd.DataFrame`,
`ucc_ml.labeling.read_labels(path: Path) -> pd.DataFrame` are the ONLY readers of those files. Each
checks required columns and value domains (split ∈ SPLITS, label ∈ LABELS, adjudication_status ∈ the
four, labelling_round ∈ the two, is_repeat boolean, case_id hex64, lender lists are lists) and raises
`ValueError` naming the file, column and offending values. B's validation logic moves into them;
B's and C's own readers are deleted.

## K8. Inference — owner B (`ucc_ml.inference`)
Canonical names (C uses them directly; C's adapter seams `load_bundle`, `to_case_input`, `predict`,
`prediction_to_dict`, `normalise_contributions`, `read_release_id`, `read_threshold` are deleted):
```python
RELEASE_FILES = ("pipeline.joblib", "model-manifest.json", "threshold.json", "dataset-manifest.json",
                 "split-manifest.json", "metrics.json", "model-card.md", "requirements-ml.lock.txt")
RELEASE_ID_MANIFESTS = ("model-manifest.json", "threshold.json", "dataset-manifest.json", "split-manifest.json")
def compute_release_id(directory: Path) -> str
def load_release_bundle(directory: Path, *, verify: bool = True, top_k: int = 10) -> ReleaseBundle
class CaseInput(BaseModel)      # borrower_name (1..300), lender_names (≤20, each ≤300), region Literal["CO","CT"], case_id|None, borrower_name_clean|None, lender_names_clean|None; extra forbidden
class Prediction(BaseModel)     # case_id|None, input_hash, release_id, score, score_type, decision, threshold, baseline_qualifies, baseline_route, top_feature_contributions: list[tuple[str, float]]  (names "block__token")
def predict_cases(cases: list[CaseInput], bundle: ReleaseBundle) -> list[Prediction]
```
- The release id comes from `ReleaseBundle.release_id` (the loader checks it equals the directory
  name). Nothing reads a `release_id` out of `model-manifest.json` (it cannot contain it).
- `load_release_bundle` calls `verify_sha256sums(directory, required=RELEASE_FILES)` when `verify`.
- B's baseline for live input uses `ucc_ml.legacy.baseline_qualifies/baseline_route` (no private copy).
- API response `top_feature_contributions` = list of `{block, feature, contribution}` with
  `block, feature = name.split("__", 1)`.
- B's run functions take only the config path and derive every path from `artefact_paths`:
  `run_train(config_path: Path) -> dict`, `run_freeze_candidate(config_path: Path, *, force: bool = False) -> dict`,
  `run_evaluate_final(config_path: Path, *, force_i_know: bool = False) -> dict`,
  `run_build_release(config_path: Path) -> Path`, `run_score_batch(config_path: Path, release_dir: Path | None = None) -> Path`.
- `run_score_batch` is the ONLY batch scorer (every candidate → `predictions_dir/<release_id>.parquet`
  + `.summary.json`). C deletes `score_pool`, `score_example_pool`, `write_batch_predictions` and
  reads B's predictions parquet for curated examples.

## K9. Public metrics — producer B, projector C
B's release `metrics.json` is `{release_id, created_at, validation: <frozen validation-metrics.json>,
test: <final metrics.json>}`; the final metrics carry `protocol, split, evaluated_at, forced, frozen,
evaluation{n, positives, negatives, weights, model, rules, delta_model_minus_rules, per_region,
per_stratum, bootstrap}, calibration, unresolved, labels, verdict, provenance`. C's
`project_public_metrics(release_metrics: dict, release_id: str, generated_at: str) -> dict` is written
against THAT shape (read B Tasks 6–8 and 10–12 for every nested key) as an explicit whitelist. Never
published: `frozen.frozen_dir`, anything under `provenance` except commit SHAs and package versions,
absolute paths, per-case rows. The public document carries `label_provenance` plus the K3 label
statistics. The Lab page renders the public shape only.

## K10. Synthetic world — owner B (`ml/src/ucc_ml/synthetic.py`, created by B Task 2)
B's `ml/tests/_synth.py` moves to `ml/src/ucc_ml/synthetic.py` (importable by tests, CI, Dagster tests
and the Lab preview tool). Contents: `World`, `make_world(seed: int = 7, groups_per_region: int = 150) -> World`,
`write_world(world: World, root: Path) -> ArtefactPaths` (writes candidates/splits/labels/labels_manifest/
lock file at the K5 locations under `root`), `write_config(root: Path, **overrides) -> Path` (writes
`<root>/ml/configs/v1.yaml` loadable by A's `load_config`, with a fast profile: `cv_folds: 3`,
`n_resamples: 50`, overridable), and `build_synthetic_release(root: Path) -> Path` (added in B Task 12)
which runs the REAL `run_train → run_freeze_candidate → run_evaluate_final → run_build_release` on the
world and returns the bundle directory. Baseline columns come from `ucc_ml.legacy`. C deletes its
hand-rolled `write_synthetic_bundle`, `fit_synthetic_pipeline`, `synthetic_metrics`, `baseline_for`,
`synthetic_candidates/splits/labels`, `frame_for`, `FEATURE_COLUMNS`; C's tests use a session-scoped
fixture `synthetic_release_dir` built with `build_synthetic_release(tmp_path_factory.mktemp("release"))`.

## K11. CLI — owner A's pattern (`ml/src/ucc_ml/cli.py`)
A's pattern is canonical: inside `build_parser()`, below the marker comment,
`sp = sub.add_parser("<name>", help="…"); _add_config_arg(sp); sp.set_defaults(func=cmd_<name>)`, and a
module-level `def cmd_<name>(ns: argparse.Namespace) -> int:` that imports its implementation LAZILY
inside the function (so `--help` never imports sklearn/mlflow/fastapi). `main(argv)` is unchanged.
B adds `train, freeze-candidate, evaluate-final, build-release, score-batch`; C adds `export-public`;
each with a `Replace in \`ml/src/ucc_ml/cli.py\`` step whose first block is the marker comment line
(re-emitted unchanged at the end of the replacement so the next plan can anchor on it).

## K12. Pilot groups go to TRAIN — owner A (`ucc_ml.splitting.assign_splits`)
Every group containing a pilot case is assigned `train` (not "train or validation"). Non-pilot groups
fill test, then validation, then train by seeded hash order. Reason: validation (which sets the
weighted-precision threshold) and test must both be probability samples of their split populations
with known inclusion probabilities; the pilot was drawn from the whole candidate population at
different rates. A's Task 11 tests change accordingly (`all(a[g] == "train" for g in pilot)`).

## K13. Main labelling round — owner A (new tasks after the pilot)
1. Pilot report (`ml/data/pilot/v1/pilot_report.json`, printed): labelability (share
   INSUFFICIENT_EVIDENCE by stratum), RELEVANT prevalence by stratum, pass agreement, founder audit
   agreement, blind-repeat consistency.
2. FOUNDER GATE: approve the main-round allocation. Default proposal (config `budget.main_round`):
   train 1,200 / validation 400 / test 800 spread over the TWELVE (stratum, cell) combinations of
   each split, not evenly across four strata. The rule is deliberately uninformed — before a yield
   probe measures anything, no cell is assumed more valuable, so each cell of a split gets an equal
   base share, shares are capped by population, the shortfall is redistributed to cells with room,
   and `B4_remainder` is raised to its floor last. `propose_main_allocation` REFUSES two designs
   rather than returning them quietly: any cell at `n_h = 0` (a cell with no draw has no inclusion
   probability and silently drops its whole population out of every estimate) and `B4_remainder`
   below its floor. A founder override replaces one cell exactly and the rest share what is left.
3. `make-main-round`: from NON-pilot cases, within the frozen splits, a seeded draw per
   (split, stratum, CELL) with `N_h` = that cell's population, `pool_h` after excluding the pilot,
   `n_h`, `inclusion_probability = n_h / pool_h` computed PER CELL, `sampling_stratum` in the K2
   colon form and `screen_cell` alongside it → `main_cases.parquet` + `main_manifest.json`.
   `draw_stratified` needs no signature change: a cell is a filter on the frame handed to it, never
   a key. The screen itself is pre-registered first as `docs/data/ml/screen_v1.sha256`.
4. Pre-registration: `docs/data/ml/main_v1.sha256` committed BEFORE any main-round label exists.
5. `label-blind --round main_v1 --chunk-size 200`: chunked blind queue files
   (`queue_main_v1_part_NNN.csv`, 10% hidden repeats within each chunk) + private key.
6. Two independent blind passes (K14), import, agreement report, founder review workbook = every
   disagreement + an audit sample of agreed rows (config `labelling.founder_audit_per_split_stratum`,
   default 10 per (split, stratum)), FOUNDER GATE (review hand-off).
7. `validate-labels` merges pilot_v1 and main_v1 rows into ONE `labels.csv` + `labels_manifest.json`
   (K3), refusing while any disagreement is undecided.

## K14. The blind labeller — owner A (`ml/specs/labeller_prompt_v1.md`)
The labeller is an orchestrator-run Claude subagent. It receives ONE queue chunk INLINE in its prompt
(case_id, borrower_name_raw, lender_names_raw, city, state) plus the policy text, uses NO tools (no
file reads, no web search — the no-lookup rule), and returns its rows through structured output
(`case_id, label, reason_code, reason`). The orchestrator — not the labeller — writes
`ml/data/labels/v1/raw/labeller_output_<round>_pass_<a|b>_part_<NNN>.csv`. Pass A and pass B are separate
agents that never see each other's output. The prompt file states all of this; A's plan tells the
executor exactly how to run the passes (one agent per chunk per pass) and the CSV header it writes.

## K15. Honesty in every public document
Every public document carries the literal `model-labelled, founder-adjudicated` AND the K3 label
statistics (counts by status and round, pass agreement, founder audit agreement, repeat consistency).
Curated examples carry `curated: true, precomputed: true` and are never a benchmark. The founder
approves the curated example list at C's real-data layout gate (named businesses next to labels).

## K16. Things that do not change
Everything in `context_pack.md` §0–§3 still holds: read-only DuckDB under `.venv` only, region-scoped
`group_id`, exact `build_scope` eligibility, no lookups, Python 3.14.7, separate page, static first,
never touch `scope_all`/headline counts/the map beyond C's single link, commit by explicit path,
`./.venv/bin/python -m pytest tests/ -q` → `225 passed` after every task.
