# SESSION 2 BRIEF — Tandem A, P2 normalizer lane

You are **Session 2**. Session 1 is concurrently running the P1 Socrata ingest and is the **sole writer**
of the DuckDB file. This brief is self-contained: **do NOT read `UCC_DEMO_RUNBOOK.md`** (60KB — the project
has a hard, non-renewable token budget and re-reading it is forbidden for exactly that reason).

## File ownership — absolute

**YOU OWN (write freely):**
- `src/normalize.py`
- `tests/test_normalize.py`
- `tests/fixtures/messy_names.csv`

**YOU MUST NEVER TOUCH:** `*.duckdb` (any, for any reason, including read-only), `src/ingest.py`,
`raw_pages/` (read-only is fine, never write), `docs/`, `STATUS.md`, `DECISIONS.md`,
`UCC_DEMO_RUNBOOK.md`, `requirements*.txt`, `.gitignore`, `LICENSE`.

**Do not `git commit` or `git push`.** Session 1 owns the commit for this phase and will merge your files.
Leave your work in the working tree and report when done.

## Environment
```
cd /Users/user/Downloads/UCC-Demo
./.venv/bin/python      # 3.14.7 — splink 4.0.16, duckdb 1.4.5, pandas 2.3.3, pytest
./.venv/bin/pytest tests/ -q
```
Do not install anything. Do not upgrade anything. The pins are load-bearing.

## Your input data
`raw_pages/sample_debtors_50k.csv` — 50,000 real debtor rows, already fetched, drawn at five spread
offsets (0 / 400k / 800k / 1.2M / 1.6M) so it is not biased to the oldest filings.
Columns: `debtorid, organizationname, address1, city, state, zipcode, fileid, actiontype, recordstatus, efsuniqueid`

**Known facts about this data — do not re-derive them:**
- **44.8% of rows have a blank `organizationname`.** Colorado publishes no individual-name column.
  `normalize_name` must handle null/empty/whitespace-only input and return a null-ish result cleanly,
  never raise. This is the single most likely crash in the whole pipeline.
- Non-ASCII organization names: 2 in 50,000 (0.004%). Do **not** build a Unicode confusable map — a stage
  that provably never fires is padding. Handle non-ASCII gracefully and move on.
- Address fields are well populated (address1 blank 0.6%, city 0.5%, zip 1.5%).

## The task

### 1. `src/normalize.py` — a pure function, no I/O, no globals
```python
def normalize_name(raw: str | None) -> tuple[str | None, str | None]:
    """Returns (name_clean, suffix). Never raises. Never mutates input."""
```
Rules:
- Uppercase; strip punctuation; `&` → ` AND `; collapse all whitespace runs to one space; trim.
- Recognise and **strip into the separate `suffix` return value** (do NOT delete, do NOT leave inline):
  `LLC, L.L.C., INC, INCORPORATED, CORP, CORPORATION, CO, COMPANY, LTD, LIMITED, LP, L.P., LLP, PLLC, PC, P.C., NA, N.A.`
  Suffix is **signal for comparison, noise for blocking** — that is the entire reason it is split out.
- Only strip a suffix when it is a trailing token. `CO` inside `COLORADO FOLIAGE INC` must not be touched;
  `ACME CO` must yield `("ACME","CO")`.
- If multiple trailing suffixes appear, strip them all and return them joined by a space in order.
- Preserve the raw name — callers keep `organizationname` alongside `name_clean`. Never destroy the original.
- Null/empty/whitespace-only → `(None, None)`.

Also export a per-table column map, because **the two party tables are NOT column-compatible**
(debtors have `debtorid`; secured parties have `spid`; and their record-status column names differ):
```python
COLUMN_MAP = {"debtors": {...}, "secured_parties": {...}}
```
The name-string function is shared by both tables; the row-level pipeline around it is not.

### 2. `tests/fixtures/messy_names.csv` — 50 real messy names
Pick them **by rule, from the real sample**, not by taste — e.g. the longest names, names with the most
punctuation, names with double spaces, names ending in each suffix variant, at least 3 blank/null rows,
the 2 non-ASCII rows, and names where a suffix-like token appears mid-string. Record the selection rule
as a comment at the top of the file. Columns: `raw,expected_name_clean,expected_suffix`.
**Write the expected values BEFORE writing `normalize.py`**, so the test is a genuine external check.

### 3. `tests/test_normalize.py`
Drive every case from the fixture file. Add explicit tests for: `None`, `""`, `"   "`, a name that is
*only* a suffix (`"LLC"`), idempotency (`normalize_name(normalize_name(x)[0]) == (x_clean, None)`),
and purity (input string is unchanged after the call).

## Definition of done
`./.venv/bin/pytest tests/ -q` is green, all 50 fixture cases pass, and you report back:
the count of tests, any name in the sample your rules handle badly, and any judgement call you made
that a reviewer might disagree with. **Report disagreements — do not silently pick one.**

## Known trap in the blocking design (context only — do NOT act on it)
Session 1 will block on `zipcode` and `substr(name_clean, 1, 4)`. A name-stem variant one character short
of the key (`ACM EXCAVATION` vs `ACME EXCAVATING`) lands in different blocks and is never compared. This
is measured in P5, not fixed in P2. Do not add phonetic or token logic to the normalizer to compensate.
