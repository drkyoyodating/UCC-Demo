# UCC Heavy Construction Equipment Finance — Colorado & Connecticut

**Live demo → https://drkyoyodating.github.io/UCC-Demo/**

Turning 3.4 million raw public lien filings into a clean, queryable market of firms
that own and operate heavy construction equipment — and proving the filter is right
rather than asserting it.

---

## What it does

US states publish UCC-1 financing statements: a public record every time a lender
takes a security interest in a borrower's equipment. The registers are enormous,
uncategorised, and mostly irrelevant — Colorado alone carries 2.6M filings covering
everything from office copiers to cattle.

This project reduces that to **74,308 filings by 21,221 borrowers** who verifiably
finance job-site machinery, resolves the duplicate name variants into **~21,000
distinct firms**, and publishes the result as a map you can interrogate.

| | |
|---|---:|
| source filings ingested | 3,432,167 |
| qualifying filings | 74,308 |
| distinct borrowers | 21,221 |
| distinct lenders | 3,405 |
| resolved entities | ~21,000 |
| years covered | 1990–2026 |

## How it works

**1 · Ingest.** Colorado and Connecticut registers pulled once from their Socrata
APIs into a local DuckDB snapshot. Three other jurisdictions were ingested and then
**cut for stated reasons**: New York City (11,035,386 party rows ingested to discover
it is 0.17% machinery — structurally, under §9-501(a)(1), county recorders hold
fixture and real-property filings, not equipment finance), Oregon (no borrower
column at all), Philadelphia (8 machinery records).

**2 · Qualify.** A row is added only if it meets one of **two criteria**:

- **the lender** is a *named* heavy-construction equipment maker or dealer, or
- **the borrower's own name** states an explicit job-site equipment category.

Either alone is sufficient; neither means the row is not added. **There is no reject
list.** Nothing is "blocked" — a row simply fails to meet a criterion. A bank lender
is not denied, it just doesn't satisfy the lender criterion, so the borrower name has
to carry it.

**3 · Resolve.** Splink over the qualifying corpus links name and address variants
into single firms. `SEMA CONSTRUCTION, INC.` filing 338 times at one address is one
firm with a 338-loan history, not 338 firms.

**4 · Verify.** The criteria are a classifier, so they are measured like one — an
adjudicated sample of qualifying rows, reviewed independently and audited
adversarially in both directions.

## Accuracy

Measured on a stratified sample of the live pull, reviewed row by row and audited
with zero overturns in either direction:

| | |
|---|---:|
| **precision** | **99.4%** (357/359 resolvable) |
| rows meeting neither criterion | **0 of 74,308** |
| non-degeneracy (largest cluster) | 0.62% — passes the <1% bar |

The number moved 72.19% → 98.0% → 99.4% across three rounds. Each round used the
failures as the blueprint: every defect was traced to the exact term that admitted
it, the term was tightened, and the pull was re-measured. Defects closed this way
included bank leasing arms riding a generic phrase, an agricultural dealership added
in error, `WRECKING` pulling auto-salvage yards, surnames posing as equipment words
(`CRANE, ROBERT GALE`), and a Deere entity re-entering as a substring of a listed
dealer's name.

## Try it

Open the demo and:

1. **Click a state.** It brightens and draws 1,300 qualifying filings as dots,
   clustered by ZIP so co-located filings become one larger dot.
2. **Click a dot** to read its filings — borrower, address, city, state, ZIP, lender,
   year, and which criterion admitted it.
3. **Click a lender** in that table to trace it across the map: every dot it financed
   highlights, the rest dim. That is the lender→borrower graph.
4. **Filter by criterion** to see the classifier's own split — lender-qualified,
   borrower-qualified, or both.
5. **Press Refresh Data.** A new sample is drawn from the qualifying pool and the map
   redraws in single-digit milliseconds. Every row is correct on every draw; only
   *which* correct rows you see changes.
6. **SQL views** and **Cross-register** tabs show the relational analytics with the
   queries printed, and the firms that file in *both* registers — deliberately **not**
   merged, because a shared name across two states is not evidence of a shared firm.

## Caveats, stated plainly

- **Dots are ZIP centroids, not rooftops.** At whole-state zoom a ZIP centroid and a
  street address land on the same pixel, and 11.6% of rows are PO Boxes with no
  rooftop to find. 95.0% of rows resolve to a coordinate.
- **~3% of rows are relocated.** A filing in one register by a borrower headquartered
  out of state has no coordinate inside the outline, so its dot is moved in-state.
  The table always shows the real address.
- **No entity is resolved across registers.** Deliberately. See the Cross-register tab.
- **Lenders may be blank.** Banks will not disclose; the borrower can be contacted.

## Reproduce it

```bash
./.venv/bin/python src/build_scope.py         # apply the criteria to the snapshot
./.venv/bin/python src/build_scope_corpus.py  # party-record corpus (CO + CT)
./.venv/bin/python src/build_geo.py           # attach ZIP coordinates
./.venv/bin/python run_p5_scope_all.py        # entity resolution
./.venv/bin/python src/build_page_data.py     # export the page payload
./.venv/bin/python src/build_page_stats.py    # export the SQL views
```

The pull is deterministic: two runs over the same snapshot produce byte-identical
output. Pins in `requirements.lock.txt` are load-bearing.

`src/heavy_filter.py` holds the criteria — two lists and nothing else.
`docs/LABELLING_CRITERIA.md` is the full written reference.

## Licence

MIT. Source data is public record.
