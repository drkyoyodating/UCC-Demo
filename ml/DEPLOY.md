# Deploying the UCC ML Lab

The operating record for the public side of the ML Lab: the static page on GitHub Pages, the CI
check, and the live prediction API on Cloud Run, which exists only if the founder approves it.
Every command runs from the repository root.

## 1. What is public

The GitHub Pages site is the `docs/` directory of `main`, served at
https://drkyoyodating.github.io/UCC-Demo/. The Lab page is `docs/ml/index.html` with
`docs/assets/ml-lab.js` and `docs/assets/ml-lab.css`, served at
https://drkyoyodating.github.io/UCC-Demo/ml/. Its data is `docs/data/ml/manifest.json` plus
`docs/data/ml/releases/<release_id>/metrics.json`, `examples.json` and `model-card.json`, and only
`export-public` writes those files: a whitelist of the release's metrics, the curated examples and
the model card.

Never public: `ml/data/`, `ml/artifacts/`, `ml/mlflow/` and `ml/reports/` (all gitignored), the
labels, per-case predictions and the release bundle itself. The map page links to the Lab once, in
its footer; the Lab never changes `scope_all`, the headline counts or the map.

## 2. Pages source

The Pages source is checked with the GitHub API, which needs a signed-in `gh`:

    gh api repos/drkyoyodating/UCC-Demo/pages --jq '.status, .source.branch, .source.path, .html_url'

It prints `built`, `main`, `/docs` and `https://drkyoyodating.github.io/UCC-Demo/`, one per line.
If that ever changes, stop: every path in this file assumes it.

That command was **not** run for this release: `gh` is installed here but signed out, and the founder
gate for signing in was not taken because the live API is out of scope. What was verified instead,
without `gh`: the site root and `https://drkyoyodating.github.io/UCC-Demo/ml/` both answer 200, a file
committed under `docs/` is served verbatim at its `docs/`-relative path, `docs/.nojekyll` is present,
and the branch publishing it is `main`. That establishes the substance — Pages serves `docs/` from
`main` — but not the API's own `status` field. Run the command above once `gh` is signed in.

## 3. GitHub CLI authentication

Pushing a file under `.github/workflows/` over HTTPS needs a token with the `workflow` scope. The
founder signs in once, in his own terminal:

    gh auth login --hostname github.com --git-protocol https --web --scopes workflow

and answers yes to "Authenticate Git with your GitHub credentials?", so `git push` uses the same
token. `gh auth status` then lists `workflow` among the token scopes.

## 4. Previewing the layout without real data

    ./.venv-ml/bin/python ml/tools/make_lab_preview.py --out /tmp/ucc-lab-preview
    ./.venv-ml/bin/python -m http.server 8080 --bind 127.0.0.1 --directory /tmp/ucc-lab-preview/site

Open http://127.0.0.1:8080/ml/. The preview runs the real pipeline on the fictitious synthetic
world, shows a SYNTHETIC banner, and is never copied under `docs/`.

## 5. Publishing a release to the static page

Plan B has built `ml/artifacts/releases/<release_id>/`, scored every candidate with `score-batch`,
and committed `docs/data/ml/release_v1.sha256`, whose first line is `release_id=<release_id>`.
Export the public documents for exactly that release:

    RELEASE_ID=$(sed -n 's/^release_id=//p' docs/data/ml/release_v1.sha256)
    ./.venv-ml/bin/python -m ucc_ml.cli export-public --config ml/configs/v1.yaml --release-dir "ml/artifacts/releases/$RELEASE_ID"

Check the result locally before anything is committed:

    ./.venv-ml/bin/python -m pytest ml/tests/test_lab_page.py -q
    ./.venv-ml/bin/python -m http.server 8080 --bind 127.0.0.1 --directory docs

Open http://127.0.0.1:8080/ml/ and http://127.0.0.1:8080/. The founder approves the layout with
the real numbers and the curated example list (named businesses appear next to their labels)
before the files are committed. Commit by explicit path, never `git add -A`, then push and confirm
the live page:

    git add docs/data/ml/manifest.json "docs/data/ml/releases/$RELEASE_ID/metrics.json" "docs/data/ml/releases/$RELEASE_ID/examples.json" "docs/data/ml/releases/$RELEASE_ID/model-card.json"
    git push origin main
    gh run watch "$(gh run list --workflow pages-build-deployment --branch main --limit 1 --json databaseId --jq '.[0].databaseId')" --exit-status
    curl -sS -o /dev/null -w '%{http_code}\n' https://drkyoyodating.github.io/UCC-Demo/ml/

## 6. Continuous integration

`.github/workflows/ml-checks.yml` installs `ml/requirements-ml.lock.txt` on Python 3.14, runs
`pytest ml/tests` on synthetic fixtures only, and checks `docs/assets/ml-lab.js` with node 22. It
never reads `ml/data/` or the database, and a green run says nothing about model accuracy.

    gh run watch "$(gh run list --workflow ml-checks.yml --branch main --limit 1 --json databaseId --jq '.[0].databaseId')" --exit-status

## 7. Rolling back the static page

A release's public files are ordinary commits, so the previous state of the page is one revert
away. This reverts the most recent change to the manifest, which moves `current` back with it:

    git revert --no-edit "$(git log -1 --format=%H -- docs/data/ml/manifest.json)"
    git push origin main

## 8. Live API

Not deployed. Plan C Task 11 records the founder's decision in `ml/deploy_decision.json`. If he
approves a deployment, plan C Task 12 replaces this section with the Cloud Run procedure, its
rollback and its disable switch.
