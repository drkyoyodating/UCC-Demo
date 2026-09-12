# ucc_ml

The ML Lab package for the UCC heavy-construction-equipment demo. Install editable into `.venv-ml`
(`./.venv-ml/bin/python -m pip install -e ml/`), run `./.venv-ml/bin/python -m ucc_ml.cli --help`,
and run the tests with `./.venv-ml/bin/python -m pytest ml/tests -q`.
Plans: `docs/superpowers/plans/2026-09-12-ucc-ml-*.md`. Label provenance is PER ROUND, and is stated per round in `docs/data/ml/labels_v1.sha256`: `pilot_v1`
(240 rows) is model-labelled and founder-adjudicated, while `main_v1` (2,880 rows) is model-labelled by
two independent blind passes with disagreements retained as unresolved rather than decided.
