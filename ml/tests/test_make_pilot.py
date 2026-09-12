"""make-pilot draws per_stratum cases from each of the four strata and records the design."""
import hashlib
import json

from conftest import balanced_case_rows, write_candidates_fixture
from test_config import MINIMAL


def _config(tmp_path, per_stratum=3):
    cfg = tmp_path / "ml" / "configs" / "v1.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(MINIMAL.replace("pilot_per_stratum: 50", f"pilot_per_stratum: {per_stratum}"))
    return cfg


def test_make_pilot_function():
    from ucc_ml.contracts import CASE_COLUMNS
    from ucc_ml.sampling import SAMPLE_COLUMNS, make_pilot
    import pandas as pd

    cases = pd.DataFrame(balanced_case_rows(6))
    pilot = make_pilot(cases, per_stratum=4, seed=20260912)
    assert len(pilot) == 16 and pilot.case_id.is_unique
    assert list(pilot.columns) == list(CASE_COLUMNS) + [c for c in SAMPLE_COLUMNS if c != "case_id"]
    assert pilot.groupby(["region", "baseline_qualifies"]).size().tolist() == [4, 4, 4, 4]
    assert pilot.sampling_stratum.tolist()[:4] == ["CO:accepted"] * 4          # STRATA order
    assert (pilot.N_h == 6).all() and (pilot.n_h == 4).all()
    assert pilot.inclusion_probability.iloc[0] == 4 / 6


def test_cli_make_pilot_writes_parquet_and_manifest(tmp_path, capsys):
    from ucc_ml.cli import main
    from ucc_ml.dataset import read_frame

    cfg = _config(tmp_path)
    cand = tmp_path / "ml/data/candidates/v1/candidates.parquet"
    cand.parent.mkdir(parents=True)
    write_candidates_fixture(cand, balanced_case_rows(5))
    assert main(["make-pilot", "--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "pilot rows=12" in out
    pilot = read_frame(tmp_path / "ml/data/pilot/v1/pilot_cases.parquet")
    assert len(pilot) == 12 and isinstance(pilot.lender_names_raw.iloc[0], list)
    m = json.loads((tmp_path / "ml/data/pilot/v1/pilot_manifest.json").read_text())
    assert m["purpose"] == "pilot_v1" and m["seed"] == 20260912 and m["per_stratum"] == 3
    assert m["strata"]["CT:rejected"] == {"N_h": 5, "pool_h": 5, "n_h": 3, "inclusion_probability": 0.6}
    assert m["by_region"] == {"CO": 6, "CT": 6}
    assert m["candidates_sha256"] == hashlib.sha256(cand.read_bytes()).hexdigest()
    expect = hashlib.sha256(("\n".join(sorted(pilot.case_id)) + "\n").encode()).hexdigest()
    assert m["case_id_digest"] == expect
    # Pin the N_h / n_h hazard: DuckDB folds the two names together, so this file is pyarrow-only.
    import duckdb

    parquet = (tmp_path / "ml/data/pilot/v1/pilot_cases.parquet").as_posix()
    con = duckdb.connect()
    try:
        names = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet}')").fetchall()]
        assert names != list(pilot.columns)                       # one of N_h / n_h was renamed
        wrong = con.execute(f"SELECT n_h FROM read_parquet('{parquet}') LIMIT 1").fetchone()[0]
        assert wrong == int(pilot.N_h.iloc[0]) != int(pilot.n_h.iloc[0])   # SELECT n_h returns N_h
    finally:
        con.close()


def test_cli_make_pilot_fails_loudly_when_a_stratum_is_short(tmp_path):
    from ucc_ml.cli import main
    import pytest

    cfg = _config(tmp_path, per_stratum=9)
    cand = tmp_path / "ml/data/candidates/v1/candidates.parquet"
    cand.parent.mkdir(parents=True)
    write_candidates_fixture(cand, balanced_case_rows(5))
    with pytest.raises(ValueError, match="asked for 9"):
        main(["make-pilot", "--config", str(cfg)])
