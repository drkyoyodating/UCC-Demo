#!/usr/bin/env python
"""SELECTION LEAD combination test. Additive: touches no frozen file.

Builds two AND-combinations of already-fitted, train-selected components and
writes them on the SHIPPED weight scale so score.py can read them unchanged:

  combo_pf  = shipped weight, vetoed unless the pair passes BOTH the Strategy D
              person/organisation gate AND the Strategy B jw>=0.92 name floor.
  combo_pc  = Strategy C comparison-model weight, vetoed unless the pair passes
              the Strategy D person gate.

Both are ANDs of components each selected on labels_train.csv. No new parameter
is fitted here and labels_test.csv is not read by this module.
"""
from __future__ import annotations
import sys
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "parquet"
VETO = -999.0


def build():
    d = duckdb.connect()
    d.execute("SET memory_limit='2GB'")
    d.execute(f"""create or replace view j as
      select s.unique_id_l l, s.unique_id_r r, s.match_weight ws,
             f.match_weight wf, p.match_weight wp, c.match_weight wc
      from '{P}/predictions_debtors.parquet' s
      join '{P}/predictions_name_floor.parquet' f
        on f.unique_id_l=s.unique_id_l and f.unique_id_r=s.unique_id_r
      join '{P}/predictions_person.parquet' p
        on p.unique_id_l=s.unique_id_l and p.unique_id_r=s.unique_id_r
      join '{P}/predictions_comparison.parquet' c
        on c.unique_id_l=s.unique_id_l and c.unique_id_r=s.unique_id_r""")

    for tag, expr in [
        ("combo_pf", f"case when wf > -900 and wp > -90 then ws else {VETO} end"),
        ("combo_pc", f"case when wp > -90 then wc else {VETO} end"),
    ]:
        d.execute(f"""copy (
            select l as unique_id_l, r as unique_id_r,
                   {expr} as match_weight,
                   1.0/(1.0+pow(2.0, -({expr}))) as match_probability
            from j
        ) to '{P}/predictions_{tag}.parquet' (format parquet)""")
        n = d.execute(f"select count(*) from '{P}/predictions_{tag}.parquet'").fetchone()[0]
        print(f"wrote parquet/predictions_{tag}.parquet  rows={n:,}")
    d.close()


if __name__ == "__main__":
    build()
