#!/usr/bin/env python
"""P5 re-run on the CURRENT scope (CO + CT). Additive runner; edits no frozen file."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from resolve import resolve, THRESHOLD_OVERRIDE
r = resolve("corpus_scope_all", "debtors", tag="scope_all",
            threshold=THRESHOLD_OVERRIDE.get("debtors"))
print("\n" + json.dumps(r, indent=2))
print(f"\nNON-DEGENERACY BAR (no cluster >1%): largest={r['largest_pct']:.3f}% -> "
      f"{'PASS' if r['largest_pct'] <= 1.0 else 'FAIL'}")
