"""
run_examples.py
===============

Replays the inputs in ``examples/sample_inputs.json`` against the trained
calibrated SVM pipeline and prints a compact comparison of predicted vs
expected disease/severity. Useful as a regression test and as the
"example input/output" deliverable required by the coursework brief.

Run from project root::

    python -m examples.run_examples
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `triage_engine` importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from triage_engine import predict, load_or_train


def main() -> int:
    pipe, _ = load_or_train()
    data = json.loads((ROOT / "examples" / "sample_inputs.json").read_text())

    print(f"{'#':>2}  {'id':<28} {'pred':<28} {'sev':<10}  src               OK")
    print("-" * 110)

    fails = 0
    for i, ex in enumerate(data["examples"], 1):
        p = predict(ex["input"], pipe)
        top3 = [d for d, _ in p.differential]
        expected = ex["expected"]

        # Disease check — top-3 is the bar (medical triage rarely needs top-1)
        ok_disease = any(d in top3 for d in expected.get("disease_in_top3", []))
        ok_severity = expected.get("severity") == p.severity
        # Severity source check — if expected is set
        exp_src = expected.get("severity_source")
        ok_src = (exp_src is None) or (exp_src == p.severity_source)
        # Red-flags check — every expected flag must appear
        exp_flags = set(expected.get("red_flags_hit", []))
        ok_flags = exp_flags.issubset(set(p.red_flags_hit))

        ok = ok_disease and ok_severity and ok_src and ok_flags
        if not ok:
            fails += 1

        flag = "✅" if ok else "❌"
        print(
            f"{i:>2}  {ex['id']:<28} {p.disease:<28} {p.severity:<10}  "
            f"{p.severity_source:<18}{flag}"
        )
        if not ok:
            if not ok_disease:
                print(f"      ↳ expected disease in top3: {expected['disease_in_top3']} | got top3: {top3}")
            if not ok_severity:
                print(f"      ↳ expected severity: {expected['severity']} | got: {p.severity}")
            if not ok_src:
                print(f"      ↳ expected severity_source: {exp_src} | got: {p.severity_source}")
            if not ok_flags:
                print(f"      ↳ expected red flags >= {exp_flags} | got: {p.red_flags_hit}")

    print("-" * 110)
    print(f"Passed {len(data['examples']) - fails}/{len(data['examples'])}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
