#!/usr/bin/env python3
"""Compare a freshly generated metrics file against the committed one.

    python scripts/run_check_reproducibility.py reports/metrics.json reports/metrics_ci.json

Answers one question: do the published numbers still hold on a machine that did
not produce them? Exits non-zero if a performance metric moved more than the
tolerance, or if the two files no longer have the same shape.

Two classes of value, treated differently. Performance metrics (concordance,
Brier, the selected parameters) are compared against the tolerance: they are
what the reports publish. Composition-sensitive diagnostics are skipped, with
a printed notice: the calibration table's bins are defined by predicted
probability, so a prediction drifting by a millionth can move a row across a
bin edge and shift that bin's observed value by a large amount, and the SHAP
ranking contains near-ties that swap order under the same drift. Neither
shift means the model changed; both are the bin or rank lens magnifying
noise. The one SHAP claim the reports actually make, that the three
walk-forward statistics lead the attribution, is checked directly instead.

On the tolerance. Exact equality is the wrong test across machines: the same
arithmetic in a different order gives a slightly different answer, and a
different processor reorders floating-point summation. The first run on
foreign hardware (CI, Linux, 2026-08-04) measured a worst performance-metric
drift of 8.8e-4 on a per-fold concordance against the Windows-produced
committed file. The tolerance is 2e-3: about double the measured worst case,
and small enough that any genuine change in data, code, or selected
hyperparameters still fails loudly. A printed third decimal can flip at a
rounding boundary within this band; the reproducibility notes say so.

A deviation of exactly 0.0 is the expected result on the same machine, and the
script prints the largest deviation either way, because "it passed" is less
useful than "it passed and the worst value moved by 3e-16".
"""

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_TOLERANCE = 2e-3

# Bin membership and near-tied ranks amplify float noise; see the docstring.
SKIP_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^\.calibration_\d+d\["),
    re.compile(r"^\.shap_top\["),
)
SHAP_TOP_N = 3


def leaves(obj, path=""):
    """Flatten nested JSON to (dotted path, value) pairs."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from leaves(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from leaves(value, f"{path}[{i}]")
    else:
        yield path, obj


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_check_reproducibility.py",
        description="Fail if a regenerated metrics file disagrees with the committed one.",
    )
    parser.add_argument("reference", help="the committed metrics file")
    parser.add_argument("candidate", help="the freshly generated one")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    ref_path, cand_path = Path(args.reference), Path(args.candidate)
    for p in (ref_path, cand_path):
        if not p.exists():
            raise SystemExit(f"no such file: {p}")

    ref_raw = json.loads(ref_path.read_text())
    cand_raw = json.loads(cand_path.read_text())
    ref = dict(leaves(ref_raw))
    cand = dict(leaves(cand_raw))

    problems = []
    # A key appearing or vanishing is a structural change, not a numeric one,
    # and neither tolerance nor skipping has anything to say about it.
    for key in sorted(set(ref) - set(cand)):
        problems.append(f"missing from {cand_path.name}: {key}")
    for key in sorted(set(cand) - set(ref)):
        problems.append(f"not present in {ref_path.name}: {key}")

    worst_delta, worst_key = 0.0, None
    mismatches = []
    skipped = 0
    for key in sorted(set(ref) & set(cand)):
        if any(pat.search(key) for pat in SKIP_PATTERNS):
            skipped += 1
            continue
        a, b = ref[key], cand[key]
        numeric = isinstance(a, (int, float)) and isinstance(b, (int, float))
        if numeric and not isinstance(a, bool) and not isinstance(b, bool):
            delta = abs(float(a) - float(b))
            if delta > worst_delta:
                worst_delta, worst_key = delta, key
            if delta > args.tolerance:
                mismatches.append(f"{key}: {a} vs {b}  (moved {delta:.3g})")
        elif a != b:
            mismatches.append(f"{key}: {a!r} vs {b!r}")

    # The reports' one attribution claim, checked at the level it is made:
    # the same features lead, regardless of the order near-ties settle in.
    ref_top = [r["feature"] for r in ref_raw.get("shap_top", [])[:SHAP_TOP_N]]
    cand_top = [r["feature"] for r in cand_raw.get("shap_top", [])[:SHAP_TOP_N]]
    if set(ref_top) != set(cand_top):
        mismatches.append(f"top-{SHAP_TOP_N} SHAP features changed: {ref_top} vs {cand_top}")

    print(f"compared {len(set(ref) & set(cand)) - skipped} values")
    print(
        f"skipped {skipped} composition-sensitive values (calibration bins, "
        f"SHAP rank order); top-{SHAP_TOP_N} SHAP feature set checked instead"
    )
    if worst_key is not None:
        print(f"largest numeric deviation: {worst_delta:.3g} at {worst_key}")
    print(f"tolerance: {args.tolerance:g}")

    problems.extend(mismatches)
    if problems:
        print(f"\nFAILED, {len(problems)} problems:")
        for p in problems[:50]:
            print(f"  - {p}")
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more")
        sys.exit(1)
    print("\nOK: the committed numbers reproduce here.")


if __name__ == "__main__":
    main()
