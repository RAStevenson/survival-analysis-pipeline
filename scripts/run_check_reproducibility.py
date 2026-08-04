#!/usr/bin/env python3
"""Compare a freshly generated metrics file against the committed one.

    python scripts/run_check_reproducibility.py reports/metrics.json reports/metrics_ci.json

Answers one question: do the published numbers still hold on a machine that did
not produce them? Exits non-zero if any value moved more than the tolerance, or
if the two files no longer have the same shape.

On the tolerance. Exact equality is the wrong test across machines. The same
arithmetic in a different order gives a slightly different answer, and a
different processor or maths library reorders it. What the repository actually
claims is the numbers it prints, which are quoted to three decimals, so the
default tolerance is 5e-4: half a unit in the last published place. Anything
smaller is invisible in every report and figure; anything larger is a real
change and should stop the build.

A deviation of exactly 0.0 is the expected result on the same machine, and the
script prints the largest deviation either way, because "it passed" is less
useful than "it passed and the worst value moved by 3e-16".
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_TOLERANCE = 5e-4


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

    ref = dict(leaves(json.loads(ref_path.read_text())))
    cand = dict(leaves(json.loads(cand_path.read_text())))

    problems = []
    # A key appearing or vanishing is a structural change, not a numeric one,
    # and tolerance has nothing to say about it.
    for key in sorted(set(ref) - set(cand)):
        problems.append(f"missing from {cand_path.name}: {key}")
    for key in sorted(set(cand) - set(ref)):
        problems.append(f"not present in {ref_path.name}: {key}")

    worst_delta, worst_key = 0.0, None
    mismatches = []
    for key in sorted(set(ref) & set(cand)):
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

    print(f"compared {len(set(ref) & set(cand))} values")
    if worst_key is not None:
        print(f"largest numeric deviation: {worst_delta:.3g} at {worst_key}")
    print(f"tolerance: {args.tolerance:g}")

    problems.extend(mismatches)
    if problems:
        print(f"\nFAILED, {len(problems)} problems:")
        for p in problems[:25]:
            print(f"  - {p}")
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more")
        sys.exit(1)
    print("\nOK: the committed numbers reproduce here.")


if __name__ == "__main__":
    main()
