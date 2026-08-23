#!/usr/bin/env python3
"""Compare a freshly generated metrics file against the committed one.

    python scripts/run_check_reproducibility.py old_metrics.json reports/synthetic/metrics.json

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

On the tolerances, plural. Exact equality is the wrong test across machines:
the same arithmetic in a different order gives a slightly different answer, a
different processor reorders floating-point summation, and a different math
library can flip a tree-split decision sitting near a tie. There are two
tolerances because the report itself makes two kinds of claim. Pooled and
fold-mean figures are the headline numbers and hold the strict tolerance.
Individual fold values are described by the report's own fold-figure caption
as indicative rather than exact, because one flipped split in a small
training window moves a single fold's concordance in the third decimal while
the aggregates absorb it; they get a looser one.

Measured history behind the values. 2026-08-04 (CI, Linux, the pre-unification
path): worst drift 8.8e-4, on a per-fold concordance. 2026-08-16 (CI, Linux,
the unified path): worst drift 2.2e-3, on the smallest training window's fold
concordance, byte-identical across two runs of the same runner, so the drift
is deterministic per platform rather than noise; every pooled figure held
inside the strict tolerance. The strict tolerance is 2e-3 and the fold
tolerance 5e-3, each roughly double its measured worst case, and both small
enough that a genuine change in data, code, or selected hyperparameters still
fails loudly. A printed third decimal can flip at a rounding boundary within
these bands.

A deviation of exactly 0.0 is the expected result on the same machine, and the
script prints the largest deviation either way, because "it passed" is less
useful than "it passed and the worst value moved by 3e-16".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import re

DEFAULT_TOLERANCE = 2e-3
DEFAULT_FOLD_TOLERANCE = 5e-3

# Values inside a single fold's record; see the docstring for why these get
# their own tolerance. Everything else, the pooled block included, is strict.
_FOLD_RE = re.compile(r"^\.folds\[\d+\]\.")

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
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="for pooled and every other headline value",
    )
    parser.add_argument(
        "--fold-tolerance",
        type=float,
        default=DEFAULT_FOLD_TOLERANCE,
        help="for values inside a single fold's record",
    )
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

    worst: dict[str, tuple[float, str | None]] = {"strict": (0.0, None), "fold": (0.0, None)}
    mismatches = []
    skipped = 0
    for key in sorted(set(ref) & set(cand)):
        if any(pat.search(key) for pat in SKIP_PATTERNS):
            skipped += 1
            continue
        a, b = ref[key], cand[key]
        numeric = isinstance(a, (int, float)) and isinstance(b, (int, float))
        if numeric and not isinstance(a, bool) and not isinstance(b, bool):
            kind = "fold" if _FOLD_RE.match(key) else "strict"
            tolerance = args.fold_tolerance if kind == "fold" else args.tolerance
            delta = abs(float(a) - float(b))
            if delta > worst[kind][0]:
                worst[kind] = (delta, key)
            if delta > tolerance:
                mismatches.append(
                    f"{key}: {a} vs {b}  (moved {delta:.3g}, tolerance {tolerance:g})"
                )
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
    for kind, tolerance in (("strict", args.tolerance), ("fold", args.fold_tolerance)):
        delta, key = worst[kind]
        at = f" at {key}" if key is not None else ""
        print(f"{kind} values: largest deviation {delta:.3g}{at} (tolerance {tolerance:g})")

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
