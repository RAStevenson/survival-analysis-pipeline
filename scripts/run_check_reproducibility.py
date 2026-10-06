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
and Cox coefficient rankings contain near-ties that swap order under the
same drift. None of these shifts mean the model changed; each is binning or
ranking magnifying noise. The claims the reports actually make at that
level, which features lead each ranking, are checked directly instead as
top-three feature sets.

On the tolerances, plural. Exact equality is the wrong test across machines:
the same arithmetic in a different order gives a slightly different answer, a
different processor reorders floating-point summation, and a different math
library can flip a tree-split decision sitting near a tie. There are two
tolerances because the report itself makes two kinds of claim. Pooled and
fold-mean figures are the numbers the report leads with and hold the strict tolerance.
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
# Both models' calibration blocks are bin-defined, and the Cox coefficient
# ranking has the same near-tie sensitivity as the SHAP one, so each gets
# the same skip-and-check-the-top-set treatment.
SKIP_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^\.calibration_[^.]*\["),
    re.compile(r"^\.shap_top\["),
    re.compile(r"^\.cox_top\["),
)
SHAP_TOP_N = 3


def leaves(node, path=""):
    """Flatten nested JSON to (dotted path, value) pairs."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from leaves(value, f"{path}[{i}]")
    else:
        yield path, node


def main() -> None:
    """Compare two metrics files within the tolerances and exit non-zero on any disagreement."""
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
        help="for pooled, fold-mean, and every other value outside a fold record",
    )
    parser.add_argument(
        "--fold-tolerance",
        type=float,
        default=DEFAULT_FOLD_TOLERANCE,
        help="for values inside a single fold's record",
    )
    args = parser.parse_args()

    reference_path, candidate_path = Path(args.reference), Path(args.candidate)
    for path in (reference_path, candidate_path):
        if not path.exists():
            raise SystemExit(f"no such file: {path}")

    reference_raw = json.loads(reference_path.read_text())
    candidate_raw = json.loads(candidate_path.read_text())
    reference = dict(leaves(reference_raw))
    candidate = dict(leaves(candidate_raw))

    problems = []
    # A key appearing or vanishing is a structural change, not a numeric one,
    # and neither tolerance nor skipping has anything to say about it.
    for key in sorted(set(reference) - set(candidate)):
        problems.append(f"missing from {candidate_path.name}: {key}")
    for key in sorted(set(candidate) - set(reference)):
        problems.append(f"not present in {reference_path.name}: {key}")

    worst: dict[str, tuple[float, str | None]] = {"strict": (0.0, None), "fold": (0.0, None)}
    mismatches = []
    skipped = 0
    for key in sorted(set(reference) & set(candidate)):
        if any(pattern.search(key) for pattern in SKIP_PATTERNS):
            skipped += 1
            continue
        reference_value, candidate_value = reference[key], candidate[key]
        numeric = isinstance(reference_value, (int, float)) and isinstance(
            candidate_value, (int, float)
        )
        if (
            numeric
            and not isinstance(reference_value, bool)
            and not isinstance(candidate_value, bool)
        ):
            kind = "fold" if _FOLD_RE.match(key) else "strict"
            tolerance = args.fold_tolerance if kind == "fold" else args.tolerance
            delta = abs(float(reference_value) - float(candidate_value))
            if delta > worst[kind][0]:
                worst[kind] = (delta, key)
            if delta > tolerance:
                mismatches.append(
                    f"{key}: {reference_value} vs {candidate_value}  "
                    f"(moved {delta:.3g}, tolerance {tolerance:g})"
                )
        elif reference_value != candidate_value:
            mismatches.append(f"{key}: {reference_value!r} vs {candidate_value!r}")

    # The reports' one attribution claim, checked at the level it is made:
    # the same features lead, regardless of the order near-ties settle in.
    for block, label in (("shap_top", "SHAP"), ("cox_top", "Cox coefficient")):
        reference_top = [record["feature"] for record in reference_raw.get(block, [])[:SHAP_TOP_N]]
        candidate_top = [record["feature"] for record in candidate_raw.get(block, [])[:SHAP_TOP_N]]
        if set(reference_top) != set(candidate_top):
            mismatches.append(
                f"top-{SHAP_TOP_N} {label} features changed: {reference_top} vs {candidate_top}"
            )

    print(f"compared {len(set(reference) & set(candidate)) - skipped} values")
    print(
        f"skipped {skipped} composition-sensitive values (calibration bins, "
        f"SHAP and Cox coefficient rank order); top-{SHAP_TOP_N} feature sets "
        "checked instead"
    )
    for kind, tolerance in (("strict", args.tolerance), ("fold", args.fold_tolerance)):
        delta, key = worst[kind]
        location_note = f" at {key}" if key is not None else ""
        print(
            f"{kind} values: largest deviation {delta:.3g}{location_note} (tolerance {tolerance:g})"
        )

    problems.extend(mismatches)
    if problems:
        print(f"\nFAILED, {len(problems)} problems:")
        for problem in problems[:50]:
            print(f"  - {problem}")
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more")
        sys.exit(1)
    print("\nOK: the committed numbers reproduce here.")


if __name__ == "__main__":
    main()
