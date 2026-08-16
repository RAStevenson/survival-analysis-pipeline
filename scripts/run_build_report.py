#!/usr/bin/env python3
"""Rebuild a run's report from its metrics.json and figures.

    python scripts/run_build_report.py                        # the synthetic run
    python scripts/run_build_report.py --run runs/x           # any other run

A run is a folder holding metrics.json, figures/, notes/, and the report
built from them, so every run renders the same way and the default is just
the committed synthetic one. Both variants render through one template in
survival_analysis_pipeline.report; the variant is chosen by what the metrics
carry rather than by a flag, and sections that need ground truth appear only
when the ground truth is there. Every number is read from metrics.json rather
than transcribed, and a figure the prose never cites fails the build. Figures
are embedded as base64 data URIs, so the HTML is one file with no external
dependencies. Writes <run>/report.html and, when Chrome is available, prints
it to <run>/report.pdf headlessly.

Authored notes live in <run>/notes/ (override with --notes). One markdown
file per anchor; see survival_analysis_pipeline/notes.py for the anchors and
the @val token syntax. An unresolvable token or an unknown anchor filename
fails the build.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import os

from survival_analysis_pipeline.report import (
    compose_report,
    emit_pdf,
    real_context,
    synthetic_context,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = Path("reports") / "synthetic"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_build_report.py",
        description="Rebuild a run's report; defaults to the committed synthetic run.",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="run directory from run_fit_evaluate.py; defaults to reports/synthetic",
    )
    parser.add_argument(
        "--notes", default=None, help="notes directory override; defaults to <run>/notes"
    )
    args = parser.parse_args()
    os.chdir(ROOT)

    run_dir = Path(args.run) if args.run else DEFAULT_RUN
    metrics = run_dir / "metrics.json"
    if not metrics.exists():
        raise SystemExit(f"no metrics.json in {run_dir}; run the pipeline first")
    m = json.loads(metrics.read_text())
    notes_dir = Path(args.notes) if args.notes else run_dir / "notes"

    # Presence, not a flag: only a run with a generating process behind it can
    # carry a generator block, and only that run can show ground truth.
    if "generator" in m:
        ctx = synthetic_context(m, run_dir, notes_dir=notes_dir)
    else:
        ctx = real_context(m, run_dir, notes_dir=notes_dir)

    out = run_dir / "report.html"
    out.write_text(compose_report(ctx), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, self-contained)")
    pdf = out.with_suffix(".pdf")
    if emit_pdf(out, pdf):
        print(f"wrote {pdf} ({pdf.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
