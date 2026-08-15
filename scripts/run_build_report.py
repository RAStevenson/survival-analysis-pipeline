#!/usr/bin/env python3
"""Rebuild a report from a metrics.json and its figures.

    python scripts/run_build_report.py                 # synthetic, reports/
    python scripts/run_build_report.py --run runs/x    # a real-data run

Both variants render through one template in survival_analysis_pipeline.report;
sections that need ground truth appear only when the metrics carry it.
Every number is read from metrics.json rather than transcribed, and a
figure the prose never cites fails the build. Figures are embedded as
base64 data URIs, so the HTML is one file with no external dependencies.

Synthetic mode: if reports/metrics_seed8.json exists (a pipeline run with
--seed 8, saved under that name), the seed-dependence reading in Addendum A
is generated from it; otherwise the report states the single-seed limitation
plainly. Writes reports/synthetic_report.html and, when Chrome is
available, prints it to reports/synthetic_report.pdf headlessly.

Authored notes: the synthetic run reads notes/synthetic/, a real run reads
<run>/notes/ (override with --notes). One markdown file per anchor; see
survival_analysis_pipeline/notes.py for anchors and the @val token syntax.
An unresolvable token or an unknown anchor filename fails the build.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json

from survival_analysis_pipeline.report import (
    compose_report,
    emit_pdf,
    real_context,
    synthetic_context,
)

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_build_report.py",
        description="Rebuild the synthetic report, or a real-data run's report with --run.",
    )
    parser.add_argument("--run", default=None, help="run directory from run_fit_evaluate.py")
    parser.add_argument(
        "--notes",
        default=None,
        help="notes directory override; defaults to notes/synthetic/ for the "
        "synthetic report and <run>/notes/ for a real-data run",
    )
    args = parser.parse_args()
    notes_dir = Path(args.notes) if args.notes else None

    if args.run is None:
        m = json.loads((REPORTS / "metrics.json").read_text())
        seed8 = REPORTS / "metrics_seed8.json"
        m8 = json.loads(seed8.read_text()) if seed8.exists() else None
        ctx = synthetic_context(
            m, m8, REPORTS, notes_dir=notes_dir or REPORTS.parent / "notes" / "synthetic"
        )
        out = REPORTS / "synthetic_report.html"
    else:
        run_dir = Path(args.run)
        metrics = run_dir / "metrics.json"
        if not metrics.exists():
            raise SystemExit(f"no metrics.json in {run_dir}; run scripts/run_fit_evaluate.py first")
        m = json.loads(metrics.read_text())
        ctx = real_context(m, run_dir, notes_dir=notes_dir)
        out = run_dir / "report.html"

    out.write_text(compose_report(ctx), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, self-contained)")
    pdf = out.with_suffix(".pdf")
    if emit_pdf(out, pdf):
        print(f"wrote {pdf} ({pdf.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
