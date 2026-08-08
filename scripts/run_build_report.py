#!/usr/bin/env python3
"""Rebuild a report from a metrics.json and its figures.

    python scripts/run_build_report.py                 # synthetic, reports/
    python scripts/run_build_report.py --run runs/x    # a real-data run

Both variants render through one template in strategy_survival.report;
sections that need ground truth appear only when the metrics carry it.
Every number is read from metrics.json rather than transcribed, and a
figure the prose never cites fails the build. Figures are embedded as
base64 data URIs, so the HTML is one file with no external dependencies.

Synthetic mode: if reports/metrics_seed8.json exists (a pipeline run with
--seed 8, saved under that name), the seed-dependence reading in Addendum A
is generated from it; otherwise the report states the single-seed limitation
plainly. Writes reports/strategy_survival_report.html and, when Chrome is
available, prints it to reports/strategy_survival_report.pdf headlessly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json

from strategy_survival.report import (
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
    args = parser.parse_args()

    if args.run is None:
        m = json.loads((REPORTS / "metrics.json").read_text())
        seed8 = REPORTS / "metrics_seed8.json"
        m8 = json.loads(seed8.read_text()) if seed8.exists() else None
        ctx = synthetic_context(m, m8, REPORTS)
        out = REPORTS / "strategy_survival_report.html"
    else:
        run_dir = Path(args.run)
        metrics = run_dir / "metrics.json"
        if not metrics.exists():
            raise SystemExit(f"no metrics.json in {run_dir}; run scripts/run_fit_evaluate.py first")
        m = json.loads(metrics.read_text())
        ctx = real_context(m, run_dir)
        out = run_dir / "report.html"

    out.write_text(compose_report(ctx), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, self-contained)")
    pdf = out.with_suffix(".pdf")
    if emit_pdf(out, pdf):
        print(f"wrote {pdf} ({pdf.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
