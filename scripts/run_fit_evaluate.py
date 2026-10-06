#!/usr/bin/env python3
"""Fit and evaluate the survival model on your own duration data.

    python scripts/run_fit_evaluate.py --data churn.csv --name churn \
        --id-col customer_id --date-col signup_date \
        --duration-col days_active --event-col churned

The CSV needs four columns under any names: a unique id, a start date, an
observed duration (positive), and an event flag (1 = the ending was
observed, 0 = censored / still going). Durations default to days; a dataset
measured in another unit declares it with --time-unit, and the duration
column and --horizons must both be in that one unit. The start-date column
stays a calendar date. Every other column is treated as a
feature: numeric columns pass through (missing values allowed), text columns
are one-hot encoded, constant and all-null columns are dropped with a notice.
Use --drop-cols for columns that must not become features, such as anything
recorded after the outcome, and --categorical-cols for codes that read as
numbers but are labels, such as a ward or zip or product id: left numeric, a
linear model fits them one monotonic slope.

The evaluation is the same one the synthetic pipeline runs: expanding-window
temporal folds, training labels re-censored at each split date, held-out
likelihood selection, calibrated predictive scale. The oracle row and the
attribution-versus-truth check exist only for the synthetic run, where the
ground truth is known.

Outputs land in runs/<name>/ (or --out): metrics.json, figures, model/ for
scripts/run_predict.py, the rendered report, and readout.txt beside it (the
summary printed at the end). --no-report skips the report and prints the summary
without saving it; build both later, without refitting, with:

    python scripts/run_build_report.py --run runs/<name>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import subprocess
import traceback

from survival_analysis_pipeline.fit_evaluate import fit_evaluate
from survival_analysis_pipeline.metrics_readout import readout_lines
from survival_analysis_pipeline.time_units import TIME_UNITS


def main() -> None:
    """Parse the column flags, run the fit and evaluation, print the summary numbers, and build the
    report unless told not to.
    """
    parser = argparse.ArgumentParser(
        prog="run_fit_evaluate.py",
        description="Fit and evaluate the survival model on a right-censored duration CSV.",
    )
    parser.add_argument("--data", required=True, help="path to the CSV")
    parser.add_argument("--name", required=True, help="run name; outputs go to runs/<name>/")
    parser.add_argument("--id-col", required=True, help="unique id column")
    parser.add_argument("--date-col", required=True, help="start date column")
    parser.add_argument(
        "--duration-col", required=True, help="observed duration, in the --time-unit unit"
    )
    parser.add_argument("--event-col", required=True, help="1 = event observed, 0 = censored")
    parser.add_argument(
        "--drop-cols",
        default="",
        help="comma-separated columns to exclude from features (post-outcome columns leak)",
    )
    parser.add_argument(
        "--categorical-cols",
        default="",
        help="comma-separated columns to treat as labels even though they read as numbers "
        "(administrative codes: ward, zip, district, product id)",
    )
    parser.add_argument(
        "--km-col",
        default=None,
        help="categorical column for a Kaplan-Meier by-group figure in the report",
    )
    parser.add_argument("--folds", type=int, default=5, help="temporal CV folds")
    parser.add_argument(
        "--horizons",
        default="90,180,365",
        help="comma-separated survival horizons, in the --time-unit unit; "
        "calibration uses the middle one",
    )
    parser.add_argument(
        "--time-unit",
        default="days",
        choices=list(TIME_UNITS),
        help="unit of the duration column and --horizons; the whole dataset must use "
        "one unit. A too-coarse declaration is caught at load, because rows would end "
        "in the future; other mismatches silently corrupt the evaluation",
    )
    parser.add_argument("--out", default=None, help="output directory (default runs/<name>)")
    parser.add_argument(
        "--no-report", action="store_true", help="skip rendering the HTML/PDF report"
    )
    args = parser.parse_args()

    drop_cols = tuple(column.strip() for column in args.drop_cols.split(",") if column.strip())
    categorical_cols = tuple(
        column.strip() for column in args.categorical_cols.split(",") if column.strip()
    )
    horizons = tuple(float(horizon) for horizon in args.horizons.split(","))
    run_dir = Path(args.out) if args.out else Path("runs") / args.name

    try:
        metrics = fit_evaluate(
            args.data,
            name=args.name,
            id_col=args.id_col,
            date_col=args.date_col,
            duration_col=args.duration_col,
            event_col=args.event_col,
            drop_cols=drop_cols,
            categorical_cols=categorical_cols,
            n_folds=args.folds,
            horizons=horizons,
            out_dir=args.out,
            km_col=args.km_col,
            time_unit=args.time_unit,
        )
    except ValueError:
        traceback.print_exc()
        print(
            f"The run stopped, so its outputs in {run_dir} are missing or incomplete. Fix the "
            "problems named in the error above and rerun this command.",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"outputs in {run_dir}: metrics.json, figures/, model/")
    print(f"predict: python scripts/run_predict.py --model {run_dir} --data new_rows.csv")

    if args.no_report:
        # The report builder prints and saves the readout when it runs. Without it the readout
        # is only printed, so a saved readout.txt always sits beside the report it points to.
        print()
        print("\n".join(readout_lines(metrics, run_dir)))
        print(f"report skipped; build it with: python scripts/run_build_report.py --run {run_dir}")
    else:
        # Separate process on purpose, same as the synthetic pipeline: the
        # report builder is its own entry script, and a report-build failure
        # should not read as a fit failure.
        report_script = Path(__file__).resolve().parent / "run_build_report.py"
        subprocess.run([sys.executable, str(report_script), "--run", str(run_dir)], check=True)


if __name__ == "__main__":
    main()
