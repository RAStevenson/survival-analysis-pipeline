#!/usr/bin/env python3
"""Fit and evaluate the survival model on your own duration data.

    python scripts/run_fit_evaluate.py --data churn.csv --name churn \
        --id-col customer_id --date-col signup_date \
        --duration-col days_active --event-col churned

The CSV needs four columns under any names: a unique id, a start date, an
observed duration in days (positive), and an event flag (1 = the ending was
observed, 0 = censored / still going). Every other column is treated as a
feature: numeric columns pass through (missing values allowed), text columns
are one-hot encoded, constant and all-null columns are dropped with a notice.
Use --drop-cols for columns that must not become features, such as anything
recorded after the outcome, and --categorical-cols for codes that read as
numbers but are labels, such as a ward or zip or product id: left numeric, a
linear model fits them one monotonic slope.

The evaluation is the same one the synthetic pipeline runs: expanding-window
temporal folds, training labels re-censored at each split date, held-out
likelihood selection, calibrated predictive scale. There is no oracle row and
no attribution-versus-truth check here, because real data has no known ground
truth; the report states that rather than omitting it silently.

Outputs land in runs/<name>/ (or --out): metrics.json, figures, and model/
for scripts/run_predict.py. Build the report with:

    python scripts/run_build_report.py --run runs/<name>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from strategy_survival.realdata import fit_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_fit_evaluate.py",
        description="Fit and evaluate the survival model on a right-censored duration CSV.",
    )
    parser.add_argument("--data", required=True, help="path to the CSV")
    parser.add_argument("--name", required=True, help="run name; outputs go to runs/<name>/")
    parser.add_argument("--id-col", required=True, help="unique id column")
    parser.add_argument("--date-col", required=True, help="start date column")
    parser.add_argument("--duration-col", required=True, help="observed duration in days")
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
    parser.add_argument("--folds", type=int, default=5, help="temporal CV folds")
    parser.add_argument(
        "--horizons",
        default="90,180,365",
        help="comma-separated survival horizons in days; calibration uses the middle one",
    )
    parser.add_argument("--out", default=None, help="output directory (default runs/<name>)")
    args = parser.parse_args()

    drop_cols = tuple(c.strip() for c in args.drop_cols.split(",") if c.strip())
    categorical_cols = tuple(c.strip() for c in args.categorical_cols.split(",") if c.strip())
    horizons = tuple(float(h) for h in args.horizons.split(","))

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
            horizons_days=horizons,
            out_dir=args.out,
        )
    except ValueError as err:
        print(err)
        raise SystemExit(2) from None

    pooled = metrics["pooled"]
    out = Path(args.out) if args.out else Path("runs") / args.name
    aft_fold, cox_fold = pooled["c_xgb_by_fold_mean"], pooled["c_cox_by_fold_mean"]
    print(
        f"pooled C-index  xgb {pooled['c_xgb']:.3f} "
        f"[{pooled['c_xgb_ci'][0]:.3f}, {pooled['c_xgb_ci'][1]:.3f}]"
    )
    print(f"fold-mean C     xgb {aft_fold:.3f}   cox {cox_fold:.3f}")
    # Name it the way --model-type spells it, or the obvious next command fails.
    winner = "cox" if cox_fold > aft_fold else "aft"
    print(f"both models saved; {winner} scored higher and is the default for run_predict.py")
    print(f"outputs in {out}: metrics.json, figures/, model/")
    print(f"report:  python scripts/run_build_report.py --run {out}")
    print(f"predict: python scripts/run_predict.py --model {out} --data new_rows.csv")


if __name__ == "__main__":
    main()
