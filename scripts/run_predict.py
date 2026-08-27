#!/usr/bin/env python3
"""Score new rows with a model saved by run_fit_evaluate.py.

    python scripts/run_predict.py --model runs/<name> --data new_rows.csv ^
        --horizons 90,180,365

The CSV must carry the id column and every feature column the model was
trained on, under the same names; a missing feature column is an error naming
exactly what is absent. Outcome columns (duration, event, date) are not
needed and are ignored if present. Writes <data>_predictions.csv next to the
input (or --out): one row per input row with the predicted median survival
time and P(survive > h) for each horizon. Horizons and predictions are in
the time unit the model was trained with (--time-unit at fit time, days by
default), and the output column names carry that unit.

A run saves two fitted models, the boosted AFT model and the Cox baseline.
This script uses whichever scored higher on out-of-time concordance during
that run, and prints which one it used. Override with --model-type. A median
of inf means the model's survival curve for that row never reaches 0.5 inside
the observed follow-up, which is the honest answer rather than a guess.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from survival_analysis_pipeline.fit_evaluate import predict


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_predict.py",
        description="Score new rows with a saved survival model bundle.",
    )
    parser.add_argument("--model", required=True, help="run directory (or its model/ subdir)")
    parser.add_argument("--data", required=True, help="CSV of new rows to score")
    parser.add_argument(
        "--horizons",
        default="90,180,365",
        help="comma-separated horizons, in the time unit the model was trained with",
    )
    parser.add_argument(
        "--model-type",
        choices=["aft", "cox"],
        default=None,
        help="which saved model to use; default is whichever scored higher out of time",
    )
    parser.add_argument("--out", default=None, help="output CSV (default <data>_predictions.csv)")
    args = parser.parse_args()

    horizons = tuple(float(h) for h in args.horizons.split(","))
    try:
        frame = predict(args.model, args.data, horizons=horizons, model_type=args.model_type)
    except ValueError as err:
        print(err)
        raise SystemExit(2) from None

    data_path = Path(args.data)
    out = Path(args.out) if args.out else data_path.with_name(data_path.stem + "_predictions.csv")
    frame.to_csv(out, index=False)
    print(f"wrote {out} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
