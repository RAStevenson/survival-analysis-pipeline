#!/usr/bin/env python3
"""Rebuild datasets/flchain.csv.gz from the public R-datasets mirror.

    python scripts/run_prepare_flchain.py

Source, citation and terms are in datasets/README.md. The committed file is
this script's output, so every construction decision is written down here.

The data is the flchain cohort shipped with R's survival package: a
stratified random half-sample of a Mayo Clinic study of serum free light
chain and mortality in Olmsted County residents aged 50 and over. Each row
is one subject, observed from their blood sample until death (death = 1)
or the end of follow-up. Unlike a live portal, the source is a fixed
research dataset, so re-running this script reproduces the committed file.

Construction, in order:

1. Download the CSV export of survival::flchain from the Rdatasets mirror.
2. Rename R-style columns to explicit names (sample.yr to sample_year,
   kappa to kappa_flc, lambda to lambda_flc, flc.grp to flc_group) and the
   row-number column to subject_id.
3. Drop the chapter column. It records the chapter of the death cause,
   which only exists once a subject has died, so keeping it would hand the
   model the outcome.
4. Drop the three subjects whose follow-up is zero days (died on the
   sample day). The pipeline requires positive durations, and a zero-day
   observation carries no survival information from intake anyway.
5. Derive flc_band, a three-level text banding of the assay's ten groups
   (low covers groups 1-7, mid 8-9, top 10), for the report's Kaplan-Meier
   figure and decomposition. The fit command excludes it from the features
   with --drop-cols, since it carries nothing flc_group does not.
6. Shuffle the rows with a fixed seed, then stable-sort by sample year.
   The source file is ordered by age within each year, and the true
   within-year order is unknowable, so the shuffle makes it genuinely
   arbitrary instead of secretly age-sorted. Without this, a fold
   boundary that lands inside a year cuts the cohort on age, the
   dominant feature, and manufactures fold-to-fold swings.
7. Keep missing values as they are; the pipeline passes them through.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import io
import urllib.request

import pandas as pd

SOURCE = (
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/flchain.csv"
)
OUT = Path(__file__).resolve().parents[1] / "datasets" / "flchain.csv.gz"

RENAMES = {
    "rownames": "subject_id",
    "sample.yr": "sample_year",
    "kappa": "kappa_flc",
    "lambda": "lambda_flc",
    "flc.grp": "flc_group",
}
# Recorded at death, so it is the outcome wearing a category label.
POST_OUTCOME = ("chapter",)


def main() -> None:
    print(f"downloading flchain from {SOURCE}")
    with urllib.request.urlopen(SOURCE, timeout=300) as response:
        raw = pd.read_csv(io.BytesIO(response.read()))
    frame = raw.rename(columns=RENAMES).drop(columns=list(POST_OUTCOME))
    zero = frame["futime"] <= 0
    print(f"dropping {int(zero.sum())} subjects with zero days of follow-up")
    frame = frame[~zero].reset_index(drop=True)
    frame["flc_band"] = pd.cut(
        frame["flc_group"],
        bins=[0, 7, 9, 10],
        labels=["low (groups 1-7)", "mid (groups 8-9)", "top (group 10)"],
    ).astype(str)
    frame = (
        frame.sample(frac=1.0, random_state=7)
        .sort_values("sample_year", kind="stable")
        .reset_index(drop=True)
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False, compression="gzip")
    events = int(frame["death"].sum())
    print(
        f"wrote {OUT} ({len(frame)} subjects, {events} deaths, "
        f"{100 * (1 - events / len(frame)):.1f}% censored)"
    )


if __name__ == "__main__":
    main()
