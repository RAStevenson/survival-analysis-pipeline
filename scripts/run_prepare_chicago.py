#!/usr/bin/env python3
"""Rebuild datasets/chicago_licences.csv.gz from the City of Chicago portal.

    python scripts/run_prepare_chicago.py

Source, citation and terms of use are in datasets/README.md. The committed
file is this script's output, so every construction decision is written down
here rather than done by hand.

What each row becomes: one Chicago business licence, observed from the day it
was first issued until the business stopped holding it (event = 1) or the
data cutoff, whichever came first.

Why this dataset suits a temporal-CV tool. The city records issuance,
renewal and cancellation continuously in one system, so a licence issued in
2004 has its whole life on file. Registry snapshots do not have that
property: they list what exists now plus recent closures, so older entities
carry years of unobserved history before the records begin. That is left
truncation, this pipeline assumes observation starts at time zero, and a
dataset without complete follow-up silently starves the early folds of
events. Two registries were tried and rejected for exactly this reason,
which is noted in datasets/README.md.

The same test has to be applied inside this dataset, not just across
candidate datasets, and step 2 below is where it happens. The pull starts at
2002 because that is where the portal's status history becomes complete, but
a licence that was already running in 2001 still appears, entering at its
first post-2002 renewal. Its recorded start is that renewal date and its
recorded span is what was left of its life, conditioned on having already
survived at least one term. Left uncorrected that was 79,690 rows, 23 percent
of the pull, it put a spike of 61,351 licences on the 2002 boundary while genuine first
issues ran flat at about 14,000 a year, and it taught the model that a
renewal predicts long life when all it really marks is a survivor. Keeping
only licences whose earliest transaction is a genuine ISSUE removes it.

Construction, in order:

1. Pull every licence transaction starting after 2002-01-01, the point from
   which the portal's status history is complete. One row per issuance or
   renewal; roughly three and a half per licence.
2. Group by licence number, then keep only licences whose earliest
   transaction is an ISSUE. Anything else means the licence predates the
   pull window and is left-truncated. Start is the earliest licence start
   date.
3. End is the cancellation or revocation date where one exists, otherwise the
   latest expiry on record. A business that simply stops renewing has closed,
   and that is by far the common case, so counting only explicit
   cancellations would miss most closures.
4. A licence whose end date has passed by the cutoff is an observed closure
   (event = 1). One still current is censored at the cutoff (event = 0).
5. Licence types that are event-scoped by construction (the Special Event,
   Pop-Up, and Itinerant variants) are dropped. Their durations are set by
   the permit's own terms, so a model that learns them learns to recognize
   permit types, not business failure.
6. Features are restricted to what was knowable on the first day, taken from
   the earliest transaction. Renewal count is deliberately excluded: more
   renewals means a longer life, so it is the outcome in disguise.

The output is gzipped. It is 35 MB as plain text and under 7 MB compressed,
and pandas reads either transparently.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import io
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ENDPOINT = "https://data.cityofchicago.org/resource/r5kz-chrr.csv"
OUT = Path(__file__).resolve().parents[1] / "datasets" / "chicago_licences.csv.gz"

# Fixed so the committed file is reproducible: a floating "today" would move
# every censoring boundary and silently change the dataset between runs.
CUTOFF = pd.Timestamp("2026-08-01")
HISTORY_STARTS = "2002-01-01"

PULL_COLUMNS = (
    "license_number,license_description,license_start_date,expiration_date,"
    "license_status,license_status_change_date,application_type,"
    "conditional_approval,ward,community_area,police_district,zip_code,"
    "latitude,longitude"
)
CLOSED_STATUSES = ("AAC", "REV")
# Licence types that are event-scoped by construction. The licence is issued
# for days or weeks, so its recorded lifespan measures the permit's term, not
# the business's survival. A temporary food licence was always going to
# expire within days, while the handyman whose business folded never meant
# it to end, and only the second belongs in a survival study. On the
# 2026-08-01 pull this removes 23,042 licences across 12 types (median life
# 4 days, 98.3% closed).
EVENT_SCOPED_TERMS = ("Special Event", "Pop-Up", "Itinerant")
# A licence whose earliest transaction is not a first issue was already running
# before the pull window, so its recorded start is a renewal date rather than
# its time zero. See the module docstring.
FIRST_ISSUE_CODE = "ISSUE"
# Administrative codes, not quantities: ward 50 is not five times ward 10, and
# left numeric the zip code's five-digit scale would swamp a penalised linear
# model. Writing them as text is not enough to keep them that way, because the
# next read_csv re-infers "42" as an integer. The fit command has to name them
# with --categorical-cols; see datasets/README.md.
CODE_COLUMNS = ("ward", "community_area", "police_district", "zip_code")
FIRST_ROW_FEATURES = (
    "license_description",
    "application_type",
    "conditional_approval",
    *CODE_COLUMNS,
    "latitude",
    "longitude",
)


def main() -> None:
    """Pull the licence transactions from the city portal, apply the cleaning rules, and write the
    committed dataset.
    """
    params = {
        "$select": PULL_COLUMNS,
        "$where": f"license_start_date > '{HISTORY_STARTS}'",
        "$limit": "1500000",
    }
    url = f"{ENDPOINT}?{urllib.parse.urlencode(params)}"
    print("downloading Chicago business licences from data.cityofchicago.org")
    with urllib.request.urlopen(url, timeout=900) as response:
        raw = pd.read_csv(io.BytesIO(response.read()), low_memory=False)
    print(
        f"{len(raw)} licence transactions, {raw['license_number'].nunique()} licences, "
        f"{raw['license_description'].nunique()} licence types"
    )

    for col in ("license_start_date", "expiration_date", "license_status_change_date"):
        raw[col] = pd.to_datetime(raw[col], errors="coerce")
    undated = raw["license_start_date"].isna()
    if undated.any():
        print(f"dropping {int(undated.sum())} transactions with no start date")
        raw = raw[~undated]

    cancelled = raw["license_status"].isin(CLOSED_STATUSES)
    cancelled_on = raw[cancelled].groupby("license_number")["license_status_change_date"].min()
    groups = raw.groupby("license_number")
    frame = pd.DataFrame(
        {
            "first_issued": groups["license_start_date"].min(),
            "last_expiry": groups["expiration_date"].max(),
            "cancelled_on": cancelled_on,
        }
    )
    # The earliest transaction's own row supplies the covariates, nulls
    # included. drop_duplicates keeps that whole row; groupby.first() would
    # take the first non-null per column, letting a later renewal fill a gap
    # with post-start information.
    first_rows = (
        raw.sort_values("license_start_date", kind="stable")
        .drop_duplicates("license_number")
        .set_index("license_number")
    )
    for col in FIRST_ROW_FEATURES:
        frame[col] = first_rows[col]
    frame = frame.reset_index()

    truncated = frame["application_type"] != FIRST_ISSUE_CODE
    print(
        f"dropping {int(truncated.sum())} licences whose earliest transaction is a "
        f"{'/'.join(sorted(frame.loc[truncated, 'application_type'].unique()))} rather than an "
        f"{FIRST_ISSUE_CODE}: they were already running before the pull window, so their "
        "recorded start is not their time zero"
    )
    frame = frame[~truncated].reset_index(drop=True)

    end = np.where(frame["cancelled_on"].notna(), frame["cancelled_on"], frame["last_expiry"])
    frame["ended_on"] = pd.to_datetime(end)
    frame["closed"] = (frame["ended_on"] < CUTOFF).astype(int)
    frame.loc[frame["ended_on"] >= CUTOFF, "ended_on"] = CUTOFF
    frame["licensed_days"] = (frame["ended_on"] - frame["first_issued"]).dt.days

    bad = ~(frame["licensed_days"] > 0) | (frame["first_issued"] >= CUTOFF)
    print(f"dropping {int(bad.sum())} licences with a non-positive or future observed span")
    frame = frame[~bad].reset_index(drop=True)

    event_scoped = frame["license_description"].str.contains(
        "|".join(EVENT_SCOPED_TERMS), case=False, na=False
    )
    print(
        f"dropping {int(event_scoped.sum())} event-scoped licences "
        f"({', '.join(sorted(frame.loc[event_scoped, 'license_description'].unique()))}): "
        "issued to expire, so their short lives are intent rather than failure"
    )
    frame = frame[~event_scoped].reset_index(drop=True)

    for col in CODE_COLUMNS:
        frame[col] = (
            frame[col].astype(str).str.replace(r"\.0$", "", regex=True).replace({"nan": np.nan})
        )
    frame["first_issued"] = frame["first_issued"].dt.strftime("%Y-%m-%d")
    frame = frame.drop(columns=["last_expiry", "cancelled_on", "ended_on"])
    frame = frame.rename(columns={"license_number": "licence_id"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False, compression="gzip")
    events = int(frame["closed"].sum())
    size_mb = OUT.stat().st_size / 1e6
    print(
        f"wrote {OUT} ({len(frame)} licences, {events} closures, "
        f"{100 * (1 - events / len(frame)):.1f}% censored, cutoff {CUTOFF.date()}, "
        f"{size_mb:.1f} MB gzipped)"
    )


if __name__ == "__main__":
    main()
