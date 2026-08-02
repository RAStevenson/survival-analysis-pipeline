#!/usr/bin/env python3
"""Rebuild datasets/chicago_licences.csv.gz from the City of Chicago portal.

    python scripts/run_prepare_chicago.py

Source, citation and the outstanding terms-of-use question are in
datasets/README.md. The committed file is this script's output, so every
construction decision is written down here rather than done by hand.

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

Construction, in order:

1. Pull every licence transaction starting after 2002-01-01, the point from
   which the portal's status history is complete. One row per issuance or
   renewal; roughly three and a half per licence.
2. Group by licence number. Start is the earliest licence start date.
3. End is the cancellation or revocation date where one exists, otherwise the
   latest expiry on record. A business that simply stops renewing has closed,
   and that is by far the common case, so counting only explicit
   cancellations would miss most closures.
4. A licence whose end date has passed by the cutoff is an observed closure
   (event = 1). One still current is censored at the cutoff (event = 0).
5. Features are restricted to what was knowable on the first day, taken from
   the earliest transaction. Renewal count is deliberately excluded: more
   renewals means a longer life, so it is the outcome in disguise.

The output is gzipped. It is 35 MB as plain text and under 7 MB compressed,
and pandas reads either transparently.
"""

import io
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
# Administrative codes, not quantities: ward 50 is not five times ward 10, and
# left numeric the zip code's five-digit scale would swamp a penalised linear
# model. Carried as labels so they are one-hot encoded downstream.
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
    # Earliest transaction supplies the covariates, so nothing recorded after
    # the licence began can reach the model.
    first_rows = raw.sort_values("license_start_date").groupby("license_number").first()
    for col in FIRST_ROW_FEATURES:
        frame[col] = first_rows[col]
    frame = frame.reset_index()

    end = np.where(frame["cancelled_on"].notna(), frame["cancelled_on"], frame["last_expiry"])
    frame["ended_on"] = pd.to_datetime(end)
    frame["closed"] = (frame["ended_on"] < CUTOFF).astype(int)
    frame.loc[frame["ended_on"] >= CUTOFF, "ended_on"] = CUTOFF
    frame["licensed_days"] = (frame["ended_on"] - frame["first_issued"]).dt.days

    bad = ~(frame["licensed_days"] > 0) | (frame["first_issued"] >= CUTOFF)
    print(f"dropping {int(bad.sum())} licences with a non-positive or future observed span")
    frame = frame[~bad].reset_index(drop=True)

    for col in CODE_COLUMNS:
        frame[col] = (
            frame[col].astype(str).str.replace(r"\.0$", "", regex=True).replace({"nan": None})
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
