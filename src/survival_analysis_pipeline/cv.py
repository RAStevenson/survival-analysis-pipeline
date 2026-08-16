"""Temporal cross-validation for survival labels.

The subtlety this module exists for: a row that started long before a fold's
split date may have ended *after* that date. Training on its final label leaks
the future. `recensor` rewrites training labels to what was knowable at the
split date -- still running then means censored then, regardless of what the
full dataset later recorded. A trading strategy discovered in 2022 and retired
in 2025 is the easy case to picture, but the same leak is a licence, a
subscription, or a machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .units import unit_seconds


@dataclass(frozen=True)
class TemporalFold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    split_date: pd.Timestamp


def temporal_folds(
    discovery_dates: pd.Series, n_folds: int = 5, min_train_frac: float = 0.4
) -> list[TemporalFold]:
    """Expanding-window folds ordered by discovery date.

    The earliest `min_train_frac` of strategies is burn-in and never tested.
    Each fold trains on everything discovered strictly before its split date,
    so train and test never overlap in time. Positional indices returned; the
    caller's frame must have a default RangeIndex.
    """
    if not discovery_dates.index.equals(pd.RangeIndex(len(discovery_dates))):
        raise ValueError("discovery_dates must have a default RangeIndex")
    order = discovery_dates.sort_values(kind="stable").index.to_numpy()
    burn_in = int(len(order) * min_train_frac)
    test_chunks = np.array_split(order[burn_in:], n_folds)

    folds: list[TemporalFold] = []
    for chunk in test_chunks:
        split_date = discovery_dates.iloc[chunk].min()
        train_mask = discovery_dates < split_date
        folds.append(
            TemporalFold(
                train_idx=np.flatnonzero(train_mask.to_numpy()),
                test_idx=np.asarray(chunk),
                split_date=split_date,
            )
        )

    # The split is strict, so a tie group straddling a chunk boundary can leave
    # a fold with nothing before its split date. Caught here because the
    # downstream symptom is misleading: XGBoost trains on an empty matrix with
    # only a warning, and the run dies later inside the Cox baseline reporting
    # zero covariates and zero events, whose suggested causes are all wrong.
    starved = [i for i, f in enumerate(folds) if len(f.train_idx) == 0]
    if starved:
        n_unique = int(discovery_dates.nunique())
        raise ValueError(
            f"folds {', '.join(str(i + 1) for i in starved)} of {n_folds} have no training rows: "
            f"{len(discovery_dates)} rows carry only {n_unique} distinct dates, so a block of "
            "tied dates spans a fold boundary and nothing falls strictly before the split. Use "
            "fewer folds, a smaller burn-in, or a start column with finer granularity than the "
            "one supplied."
        )
    return folds


def recensor(
    duration: np.ndarray,
    event: np.ndarray,
    discovery_dates: pd.Series,
    as_of: pd.Timestamp,
    time_unit: str = "days",
) -> tuple[np.ndarray, np.ndarray]:
    """Labels as they were observable at `as_of`.

    A death recorded after `as_of` becomes a censoring at `as_of`. Durations
    are floored at 1.0 timestep so AFT lower bounds stay positive for
    strategies discovered immediately before the split.

    `time_unit` is the unit the duration column is measured in. This is the
    one function where that unit is load-bearing: follow-up comes from
    calendar arithmetic and durations come from the user's column, and the
    two are compared directly. Under a mismatched unit the comparison does
    not fail, it silently either truncates every training label (durations
    finer than the assumed unit) or stops re-censoring at all and leaks the
    future (durations coarser than it).
    """
    # total_seconds rather than .dt.days, so a timestamped start column keeps
    # its sub-day precision in any unit. On date-resolution columns the two
    # agree exactly in days.
    follow_up = (as_of - discovery_dates).dt.total_seconds().to_numpy(dtype=float) / unit_seconds(
        time_unit
    )
    if (follow_up < 0).any():
        raise ValueError("as_of precedes some discovery dates; fold construction is broken")
    new_duration = np.minimum(duration, follow_up)
    new_event = ((event == 1) & (duration <= follow_up)).astype(int)
    return np.maximum(new_duration, 1.0), new_event
