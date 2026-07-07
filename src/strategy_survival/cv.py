"""Temporal cross-validation for survival labels.

The subtlety this module exists for: a strategy discovered long before a fold's
split date may have died *after* that date. Training on its final label leaks
the future. `recensor` rewrites training labels to what was knowable at the
split date -- still alive then means censored then, regardless of what the full
dataset later recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
    return folds


def recensor(
    duration_days: np.ndarray,
    event: np.ndarray,
    discovery_dates: pd.Series,
    as_of: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    """Labels as they were observable at `as_of`.

    A death recorded after `as_of` becomes a censoring at `as_of`. Durations
    are floored at 1.0 day so AFT lower bounds stay positive for strategies
    discovered immediately before the split.
    """
    follow_up = (as_of - discovery_dates).dt.days.to_numpy(dtype=float)
    if (follow_up < 0).any():
        raise ValueError("as_of precedes some discovery dates; fold construction is broken")
    new_duration = np.minimum(duration_days, follow_up)
    new_event = ((event == 1) & (duration_days <= follow_up)).astype(int)
    return np.maximum(new_duration, 1.0), new_event
