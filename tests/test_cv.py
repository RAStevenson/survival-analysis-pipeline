from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_survival.cv import recensor, temporal_folds


def test_folds_never_leak_time(small_data):
    df, _ = small_data
    folds = temporal_folds(df["discovery_date"], n_folds=4)
    assert len(folds) == 4
    for fold in folds:
        train_dates = df["discovery_date"].iloc[fold.train_idx]
        test_dates = df["discovery_date"].iloc[fold.test_idx]
        assert train_dates.max() < fold.split_date
        assert test_dates.min() >= fold.split_date


def test_folds_disjoint_and_cover(small_data):
    df, _ = small_data
    folds = temporal_folds(df["discovery_date"], n_folds=4, min_train_frac=0.4)
    all_test = np.concatenate([f.test_idx for f in folds])
    assert len(all_test) == len(np.unique(all_test))
    assert len(all_test) == len(df) - int(len(df) * 0.4)


def test_folds_expand(small_data):
    df, _ = small_data
    folds = temporal_folds(df["discovery_date"], n_folds=4)
    sizes = [len(f.train_idx) for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[0] >= int(len(df) * 0.4) - 1


def test_folds_require_range_index(small_data):
    df, _ = small_data
    shuffled = df.sample(frac=1.0, random_state=0)
    with pytest.raises(ValueError, match="RangeIndex"):
        temporal_folds(shuffled["discovery_date"])


def test_recensor_hand_case():
    discovery = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-01", "2024-03-01"]))
    duration = np.array([100.0, 200.0, 40.0])
    event = np.array([1, 1, 0])
    as_of = pd.Timestamp("2024-05-30")

    new_dur, new_ev = recensor(duration, event, discovery, as_of)
    # 150 days of follow-up for the first two rows, 90 for the third.
    assert new_dur.tolist() == [100.0, 150.0, 40.0]
    assert new_ev.tolist() == [1, 0, 0]


def test_recensor_floors_duration():
    discovery = pd.Series(pd.to_datetime(["2024-01-01"]))
    new_dur, new_ev = recensor(
        np.array([300.0]), np.array([1]), discovery, pd.Timestamp("2024-01-01")
    )
    assert new_dur[0] == 1.0
    assert new_ev[0] == 0


def test_recensor_rejects_future_discovery():
    discovery = pd.Series(pd.to_datetime(["2024-06-01"]))
    with pytest.raises(ValueError, match="precedes"):
        recensor(np.array([10.0]), np.array([1]), discovery, pd.Timestamp("2024-01-01"))
