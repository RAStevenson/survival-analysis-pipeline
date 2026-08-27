from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survival_analysis_pipeline.cv import recensor, temporal_folds


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


def test_tied_dates_that_starve_a_fold_are_refused_with_the_real_cause():
    """A year-granularity start column is an input the loader documents
    supporting. The split is strict, so a block of tied dates spanning a fold
    boundary leaves nothing before the split date. Before this check the
    symptom was XGBoost training on an empty matrix with only a warning, then a
    failure inside the Cox baseline reporting zero covariates and zero events
    and suggesting three causes, none of them this one."""
    dates = pd.Series(pd.to_datetime(["2018-01-01"] * 1200 + ["2019-01-01"] * 800))
    with pytest.raises(ValueError, match="have no training rows"):
        temporal_folds(dates, n_folds=5, min_train_frac=0.4)


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


def test_recensor_hand_case_in_hours():
    """The unit is load-bearing only here, where calendar spans meet the
    duration column, so the hand case is repeated in a non-day unit."""
    discovery = pd.Series(
        pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-05 12:00"])
    )
    duration = np.array([100.0, 300.0, 40.0])  # hours
    event = np.array([1, 1, 0])
    as_of = pd.Timestamp("2024-01-11 00:00")

    new_dur, new_ev = recensor(duration, event, discovery, as_of, time_unit="hours")
    # 240 hours of follow-up for the first two rows, 132 for the third.
    assert new_dur.tolist() == [100.0, 240.0, 40.0]
    assert new_ev.tolist() == [1, 0, 0]


def test_recensor_hand_case_in_months():
    # 2020-01-01 to 2024-01-01 is 1461 days, exactly 48 average months.
    discovery = pd.Series(pd.to_datetime(["2020-01-01", "2023-01-01"]))
    duration = np.array([50.0, 10.0])  # months
    event = np.array([1, 1])
    as_of = pd.Timestamp("2024-01-01")

    new_dur, new_ev = recensor(duration, event, discovery, as_of, time_unit="months")
    assert new_dur[0] == pytest.approx(48.0)
    assert new_dur[1] == 10.0
    assert new_ev.tolist() == [0, 1]


def test_recensor_keeps_subday_precision_of_timestamped_starts():
    """The original implementation truncated follow-up to whole days, which
    was invisible on date-resolution columns but wrong for timestamps."""
    discovery = pd.Series(pd.to_datetime(["2024-01-01 18:00"]))
    new_dur, new_ev = recensor(
        np.array([5.0]), np.array([1]), discovery, pd.Timestamp("2024-01-03 06:00")
    )
    assert new_dur[0] == 1.5
    assert new_ev[0] == 0


def test_recensor_rejects_future_discovery():
    discovery = pd.Series(pd.to_datetime(["2024-06-01"]))
    with pytest.raises(ValueError, match="precedes"):
        recensor(np.array([10.0]), np.array([1]), discovery, pd.Timestamp("2024-01-01"))


def test_tied_dates_merge_folds_sharing_a_split():
    """Start dates coarser than the fold grid snap several chunks to one
    split date; unmerged they would train identical models presented as
    distinct folds, which the flchain year-dated run exposed."""
    dates = pd.Series(
        pd.to_datetime(["2019-01-01"] * 40 + ["2020-01-01"] * 100 + ["2021-01-01"] * 60)
    )
    folds = temporal_folds(dates, n_folds=5, min_train_frac=0.2)
    splits = [f.split_date for f in folds]
    assert len(folds) < 5
    assert len(splits) == len(set(splits))
    all_test = np.concatenate([f.test_idx for f in folds])
    assert len(all_test) == len(np.unique(all_test)) == 160
    trains = [len(f.train_idx) for f in folds]
    assert trains == sorted(trains) and len(set(trains)) == len(trains)
