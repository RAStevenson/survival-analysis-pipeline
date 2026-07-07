from __future__ import annotations

import numpy as np

from strategy_survival.evaluate import bootstrap_ci, calibration_bins, harrell_c, ipcw_brier


def test_harrell_c_perfect_and_reversed():
    durations = np.array([10.0, 20.0, 30.0, 40.0])
    events = np.ones(4, dtype=int)
    assert harrell_c(durations, events, durations) == 1.0
    assert harrell_c(durations, events, -durations) == 0.0


def test_harrell_c_ignores_uncomparable_censored_pairs():
    # Censored at 15 vs death at 10: comparable. Censored at 5 vs death at 10: not.
    durations = np.array([10.0, 15.0, 5.0])
    events = np.array([1, 0, 0])
    predicted = np.array([1.0, 2.0, 0.5])
    assert harrell_c(durations, events, predicted) == 1.0


def test_ipcw_brier_matches_plain_brier_without_censoring():
    rng = np.random.default_rng(0)
    durations = rng.uniform(10, 400, 200)
    events = np.ones(200, dtype=int)
    pred = rng.uniform(0, 1, 200)
    h = 180.0
    plain = float(np.mean(((durations > h).astype(float) - pred) ** 2))
    assert abs(ipcw_brier(durations, events, pred, h) - plain) < 1e-9


def test_ipcw_brier_rewards_perfect_predictions():
    durations = np.array([50.0, 90.0, 300.0, 400.0])
    events = np.ones(4, dtype=int)
    perfect = (durations > 180.0).astype(float)
    assert ipcw_brier(durations, events, perfect, 180.0) == 0.0
    assert ipcw_brier(durations, events, 1.0 - perfect, 180.0) > 0.5


def test_calibration_bins_cover_all_rows():
    rng = np.random.default_rng(1)
    n = 500
    pred = rng.uniform(0, 1, n)
    durations = rng.uniform(1, 700, n)
    events = rng.integers(0, 2, n)
    bins = calibration_bins(durations, events, pred, 180.0, n_bins=10)
    assert bins["n"].sum() == n
    assert bins["predicted"].is_monotonic_increasing
    assert bins["observed_km"].between(0, 1).all()


def test_bootstrap_ci_brackets_point_estimate():
    values = np.random.default_rng(2).normal(5.0, 1.0, 400)
    lo, hi = bootstrap_ci(lambda idx: float(values[idx].mean()), len(values), n_boot=300)
    assert lo < values.mean() < hi
    assert hi - lo < 0.5
