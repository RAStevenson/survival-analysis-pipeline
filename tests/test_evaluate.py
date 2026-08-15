from __future__ import annotations

import numpy as np
import pandas as pd

from survival_analysis_pipeline.evaluate import (
    bootstrap_ci,
    calibration_bins,
    harrell_c,
    ipcw_brier,
    within_group_concordance,
)


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


def test_within_group_concordance_separates_group_and_row_skill():
    # Two groups whose typical durations differ by an order of magnitude.
    # Scores carry the group mean plus row-level noise uncorrelated with
    # duration: group membership ranks almost everything, rows add nothing.
    rng = np.random.default_rng(3)
    n = 400
    group = pd.Series(["short"] * n + ["long"] * n)
    durations = np.concatenate([rng.uniform(5, 50, n), rng.uniform(500, 5000, n)])
    events = np.ones(2 * n, dtype=int)
    scores = np.where(group == "short", 10.0, 1000.0) + rng.normal(0, 1, 2 * n)
    out = within_group_concordance(durations, events, scores, group, min_n=50, min_events=10)
    assert out is not None
    assert out["n_groups"] == 2
    # Within-group pairs are score-ties under a group-mean ranking and count
    # 0.5 in Harrell C, so even perfect cross-group separation tops out well
    # below 1.0 here; the point is the gap against c_within at the coin flip.
    assert out["c_group_mean"] > 0.7
    assert abs(out["c_within"] - 0.5) < 0.05


def test_within_group_concordance_detects_row_skill():
    # Scores equal durations exactly: within-group ranking is perfect in
    # every group, so the pair-weighted within figure is 1.0.
    rng = np.random.default_rng(4)
    n = 300
    group = pd.Series(rng.choice(["a", "b", "c"], size=n))
    durations = rng.uniform(10, 1000, n)
    events = np.ones(n, dtype=int)
    out = within_group_concordance(durations, events, durations, group, min_n=20, min_events=5)
    assert out is not None
    assert out["c_within"] == 1.0


def test_within_group_concordance_filters_small_groups():
    group = pd.Series(["big"] * 100 + ["tiny"] * 5)
    durations = np.arange(105, dtype=float) + 1
    events = np.ones(105, dtype=int)
    out = within_group_concordance(durations, events, durations, group, min_n=50, min_events=10)
    assert out is not None
    assert out["n_groups"] == 1
    out_none = within_group_concordance(
        durations, events, durations, group, min_n=500, min_events=10
    )
    assert out_none is None
