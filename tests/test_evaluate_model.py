from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survival_analysis_pipeline.evaluate_model import (
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
    random_generator = np.random.default_rng(0)
    durations = random_generator.uniform(10, 400, 200)
    events = np.ones(200, dtype=int)
    predicted_survival = random_generator.uniform(0, 1, 200)
    horizon = 180.0
    plain = float(np.mean(((durations > horizon).astype(float) - predicted_survival) ** 2))
    assert abs(ipcw_brier(durations, events, predicted_survival, horizon) - plain) < 1e-9


def test_ipcw_brier_hand_case_with_censoring():
    """The weights are the estimator, and no other test exercises them: a row
    dead by the horizon contributes s^2 / G(t-), a row still alive at the
    horizon (1-s)^2 / G(h), and a row censored before the horizon nothing,
    with G the Kaplan-Meier curve of the censoring process. Here G steps only
    at the censoring at t=100 with three rows at risk, so G(50-) = 1 and
    G(150) = 2/3, and the score is computable by hand."""
    durations = np.array([50.0, 100.0, 200.0, 300.0])
    events = np.array([1, 0, 1, 0])
    predicted_survival = np.array([0.2, 0.9, 0.8, 0.7])

    expected = (0.2**2 / 1.0 + (1 - 0.8) ** 2 / (2 / 3) + (1 - 0.7) ** 2 / (2 / 3)) / 4
    assert ipcw_brier(durations, events, predicted_survival, 150.0) == pytest.approx(expected)

    # The censored row's prediction must not matter: its weight is zero.
    moved = predicted_survival.copy()
    moved[1] = 0.1
    assert ipcw_brier(durations, events, moved, 150.0) == pytest.approx(expected)


def test_ipcw_brier_rewards_perfect_predictions():
    durations = np.array([50.0, 90.0, 300.0, 400.0])
    events = np.ones(4, dtype=int)
    perfect = (durations > 180.0).astype(float)
    assert ipcw_brier(durations, events, perfect, 180.0) == 0.0
    assert ipcw_brier(durations, events, 1.0 - perfect, 180.0) > 0.5


def test_calibration_bins_cover_all_rows():
    random_generator = np.random.default_rng(1)
    n_rows = 500
    predicted_survival = random_generator.uniform(0, 1, n_rows)
    durations = random_generator.uniform(1, 700, n_rows)
    events = random_generator.integers(0, 2, n_rows)
    bins = calibration_bins(durations, events, predicted_survival, 180.0, n_bins=10)
    assert bins["n"].sum() == n_rows
    assert bins["predicted"].is_monotonic_increasing
    assert bins["observed_km"].between(0, 1).all()


def test_bootstrap_ci_brackets_point_estimate():
    values = np.random.default_rng(2).normal(5.0, 1.0, 400)
    lower, upper = bootstrap_ci(
        lambda sample_rows: float(values[sample_rows].mean()), len(values), n_resamples=300
    )
    assert lower < values.mean() < upper
    assert upper - lower < 0.5


def test_within_group_concordance_separates_group_and_row_skill():
    # Two groups whose typical durations differ by an order of magnitude.
    # Scores carry the group mean plus row-level noise uncorrelated with
    # duration: group membership ranks almost everything, rows add nothing.
    random_generator = np.random.default_rng(3)
    n_per_group = 400
    group = pd.Series(["short"] * n_per_group + ["long"] * n_per_group)
    durations = np.concatenate(
        [
            random_generator.uniform(5, 50, n_per_group),
            random_generator.uniform(500, 5000, n_per_group),
        ]
    )
    events = np.ones(2 * n_per_group, dtype=int)
    scores = np.where(group == "short", 10.0, 1000.0) + random_generator.normal(
        0, 1, 2 * n_per_group
    )
    decomposition = within_group_concordance(
        durations, events, scores, group, min_rows=50, min_events=10
    )
    assert decomposition is not None
    assert decomposition["n_groups"] == 2
    # Within-group pairs are score-ties under a group-mean ranking and count
    # 0.5 in Harrell C, so even perfect cross-group separation tops out well
    # below 1.0 here; the point is the gap against c_within at the coin flip.
    assert decomposition["c_group_mean"] > 0.7
    assert abs(decomposition["c_within"] - 0.5) < 0.05


def test_within_group_concordance_detects_row_skill():
    # Scores equal durations exactly: within-group ranking is perfect in
    # every group, so the pair-weighted within figure is 1.0.
    random_generator = np.random.default_rng(4)
    n_rows = 300
    group = pd.Series(random_generator.choice(["a", "b", "c"], size=n_rows))
    durations = random_generator.uniform(10, 1000, n_rows)
    events = np.ones(n_rows, dtype=int)
    decomposition = within_group_concordance(
        durations, events, durations, group, min_rows=20, min_events=5
    )
    assert decomposition is not None
    assert decomposition["c_within"] == 1.0


def test_within_group_concordance_filters_small_groups():
    group = pd.Series(["big"] * 100 + ["tiny"] * 5)
    durations = np.arange(105, dtype=float) + 1
    events = np.ones(105, dtype=int)
    decomposition = within_group_concordance(
        durations, events, durations, group, min_rows=50, min_events=10
    )
    assert decomposition is not None
    assert decomposition["n_groups"] == 1
    out_none = within_group_concordance(
        durations, events, durations, group, min_rows=500, min_events=10
    )
    assert out_none is None
