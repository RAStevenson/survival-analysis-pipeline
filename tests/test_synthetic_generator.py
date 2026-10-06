from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survival_analysis_pipeline.synthetic_generator import GeneratorConfig, generate
from survival_analysis_pipeline.synthetic_schema import (
    LATENT_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMNS,
)


def test_schema(small_data):
    strategies, latents = small_data
    assert list(strategies.columns) == [*METADATA_COLUMNS, *TARGET_COLUMNS]
    assert list(latents.columns) == ["strategy_id", *LATENT_COLUMNS]
    assert len(strategies) == 600
    assert strategies["strategy_id"].is_unique


def test_reproducible():
    generator_config = GeneratorConfig(n_strategies=200, seed=42)
    first_strategies, first_latents = generate(generator_config)
    second_strategies, second_latents = generate(generator_config)
    pd.testing.assert_frame_equal(first_strategies, second_strategies)
    pd.testing.assert_frame_equal(first_latents, second_latents)


def test_seed_changes_data():
    first_strategies, _ = generate(GeneratorConfig(n_strategies=200, seed=1))
    second_strategies, _ = generate(GeneratorConfig(n_strategies=200, seed=2))
    assert not first_strategies["val_sharpe"].equals(second_strategies["val_sharpe"])


def test_regime_concentration_matches_the_implied_third_fraction(small_data):
    """Only two of the three regime fractions are emitted, because the full
    simplex is collinear with a linear model's intercept. The third is still
    recoverable, and regime_concentration must be the max over all three, not
    over the two that survived."""
    strategies, _ = small_data
    trend, chop = strategies["frac_regime_trend"], strategies["frac_regime_chop"]
    implied_highvol = 1.0 - trend - chop
    assert (implied_highvol >= -1e-9).all()
    expected = np.maximum(np.maximum(trend, chop), implied_highvol)
    assert np.allclose(strategies["regime_concentration"], expected)


def test_selection_threshold(small_data):
    strategies, _ = small_data
    assert (strategies["val_sharpe"] >= GeneratorConfig().selection_sharpe).all()


def test_family_flags_match_count(small_data):
    strategies, _ = small_data
    flag_cols = [column for column in strategies.columns if column.startswith("uses_")]
    assert (strategies[flag_cols].sum(axis=1) == strategies["n_feature_families"]).all()


def test_censoring_consistency(small_data):
    strategies, latents = small_data
    cutoff = pd.Timestamp(GeneratorConfig().observation_cutoff)
    follow_up = (cutoff - strategies["discovery_date"]).dt.days.to_numpy(dtype=float)
    assert (strategies["duration_days"].to_numpy() <= follow_up + 0.11).all()
    assert set(strategies["event"].unique()) <= {0, 1}

    true_dur = latents["true_duration_days"].to_numpy()
    events = strategies["event"].to_numpy() == 1
    assert np.allclose(strategies["duration_days"].to_numpy()[events], true_dur[events], atol=0.06)
    assert (strategies["duration_days"].to_numpy()[~events] <= true_dur[~events]).all()


def test_latents_aligned(small_data):
    strategies, latents = small_data
    assert (strategies["strategy_id"].to_numpy() == latents["strategy_id"].to_numpy()).all()


def test_walk_forward_consistency_predicts_survival(medium_data):
    """The core generative claim: consistent walk-forward results mark real
    edge, so uncensored survivors with high wf_positive_fraction last longer."""
    strategies, _ = medium_data
    dead = strategies[strategies["event"] == 1]
    high = dead[dead["wf_positive_fraction"] >= 0.75]["duration_days"]
    low = dead[dead["wf_positive_fraction"] <= 0.5]["duration_days"]
    assert high.mean() > low.mean() * 1.2


def test_impossible_selection_raises():
    with pytest.raises(RuntimeError, match="selection threshold"):
        generate(GeneratorConfig(n_strategies=50, seed=0, selection_sharpe=50.0))
