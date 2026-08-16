from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survival_analysis_pipeline.generate import GeneratorConfig, generate
from survival_analysis_pipeline.schema import LATENT_COLUMNS, METADATA_COLUMNS, TARGET_COLUMNS


def test_schema(small_data):
    df, latents = small_data
    assert list(df.columns) == [*METADATA_COLUMNS, *TARGET_COLUMNS]
    assert list(latents.columns) == ["strategy_id", *LATENT_COLUMNS]
    assert len(df) == 600
    assert df["strategy_id"].is_unique


def test_reproducible():
    cfg = GeneratorConfig(n_strategies=200, seed=42)
    df_a, lat_a = generate(cfg)
    df_b, lat_b = generate(cfg)
    pd.testing.assert_frame_equal(df_a, df_b)
    pd.testing.assert_frame_equal(lat_a, lat_b)


def test_seed_changes_data():
    df_a, _ = generate(GeneratorConfig(n_strategies=200, seed=1))
    df_b, _ = generate(GeneratorConfig(n_strategies=200, seed=2))
    assert not df_a["val_sharpe"].equals(df_b["val_sharpe"])


def test_regime_concentration_matches_the_implied_third_fraction(small_data):
    """Only two of the three regime fractions are emitted, because the full
    simplex is collinear with a linear model's intercept. The third is still
    recoverable, and regime_concentration must be the max over all three, not
    over the two that survived."""
    df, _ = small_data
    trend, chop = df["frac_regime_trend"], df["frac_regime_chop"]
    implied_highvol = 1.0 - trend - chop
    assert (implied_highvol >= -1e-9).all()
    expected = np.maximum(np.maximum(trend, chop), implied_highvol)
    assert np.allclose(df["regime_concentration"], expected)


def test_selection_threshold(small_data):
    df, _ = small_data
    assert (df["val_sharpe"] >= GeneratorConfig().selection_sharpe).all()


def test_family_flags_match_count(small_data):
    df, _ = small_data
    flag_cols = [c for c in df.columns if c.startswith("uses_")]
    assert (df[flag_cols].sum(axis=1) == df["n_feature_families"]).all()


def test_censoring_consistency(small_data):
    df, latents = small_data
    cutoff = pd.Timestamp(GeneratorConfig().observation_cutoff)
    follow_up = (cutoff - df["discovery_date"]).dt.days.to_numpy(dtype=float)
    assert (df["duration_days"].to_numpy() <= follow_up + 0.11).all()
    assert set(df["event"].unique()) <= {0, 1}

    true_dur = latents["true_duration_days"].to_numpy()
    events = df["event"].to_numpy() == 1
    assert np.allclose(df["duration_days"].to_numpy()[events], true_dur[events], atol=0.06)
    assert (df["duration_days"].to_numpy()[~events] <= true_dur[~events]).all()


def test_latents_aligned(small_data):
    df, latents = small_data
    assert (df["strategy_id"].to_numpy() == latents["strategy_id"].to_numpy()).all()


def test_walk_forward_consistency_predicts_survival(medium_data):
    """The core generative claim: consistent walk-forward results mark real
    edge, so uncensored survivors with high wf_positive_fraction last longer."""
    df, _ = medium_data
    dead = df[df["event"] == 1]
    high = dead[dead["wf_positive_fraction"] >= 0.75]["duration_days"]
    low = dead[dead["wf_positive_fraction"] <= 0.5]["duration_days"]
    assert high.mean() > low.mean() * 1.2


def test_impossible_selection_raises():
    with pytest.raises(RuntimeError, match="selection threshold"):
        generate(GeneratorConfig(n_strategies=50, seed=0, selection_sharpe=50.0))
