from __future__ import annotations

import numpy as np

from survival_analysis_pipeline.features import FEATURE_COLUMNS
from survival_analysis_pipeline.schema import ASSET_CLASSES


def test_column_order_is_stable(small_features):
    assert list(small_features.columns) == list(FEATURE_COLUMNS)


def test_all_finite(small_features):
    assert np.isfinite(small_features.to_numpy()).all()


def test_asset_one_hot_partitions(small_features):
    onehot = small_features[[f"asset_{a}" for a in ASSET_CLASSES]]
    assert (onehot.sum(axis=1) == 1).all()


def test_regime_concentration_is_max(small_data, small_features):
    df, _ = small_data
    fracs = df[["frac_regime_trend", "frac_regime_chop", "frac_regime_highvol"]]
    assert np.allclose(small_features["regime_concentration"], fracs.max(axis=1))
