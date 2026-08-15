"""Model feature matrix from raw metadata.

FEATURE_COLUMNS fixes the column order; the model, SHAP analysis, and tests
all rely on it being stable. Everything is float64 and finite by construction,
so downstream code does no imputation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import ASSET_CLASSES, FAMILY_FLAG_COLUMNS

FEATURE_COLUMNS: tuple[str, ...] = (
    "val_sharpe",
    "val_sortino",
    "val_calmar",
    "log_n_trades_val",
    "n_years_val",
    "log_avg_holding_hours",
    "frac_regime_trend",
    "frac_regime_chop",
    "frac_regime_highvol",
    "regime_concentration",
    "wf_positive_fraction",
    "wf_sharpe_std",
    "wf_sharpe_decay",
    *FAMILY_FLAG_COLUMNS,
    "n_feature_families",
    "log_n_candidates_tested",
    "n_params",
    *(f"asset_{a}" for a in ASSET_CLASSES),
)

# One column out of each simplex in this matrix: the regime fractions sum to 1
# and the asset one-hots sum to 1, either of which is collinear with a linear
# model's intercept and makes the design matrix singular. Real-data runs derive
# the same thing from their encoding recipe, so this list belongs with the
# synthetic feature schema rather than inside the Cox baseline.
COX_REFERENCE_COLUMNS: tuple[str, ...] = ("frac_regime_highvol", "asset_fx_majors")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["val_sharpe"] = df["val_sharpe"]
    x["val_sortino"] = df["val_sortino"]
    x["val_calmar"] = df["val_calmar"]
    x["log_n_trades_val"] = np.log10(df["n_trades_val"])
    x["n_years_val"] = df["n_years_val"]
    x["log_avg_holding_hours"] = np.log10(df["avg_holding_hours"])
    x["frac_regime_trend"] = df["frac_regime_trend"]
    x["frac_regime_chop"] = df["frac_regime_chop"]
    x["frac_regime_highvol"] = df["frac_regime_highvol"]
    x["regime_concentration"] = df[
        ["frac_regime_trend", "frac_regime_chop", "frac_regime_highvol"]
    ].max(axis=1)
    x["wf_positive_fraction"] = df["wf_positive_fraction"]
    x["wf_sharpe_std"] = df["wf_sharpe_std"]
    x["wf_sharpe_decay"] = df["wf_sharpe_decay"]
    for flag in FAMILY_FLAG_COLUMNS:
        x[flag] = df[flag]
    x["n_feature_families"] = df["n_feature_families"]
    x["log_n_candidates_tested"] = np.log10(df["n_candidates_tested"])
    x["n_params"] = df["n_params"]
    for asset in ASSET_CLASSES:
        x[f"asset_{asset}"] = (df["asset_class"] == asset).astype(int)

    x = x[list(FEATURE_COLUMNS)].astype(np.float64)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("non-finite values in feature matrix")
    return x
