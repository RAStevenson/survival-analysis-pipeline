"""Column names shared by the generator, feature builder, and tests.

METADATA_COLUMNS is the public schema: everything a strategy-search system
knows about a strategy on the day it is deployed. TARGET_COLUMNS is the
survival label observed later. Latent columns exist only for the oracle
ceiling computation and never feed the model.
"""

from __future__ import annotations

FEATURE_FAMILIES: tuple[str, ...] = (
    "momentum",
    "mean_reversion",
    "volatility_premium",
    "value_carry",
    "seasonality",
    "microstructure",
)

ASSET_CLASSES: tuple[str, ...] = (
    "equity_index_futures",
    "fx_majors",
    "rates_futures",
    "crypto",
)

FAMILY_FLAG_COLUMNS: tuple[str, ...] = tuple(f"uses_{f}" for f in FEATURE_FAMILIES)

METADATA_COLUMNS: tuple[str, ...] = (
    "strategy_id",
    "discovery_date",
    "asset_class",
    "val_sharpe",
    "val_sortino",
    "val_calmar",
    "n_trades_val",
    "n_years_val",
    "avg_holding_hours",
    "frac_regime_trend",
    "frac_regime_chop",
    "frac_regime_highvol",
    "wf_n_folds",
    "wf_positive_fraction",
    "wf_sharpe_std",
    "wf_sharpe_decay",
    *FAMILY_FLAG_COLUMNS,
    "n_feature_families",
    "n_candidates_tested",
    "n_params",
)

TARGET_COLUMNS: tuple[str, ...] = ("duration_days", "event")

LATENT_COLUMNS: tuple[str, ...] = ("true_sharpe", "overfit", "log_time_eta", "true_duration_days")
