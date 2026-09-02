"""Column names shared by the generator and the tests.

METADATA_COLUMNS is the public schema: everything a strategy-search system
knows about a strategy on the day it is deployed. TARGET_COLUMNS is the
survival label observed later. Latent columns exist only for the oracle
ceiling computation and never feed the model.

The schema is the model's feature set, already prepared. Counts are emitted
on the log scale they are modeled on and `regime_concentration` is emitted
as its own column, because the synthetic CSV goes through the same public
door as a user's file and that door does no feature engineering. Only two of
the three regime fractions are emitted: they sum to one, and a full simplex
is collinear with a linear model's intercept, which makes the Cox baseline's
design matrix singular. `asset_class` stays as text so the door's own
one-hot encoding handles it and derives the reference level itself.
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
    "stocks",
    "currencies",
    "bonds",
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
    "log_n_trades_val",
    "n_years_val",
    "log_avg_holding_hours",
    "frac_regime_trend",
    "frac_regime_chop",
    "regime_concentration",
    "wf_positive_fraction",
    "wf_sharpe_std",
    "wf_sharpe_decay",
    *FAMILY_FLAG_COLUMNS,
    "n_feature_families",
    "log_n_candidates_tested",
    "n_params",
)

TARGET_COLUMNS: tuple[str, ...] = ("duration_days", "event")

LATENT_COLUMNS: tuple[str, ...] = ("true_sharpe", "overfit", "log_time_eta", "true_duration_days")
