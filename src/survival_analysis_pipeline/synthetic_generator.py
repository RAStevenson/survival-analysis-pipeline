"""Synthetic strategy metadata with a survival target.

The generator imitates the selection process of an automated strategy search.
Candidates carry two latent quantities: true forward Sharpe (real edge) and an
overfit component inflated by search intensity and parameter count. Validation
Sharpe is the sum of both plus measurement noise, and only candidates clearing
a validation-Sharpe threshold are "deployed". Conditional on selection, high
validation Sharpe is therefore ambiguous between edge and overfitting, which
installs one misleading feature beside honest proxies like walk-forward
consistency, trade counts, and search intensity.

Survival time is log-normal AFT in the latents plus a few directly observable
effects (feature family, asset class, regime concentration, holding period,
search intensity). Observed durations are right-censored two ways: strategies
still alive at the observation cutoff, and a small rate of administrative
retirement (capacity reallocation) independent of performance.

The returned latent frame exists only to compute an oracle concordance ceiling
and to sanity-check SHAP directions against the generative truth. It must never
be used as model input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .synthetic_schema import ASSET_CLASSES, FEATURE_FAMILIES, LATENT_COLUMNS, METADATA_COLUMNS

# Log-time multipliers. Crowded or fragile families (momentum, seasonality,
# microstructure) decay faster; slow risk-premium families persist.
FAMILY_LOG_TIME_EFFECT: dict[str, float] = {
    "momentum": -0.22,
    "mean_reversion": -0.08,
    "volatility_premium": 0.06,
    "value_carry": 0.18,
    "seasonality": -0.32,
    "microstructure": -0.45,
}

ASSET_LOG_TIME_EFFECT: dict[str, float] = {
    "stocks": 0.05,
    "currencies": 0.0,
    "bonds": 0.10,
    "crypto": -0.30,
}

ASSET_CLASS_WEIGHTS: tuple[float, ...] = (0.35, 0.30, 0.20, 0.15)

FAMILY_COUNT_WEIGHTS: tuple[float, ...] = (0.50, 0.35, 0.15)


@dataclass(frozen=True)
class GeneratorConfig:
    n_strategies: int = 5000
    seed: int = 7
    discovery_start: str = "2021-07-01"
    discovery_end: str = "2026-06-01"
    observation_cutoff: str = "2026-07-01"
    selection_sharpe: float = 0.8
    wf_n_folds: int = 8
    admin_censor_rate: float = 0.06
    log_time_sigma: float = 0.55


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _draw_candidates(rng: np.random.Generator, n: int, cfg: GeneratorConfig) -> pd.DataFrame:
    n_candidates = np.round(10 ** rng.uniform(2.0, 5.0, n)).astype(int)
    search = np.log10(n_candidates) - 2.0

    avg_holding_hours = np.clip(10 ** rng.normal(1.3, 0.7, n), 0.25, 2000.0)
    n_years_val = rng.uniform(2.0, 5.0, n)
    activity = rng.uniform(0.15, 0.6, n)
    n_trades_val = np.clip(
        np.round(n_years_val * 8760.0 / avg_holding_hours * activity), 30, 20000
    ).astype(int)

    n_params = rng.integers(4, 61, n)

    true_sharpe = rng.normal(0.25, 0.35, n)
    overfit_scale = (
        0.22
        * (1.0 + 0.30 * search + 0.15 * np.clip(np.log(n_params / 8.0), 0.0, None))
        * (800.0 / n_trades_val) ** 0.15
    )
    overfit = rng.gamma(1.3, 1.0, n) * overfit_scale
    # Sharpe standard error scales roughly with 1/sqrt(window length in years).
    measurement_noise = rng.normal(0.0, 1.0, n) / np.sqrt(n_years_val)
    val_sharpe = true_sharpe + overfit + measurement_noise

    val_sortino = val_sharpe * rng.normal(1.40, 0.12, n) + rng.normal(0.0, 0.05, n)
    val_calmar = np.clip(val_sharpe * rng.normal(0.55, 0.15, n), 0.05, None)

    regime_fracs = rng.dirichlet((1.9, 1.7, 1.2), n)

    p_positive = _sigmoid(0.35 + 1.9 * true_sharpe - 1.5 * overfit)
    wf_positive_fraction = rng.binomial(cfg.wf_n_folds, p_positive) / cfg.wf_n_folds
    wf_sharpe_std = np.exp(rng.normal(np.log(0.30 + 0.45 * overfit), 0.30))
    wf_sharpe_decay = rng.normal(-0.03 - 0.28 * overfit + 0.08 * true_sharpe, 0.10)

    n_families = rng.choice((1, 2, 3), n, p=FAMILY_COUNT_WEIGHTS)
    family_rank = np.argsort(rng.random((n, len(FEATURE_FAMILIES))), axis=1)
    flags = family_rank < n_families[:, None]
    effects = np.array([FAMILY_LOG_TIME_EFFECT[f] for f in FEATURE_FAMILIES])
    family_effect = (flags * effects).sum(axis=1) / n_families + 0.05 * (n_families - 1)

    asset_class = rng.choice(ASSET_CLASSES, n, p=ASSET_CLASS_WEIGHTS)
    asset_effect = np.array([ASSET_LOG_TIME_EFFECT[a] for a in asset_class])

    regime_concentration = regime_fracs.max(axis=1)
    holding_effect = np.clip(0.10 * np.log10(avg_holding_hours / 24.0), -0.20, 0.12)

    log_time_eta = (
        np.log(210.0)
        + 0.85 * true_sharpe
        - 1.15 * overfit
        + family_effect
        + asset_effect
        - 1.1 * np.clip(regime_concentration - 0.45, 0.0, None)
        - 0.07 * search
        + holding_effect
    )
    true_duration = np.clip(
        np.exp(log_time_eta + cfg.log_time_sigma * rng.normal(size=n)), 3.0, None
    )

    # Emitted as the prepared feature set (see schema.py): counts on the log
    # scale they are modeled on, regime_concentration as its own column, and
    # only two of the three regime fractions.
    df = pd.DataFrame(
        {
            "asset_class": asset_class,
            "val_sharpe": val_sharpe,
            "val_sortino": val_sortino,
            "val_calmar": val_calmar,
            "log_n_trades_val": np.log10(n_trades_val),
            "n_years_val": n_years_val,
            "log_avg_holding_hours": np.log10(avg_holding_hours),
            "frac_regime_trend": regime_fracs[:, 0],
            "frac_regime_chop": regime_fracs[:, 1],
            "regime_concentration": regime_concentration,
            "wf_positive_fraction": wf_positive_fraction,
            "wf_sharpe_std": wf_sharpe_std,
            "wf_sharpe_decay": wf_sharpe_decay,
            "n_feature_families": n_families,
            "log_n_candidates_tested": np.log10(n_candidates),
            "n_params": n_params,
            "true_sharpe": true_sharpe,
            "overfit": overfit,
            "log_time_eta": log_time_eta,
            "true_duration_days": true_duration,
        }
    )
    for i, family in enumerate(FEATURE_FAMILIES):
        df[f"uses_{family}"] = flags[:, i].astype(int)
    return df


def generate(cfg: GeneratorConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (metadata, latents), row-aligned, sorted by discovery date.

    Metadata has METADATA_COLUMNS + TARGET_COLUMNS. Latents has strategy_id +
    LATENT_COLUMNS and must not be used as model input.
    """
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)

    selected: list[pd.DataFrame] = []
    n_selected = 0
    for _ in range(50):
        batch = _draw_candidates(rng, max(2 * cfg.n_strategies, 1000), cfg)
        batch = batch[batch["val_sharpe"] >= cfg.selection_sharpe]
        selected.append(batch)
        n_selected += len(batch)
        if n_selected >= cfg.n_strategies:
            break
    if n_selected < cfg.n_strategies:
        raise RuntimeError(
            f"selection threshold {cfg.selection_sharpe} too strict: "
            f"only {n_selected} of {cfg.n_strategies} strategies accepted"
        )
    df = pd.concat(selected, ignore_index=True).iloc[: cfg.n_strategies].copy()
    n = len(df)

    start = pd.Timestamp(cfg.discovery_start)
    end = pd.Timestamp(cfg.discovery_end)
    cutoff = pd.Timestamp(cfg.observation_cutoff)
    offsets = rng.integers(0, (end - start).days + 1, n)
    df["discovery_date"] = start + pd.to_timedelta(offsets, unit="D")
    df = df.sort_values("discovery_date", ignore_index=True)
    df["strategy_id"] = [f"S{i:05d}" for i in range(n)]

    follow_up = (cutoff - df["discovery_date"]).dt.days.to_numpy(dtype=float)
    admin_censor = np.where(
        rng.random(n) < cfg.admin_censor_rate, rng.uniform(30.0, 700.0, n), np.inf
    )
    censor_time = np.minimum(follow_up, admin_censor)
    true_duration = df["true_duration_days"].to_numpy()
    df["duration_days"] = np.round(np.minimum(true_duration, censor_time), 1)
    df["event"] = (true_duration <= censor_time).astype(int)

    latents = df[["strategy_id", *LATENT_COLUMNS]].copy()
    metadata = df[[*METADATA_COLUMNS, "duration_days", "event"]].copy()
    return metadata, latents
