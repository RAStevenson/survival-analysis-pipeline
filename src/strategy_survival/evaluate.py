"""Censoring-aware evaluation: concordance, IPCW Brier score, calibration.

All metrics take durations and event flags as observed (censored rows count),
never a filtered uncensored subset -- dropping censored rows biases every one
of these toward optimism.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index


def harrell_c(duration_days: np.ndarray, event: np.ndarray, predicted_score: np.ndarray) -> float:
    """Harrell's C. `predicted_score` must be increasing in predicted survival
    time (a predicted duration or a negated hazard, not a raw hazard)."""
    return float(concordance_index(duration_days, predicted_score, event))


def censoring_survival(duration_days: np.ndarray, event: np.ndarray) -> KaplanMeierFitter:
    """Kaplan-Meier estimate of the censoring distribution G(t), used as IPCW
    weights. Note the flipped event indicator: a death is a 'censoring' of the
    censoring process."""
    kmf = KaplanMeierFitter()
    kmf.fit(duration_days, event_observed=1 - np.asarray(event))
    return kmf


def ipcw_brier(
    duration_days: np.ndarray,
    event: np.ndarray,
    predicted_survival_at_h: np.ndarray,
    horizon_days: float,
) -> float:
    """Brier score at a horizon, inverse-probability-of-censoring weighted.

    Rows censored before the horizon get zero weight; the weights of the rest
    are inflated by 1/G so the expectation matches the uncensored population.
    Degenerates to the plain Brier score when nothing is censored.
    """
    duration_days = np.asarray(duration_days, dtype=float)
    event = np.asarray(event)
    s_hat = np.asarray(predicted_survival_at_h, dtype=float)

    g = censoring_survival(duration_days, event)
    # G evaluated just before the death time, per Graf et al. (1999).
    g_at_death = np.maximum(g.predict(np.maximum(duration_days - 1e-8, 0.0)), 1e-4)
    g_at_horizon = max(float(g.predict(horizon_days)), 1e-4)

    died_by_h = (duration_days <= horizon_days) & (event == 1)
    alive_at_h = duration_days > horizon_days

    contrib = np.zeros_like(s_hat)
    contrib[died_by_h] = s_hat[died_by_h] ** 2 / g_at_death[died_by_h]
    contrib[alive_at_h] = (1.0 - s_hat[alive_at_h]) ** 2 / g_at_horizon
    return float(contrib.mean())


def calibration_bins(
    duration_days: np.ndarray,
    event: np.ndarray,
    predicted_survival_at_h: np.ndarray,
    horizon_days: float,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Predicted vs Kaplan-Meier-observed survival at a horizon, by predicted
    decile. The KM estimate inside each bin handles censoring; if a bin's last
    observed time falls short of the horizon its estimate carries forward the
    last value, which the small-bin caveat in the report covers.
    """
    frame = pd.DataFrame(
        {
            "duration": np.asarray(duration_days, dtype=float),
            "event": np.asarray(event),
            "pred": np.asarray(predicted_survival_at_h, dtype=float),
        }
    )
    frame["bin"] = pd.qcut(frame["pred"], q=n_bins, labels=False, duplicates="drop")
    if frame["bin"].isna().all():
        raise ValueError(
            "predicted survival probabilities are (near) constant; cannot form "
            "calibration bins - the model has learned nothing to separate rows by"
        )

    rows = []
    for bin_id, group in frame.groupby("bin"):
        kmf = KaplanMeierFitter()
        kmf.fit(group["duration"], event_observed=group["event"])
        rows.append(
            {
                "bin": int(bin_id),
                "n": len(group),
                "predicted": float(group["pred"].mean()),
                "observed_km": float(kmf.predict(horizon_days)),
            }
        )
    return pd.DataFrame(rows).sort_values("predicted", ignore_index=True)


def bootstrap_ci(
    metric: Callable[[np.ndarray], float],
    n_rows: int,
    n_boot: int = 500,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap over row indices. `metric` receives an index array
    and must be a pure function of it."""
    rng = np.random.default_rng(seed)
    alpha = (1.0 - level) / 2.0
    stats = [metric(rng.integers(0, n_rows, n_rows)) for _ in range(n_boot)]
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha))
