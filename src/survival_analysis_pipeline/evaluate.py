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


def harrell_c(duration: np.ndarray, event: np.ndarray, predicted_score: np.ndarray) -> float:
    """Harrell's C. `predicted_score` must be increasing in predicted survival
    time (a predicted duration or a negated hazard, not a raw hazard)."""
    return float(concordance_index(duration, predicted_score, event))


def _comparable_pairs(duration: np.ndarray, event: np.ndarray) -> int:
    """Comparable pairs in the survival sense: for each observed ending at
    time t, every row with duration strictly greater than t. Used as a
    weight, so the strict-inequality tie convention is acceptable."""
    s = np.sort(duration)
    ends = duration[np.asarray(event).astype(bool)]
    return int(np.sum(len(s) - np.searchsorted(s, ends, side="right")))


def within_group_concordance(
    duration: np.ndarray,
    event: np.ndarray,
    predicted_score: np.ndarray,
    groups: pd.Series,
    min_n: int = 50,
    min_events: int = 10,
) -> dict | None:
    """Decompose ranking skill by a grouping column.

    `c_group_mean` ranks every row by its group's mean prediction alone, so
    it measures how far group membership carries. `c_within` restricts
    comparisons to rows in the same group, pair-weighted across groups with
    at least `min_n` rows and `min_events` observed endings, so it measures
    what the model adds beyond group membership. Returns None when no group
    qualifies.
    """
    frame = pd.DataFrame(
        {
            "dur": np.asarray(duration, dtype=float),
            "ev": np.asarray(event),
            "score": np.asarray(predicted_score, dtype=float),
            "g": groups.where(groups.notna(), "(missing)").astype(str).to_numpy(),
        }
    )
    group_mean = frame.groupby("g")["score"].transform("mean")
    c_group_mean = harrell_c(frame["dur"].to_numpy(), frame["ev"].to_numpy(), group_mean.to_numpy())

    weighted, total_pairs, n_groups = 0.0, 0, 0
    for _, sub in frame.groupby("g"):
        if len(sub) < min_n or int(sub["ev"].sum()) < min_events:
            continue
        pairs = _comparable_pairs(sub["dur"].to_numpy(), sub["ev"].to_numpy())
        if pairs == 0:
            continue
        c = harrell_c(sub["dur"].to_numpy(), sub["ev"].to_numpy(), sub["score"].to_numpy())
        weighted += pairs * c
        total_pairs += pairs
        n_groups += 1
    if n_groups == 0:
        return None
    return {
        "c_group_mean": c_group_mean,
        "c_within": weighted / total_pairs,
        "n_groups": n_groups,
        "n_pairs": total_pairs,
        "min_n": min_n,
        "min_events": min_events,
    }


def censoring_survival(duration: np.ndarray, event: np.ndarray) -> KaplanMeierFitter:
    """Kaplan-Meier estimate of the censoring distribution G(t), used as IPCW
    weights. Note the flipped event indicator: a death is a 'censoring' of the
    censoring process."""
    kmf = KaplanMeierFitter()
    kmf.fit(duration, event_observed=1 - np.asarray(event))
    return kmf


def ipcw_brier(
    duration: np.ndarray,
    event: np.ndarray,
    predicted_survival_at_h: np.ndarray,
    horizon: float,
) -> float:
    """Brier score at a horizon, inverse-probability-of-censoring weighted.

    Rows censored before the horizon get zero weight; the weights of the rest
    are inflated by 1/G so the expectation matches the uncensored population.
    Degenerates to the plain Brier score when nothing is censored.
    """
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event)
    s_hat = np.asarray(predicted_survival_at_h, dtype=float)

    g = censoring_survival(duration, event)
    # G evaluated just before the death time, per Graf et al. (1999).
    g_at_death = np.maximum(g.predict(np.maximum(duration - 1e-8, 0.0)), 1e-4)
    g_at_horizon = max(float(g.predict(horizon)), 1e-4)

    died_by_h = (duration <= horizon) & (event == 1)
    alive_at_h = duration > horizon

    contrib = np.zeros_like(s_hat)
    contrib[died_by_h] = s_hat[died_by_h] ** 2 / g_at_death[died_by_h]
    contrib[alive_at_h] = (1.0 - s_hat[alive_at_h]) ** 2 / g_at_horizon
    return float(contrib.mean())


def calibration_bins(
    duration: np.ndarray,
    event: np.ndarray,
    predicted_survival_at_h: np.ndarray,
    horizon: float,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Predicted vs Kaplan-Meier-observed survival at a horizon, by predicted
    decile. The KM estimate inside each bin handles censoring; if a bin's last
    observed time falls short of the horizon its estimate carries forward the
    last value, which the small-bin caveat in the report covers.
    """
    frame = pd.DataFrame(
        {
            "duration": np.asarray(duration, dtype=float),
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
                "observed_km": float(kmf.predict(horizon)),
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
