"""Censoring-aware evaluation: concordance, IPCW Brier score, calibration.

All metrics take durations and event flags as observed (censored rows count),
never a filtered uncensored subset -- dropping censored rows biases every one
of these toward optimism.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

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
    sorted_durations = np.sort(duration)
    ends = duration[np.asarray(event).astype(bool)]
    return int(
        np.sum(len(sorted_durations) - np.searchsorted(sorted_durations, ends, side="right"))
    )


def within_group_concordance(
    duration: np.ndarray,
    event: np.ndarray,
    predicted_score: np.ndarray,
    groups: pd.Series,
    min_rows: int = 50,
    min_events: int = 10,
) -> dict | None:
    """Decompose ranking skill by a grouping column.

    `c_group_mean` ranks every row by its group's mean prediction alone, so
    it measures how far group membership carries. `c_within` restricts
    comparisons to rows in the same group, pair-weighted across groups with
    at least `min_rows` rows and `min_events` observed endings, so it measures
    what the model adds beyond group membership. Returns None when no group
    qualifies.
    """
    scored_rows = pd.DataFrame(
        {
            "duration": np.asarray(duration, dtype=float),
            "event": np.asarray(event),
            "score": np.asarray(predicted_score, dtype=float),
            "group": groups.where(groups.notna(), "(missing)").astype(str).to_numpy(),
        }
    )
    group_mean = scored_rows.groupby("group")["score"].transform("mean")
    c_group_mean = harrell_c(
        scored_rows["duration"].to_numpy(), scored_rows["event"].to_numpy(), group_mean.to_numpy()
    )

    weighted, total_pairs, n_groups = 0.0, 0, 0
    for _, group_rows in scored_rows.groupby("group"):
        if len(group_rows) < min_rows or int(group_rows["event"].sum()) < min_events:
            continue
        pairs = _comparable_pairs(group_rows["duration"].to_numpy(), group_rows["event"].to_numpy())
        if pairs == 0:
            continue
        group_c_index = harrell_c(
            group_rows["duration"].to_numpy(),
            group_rows["event"].to_numpy(),
            group_rows["score"].to_numpy(),
        )
        weighted += pairs * group_c_index
        total_pairs += pairs
        n_groups += 1
    if n_groups == 0:
        return None
    return {
        "c_group_mean": c_group_mean,
        "c_within": weighted / total_pairs,
        "n_groups": n_groups,
        "n_pairs": total_pairs,
        "min_n": min_rows,
        "min_events": min_events,
    }


def censoring_survival(duration: np.ndarray, event: np.ndarray) -> KaplanMeierFitter:
    """Kaplan-Meier estimate of the censoring distribution G(t), used as IPCW
    weights. Note the flipped event indicator: a death is a 'censoring' of the
    censoring process."""
    censoring_curve = KaplanMeierFitter()
    censoring_curve.fit(duration, event_observed=1 - np.asarray(event))
    return censoring_curve


def ipcw_brier(
    duration: np.ndarray,
    event: np.ndarray,
    predicted_survival_at_horizon: np.ndarray,
    horizon: float,
) -> float:
    """Brier score at a horizon, inverse-probability-of-censoring weighted.

    Rows censored before the horizon get zero weight; the weights of the rest
    are inflated by 1/G so the expectation matches the uncensored population.
    Degenerates to the plain Brier score when nothing is censored.
    """
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event)
    predicted_survival = np.asarray(predicted_survival_at_horizon, dtype=float)

    censoring_curve = censoring_survival(duration, event)
    # G evaluated just before the death time, per Graf et al. (1999).
    censoring_at_death = np.maximum(
        np.asarray(censoring_curve.predict(np.maximum(duration - 1e-8, 0.0))), 1e-4
    )
    censoring_at_horizon = max(float(np.asarray(censoring_curve.predict(horizon))), 1e-4)

    died_by_horizon = (duration <= horizon) & (event == 1)
    alive_at_horizon = duration > horizon

    contributions = np.zeros_like(predicted_survival)
    contributions[died_by_horizon] = (
        predicted_survival[died_by_horizon] ** 2 / censoring_at_death[died_by_horizon]
    )
    contributions[alive_at_horizon] = (
        1.0 - predicted_survival[alive_at_horizon]
    ) ** 2 / censoring_at_horizon
    return float(contributions.mean())


def calibration_bins(
    duration: np.ndarray,
    event: np.ndarray,
    predicted_survival_at_horizon: np.ndarray,
    horizon: float,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Predicted vs Kaplan-Meier-observed survival at a horizon, by predicted
    decile. The KM estimate inside each bin handles censoring; if a bin's last
    observed time falls short of the horizon its estimate carries forward the
    last value, which the small-bin caveat in the report covers.
    """
    scored_rows = pd.DataFrame(
        {
            "duration": np.asarray(duration, dtype=float),
            "event": np.asarray(event),
            "predicted_survival": np.asarray(predicted_survival_at_horizon, dtype=float),
        }
    )
    scored_rows["bin"] = pd.qcut(
        scored_rows["predicted_survival"], q=n_bins, labels=False, duplicates="drop"
    )
    if scored_rows["bin"].isna().all():
        raise ValueError(
            "predicted survival probabilities are (near) constant; cannot form "
            "calibration bins - the model has learned nothing to separate rows by"
        )

    rows = []
    for bin_number, bin_rows in scored_rows.groupby("bin"):
        kaplan_meier = KaplanMeierFitter()
        kaplan_meier.fit(bin_rows["duration"], event_observed=bin_rows["event"])
        rows.append(
            {
                "bin": int(cast(np.integer, bin_number)),
                "n": len(bin_rows),
                "predicted": float(bin_rows["predicted_survival"].to_numpy().mean()),
                "observed_km": float(np.asarray(kaplan_meier.predict(horizon))),
            }
        )
    return pd.DataFrame(rows).sort_values("predicted", ignore_index=True)


def bootstrap_ci(
    metric: Callable[[np.ndarray], float],
    n_rows: int,
    n_resamples: int = 500,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap over row indices. `metric` receives an index array
    and must be a pure function of it."""
    random_generator = np.random.default_rng(seed)
    tail_probability = (1.0 - level) / 2.0
    resampled_values = [
        metric(random_generator.integers(0, n_rows, n_rows)) for _ in range(n_resamples)
    ]
    return float(np.quantile(resampled_values, tail_probability)), float(
        np.quantile(resampled_values, 1.0 - tail_probability)
    )
