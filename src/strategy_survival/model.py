"""XGBoost accelerated-failure-time model.

Censoring enters through interval labels: an observed death is the interval
[t, t], a right-censored strategy is [t, +inf). Predictions come back in the
time domain, in whatever unit the durations were supplied in; the model is
scale-free and never needs to know which unit that is. The predict_median_days
name is the day-based original and reads as "median survival time" in the
dataset's unit. Survival probabilities are log-normal around the predicted
median; the scale defaults to the training loss scale but should be replaced
by `calibrate_predictive_sigma` on held-out data, because the loss scale that
ranks best is usually too wide to be calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm


@dataclass(frozen=True)
class AFTParams:
    max_depth: int = 3
    learning_rate: float = 0.05
    n_rounds: int = 500
    min_child_weight: float = 5.0
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    aft_sigma: float = 0.9
    early_stopping_rounds: int = 50
    nthread: int = 1
    seed: int = 0


def censored_lognormal_nll(
    median_days: np.ndarray, sigma: float, duration_days: np.ndarray, event: np.ndarray
) -> float:
    """Mean negative log-likelihood of a log-normal centred at `median_days`.

    Deaths contribute the density, censored rows the survival function -- the
    standard right-censored likelihood, so this is minimized by a calibrated
    scale rather than by hedging every prediction toward 0.5.
    """
    z = (np.log(duration_days) - np.log(median_days)) / sigma
    event = np.asarray(event) == 1
    ll = np.where(
        event,
        norm.logpdf(z) - np.log(sigma * duration_days),
        norm.logsf(z),
    )
    return float(-ll.mean())


def fit_predictive_sigma(
    median_days: np.ndarray,
    duration_days: np.ndarray,
    event: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    grid = grid if grid is not None else np.arange(0.20, 2.01, 0.02)
    nlls = [censored_lognormal_nll(median_days, s, duration_days, event) for s in grid]
    return float(grid[int(np.argmin(nlls))])


def aft_labels(duration_days: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(duration_days, dtype=float)
    upper = np.where(np.asarray(event) == 1, lower, np.inf)
    return lower, upper


def _dmatrix(x: pd.DataFrame, duration: np.ndarray | None, event: np.ndarray | None) -> xgb.DMatrix:
    d = xgb.DMatrix(x, feature_names=list(x.columns))
    if duration is not None:
        assert event is not None
        lower, upper = aft_labels(duration, event)
        d.set_float_info("label_lower_bound", lower)
        d.set_float_info("label_upper_bound", upper)
    return d


class XGBoostAFT:
    def __init__(self, params: AFTParams | None = None) -> None:
        self.params = params or AFTParams()
        self.booster: xgb.Booster | None = None
        self.predictive_sigma: float | None = None

    def fit(
        self,
        x: pd.DataFrame,
        duration_days: np.ndarray,
        event: np.ndarray,
        eval_x: pd.DataFrame | None = None,
        eval_duration: np.ndarray | None = None,
        eval_event: np.ndarray | None = None,
    ) -> XGBoostAFT:
        """Train, with early stopping when an eval set is given.

        The eval set must be temporally later than the training rows; passing
        a random split here silently reintroduces the leakage the temporal CV
        is designed to avoid.
        """
        dtrain = _dmatrix(x, duration_days, event)
        xgb_params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": self.params.aft_sigma,
            "max_depth": self.params.max_depth,
            "learning_rate": self.params.learning_rate,
            "min_child_weight": self.params.min_child_weight,
            "subsample": self.params.subsample,
            "colsample_bytree": self.params.colsample_bytree,
            "tree_method": "hist",
            # Both pins exist so the committed numbers reproduce off this
            # machine. seed fixes the subsample and colsample draws. nthread
            # matters for a less obvious reason: hist accumulates gradient
            # histograms per thread and sums them at the end, so the thread
            # count sets the floating-point summation order and moves the
            # third decimal of every score. Left free, the reported C-index
            # depends on the reviewer's core count.
            "nthread": self.params.nthread,
            "seed": self.params.seed,
        }
        evals = []
        early_stopping = None
        if eval_x is not None:
            assert eval_duration is not None and eval_event is not None
            evals = [(_dmatrix(eval_x, eval_duration, eval_event), "eval")]
            early_stopping = self.params.early_stopping_rounds
        self.booster = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=self.params.n_rounds,
            evals=evals,
            early_stopping_rounds=early_stopping,
            verbose_eval=False,
        )
        return self

    def predict_median_days(self, x: pd.DataFrame) -> np.ndarray:
        """Predicted survival time in days (the log-normal median)."""
        if self.booster is None:
            raise RuntimeError("model not fitted")
        return self.booster.predict(_dmatrix(x, None, None))

    def predict_survival(self, x: pd.DataFrame, horizons_days: np.ndarray) -> np.ndarray:
        """P(T > h) for each row and horizon; shape (n_rows, n_horizons)."""
        sigma = self.predictive_sigma
        if sigma is None:
            sigma = self.params.aft_sigma
        median = self.predict_median_days(x)
        z = (np.log(horizons_days)[None, :] - np.log(median)[:, None]) / sigma
        return 1.0 - norm.cdf(z)

    def calibrate_predictive_sigma(
        self, x: pd.DataFrame, duration_days: np.ndarray, event: np.ndarray
    ) -> float:
        """Fit the predictive scale on held-out rows. The rows must not have
        been trained on, or the scale comes out overconfident."""
        sigma = fit_predictive_sigma(self.predict_median_days(x), duration_days, event)
        self.predictive_sigma = sigma
        return sigma

    def with_params(self, **kwargs: float | int) -> XGBoostAFT:
        return XGBoostAFT(replace(self.params, **kwargs))
