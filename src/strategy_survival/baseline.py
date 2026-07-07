"""Baselines the AFT model has to beat.

CoxBaseline is the standard semi-parametric survival model on the same
features. The validation-Sharpe ranking is the heuristic a strategy-search
system implicitly uses when it allocates by backtest quality; the generator's
winner's curse predicts it should be nearly useless, and showing that is the
point of including it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

# One column from each simplex must go: the regime fractions sum to 1 and the
# asset one-hots sum to 1, which is singular for a linear model.
_DROP_FOR_COX: tuple[str, ...] = ("frac_regime_highvol", "asset_fx_majors")


class CoxBaseline:
    def __init__(self, penalizer: float = 0.1) -> None:
        self.penalizer = penalizer
        self.fitter: CoxPHFitter | None = None

    def _design(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.drop(columns=[c for c in _DROP_FOR_COX if c in x.columns])

    def fit(self, x: pd.DataFrame, duration_days: np.ndarray, event: np.ndarray) -> CoxBaseline:
        frame = self._design(x).copy()
        frame["duration_days"] = duration_days
        frame["event"] = event
        self.fitter = CoxPHFitter(penalizer=self.penalizer)
        self.fitter.fit(frame, duration_col="duration_days", event_col="event")
        return self

    def predict_neg_risk(self, x: pd.DataFrame) -> np.ndarray:
        """Negated partial hazard: higher means expected to survive longer,
        so it is orientation-compatible with predicted survival times."""
        if self.fitter is None:
            raise RuntimeError("model not fitted")
        return -self.fitter.predict_partial_hazard(self._design(x)).to_numpy()

    def predict_survival(self, x: pd.DataFrame, horizons_days: np.ndarray) -> np.ndarray:
        if self.fitter is None:
            raise RuntimeError("model not fitted")
        surv = self.fitter.predict_survival_function(self._design(x), times=horizons_days)
        return surv.to_numpy().T
