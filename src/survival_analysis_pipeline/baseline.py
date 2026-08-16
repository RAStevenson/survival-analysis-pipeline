"""The baseline the AFT model has to beat.

CoxBaseline is the standard semi-parametric survival model, fitted on the same
features and the same windows, and it answers whether the boosted model earned
its place. Every run reports both. The synthetic study additionally scores a
validation-Sharpe ranking, but that one is a property of how its population was
selected rather than a model, so it lives in `synthetic_extras`.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import lifelines
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


class CoxBaseline:
    """`drop_columns` names one reference level per categorical column. Without
    it the one-hot dummies for a column sum to 1, the design matrix is singular,
    and the fit either warns or fails outright. Every run passes the reference
    columns its encoding recipe recorded. This class holds no column names of
    its own, so it stays usable on any feature matrix."""

    def __init__(self, penalizer: float = 0.1, drop_columns: tuple[str, ...] = ()) -> None:
        self.penalizer = penalizer
        self.drop_columns = tuple(drop_columns)
        self.fitted_columns: list[str] | None = None
        self.impute_values: pd.Series | None = None
        self.fitter: CoxPHFitter | None = None

    def _design(self, x: pd.DataFrame) -> pd.DataFrame:
        """Select the fitted columns and fill gaps with the training medians.

        A Cox fit cannot take missing values the way a boosted tree can, so
        the medians are learned at fit time and stored on the model. Imputing
        outside it would leave a saved model unable to score a row with a
        missing feature, silently returning NaN.
        """
        if self.fitted_columns is None:
            return x.drop(columns=[c for c in self.drop_columns if c in x.columns])
        design = x[self.fitted_columns]
        if self.impute_values is not None and design.isna().to_numpy().any():
            design = design.fillna(self.impute_values)
        return design

    def fit(self, x: pd.DataFrame, duration: np.ndarray, event: np.ndarray) -> CoxBaseline:
        design = self._design(x)
        # A column can vary across the whole dataset yet be constant inside one
        # fold's training window -- categories that only appear in later years
        # are the usual case under temporal splits. Constant columns make the
        # Hessian singular, so they are dropped per fit rather than globally.
        varying = design.columns[design.std(ddof=0).fillna(0.0) > 0]
        dropped = [c for c in design.columns if c not in set(varying)]
        if dropped:
            print(
                f"Cox baseline: dropping {len(dropped)} columns with no variation in this "
                f"training window ({', '.join(dropped[:4])}{', ...' if len(dropped) > 4 else ''})"
            )
        self.fitted_columns = list(varying)
        self.impute_values = design[self.fitted_columns].median()

        frame = design[self.fitted_columns].fillna(self.impute_values).copy()
        frame["duration"] = duration
        frame["event"] = event
        self.fitter = CoxPHFitter(penalizer=self.penalizer)
        try:
            self.fitter.fit(frame, duration_col="duration", event_col="event")
        except Exception as err:
            raise RuntimeError(
                "the Cox baseline failed to converge on this feature matrix "
                f"({len(self.fitted_columns)} covariates, {int(np.sum(event))} events). The "
                "usual causes are collinear one-hot columns (pass reference levels via "
                "drop_columns), a rare category that perfectly predicts the outcome, or a "
                f"numeric column whose scale dwarfs the others. Underlying error: {err}"
            ) from err
        return self

    def predict_neg_risk(self, x: pd.DataFrame) -> np.ndarray:
        """Negated partial hazard: higher means expected to survive longer,
        so it is orientation-compatible with predicted survival times."""
        if self.fitter is None:
            raise RuntimeError("model not fitted")
        return -self.fitter.predict_partial_hazard(self._design(x)).to_numpy()

    def predict_survival(self, x: pd.DataFrame, horizons: np.ndarray) -> np.ndarray:
        if self.fitter is None:
            raise RuntimeError("model not fitted")
        surv = self.fitter.predict_survival_function(self._design(x), times=horizons)
        return surv.to_numpy().T

    def predict_median_time(self, x: pd.DataFrame) -> np.ndarray:
        """Median survival time from the fitted baseline curve.

        Returns inf for rows whose curve never reaches 0.5 inside the observed
        follow-up. That is the honest answer under heavy censoring -- the data
        does not say when half of such rows have failed -- and callers must not
        silently turn it into a number.
        """
        if self.fitter is None:
            raise RuntimeError("model not fitted")
        return np.asarray(self.fitter.predict_median(self._design(x)), dtype=float)

    # Per-training-row arrays lifelines keeps for its own diagnostics. They
    # scale with the training set (25 MB on a 340k-row fit) and take no part
    # in prediction, so they are dropped from the saved copy. The round-trip
    # test asserts predictions are unchanged.
    _DIAGNOSTIC_ATTRS: tuple[str, ...] = (
        "_predicted_partial_hazards_",
        "durations",
        "weights",
        "event_observed",
        "entry",
    )

    def save(self, path: str | Path) -> None:
        """Pickle the fitted model. lifelines has no portable serialization
        format, so the version is recorded alongside and checked on load."""
        if self.fitter is None:
            raise RuntimeError("model not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        slim = copy.deepcopy(self.fitter)
        # Only ever strip attributes an object owns. CoxPHFitter proxies
        # unknown names through to its inner model, so hasattr would report
        # True on the wrapper and deleting there corrupts the proxy.
        for holder in (slim, getattr(slim, "_model", None)):
            if holder is None:
                continue
            for attr in self._DIAGNOSTIC_ATTRS:
                if attr in vars(holder):
                    delattr(holder, attr)

        with path.open("wb") as handle:
            pickle.dump(
                {
                    "lifelines_version": lifelines.__version__,
                    "penalizer": self.penalizer,
                    "fitted_columns": self.fitted_columns,
                    "impute_values": self.impute_values,
                    "fitter": slim,
                },
                handle,
            )

    @classmethod
    def load(cls, path: str | Path) -> CoxBaseline:
        path = Path(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        saved_version = payload.get("lifelines_version")
        if saved_version != lifelines.__version__:
            raise ValueError(
                f"{path.name} was pickled with lifelines {saved_version}, this environment has "
                f"{lifelines.__version__}; install the pinned version from requirements.txt or "
                "refit the model"
            )
        model = cls(penalizer=payload["penalizer"])
        model.fitter = payload["fitter"]
        model.fitted_columns = payload.get("fitted_columns")
        model.impute_values = payload.get("impute_values")
        return model
