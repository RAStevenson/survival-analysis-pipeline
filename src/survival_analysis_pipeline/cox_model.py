"""The baseline the AFT model has to beat.

CoxBaseline is the standard semi-parametric survival model, fitted on the same
features and the same windows, and it answers whether the boosted model earned
its place. Every run reports both.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import lifelines
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError


class CoxBaseline:
    """`drop_columns` names one reference level per categorical column. Without
    it the one-hot dummies for a column sum to 1, the design matrix is singular,
    and the fit either warns or fails outright. Every run passes the reference
    columns its encoding recipe recorded. This class holds no column names of
    its own, so it stays usable on any feature matrix."""

    def __init__(self, penalizer: float = 0.1, drop_columns: tuple[str, ...] = ()) -> None:
        """Store the ridge penalty and the reference columns to drop; nothing is fitted until fit is
        called.
        """
        self.penalizer = penalizer
        self.drop_columns = tuple(drop_columns)
        self.fitted_columns: list[str] | None = None
        self.impute_values: pd.Series | None = None
        self.fitter: CoxPHFitter | None = None

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select the fitted columns and fill gaps with the training medians.

        A Cox fit cannot take missing values the way a boosted tree can, so
        the medians are learned at fit time and stored on the model. Imputing
        outside it would leave a saved model unable to score a row with a
        missing feature, silently returning NaN.
        """
        if self.fitted_columns is None:
            return X.drop(columns=[column for column in self.drop_columns if column in X.columns])
        design = X[self.fitted_columns]
        if self.impute_values is not None and design.isna().to_numpy().any():
            design = design.fillna(self.impute_values)
        return design

    def fit(self, X: pd.DataFrame, duration: np.ndarray, event: np.ndarray) -> CoxBaseline:
        """Fit the penalized Cox model on the encoded features, dropping columns constant in this
        window and learning the median imputation values; returns self.
        """
        design = self._design(X)
        # A column can vary across the whole dataset yet be constant inside one
        # fold's training window -- categories that only appear in later years
        # are the usual case under temporal splits. Constant columns make the
        # Hessian singular, so they are dropped per fit rather than globally.
        varying = design.columns[design.std(ddof=0).fillna(0.0) > 0]
        dropped = [column for column in design.columns if column not in set(varying)]
        if dropped:
            print(
                f"Cox baseline: dropping {len(dropped)} columns with no variation in this "
                f"training window ({', '.join(dropped[:4])}{', ...' if len(dropped) > 4 else ''})"
            )
        self.fitted_columns = list(varying)
        self.impute_values = design[self.fitted_columns].median()

        training_rows = design[self.fitted_columns].fillna(self.impute_values).copy()
        training_rows["duration"] = duration
        training_rows["event"] = event
        self.fitter = CoxPHFitter(penalizer=self.penalizer)
        try:
            self.fitter.fit(training_rows, duration_col="duration", event_col="event")
        except ConvergenceError as error:
            raise RuntimeError(
                "the Cox baseline failed to converge on this feature matrix "
                f"({len(self.fitted_columns)} covariates, {int(np.sum(event))} events). The "
                "usual causes are collinear one-hot columns (pass reference levels via "
                "drop_columns), a rare category that perfectly predicts the outcome, or a "
                f"numeric column whose scale dwarfs the others. Underlying error: {error}"
            ) from error
        return self

    def top_coefficients(self, count: int = 12) -> list[dict]:
        """The strongest covariates as hazard ratios with 95% intervals.

        Ranked by |z|, the coefficient over its standard error, because raw
        coefficient sizes are not comparable across unstandardized features.
        The ridge penalizer shrinks coefficients toward zero, so the
        intervals are approximate; the report discloses that.
        """
        if self.fitter is None:
            raise RuntimeError("model not fitted")
        summary = self.fitter.summary
        top = summary.reindex(summary["z"].abs().sort_values(ascending=False).index).head(count)
        return [
            {
                "feature": str(name),
                "coef": float(row["coef"]),
                "hr": float(row["exp(coef)"]),
                "hr_lo": float(row["exp(coef) lower 95%"]),
                "hr_hi": float(row["exp(coef) upper 95%"]),
                "z": float(row["z"]),
            }
            for name, row in top.iterrows()
        ]

    def predict_neg_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Negated partial hazard: higher means expected to survive longer,
        so it is orientation-compatible with predicted survival times."""
        if self.fitter is None:
            raise RuntimeError("model not fitted")
        return -self.fitter.predict_partial_hazard(self._design(X)).to_numpy()

    def predict_survival(self, X: pd.DataFrame, horizons: np.ndarray) -> np.ndarray:
        """Survival probability for each row at each horizon, as a rows-by-horizons array."""
        if self.fitter is None:
            raise RuntimeError("model not fitted")
        survival_curves = self.fitter.predict_survival_function(self._design(X), times=horizons)
        return survival_curves.to_numpy().T

    def predict_median_time(self, X: pd.DataFrame) -> np.ndarray:
        """Median survival time from the fitted baseline curve.

        Returns inf for rows whose curve never reaches 0.5 inside the observed
        follow-up. That is the correct answer under heavy censoring -- the data
        does not say when half of such rows have failed -- and callers must not
        silently turn it into a number.
        """
        if self.fitter is None:
            raise RuntimeError("model not fitted")
        return np.asarray(self.fitter.predict_median(self._design(X)), dtype=float)

    # Per-training-row arrays lifelines keeps for its own diagnostics. They
    # scale with the training set (25 MB on a 340k-row Chicago fit, measured
    # in 2026-08 before the license-type exclusions) and take no part
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

        trimmed_fitter = copy.deepcopy(self.fitter)
        # Only ever strip attributes an object owns. CoxPHFitter proxies
        # unknown names through to its inner model, so hasattr would report
        # True on the wrapper and deleting there corrupts the proxy.
        for fitter_object in (trimmed_fitter, getattr(trimmed_fitter, "_model", None)):
            if fitter_object is None:
                continue
            for attribute_name in self._DIAGNOSTIC_ATTRS:
                if attribute_name in vars(fitter_object):
                    delattr(fitter_object, attribute_name)

        with path.open("wb") as handle:
            pickle.dump(
                {
                    "lifelines_version": lifelines.__version__,
                    "penalizer": self.penalizer,
                    "fitted_columns": self.fitted_columns,
                    "impute_values": self.impute_values,
                    "fitter": trimmed_fitter,
                },
                handle,
            )

    @classmethod
    def load(cls, path: str | Path) -> CoxBaseline:
        """Read a saved baseline back, refusing a pickle from a different lifelines version rather
        than trusting it.
        """
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
