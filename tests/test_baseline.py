from __future__ import annotations

import numpy as np
import pytest

from strategy_survival.baseline import CoxBaseline
from strategy_survival.evaluate import harrell_c


@pytest.fixture(scope="module")
def fitted_cox(small_data, small_features):
    df, _ = small_data
    return CoxBaseline().fit(small_features, df["duration_days"].to_numpy(), df["event"].to_numpy())


def test_cox_risk_orientation(fitted_cox, small_data, small_features):
    df, _ = small_data
    score = fitted_cox.predict_neg_risk(small_features)
    assert np.isfinite(score).all()
    c = harrell_c(df["duration_days"].to_numpy(), df["event"].to_numpy(), score)
    assert c > 0.55


def test_cox_survival_probabilities(fitted_cox, small_features):
    horizons = np.array([90.0, 180.0])
    surv = fitted_cox.predict_survival(small_features, horizons)
    assert surv.shape == (len(small_features), 2)
    assert ((surv >= 0) & (surv <= 1)).all()
    assert (surv[:, 1] <= surv[:, 0] + 1e-12).all()


def test_predict_before_fit_raises(small_features):
    with pytest.raises(RuntimeError, match="not fitted"):
        CoxBaseline().predict_neg_risk(small_features)
