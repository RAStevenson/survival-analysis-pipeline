from __future__ import annotations

import numpy as np
import pytest

from survival_analysis_pipeline.cv import recensor, temporal_folds
from survival_analysis_pipeline.evaluate import harrell_c
from survival_analysis_pipeline.features import build_features
from survival_analysis_pipeline.model import XGBoostAFT, aft_labels, fit_predictive_sigma


def test_aft_labels():
    lower, upper = aft_labels(np.array([10.0, 20.0]), np.array([1, 0]))
    assert lower.tolist() == [10.0, 20.0]
    assert upper[0] == 10.0
    assert np.isinf(upper[1])


def test_predict_before_fit_raises(small_features):
    with pytest.raises(RuntimeError, match="not fitted"):
        XGBoostAFT().predict_median_time(small_features)


@pytest.fixture(scope="module")
def fitted(medium_data):
    df, _ = medium_data
    x = build_features(df)
    fold = temporal_folds(df["discovery_date"], n_folds=1, min_train_frac=0.7)[0]
    train_dur, train_ev = recensor(
        df["duration_days"].to_numpy()[fold.train_idx],
        df["event"].to_numpy()[fold.train_idx],
        df["discovery_date"].iloc[fold.train_idx],
        fold.split_date,
    )
    model = XGBoostAFT().fit(x.iloc[fold.train_idx], train_dur, train_ev)
    return model, x, df, fold


def test_predictions_are_positive_days(fitted):
    model, x, _, fold = fitted
    pred = model.predict_median_time(x.iloc[fold.test_idx])
    assert np.isfinite(pred).all()
    assert (pred > 0).all()
    assert np.median(pred) < 3000


def test_beats_random_out_of_time(fitted):
    model, x, df, fold = fitted
    pred = model.predict_median_time(x.iloc[fold.test_idx])
    c = harrell_c(
        df["duration_days"].to_numpy()[fold.test_idx],
        df["event"].to_numpy()[fold.test_idx],
        pred,
    )
    assert c > 0.55


def test_fit_predictive_sigma_recovers_true_scale():
    rng = np.random.default_rng(3)
    n = 4000
    median = np.exp(rng.uniform(3.0, 6.0, n))
    true_sigma = 0.5
    t = median * np.exp(true_sigma * rng.normal(size=n))
    censor = np.exp(rng.uniform(3.0, 7.0, n))
    duration = np.minimum(t, censor)
    event = (t <= censor).astype(int)
    assert abs(fit_predictive_sigma(median, duration, event) - true_sigma) < 0.05


def test_calibrated_sigma_used_by_predict_survival(fitted):
    model, x, df, fold = fitted
    test_x = x.iloc[fold.test_idx]
    before = model.predict_survival(test_x, np.array([180.0]))
    sigma = model.calibrate_predictive_sigma(
        test_x,
        df["duration_days"].to_numpy()[fold.test_idx],
        df["event"].to_numpy()[fold.test_idx],
    )
    after = model.predict_survival(test_x, np.array([180.0]))
    assert model.predictive_sigma == sigma
    if sigma != model.params.aft_sigma:
        assert not np.allclose(before, after)
    model.predictive_sigma = None


def test_survival_probabilities_monotone(fitted):
    model, x, _, fold = fitted
    horizons = np.array([30.0, 90.0, 180.0, 365.0])
    surv = model.predict_survival(x.iloc[fold.test_idx], horizons)
    assert surv.shape == (len(fold.test_idx), 4)
    assert ((surv >= 0) & (surv <= 1)).all()
    assert (np.diff(surv, axis=1) <= 1e-12).all()
