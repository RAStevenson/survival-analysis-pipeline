from __future__ import annotations

import numpy as np
import pytest

from survival_analysis_pipeline.aft_model import XGBoostAFT, aft_labels, fit_predictive_sigma
from survival_analysis_pipeline.evaluate_model import harrell_c
from survival_analysis_pipeline.temporal_folds import recensor, temporal_folds


def test_aft_labels():
    lower, upper = aft_labels(np.array([10.0, 20.0]), np.array([1, 0]))
    assert lower.tolist() == [10.0, 20.0]
    assert upper[0] == 10.0
    assert np.isinf(upper[1])


def test_predict_before_fit_raises(small_features):
    with pytest.raises(RuntimeError, match="not fitted"):
        XGBoostAFT().predict_median_time(small_features)


@pytest.fixture(scope="module")
def fitted(medium_data, medium_features):
    strategies, _ = medium_data
    features = medium_features
    fold = temporal_folds(strategies["discovery_date"], n_folds=1, min_train_frac=0.7)[0]
    train_dur, train_ev = recensor(
        strategies["duration_days"].to_numpy()[fold.train_rows],
        strategies["event"].to_numpy()[fold.train_rows],
        strategies["discovery_date"].iloc[fold.train_rows],
        fold.split_date,
    )
    model = XGBoostAFT().fit(features.iloc[fold.train_rows], train_dur, train_ev)
    return model, features, strategies, fold


def test_predictions_are_positive_days(fitted):
    model, features, _, fold = fitted
    predicted_median = model.predict_median_time(features.iloc[fold.test_rows])
    assert np.isfinite(predicted_median).all()
    assert (predicted_median > 0).all()
    assert np.median(predicted_median) < 3000


def test_beats_random_out_of_time(fitted):
    model, features, strategies, fold = fitted
    predicted_median = model.predict_median_time(features.iloc[fold.test_rows])
    c_index = harrell_c(
        strategies["duration_days"].to_numpy()[fold.test_rows],
        strategies["event"].to_numpy()[fold.test_rows],
        predicted_median,
    )
    assert c_index > 0.55


def test_fit_predictive_sigma_recovers_true_scale():
    random_generator = np.random.default_rng(3)
    n_rows = 4000
    median = np.exp(random_generator.uniform(3.0, 6.0, n_rows))
    true_sigma = 0.5
    true_lifetime = median * np.exp(true_sigma * random_generator.normal(size=n_rows))
    censor = np.exp(random_generator.uniform(3.0, 7.0, n_rows))
    duration = np.minimum(true_lifetime, censor)
    event = (true_lifetime <= censor).astype(int)
    assert abs(fit_predictive_sigma(median, duration, event) - true_sigma) < 0.05


def test_calibrated_sigma_used_by_predict_survival(fitted):
    model, features, strategies, fold = fitted
    test_features = features.iloc[fold.test_rows]
    before = model.predict_survival(test_features, np.array([180.0]))
    sigma = model.calibrate_predictive_sigma(
        test_features,
        strategies["duration_days"].to_numpy()[fold.test_rows],
        strategies["event"].to_numpy()[fold.test_rows],
    )
    after = model.predict_survival(test_features, np.array([180.0]))
    assert model.predictive_sigma == sigma
    if sigma != model.params.aft_sigma:
        assert not np.allclose(before, after)
    model.predictive_sigma = None


def test_survival_probabilities_monotone(fitted):
    model, features, _, fold = fitted
    horizons = np.array([30.0, 90.0, 180.0, 365.0])
    surv = model.predict_survival(features.iloc[fold.test_rows], horizons)
    assert surv.shape == (len(fold.test_rows), 4)
    assert ((surv >= 0) & (surv <= 1)).all()
    assert (np.diff(surv, axis=1) <= 1e-12).all()
