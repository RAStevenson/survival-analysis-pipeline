from __future__ import annotations

import numpy as np
import pytest

from survival_analysis_pipeline.cox_model import CoxBaseline
from survival_analysis_pipeline.evaluate_model import harrell_c


@pytest.fixture(scope="module")
def fitted_cox(small_data, small_loaded, small_features):
    df, _ = small_data
    return CoxBaseline(drop_columns=small_loaded.recipe.reference_columns).fit(
        small_features, df["duration_days"].to_numpy(), df["event"].to_numpy()
    )


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


def test_top_coefficients_rank_by_z_and_carry_consistent_ratios(fitted_cox):
    top = fitted_cox.top_coefficients(5)
    assert 0 < len(top) <= 5
    zs = [abs(r["z"]) for r in top]
    assert zs == sorted(zs, reverse=True)
    for r in top:
        assert r["hr"] == pytest.approx(np.exp(r["coef"]))
        assert r["hr_lo"] <= r["hr"] <= r["hr_hi"]
        assert r["feature"] in fitted_cox.fitted_columns


def test_top_coefficients_before_fit_raises(small_features):
    with pytest.raises(RuntimeError, match="not fitted"):
        CoxBaseline().top_coefficients()


def test_constant_column_is_dropped_rather_than_breaking_the_fit(
    small_data, small_loaded, small_features
):
    """A column can vary across the file yet be constant inside one fold's
    training window, which is the normal case for a category that only appears
    in later years. lifelines raises ConvergenceError on such a column, so the
    per-fit drop is load-bearing, not defensive: without it the Chicago run
    does not fit at all."""
    df, _ = small_data
    x = small_features.copy()
    x["never_varies_in_this_window"] = 1.0

    model = CoxBaseline(drop_columns=small_loaded.recipe.reference_columns).fit(
        x, df["duration_days"].to_numpy(), df["event"].to_numpy()
    )

    assert model.fitted_columns is not None
    assert "never_varies_in_this_window" not in model.fitted_columns
    assert np.isfinite(model.predict_neg_risk(x)).all()


def test_cox_save_is_slim_and_lossless(tmp_path, fitted_cox, small_features):
    """Saving drops lifelines' per-training-row diagnostic arrays, which
    dominate the file on a large fit. Predictions must not move."""
    path = tmp_path / "cox.pkl"
    fitted_cox.save(path)
    loaded = CoxBaseline.load(path)

    horizons = np.array([90.0, 180.0, 365.0])
    np.testing.assert_allclose(
        fitted_cox.predict_neg_risk(small_features), loaded.predict_neg_risk(small_features)
    )
    np.testing.assert_allclose(
        fitted_cox.predict_survival(small_features, horizons),
        loaded.predict_survival(small_features, horizons),
    )
    np.testing.assert_allclose(
        fitted_cox.predict_median_time(small_features), loaded.predict_median_time(small_features)
    )
    assert loaded.fitted_columns == fitted_cox.fitted_columns
    # The saved file must not scale with the training set.
    assert path.stat().st_size < 400_000


def test_saved_cox_carries_its_imputation_values(tmp_path, fitted_cox, small_features):
    """A saved model with no imputation values scores any row with a gap as
    NaN instead of failing, which is the silent kind of wrong."""
    path = tmp_path / "cox.pkl"
    fitted_cox.save(path)
    reloaded = CoxBaseline.load(path)

    gapped = small_features.copy()
    gapped.iloc[0, gapped.columns.get_loc("val_sharpe")] = np.nan

    assert reloaded.impute_values is not None
    assert np.isfinite(reloaded.predict_neg_risk(gapped)).all()
    assert np.isfinite(reloaded.predict_survival(gapped, np.array([90.0, 180.0]))).all()
    np.testing.assert_allclose(
        reloaded.predict_neg_risk(small_features), fitted_cox.predict_neg_risk(small_features)
    )
