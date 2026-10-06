"""The pipeline: fit and evaluate two survival models on a duration CSV.

This is the one door. `fit_evaluate` takes a right-censored duration file,
runs expanding-window temporal cross-validation with label re-censoring at
every split, scores an XGBoost AFT model against a Cox proportional hazards
baseline, and writes a run directory holding metrics.json, figures, and a
saved model bundle that `predict` can score new rows with later. The
synthetic study goes through this same function on a generated CSV; the
ground-truth extras it can additionally report are computed afterwards, in
synthetic_extras, because nothing about a user's file could supply them.

Hyperparameters are selected once, on the first fold's training window with
an inner temporal split, and reused for every fold. Selecting per-fold would
be cleaner in principle but makes fold metrics harder to compare and
sextuples the runtime for no visible gain on this data.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from .aft_model import AFTParams, XGBoostAFT
from .cox_model import CoxBaseline
from .duration_csv import (
    DURATION,
    EVENT,
    ROW_ID,
    START,
    check_minimum_data,
    encode_with_recipe,
    load_cox_from_bundle,
    load_duration_csv,
    load_model_bundle,
    make_fold_encoder,
    save_model_bundle,
)
from .evaluate_model import (
    bootstrap_ci,
    calibration_bins,
    harrell_c,
    ipcw_brier,
    within_group_concordance,
)
from .report_plots import calibration_plot, cox_hr_plot, fold_cindex_plot, km_by_group_plot
from .shap_analysis import compute_shap, write_shap_figures
from .temporal_folds import TemporalFold, recensor, temporal_folds
from .time_units import check_time_unit, horizon_label, unit_abbrev

PARAM_GRID: tuple[AFTParams, ...] = tuple(
    AFTParams(max_depth=depth, aft_sigma=sigma) for depth in (2, 3) for sigma in (0.6, 0.9, 1.2)
)

# The one numeric bound on the median observed duration, in timesteps.
# Above it the fit is genuinely scale-free: re-measured 2026-08-12 on the
# 800-row synthetic draw tests/test_synthetic_pipeline.py fits (seed 11,
# not the committed 5,000-row run) with the unit correctly declared,
# medians of 91 (days), 2.2e3 (hours), 1.3e5 (minutes), and 7.9e6
# (seconds) all score within 0.005 of each other and the Cox fold mean is
# identical to four decimals, because XGBoost estimates the AFT intercept
# from the data. (An earlier version of this guard also imposed a ceiling
# near 1e4, measured from runs whose declared unit did not match their
# durations; that measured the mismatch corruption, not a numeric limit,
# and the ceiling was removed once the runs were redone declared
# correctly.) Below a median of 1.0 the degradation is real in any unit:
# re-censored training durations are floored at one timestep, so most
# labels get fabricated, and a correctly declared years run with a median
# of 0.25 scored 0.589 against the same data's 0.749 in days. The remedy
# is declaring a finer unit, which is what the refusal names.
_MEDIAN_FLOOR = 1.0


@dataclass(frozen=True)
class PipelineConfig:
    """The run's evaluation settings: fold count, burn-in share, horizons, bootstrap resamples, and
    time unit.
    """

    n_folds: int = 5
    min_train_frac: float = 0.4
    # Horizons are in `time_unit` timesteps. The JSON emitted from these
    # still spells its keys horizons_days / calibration_horizon_days: that
    # is the metrics schema, and renaming it would orphan every committed
    # metrics.json.
    horizons: tuple[float, ...] = (90.0, 180.0, 365.0)
    calibration_horizon: float = 180.0
    time_unit: str = "days"
    n_bootstrap: int = 500
    shap_sample_n: int = 2000


def _inner_temporal_split(
    dates: pd.Series, fit_fraction: float = 0.85
) -> tuple[np.ndarray, np.ndarray]:
    """Positional (fit, eval) indices; eval is the latest (1 - fit_fraction) by date.

    Ties are broken by row order, so on coarse-dated data (year-only start
    dates) the cut can fall inside one date and the eval slice is then no
    earlier than the fit rows rather than strictly later. The window is
    already re-censored at its split date, so no test label leaks; what the
    tie changes is which rows choose the stopping round and the scale.
    """
    order = np.argsort(dates.to_numpy(), kind="stable")
    cut_position = int(len(order) * fit_fraction)
    return order[:cut_position], order[cut_position:]


def _fit_aft(
    params: AFTParams,
    features: pd.DataFrame,
    duration: np.ndarray,
    event: np.ndarray,
    dates: pd.Series,
) -> XGBoostAFT:
    """Early-stop on a temporal tail split, refit on the full window at the
    chosen round count, and carry over a predictive scale calibrated on the
    tail rows the probe model never saw."""
    fit_rows, eval_rows = _inner_temporal_split(dates)
    probe = XGBoostAFT(params).fit(
        features.iloc[fit_rows],
        duration[fit_rows],
        event[fit_rows],
        eval_features=features.iloc[eval_rows],
        eval_duration=duration[eval_rows],
        eval_event=event[eval_rows],
    )
    assert probe.booster is not None
    sigma = probe.calibrate_predictive_sigma(
        features.iloc[eval_rows], duration[eval_rows], event[eval_rows]
    )
    best_rounds = probe.booster.best_iteration + 1
    refit = XGBoostAFT(replace(params, n_rounds=best_rounds)).fit(features, duration, event)
    refit.predictive_sigma = sigma
    return refit


def _select_params(
    features: pd.DataFrame, duration: np.ndarray, event: np.ndarray, dates: pd.Series
) -> AFTParams:
    """Pick the grid point by held-out censored log-likelihood, not C-index:
    likelihood punishes a miscalibrated scale, ranking metrics cannot."""
    fit_rows, eval_rows = _inner_temporal_split(dates)
    best_params, best_neg_log_likelihood = None, np.inf
    for params in PARAM_GRID:
        model = XGBoostAFT(params).fit(
            features.iloc[fit_rows],
            duration[fit_rows],
            event[fit_rows],
            eval_features=features.iloc[eval_rows],
            eval_duration=duration[eval_rows],
            eval_event=event[eval_rows],
        )
        assert model.booster is not None
        neg_log_likelihood = float(model.booster.best_score)
        if neg_log_likelihood < best_neg_log_likelihood:
            best_params, best_neg_log_likelihood = params, neg_log_likelihood
    assert best_params is not None
    return best_params


def _evaluate_fold(
    fold: TemporalFold,
    params: AFTParams,
    dataset: pd.DataFrame,
    horizons: np.ndarray,
    date_col: str,
    time_unit: str,
    fold_encoder: Callable,
) -> dict:
    """Re-censor the training labels at the fold's split date, fit both models on that window, and
    score them on the test block; returns the fold's metrics with the raw predictions attached for
    pooling.
    """
    dates = dataset[date_col]
    train_durations, train_events = recensor(
        dataset[DURATION].to_numpy()[fold.train_rows],
        dataset["event"].to_numpy()[fold.train_rows],
        dates.iloc[fold.train_rows],
        fold.split_date,
        time_unit,
    )
    train_features, test_features, cox_drop_columns = fold_encoder(fold.train_rows, fold.test_rows)
    test_durations = dataset[DURATION].to_numpy()[fold.test_rows]
    test_events = dataset["event"].to_numpy()[fold.test_rows]

    aft = _fit_aft(
        params, train_features, train_durations, train_events, dates.iloc[fold.train_rows]
    )
    # Missing values are handled inside CoxBaseline, using medians learned on
    # this training window, so the raw frames go in unmodified.
    cox = CoxBaseline(drop_columns=cox_drop_columns).fit(
        train_features, train_durations, train_events
    )

    predicted_median = aft.predict_median_time(test_features)
    aft_survival = aft.predict_survival(test_features, horizons)
    cox_survival = cox.predict_survival(test_features, horizons)

    return {
        "split_date": str(fold.split_date.date()),
        "n_train": len(fold.train_rows),
        "n_test": len(fold.test_rows),
        "train_event_rate": float(np.mean(train_events)),
        "test_event_rate": float(np.mean(test_events)),
        "c_xgb": harrell_c(test_durations, test_events, predicted_median),
        "c_cox": harrell_c(test_durations, test_events, cox.predict_neg_risk(test_features)),
        "_test_rows": fold.test_rows,
        "_predicted_median": predicted_median,
        "_aft_survival": aft_survival,
        "_cox_survival": cox_survival,
    }


def _run_core(
    dataset: pd.DataFrame,
    features: pd.DataFrame,
    config: PipelineConfig,
    date_col: str,
    dataset_block: dict,
    cox_drop_columns: tuple[str, ...],
    fold_encoder: Callable,
) -> dict:
    """Temporal CV, pooled metrics, final fit, SHAP. Returns the metrics dict
    plus the fitted artifacts the caller writes to disk.

    Labels are taken as they stand in the file. That is exactly what was
    observable when the file was exported, which is the contract the loader
    enforces, so the final fit needs no further re-censoring.
    """
    horizons = np.asarray(config.horizons)

    folds = temporal_folds(dataset[date_col], config.n_folds, config.min_train_frac)
    first_fold = folds[0]
    selection_durations, selection_events = recensor(
        dataset[DURATION].to_numpy()[first_fold.train_rows],
        dataset["event"].to_numpy()[first_fold.train_rows],
        dataset[date_col].iloc[first_fold.train_rows],
        first_fold.split_date,
        config.time_unit,
    )
    selection_features = fold_encoder(first_fold.train_rows, first_fold.train_rows[:0])[0]
    params = _select_params(
        selection_features,
        selection_durations,
        selection_events,
        dataset[date_col].iloc[first_fold.train_rows],
    )

    fold_results = [
        _evaluate_fold(fold, params, dataset, horizons, date_col, config.time_unit, fold_encoder)
        for fold in folds
    ]

    test_rows = np.concatenate([fold_result["_test_rows"] for fold_result in fold_results])
    predicted_median = np.concatenate(
        [fold_result["_predicted_median"] for fold_result in fold_results]
    )
    aft_survival = np.vstack([fold_result["_aft_survival"] for fold_result in fold_results])
    cox_survival = np.vstack([fold_result["_cox_survival"] for fold_result in fold_results])
    pooled_durations = dataset[DURATION].to_numpy()[test_rows]
    pooled_events = dataset["event"].to_numpy()[test_rows]

    n_test_rows = len(test_rows)
    pooled = {
        "n_test": n_test_rows,
        "event_rate": float(np.mean(pooled_events)),
        "c_xgb": harrell_c(pooled_durations, pooled_events, predicted_median),
        "c_xgb_ci": bootstrap_ci(
            lambda i: harrell_c(pooled_durations[i], pooled_events[i], predicted_median[i]),
            n_test_rows,
            config.n_bootstrap,
            seed=1,
        ),
    }
    pooled["c_cox_by_fold_mean"] = float(
        np.mean([fold_result["c_cox"] for fold_result in fold_results])
    )
    # Each fold refits Cox, and predict_partial_hazard returns a risk relative
    # to that fold's own training means, so the scores carry no common scale
    # across folds. A fold mean is therefore the only like-for-like comparison
    # with the AFT model; the pooled AFT figure above is not comparable to it.
    pooled["c_xgb_by_fold_mean"] = float(
        np.mean([fold_result["c_xgb"] for fold_result in fold_results])
    )

    # Marginal KM survival gives the no-skill Brier reference: same probability
    # for every row, censoring handled the same way.
    # Horizon keys carry the unit's abbreviation ("90d", "24h"); day-based
    # runs keep the exact keys every committed metrics.json already has.
    unit_suffix = unit_abbrev(config.time_unit)
    brier = {}
    kaplan_meier = KaplanMeierFitter().fit(pooled_durations, event_observed=pooled_events)
    for j, horizon in enumerate(horizons):
        marginal_survival = float(kaplan_meier.predict(horizon))
        brier[f"{horizon_label(horizon)}{unit_suffix}"] = {
            "xgb": ipcw_brier(pooled_durations, pooled_events, aft_survival[:, j], horizon),
            "cox": ipcw_brier(pooled_durations, pooled_events, cox_survival[:, j], horizon),
            "km_marginal": ipcw_brier(
                pooled_durations, pooled_events, np.full(n_test_rows, marginal_survival), horizon
            ),
        }

    calibration_horizon = config.calibration_horizon
    calibration_column = int(np.argmin(np.abs(horizons - calibration_horizon)))
    calibration = calibration_bins(
        pooled_durations, pooled_events, aft_survival[:, calibration_column], calibration_horizon
    )
    # The Cox survival probabilities at the same horizon already exist (the
    # Brier table grades them), so the baseline gets the same calibration
    # binning, on its own predicted deciles.
    cox_calibration = calibration_bins(
        pooled_durations, pooled_events, cox_survival[:, calibration_column], calibration_horizon
    )

    fold_metrics = pd.DataFrame(
        [
            {key: value for key, value in fold_result.items() if not key.startswith("_")}
            for fold_result in fold_results
        ]
    )
    fold_metrics["fold_label"] = [
        f"F{i + 1}\n{fold_result['split_date'][:7]}" for i, fold_result in enumerate(fold_results)
    ]

    all_durations = dataset[DURATION].to_numpy()
    all_events = dataset["event"].to_numpy()
    final_model = _fit_aft(params, features, all_durations, all_events, dataset[date_col])
    sampled_features, shap_values, shap_importance = compute_shap(
        final_model, features, config.shap_sample_n
    )
    final_cox = CoxBaseline(drop_columns=cox_drop_columns).fit(features, all_durations, all_events)

    metrics = {
        "params": {
            "max_depth": params.max_depth,
            "aft_sigma": params.aft_sigma,
            "learning_rate": params.learning_rate,
            "predictive_sigma_final": final_model.predictive_sigma,
        },
        # Recorded so the report can state the run's conditions by reading them.
        # The builder used to carry its own copies of these as literals, which
        # meant a run at different settings produced a report describing the
        # settings the builder was written for.
        "config": {
            "n_folds": config.n_folds,
            "min_train_frac": config.min_train_frac,
            "n_bootstrap": config.n_bootstrap,
            "horizons_days": list(config.horizons),
            "calibration_horizon_days": config.calibration_horizon,
            "time_unit": config.time_unit,
        },
        "dataset": dataset_block,
        "folds": fold_metrics.drop(columns="fold_label").to_dict(orient="records"),
        "pooled": pooled,
        "ipcw_brier": brier,
        f"calibration_{horizon_label(calibration_horizon)}{unit_suffix}": calibration.to_dict(
            orient="records"
        ),
        f"calibration_cox_{horizon_label(calibration_horizon)}"
        f"{unit_suffix}": cox_calibration.to_dict(orient="records"),
        "shap_top": shap_importance.head(12).to_dict(orient="records"),
        "cox_top": final_cox.top_coefficients(12),
        # The dropped reference level per categorical column, so the report
        # can name what each one-hot hazard ratio is measured against.
        "cox_reference": list(cox_drop_columns),
    }
    return {
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "calibration": calibration,
        "cox_calibration": cox_calibration,
        "calibration_horizon": calibration_horizon,
        "final_model": final_model,
        "final_cox": final_cox,
        "sampled_features": sampled_features,
        "shap_values": shap_values,
        "mean_abs": shap_importance,
        # Out-of-fold row indices and predictions, so callers can compute
        # decompositions (e.g. within-group concordance) without refitting.
        "pooled_test_rows": test_rows,
        "pooled_predicted_median": predicted_median,
    }


def fit_evaluate(
    data_path: str | Path,
    name: str,
    id_col: str,
    date_col: str,
    duration_col: str,
    event_col: str,
    drop_cols: tuple[str, ...] = (),
    categorical_cols: tuple[str, ...] = (),
    n_folds: int = 5,
    horizons: tuple[float, ...] = (90.0, 180.0, 365.0),
    min_train_frac: float = 0.4,
    out_dir: str | Path | None = None,
    n_bootstrap: int = 500,
    km_col: str | None = None,
    time_unit: str = "days",
) -> dict:
    """Full evaluation of a duration CSV; writes a run directory and returns
    the metrics dict. Raises ValueError on a malformed file or one too small
    to support the requested folds. `km_col` names a categorical column to
    draw a Kaplan-Meier by-group figure from; the report cites it in its Data
    section when present. `time_unit` is the unit the duration column and the
    horizons are measured in; label re-censoring converts calendar spans into
    it, and the report and figures name it."""
    time_unit = check_time_unit(time_unit)
    loaded = load_duration_csv(
        data_path,
        id_col,
        date_col,
        duration_col,
        event_col,
        drop_cols,
        categorical_cols,
        time_unit=time_unit,
    )
    dataset, features = loaded.frame, loaded.features

    if km_col is not None and km_col not in dataset.columns:
        raise ValueError(
            f"--km-col {km_col!r} is not a column of this dataset. It may be a "
            "feature column or one named in --drop-cols; check for a typo."
        )

    refusal = check_minimum_data(len(dataset), int(dataset[EVENT].sum()), n_folds)
    if refusal is not None:
        raise ValueError(refusal)

    median_duration = float(dataset[DURATION].median())
    if median_duration < _MEDIAN_FLOOR:
        raise ValueError(
            f"refusing to fit: the median observed duration is {median_duration:.3g} {time_unit}, "
            "below one timestep. Re-censored training durations are floored at 1.0 "
            "timestep, so most labels would be fabricated at this scale. Declare a "
            "finer --time-unit so typical durations are tens of timesteps or more."
        )

    horizons = tuple(float(horizon) for horizon in horizons)
    config = PipelineConfig(
        n_folds=n_folds,
        min_train_frac=min_train_frac,
        horizons=horizons,
        # The calibration deep-dive happens at the middle requested horizon,
        # not a hardcoded 180 days: real datasets live on their own timescale.
        calibration_horizon=horizons[len(horizons) // 2],
        n_bootstrap=n_bootstrap,
        time_unit=time_unit,
    )
    # The per-fold encoder refits the one-hot vocabulary on each training
    # window, so early folds cannot see level frequencies from after their
    # split dates. The full-file recipe (data.recipe) still serves the
    # deployed model, whose past legitimately is the whole file.
    # Dropped columns ride in the frame for grouping figures but must never
    # reach the per-fold matrices, so the exclusion here mirrors the loader's.
    feature_cols = [
        column
        for column in dataset.columns
        if column not in (ROW_ID, START, DURATION, EVENT) and column not in drop_cols
    ]
    fold_encoder = make_fold_encoder(dataset[feature_cols], tuple(categorical_cols))

    # Said before the fit, not after, so a wrong unit is caught before minutes
    # of fitting: the loader cannot tell months in a days column apart.
    print(
        f"declared time unit: {time_unit}; median observed duration"
        f" {dataset[DURATION].median():.0f} {time_unit}"
    )

    core_results = _run_core(
        dataset,
        features,
        config,
        date_col=START,
        fold_encoder=fold_encoder,
        dataset_block={
            "n_rows": len(dataset),
            "event_rate": float(dataset[EVENT].mean()),
            # The _days key name is the metrics schema, shared with every
            # committed metrics.json; the value is in run.time_unit timesteps.
            "median_observed_duration_days": float(dataset[DURATION].median()),
            "date_min": str(dataset[START].min().date()),
            "date_max": str(dataset[START].max().date()),
            "n_features": features.shape[1],
        },
        cox_drop_columns=loaded.recipe.reference_columns,
    )
    run_dir = Path(out_dir) if out_dir is not None else Path("runs") / name
    metrics = core_results["metrics"]
    metrics["run"] = {
        "name": name,
        "source": str(data_path),
        # Recorded because the report prints the command that reproduces the
        # run. Without it the printed command omits --out and writes to the
        # default runs/<name>/, so following it rebuilds the report from the
        # untouched old metrics and looks like it reproduced when it did not.
        "out_dir": run_dir.as_posix(),
        "columns": {
            "id": id_col,
            "date": date_col,
            "duration": duration_col,
            "event": event_col,
            "dropped": list(drop_cols),
            "categorical": list(categorical_cols),
        },
        "n_folds": n_folds,
        # The _days key spellings are the metrics schema; the values are in
        # time_unit timesteps, recorded beside them.
        "horizons_days": list(horizons),
        "calibration_horizon_days": config.calibration_horizon,
        "time_unit": time_unit,
    }
    if km_col is not None:
        metrics["run"]["km_col"] = km_col
        # How much of the pooled concordance is group membership alone, and
        # how much survives when comparisons stay inside a group. Computed
        # from the same out-of-fold predictions the pooled figure uses; the
        # report renders it when present.
        test_rows = core_results["pooled_test_rows"]
        decomposition = within_group_concordance(
            dataset[DURATION].to_numpy()[test_rows],
            dataset[EVENT].to_numpy()[test_rows],
            core_results["pooled_predicted_median"],
            dataset[km_col].iloc[test_rows],
        )
        if decomposition is not None:
            metrics["within_group"] = {"col": km_col, **decomposition}

    figures_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    fold_cindex_plot(core_results["fold_metrics"], figures_dir / "fold_cindex.png")
    calibration_horizon = core_results["calibration_horizon"]
    calibration_plot(
        core_results["calibration"],
        calibration_horizon,
        figures_dir
        / f"calibration_{horizon_label(calibration_horizon)}{unit_abbrev(time_unit)}.png",
        time_unit=time_unit,
        cox_bins=core_results["cox_calibration"],
    )
    two_level_columns = {
        column for column, levels in loaded.recipe.categorical_levels.items() if len(levels) == 2
    }
    cox_hr_plot(
        pd.DataFrame(metrics["cox_top"]), figures_dir / "cox_hr.png", keep_prefix=two_level_columns
    )
    write_shap_figures(
        core_results["sampled_features"],
        core_results["shap_values"],
        core_results["mean_abs"],
        figures_dir,
        time_unit=time_unit,
        keep_prefix=two_level_columns,
    )
    if km_col is not None:
        group_values = dataset[km_col]
        group = group_values.where(group_values.notna(), "(missing)").astype(str)
        if group.nunique() > 8:
            top = group.value_counts().nlargest(7).index
            group = group.where(group.isin(top), "(other)")
        km_by_group_plot(
            dataset[DURATION].to_numpy(dtype=float),
            dataset[EVENT].to_numpy(),
            group,
            figures_dir / "km_by_group.png",
            # Axis and labels come from the data rather than any one dataset's
            # vocabulary, so the figure reads the same for every run.
            max_time=float(np.quantile(dataset[DURATION].to_numpy(dtype=float), 0.95)),
            xlabel=f"{time_unit} since start",
            ylabel="fraction surviving",
        )
    save_model_bundle(
        run_dir / "model",
        core_results["final_model"],
        loaded.recipe,
        metadata={
            "run_name": name,
            "source": str(data_path),
            "id_col": id_col,
            "n_train_rows": len(dataset),
            "training_date": datetime.date.today().isoformat(),
            # Saved so predict() can label its outputs in the unit the model
            # was trained in rather than assuming days.
            "time_unit": time_unit,
        },
        cox=core_results["final_cox"],
        scores={
            "aft": metrics["pooled"]["c_xgb_by_fold_mean"],
            "cox": metrics["pooled"]["c_cox_by_fold_mean"],
        },
    )
    return metrics


def predict(
    model_dir: str | Path,
    data_path: str | Path,
    horizons: tuple[float, ...] = (90.0, 180.0, 365.0),
    model_type: str | None = None,
) -> pd.DataFrame:
    """Score new rows with a saved model bundle. The CSV must carry the id
    column and every feature column the model was trained on; outcome columns
    are not needed and are ignored if present.

    `model_type` picks 'aft' or 'cox'; the default follows the bundle's
    recommendation, which is whichever scored higher on out-of-time fold-mean
    concordance during the run that produced it.

    Horizons are in the time unit the model was trained with, recorded in the
    bundle, and the output column names carry that unit. Bundles saved before
    the unit was recorded are day-based by construction and read as days.
    """
    model_dir = Path(model_dir)
    # Accept either the run directory or its model/ subdirectory.
    if (model_dir / "model" / "sidecar.json").exists():
        model_dir = model_dir / "model"
    aft, recipe, sidecar = load_model_bundle(model_dir)

    available = sidecar.get("models", {"aft": {}})
    chosen = model_type or sidecar.get("recommended", "aft")
    if chosen not in available:
        raise ValueError(
            f"model type {chosen!r} is not in this bundle; available: "
            + ", ".join(sorted(available))
        )
    if chosen == "cox":
        model = load_cox_from_bundle(model_dir)
        if model is None:
            raise ValueError(f"bundle claims a cox model but {model_dir / 'cox.pkl'} is missing")
    else:
        model = aft
    score = available[chosen].get("c_index_fold_mean")
    scored_note = f" (out-of-time C-index {score:.3f})" if score is not None else ""
    print(f"scoring with the {chosen} model{scored_note}")

    raw = pd.read_csv(data_path)
    id_col = sidecar["id_col"]
    if id_col not in raw.columns:
        raise ValueError(f"id column {id_col!r} (from the saved model) not found in the input")
    features = encode_with_recipe(raw.drop(columns=[id_col]), recipe)

    horizon_values = np.asarray([float(horizon) for horizon in horizons])
    median = model.predict_median_time(features)
    survival = model.predict_survival(features, horizon_values)
    if np.isinf(median).any():
        n_infinite_medians = int(np.isinf(median).sum())
        print(
            f"{n_infinite_medians} rows have no finite median: their survival curve never "
            "reaches 0.5 inside the observed follow-up, so the data cannot say when half of "
            "them fail"
        )
    time_unit = sidecar.get("time_unit", "days")
    predictions = pd.DataFrame(
        {id_col: raw[id_col], "model": chosen, f"predicted_median_{time_unit}": median}
    )
    for j, horizon in enumerate(horizon_values):
        predictions[f"p_survive_{horizon_label(horizon)}{unit_abbrev(time_unit)}"] = survival[:, j]
    return predictions
