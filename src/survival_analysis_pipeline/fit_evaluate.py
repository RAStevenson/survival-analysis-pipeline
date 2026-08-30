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
    ID,
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
# 800-row synthetic validation set with the unit correctly declared,
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


def _inner_temporal_split(dates: pd.Series, frac: float = 0.85) -> tuple[np.ndarray, np.ndarray]:
    """Positional (fit, eval) indices; eval is the latest (1 - frac) by date."""
    order = np.argsort(dates.to_numpy(), kind="stable")
    cut = int(len(order) * frac)
    return order[:cut], order[cut:]


def _fit_aft(
    params: AFTParams,
    x: pd.DataFrame,
    duration: np.ndarray,
    event: np.ndarray,
    dates: pd.Series,
) -> XGBoostAFT:
    """Early-stop on a temporal tail split, refit on the full window at the
    chosen round count, and carry over a predictive scale calibrated on the
    tail rows the probe model never saw."""
    fit_idx, eval_idx = _inner_temporal_split(dates)
    probe = XGBoostAFT(params).fit(
        x.iloc[fit_idx],
        duration[fit_idx],
        event[fit_idx],
        eval_x=x.iloc[eval_idx],
        eval_duration=duration[eval_idx],
        eval_event=event[eval_idx],
    )
    assert probe.booster is not None
    sigma = probe.calibrate_predictive_sigma(x.iloc[eval_idx], duration[eval_idx], event[eval_idx])
    best_rounds = probe.booster.best_iteration + 1
    refit = XGBoostAFT(replace(params, n_rounds=best_rounds)).fit(x, duration, event)
    refit.predictive_sigma = sigma
    return refit


def _select_params(
    x: pd.DataFrame, duration: np.ndarray, event: np.ndarray, dates: pd.Series
) -> AFTParams:
    """Pick the grid point by held-out censored log-likelihood, not C-index:
    likelihood punishes a miscalibrated scale, ranking metrics cannot."""
    fit_idx, eval_idx = _inner_temporal_split(dates)
    best_params, best_nll = None, np.inf
    for params in PARAM_GRID:
        model = XGBoostAFT(params).fit(
            x.iloc[fit_idx],
            duration[fit_idx],
            event[fit_idx],
            eval_x=x.iloc[eval_idx],
            eval_duration=duration[eval_idx],
            eval_event=event[eval_idx],
        )
        assert model.booster is not None
        nll = float(model.booster.best_score)
        if nll < best_nll:
            best_params, best_nll = params, nll
    assert best_params is not None
    return best_params


def _evaluate_fold(
    fold: TemporalFold,
    params: AFTParams,
    df: pd.DataFrame,
    horizons: np.ndarray,
    date_col: str,
    time_unit: str,
    fold_encoder: Callable,
) -> dict:
    dates = df[date_col]
    train_dur, train_ev = recensor(
        df[DURATION].to_numpy()[fold.train_idx],
        df["event"].to_numpy()[fold.train_idx],
        dates.iloc[fold.train_idx],
        fold.split_date,
        time_unit,
    )
    x_train, x_test, cox_drop_columns = fold_encoder(fold.train_idx, fold.test_idx)
    test_dur = df[DURATION].to_numpy()[fold.test_idx]
    test_ev = df["event"].to_numpy()[fold.test_idx]

    aft = _fit_aft(params, x_train, train_dur, train_ev, dates.iloc[fold.train_idx])
    # Missing values are handled inside CoxBaseline, using medians learned on
    # this training window, so the raw frames go in unmodified.
    cox = CoxBaseline(drop_columns=cox_drop_columns).fit(x_train, train_dur, train_ev)

    pred = aft.predict_median_time(x_test)
    surv = aft.predict_survival(x_test, horizons)
    cox_surv = cox.predict_survival(x_test, horizons)

    return {
        "split_date": str(fold.split_date.date()),
        "n_train": len(fold.train_idx),
        "n_test": len(fold.test_idx),
        "train_event_rate": float(np.mean(train_ev)),
        "test_event_rate": float(np.mean(test_ev)),
        "c_xgb": harrell_c(test_dur, test_ev, pred),
        "c_cox": harrell_c(test_dur, test_ev, cox.predict_neg_risk(x_test)),
        "_test_idx": fold.test_idx,
        "_pred": pred,
        "_surv": surv,
        "_cox_surv": cox_surv,
    }


def _run_core(
    df: pd.DataFrame,
    x: pd.DataFrame,
    cfg: PipelineConfig,
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
    horizons = np.asarray(cfg.horizons)

    folds = temporal_folds(df[date_col], cfg.n_folds, cfg.min_train_frac)
    first = folds[0]
    sel_dur, sel_ev = recensor(
        df[DURATION].to_numpy()[first.train_idx],
        df["event"].to_numpy()[first.train_idx],
        df[date_col].iloc[first.train_idx],
        first.split_date,
        cfg.time_unit,
    )
    x_sel = fold_encoder(first.train_idx, first.train_idx[:0])[0]
    params = _select_params(x_sel, sel_dur, sel_ev, df[date_col].iloc[first.train_idx])

    fold_results = [
        _evaluate_fold(f, params, df, horizons, date_col, cfg.time_unit, fold_encoder)
        for f in folds
    ]

    test_idx = np.concatenate([r["_test_idx"] for r in fold_results])
    pred = np.concatenate([r["_pred"] for r in fold_results])
    surv = np.vstack([r["_surv"] for r in fold_results])
    cox_surv = np.vstack([r["_cox_surv"] for r in fold_results])
    oof_dur = df[DURATION].to_numpy()[test_idx]
    oof_ev = df["event"].to_numpy()[test_idx]

    n = len(test_idx)
    pooled = {
        "n_test": n,
        "event_rate": float(np.mean(oof_ev)),
        "c_xgb": harrell_c(oof_dur, oof_ev, pred),
        "c_xgb_ci": bootstrap_ci(
            lambda i: harrell_c(oof_dur[i], oof_ev[i], pred[i]), n, cfg.n_bootstrap, seed=1
        ),
    }
    pooled["c_cox_by_fold_mean"] = float(np.mean([r["c_cox"] for r in fold_results]))
    # Each fold refits Cox, and predict_partial_hazard returns a risk relative
    # to that fold's own training means, so the scores carry no common scale
    # across folds. A fold mean is therefore the only like-for-like comparison
    # with the AFT model; the pooled AFT figure above is not comparable to it.
    pooled["c_xgb_by_fold_mean"] = float(np.mean([r["c_xgb"] for r in fold_results]))

    # Marginal KM survival gives the no-skill Brier reference: same probability
    # for every row, censoring handled the same way.
    # Horizon keys carry the unit's abbreviation ("90d", "24h"); day-based
    # runs keep the exact keys every committed metrics.json already has.
    ua = unit_abbrev(cfg.time_unit)
    brier = {}
    kmf = KaplanMeierFitter().fit(oof_dur, event_observed=oof_ev)
    for j, h in enumerate(horizons):
        marginal = float(kmf.predict(h))
        brier[f"{horizon_label(h)}{ua}"] = {
            "xgb": ipcw_brier(oof_dur, oof_ev, surv[:, j], h),
            "cox": ipcw_brier(oof_dur, oof_ev, cox_surv[:, j], h),
            "km_marginal": ipcw_brier(oof_dur, oof_ev, np.full(n, marginal), h),
        }

    h_cal = cfg.calibration_horizon
    j_cal = int(np.argmin(np.abs(horizons - h_cal)))
    cal = calibration_bins(oof_dur, oof_ev, surv[:, j_cal], h_cal)
    # The Cox survival probabilities at the same horizon already exist (the
    # Brier table grades them), so the baseline gets the same calibration
    # lens, binned on its own predicted deciles.
    cal_cox = calibration_bins(oof_dur, oof_ev, cox_surv[:, j_cal], h_cal)

    fold_metrics = pd.DataFrame(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in fold_results]
    )
    fold_metrics["fold_label"] = [
        f"F{i + 1}\n{r['split_date'][:7]}" for i, r in enumerate(fold_results)
    ]

    dur_all = df[DURATION].to_numpy()
    ev_all = df["event"].to_numpy()
    final_model = _fit_aft(params, x, dur_all, ev_all, df[date_col])
    x_sample, shap_values, mean_abs = compute_shap(final_model, x, cfg.shap_sample_n)
    final_cox = CoxBaseline(drop_columns=cox_drop_columns).fit(x, dur_all, ev_all)

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
            "n_folds": cfg.n_folds,
            "min_train_frac": cfg.min_train_frac,
            "n_bootstrap": cfg.n_bootstrap,
            "horizons_days": list(cfg.horizons),
            "calibration_horizon_days": cfg.calibration_horizon,
            "time_unit": cfg.time_unit,
        },
        "dataset": dataset_block,
        "folds": fold_metrics.drop(columns="fold_label").to_dict(orient="records"),
        "pooled": pooled,
        "ipcw_brier": brier,
        f"calibration_{horizon_label(h_cal)}{ua}": cal.to_dict(orient="records"),
        f"calibration_cox_{horizon_label(h_cal)}{ua}": cal_cox.to_dict(orient="records"),
        "shap_top": mean_abs.head(12).to_dict(orient="records"),
        "cox_top": final_cox.top_coefficients(12),
        # The dropped reference level per categorical column, so the report
        # can name what each one-hot hazard ratio is measured against.
        "cox_reference": list(cox_drop_columns),
    }
    return {
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "cal": cal,
        "cal_cox": cal_cox,
        "h_cal": h_cal,
        "final_model": final_model,
        "final_cox": final_cox,
        "x_sample": x_sample,
        "shap_values": shap_values,
        "mean_abs": mean_abs,
        # Out-of-fold row indices and predictions, so callers can compute
        # decompositions (e.g. within-group concordance) without refitting.
        "oof_test_idx": test_idx,
        "oof_pred": pred,
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
    data = load_duration_csv(
        data_path,
        id_col,
        date_col,
        duration_col,
        event_col,
        drop_cols,
        categorical_cols,
        time_unit=time_unit,
    )
    frame, x = data.frame, data.features

    if km_col is not None and km_col not in frame.columns:
        raise ValueError(
            f"--km-col {km_col!r} is not a column of this dataset. It may be a "
            "feature column or one named in --drop-cols; check for a typo."
        )

    refusal = check_minimum_data(len(frame), int(frame[EVENT].sum()), n_folds)
    if refusal is not None:
        raise ValueError(refusal)

    med = float(frame[DURATION].median())
    if med < _MEDIAN_FLOOR:
        raise ValueError(
            f"refusing to fit: the median observed duration is {med:.3g} {time_unit}, "
            "below one timestep. Re-censored training durations are floored at 1.0 "
            "timestep, so most labels would be fabricated at this scale. Declare a "
            "finer --time-unit so typical durations are tens of timesteps or more."
        )

    horizons = tuple(float(h) for h in horizons)
    cfg = PipelineConfig(
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
        c for c in frame.columns if c not in (ID, START, DURATION, EVENT) and c not in drop_cols
    ]
    fold_encoder = make_fold_encoder(frame[feature_cols], tuple(categorical_cols))

    core = _run_core(
        frame,
        x,
        cfg,
        date_col=START,
        fold_encoder=fold_encoder,
        dataset_block={
            "n_rows": len(frame),
            "event_rate": float(frame[EVENT].mean()),
            # The _days key name is the metrics schema, shared with every
            # committed metrics.json; the value is in run.time_unit timesteps.
            "median_observed_duration_days": float(frame[DURATION].median()),
            "date_min": str(frame[START].min().date()),
            "date_max": str(frame[START].max().date()),
            "n_features": x.shape[1],
        },
        cox_drop_columns=data.recipe.reference_columns,
    )
    out = Path(out_dir) if out_dir is not None else Path("runs") / name
    metrics = core["metrics"]
    metrics["run"] = {
        "name": name,
        "source": str(data_path),
        # Recorded because the report prints the command that reproduces the
        # run. Without it the printed command omits --out and writes to the
        # default runs/<name>/, so following it rebuilds the report from the
        # untouched old metrics and looks like it reproduced when it did not.
        "out_dir": out.as_posix(),
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
        "calibration_horizon_days": cfg.calibration_horizon,
        "time_unit": time_unit,
    }
    if km_col is not None:
        metrics["run"]["km_col"] = km_col
        # How much of the pooled concordance is group membership alone, and
        # how much survives when comparisons stay inside a group. Computed
        # from the same out-of-fold predictions the pooled figure uses; the
        # report renders it when present.
        test_idx = core["oof_test_idx"]
        decomposition = within_group_concordance(
            frame[DURATION].to_numpy()[test_idx],
            frame[EVENT].to_numpy()[test_idx],
            core["oof_pred"],
            frame[km_col].iloc[test_idx],
        )
        if decomposition is not None:
            metrics["within_group"] = {"col": km_col, **decomposition}

    figures = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    fold_cindex_plot(core["fold_metrics"], figures / "fold_cindex.png")
    h_cal = core["h_cal"]
    calibration_plot(
        core["cal"],
        h_cal,
        figures / f"calibration_{horizon_label(h_cal)}{unit_abbrev(time_unit)}.png",
        time_unit=time_unit,
        cox_bins_df=core["cal_cox"],
    )
    cox_hr_plot(pd.DataFrame(metrics["cox_top"]), figures / "cox_hr.png")
    write_shap_figures(
        core["x_sample"],
        core["shap_values"],
        core["mean_abs"],
        figures,
        time_unit=time_unit,
        numeric_only_dependence=True,
    )
    if km_col is not None:
        raw_group = frame[km_col]
        group = raw_group.where(raw_group.notna(), "(missing)").astype(str)
        if group.nunique() > 8:
            top = group.value_counts().nlargest(7).index
            group = group.where(group.isin(top), "(other)")
        km_by_group_plot(
            frame[DURATION].to_numpy(dtype=float),
            frame[EVENT].to_numpy(),
            group,
            figures / "km_by_group.png",
            # Axis and labels come from the data rather than any one dataset's
            # vocabulary, so the figure reads the same for every run.
            max_time=float(np.quantile(frame[DURATION].to_numpy(dtype=float), 0.95)),
            xlabel=f"{time_unit} since start",
            ylabel="fraction surviving",
        )
    save_model_bundle(
        out / "model",
        core["final_model"],
        data.recipe,
        meta={
            "run_name": name,
            "source": str(data_path),
            "id_col": id_col,
            "n_train_rows": len(frame),
            "training_date": datetime.date.today().isoformat(),
            # Saved so predict() can label its outputs in the unit the model
            # was trained in rather than assuming days.
            "time_unit": time_unit,
        },
        cox=core["final_cox"],
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
    x = encode_with_recipe(raw.drop(columns=[id_col]), recipe)

    horizon_arr = np.asarray([float(h) for h in horizons])
    median = model.predict_median_time(x)
    survival = model.predict_survival(x, horizon_arr)
    if np.isinf(median).any():
        n_inf = int(np.isinf(median).sum())
        print(
            f"{n_inf} rows have no finite median: their survival curve never reaches 0.5 "
            "inside the observed follow-up, so the data cannot say when half of them fail"
        )
    time_unit = sidecar.get("time_unit", "days")
    out = pd.DataFrame(
        {id_col: raw[id_col], "model": chosen, f"predicted_median_{time_unit}": median}
    )
    for j, h in enumerate(horizon_arr):
        out[f"p_survive_{horizon_label(h)}{unit_abbrev(time_unit)}"] = survival[:, j]
    return out
