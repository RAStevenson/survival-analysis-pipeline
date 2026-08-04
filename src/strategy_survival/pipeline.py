"""End-to-end run: generate data, temporal CV against baselines, pooled
metrics with bootstrap intervals, calibration, final fit, SHAP, figures.

Hyperparameters are selected once, on the first fold's training window with an
inner temporal split, and reused for every fold. Selecting per-fold would be
cleaner in principle but makes fold metrics harder to compare and sextuples
the runtime for no visible gain on this data.

`_run_core` is the evaluation shared by the synthetic pipeline and real-data
runs. The synthetic extras -- the validation-Sharpe anti-baseline and the
oracle ceiling -- are optional there, because real data has neither a
selection metric guaranteed to exist nor a latent truth table.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from .baseline import CoxBaseline
from .cv import TemporalFold, recensor, temporal_folds
from .evaluate import bootstrap_ci, calibration_bins, harrell_c, ipcw_brier
from .features import COX_REFERENCE_COLUMNS, build_features
from .generate import GeneratorConfig, generate
from .model import AFTParams, XGBoostAFT
from .plots import calibration_plot, fold_cindex_plot, km_by_group_plot
from .shap_analysis import compute_shap, write_shap_figures

PARAM_GRID: tuple[AFTParams, ...] = tuple(
    AFTParams(max_depth=depth, aft_sigma=sigma) for depth in (2, 3) for sigma in (0.6, 0.9, 1.2)
)


@dataclass(frozen=True)
class PipelineConfig:
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    n_folds: int = 5
    min_train_frac: float = 0.4
    horizons_days: tuple[float, ...] = (90.0, 180.0, 365.0)
    calibration_horizon_days: float = 180.0
    n_bootstrap: int = 500
    shap_sample_n: int = 2000
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    # A robustness run at another seed is only wanted for its metrics. Writing
    # it under its own name, with figures and data left alone, is what keeps
    # `--seed 8` from overwriting the seed-7 artifacts the report is built from.
    metrics_name: str = "metrics.json"
    write_figures: bool = True


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
    x: pd.DataFrame,
    df: pd.DataFrame,
    horizons: np.ndarray,
    date_col: str,
    latents: pd.DataFrame | None,
    sharpe_col: str | None,
    cox_drop_columns: tuple[str, ...],
) -> dict:
    dates = df[date_col]
    train_dur, train_ev = recensor(
        df["duration_days"].to_numpy()[fold.train_idx],
        df["event"].to_numpy()[fold.train_idx],
        dates.iloc[fold.train_idx],
        fold.split_date,
    )
    x_train = x.iloc[fold.train_idx]
    x_test = x.iloc[fold.test_idx]
    test_dur = df["duration_days"].to_numpy()[fold.test_idx]
    test_ev = df["event"].to_numpy()[fold.test_idx]

    aft = _fit_aft(params, x_train, train_dur, train_ev, dates.iloc[fold.train_idx])
    # Missing values are handled inside CoxBaseline, using medians learned on
    # this training window, so the raw frames go in unmodified.
    cox = CoxBaseline(drop_columns=cox_drop_columns).fit(x_train, train_dur, train_ev)

    pred_days = aft.predict_median_days(x_test)
    surv = aft.predict_survival(x_test, horizons)
    cox_surv = cox.predict_survival(x_test, horizons)

    result = {
        "split_date": str(fold.split_date.date()),
        "n_train": len(fold.train_idx),
        "n_test": len(fold.test_idx),
        "train_event_rate": float(np.mean(train_ev)),
        "test_event_rate": float(np.mean(test_ev)),
        "c_xgb": harrell_c(test_dur, test_ev, pred_days),
        "c_cox": harrell_c(test_dur, test_ev, cox.predict_neg_risk(x_test)),
    }
    if sharpe_col is not None:
        result["c_sharpe"] = harrell_c(test_dur, test_ev, df[sharpe_col].to_numpy()[fold.test_idx])
    if latents is not None:
        result["c_oracle"] = harrell_c(
            test_dur, test_ev, latents["log_time_eta"].to_numpy()[fold.test_idx]
        )
    result.update(
        {"_test_idx": fold.test_idx, "_pred_days": pred_days, "_surv": surv, "_cox_surv": cox_surv}
    )
    return result


def _run_core(
    df: pd.DataFrame,
    x: pd.DataFrame,
    cfg: PipelineConfig,
    date_col: str,
    dataset_block: dict,
    latents: pd.DataFrame | None = None,
    sharpe_col: str | None = None,
    final_recensor_at: pd.Timestamp | None = None,
    fit_final_cox: bool = False,
    cox_drop_columns: tuple[str, ...] = (),
) -> dict:
    """Temporal CV, pooled metrics, final fit, SHAP. Returns the metrics dict
    plus the fitted artifacts the callers write to disk. `final_recensor_at`
    re-censors the final fit's labels at a known observation cutoff; real data
    passes None because its labels are already exactly what was observable
    when the file was exported."""
    horizons = np.asarray(cfg.horizons_days)

    folds = temporal_folds(df[date_col], cfg.n_folds, cfg.min_train_frac)
    first = folds[0]
    sel_dur, sel_ev = recensor(
        df["duration_days"].to_numpy()[first.train_idx],
        df["event"].to_numpy()[first.train_idx],
        df[date_col].iloc[first.train_idx],
        first.split_date,
    )
    params = _select_params(
        x.iloc[first.train_idx], sel_dur, sel_ev, df[date_col].iloc[first.train_idx]
    )

    fold_results = [
        _evaluate_fold(f, params, x, df, horizons, date_col, latents, sharpe_col, cox_drop_columns)
        for f in folds
    ]

    test_idx = np.concatenate([r["_test_idx"] for r in fold_results])
    pred_days = np.concatenate([r["_pred_days"] for r in fold_results])
    surv = np.vstack([r["_surv"] for r in fold_results])
    cox_surv = np.vstack([r["_cox_surv"] for r in fold_results])
    oof_dur = df["duration_days"].to_numpy()[test_idx]
    oof_ev = df["event"].to_numpy()[test_idx]

    def c_of(scores: np.ndarray, idx: np.ndarray) -> float:
        return harrell_c(oof_dur[idx], oof_ev[idx], scores[idx])

    n = len(test_idx)
    pooled = {
        "n_test": n,
        "event_rate": float(np.mean(oof_ev)),
        "c_xgb": harrell_c(oof_dur, oof_ev, pred_days),
        "c_xgb_ci": bootstrap_ci(lambda i: c_of(pred_days, i), n, cfg.n_bootstrap, seed=1),
    }
    if sharpe_col is not None:
        oof_sharpe = df[sharpe_col].to_numpy()[test_idx]
        pooled["c_sharpe"] = harrell_c(oof_dur, oof_ev, oof_sharpe)
        pooled["c_sharpe_ci"] = bootstrap_ci(
            lambda i: c_of(oof_sharpe, i), n, cfg.n_bootstrap, seed=2
        )
    if latents is not None:
        oof_eta = latents["log_time_eta"].to_numpy()[test_idx]
        pooled["c_oracle"] = harrell_c(oof_dur, oof_ev, oof_eta)
    pooled["c_cox_by_fold_mean"] = float(np.mean([r["c_cox"] for r in fold_results]))
    # Each fold refits Cox, and predict_partial_hazard returns a risk relative
    # to that fold's own training means, so the scores carry no common scale
    # across folds. A fold mean is therefore the only like-for-like comparison
    # with the AFT model; the pooled AFT figure above is not comparable to it.
    pooled["c_xgb_by_fold_mean"] = float(np.mean([r["c_xgb"] for r in fold_results]))

    # Marginal KM survival gives the no-skill Brier reference: same probability
    # for every row, censoring handled the same way.
    brier = {}
    kmf = KaplanMeierFitter().fit(oof_dur, event_observed=oof_ev)
    for j, h in enumerate(horizons):
        marginal = float(kmf.predict(h))
        brier[f"{int(h)}d"] = {
            "xgb": ipcw_brier(oof_dur, oof_ev, surv[:, j], h),
            "cox": ipcw_brier(oof_dur, oof_ev, cox_surv[:, j], h),
            "km_marginal": ipcw_brier(oof_dur, oof_ev, np.full(n, marginal), h),
        }

    h_cal = cfg.calibration_horizon_days
    j_cal = int(np.argmin(np.abs(horizons - h_cal)))
    cal = calibration_bins(oof_dur, oof_ev, surv[:, j_cal], h_cal)

    fold_metrics = pd.DataFrame(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in fold_results]
    )
    fold_metrics["fold_label"] = [
        f"F{i + 1}\n{r['split_date'][:7]}" for i, r in enumerate(fold_results)
    ]

    if final_recensor_at is not None:
        dur_all, ev_all = recensor(
            df["duration_days"].to_numpy(), df["event"].to_numpy(), df[date_col], final_recensor_at
        )
    else:
        dur_all = df["duration_days"].to_numpy()
        ev_all = df["event"].to_numpy()
    final_model = _fit_aft(params, x, dur_all, ev_all, df[date_col])
    x_sample, shap_values, mean_abs = compute_shap(final_model, x, cfg.shap_sample_n)

    # Only real-data runs persist models, and only they need a Cox fitted on
    # the full window. The synthetic pipeline skips it so its runtime and its
    # committed metrics stay exactly as they were.
    final_cox = None
    if fit_final_cox:
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
            "horizons_days": list(cfg.horizons_days),
            "calibration_horizon_days": cfg.calibration_horizon_days,
        },
        "dataset": dataset_block,
        "folds": fold_metrics.drop(columns="fold_label").to_dict(orient="records"),
        "pooled": pooled,
        "ipcw_brier": brier,
        f"calibration_{int(h_cal)}d": cal.to_dict(orient="records"),
        "shap_top": mean_abs.head(12).to_dict(orient="records"),
    }
    return {
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "cal": cal,
        "h_cal": h_cal,
        "final_model": final_model,
        "final_cox": final_cox,
        "x_sample": x_sample,
        "shap_values": shap_values,
        "mean_abs": mean_abs,
    }


def run_pipeline(cfg: PipelineConfig | None = None, write_outputs: bool = True) -> dict:
    cfg = cfg or PipelineConfig()
    figures_dir = cfg.reports_dir / "figures"

    df, latents = generate(cfg.generator)
    x = build_features(df)

    core = _run_core(
        df,
        x,
        cfg,
        date_col="discovery_date",
        dataset_block={
            "n_strategies": len(df),
            "event_rate": float(df["event"].mean()),
            "median_observed_duration_days": float(df["duration_days"].median()),
        },
        latents=latents,
        sharpe_col="val_sharpe",
        cox_drop_columns=COX_REFERENCE_COLUMNS,
        final_recensor_at=pd.Timestamp(cfg.generator.observation_cutoff),
    )
    metrics = core["metrics"]
    metrics["generator"] = asdict(cfg.generator)

    if write_outputs:
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        (cfg.reports_dir / cfg.metrics_name).write_text(json.dumps(metrics, indent=2))

    if write_outputs and cfg.write_figures:
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.data_dir / "strategies.csv", index=False)
        latents.to_csv(cfg.data_dir / "latents.csv", index=False)

        fold_cindex_plot(core["fold_metrics"], figures_dir / "fold_cindex.png")
        calibration_plot(core["cal"], core["h_cal"], figures_dir / "calibration_180d.png")
        km_by_group_plot(
            df["duration_days"].to_numpy(),
            df["event"].to_numpy(),
            df["asset_class"],
            figures_dir / "km_by_asset_class.png",
        )
        write_shap_figures(core["x_sample"], core["shap_values"], core["mean_abs"], figures_dir)

    return metrics
