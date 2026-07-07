"""End-to-end run: generate data, temporal CV against baselines, pooled
metrics with bootstrap intervals, calibration, final fit, SHAP, figures.

Hyperparameters are selected once, on the first fold's training window with an
inner temporal split, and reused for every fold. Selecting per-fold would be
cleaner in principle but makes fold metrics harder to compare and sextuples
the runtime for no visible gain on this data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import CoxBaseline
from .cv import TemporalFold, recensor, temporal_folds
from .evaluate import bootstrap_ci, calibration_bins, harrell_c, ipcw_brier
from .features import build_features
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
    latents: pd.DataFrame,
    horizons: np.ndarray,
) -> dict:
    dates = df["discovery_date"]
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
    cox = CoxBaseline().fit(x_train, train_dur, train_ev)

    pred_days = aft.predict_median_days(x_test)
    surv = aft.predict_survival(x_test, horizons)
    cox_surv = cox.predict_survival(x_test, horizons)

    return {
        "split_date": str(fold.split_date.date()),
        "n_train": len(fold.train_idx),
        "n_test": len(fold.test_idx),
        "train_event_rate": float(np.mean(train_ev)),
        "test_event_rate": float(np.mean(test_ev)),
        "c_xgb": harrell_c(test_dur, test_ev, pred_days),
        "c_cox": harrell_c(test_dur, test_ev, cox.predict_neg_risk(x_test)),
        "c_sharpe": harrell_c(test_dur, test_ev, df["val_sharpe"].to_numpy()[fold.test_idx]),
        "c_oracle": harrell_c(test_dur, test_ev, latents["log_time_eta"].to_numpy()[fold.test_idx]),
        "_test_idx": fold.test_idx,
        "_pred_days": pred_days,
        "_surv": surv,
        "_cox_surv": cox_surv,
    }


def run_pipeline(cfg: PipelineConfig | None = None, write_outputs: bool = True) -> dict:
    cfg = cfg or PipelineConfig()
    horizons = np.asarray(cfg.horizons_days)
    figures_dir = cfg.reports_dir / "figures"

    df, latents = generate(cfg.generator)
    x = build_features(df)

    folds = temporal_folds(df["discovery_date"], cfg.n_folds, cfg.min_train_frac)
    first = folds[0]
    sel_dur, sel_ev = recensor(
        df["duration_days"].to_numpy()[first.train_idx],
        df["event"].to_numpy()[first.train_idx],
        df["discovery_date"].iloc[first.train_idx],
        first.split_date,
    )
    params = _select_params(
        x.iloc[first.train_idx], sel_dur, sel_ev, df["discovery_date"].iloc[first.train_idx]
    )

    fold_results = [_evaluate_fold(f, params, x, df, latents, horizons) for f in folds]

    test_idx = np.concatenate([r["_test_idx"] for r in fold_results])
    pred_days = np.concatenate([r["_pred_days"] for r in fold_results])
    surv = np.vstack([r["_surv"] for r in fold_results])
    cox_surv = np.vstack([r["_cox_surv"] for r in fold_results])
    oof_dur = df["duration_days"].to_numpy()[test_idx]
    oof_ev = df["event"].to_numpy()[test_idx]
    oof_sharpe = df["val_sharpe"].to_numpy()[test_idx]
    oof_eta = latents["log_time_eta"].to_numpy()[test_idx]

    def c_of(scores: np.ndarray, idx: np.ndarray) -> float:
        return harrell_c(oof_dur[idx], oof_ev[idx], scores[idx])

    n = len(test_idx)
    pooled = {
        "n_test": n,
        "event_rate": float(np.mean(oof_ev)),
        "c_xgb": harrell_c(oof_dur, oof_ev, pred_days),
        "c_xgb_ci": bootstrap_ci(lambda i: c_of(pred_days, i), n, cfg.n_bootstrap, seed=1),
        "c_sharpe": harrell_c(oof_dur, oof_ev, oof_sharpe),
        "c_sharpe_ci": bootstrap_ci(lambda i: c_of(oof_sharpe, i), n, cfg.n_bootstrap, seed=2),
        "c_oracle": harrell_c(oof_dur, oof_ev, oof_eta),
        "c_cox_by_fold_mean": float(np.mean([r["c_cox"] for r in fold_results])),
    }

    # Marginal KM survival gives the no-skill Brier reference: same probability
    # for every strategy, censoring handled the same way.
    from lifelines import KaplanMeierFitter

    brier = {}
    for j, h in enumerate(horizons):
        kmf = KaplanMeierFitter().fit(oof_dur, event_observed=oof_ev)
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

    dur_all, ev_all = recensor(
        df["duration_days"].to_numpy(),
        df["event"].to_numpy(),
        df["discovery_date"],
        pd.Timestamp(cfg.generator.observation_cutoff),
    )
    final_model = _fit_aft(params, x, dur_all, ev_all, df["discovery_date"])
    x_sample, shap_values, mean_abs = compute_shap(final_model, x, cfg.shap_sample_n)

    metrics = {
        "params": {
            "max_depth": params.max_depth,
            "aft_sigma": params.aft_sigma,
            "learning_rate": params.learning_rate,
            "predictive_sigma_final": final_model.predictive_sigma,
        },
        "dataset": {
            "n_strategies": len(df),
            "event_rate": float(df["event"].mean()),
            "median_observed_duration_days": float(df["duration_days"].median()),
        },
        "folds": fold_metrics.drop(columns="fold_label").to_dict(orient="records"),
        "pooled": pooled,
        "ipcw_brier": brier,
        "calibration_180d": cal.to_dict(orient="records"),
        "shap_top": mean_abs.head(12).to_dict(orient="records"),
    }

    if write_outputs:
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.data_dir / "strategies.csv", index=False)
        latents.to_csv(cfg.data_dir / "latents.csv", index=False)
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        (cfg.reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        fold_cindex_plot(fold_metrics, figures_dir / "fold_cindex.png")
        calibration_plot(cal, h_cal, figures_dir / "calibration_180d.png")
        km_by_group_plot(
            df["duration_days"].to_numpy(),
            df["event"].to_numpy(),
            df["asset_class"],
            figures_dir / "km_by_asset_class.png",
        )
        write_shap_figures(x_sample, shap_values, mean_abs, figures_dir)

    return metrics
