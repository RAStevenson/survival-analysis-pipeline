"""Fit-evaluate and predict on user-supplied duration CSVs.

This is the real-data orchestration behind scripts/run_fit_evaluate.py and
scripts/run_predict.py. It reuses the exact evaluation core the synthetic
pipeline runs -- same temporal folds, same label re-censoring, same
likelihood-based selection, same calibrated predictive scale -- with the
synthetic-only pieces (oracle ceiling, validation-Sharpe anti-baseline)
absent, because real data has no latent truth table and no guaranteed
selection metric. Outputs land in a run directory: metrics.json, figures,
and a saved model bundle that run_predict.py can score new rows with later.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import within_group_concordance
from .io import (
    DURATION,
    EVENT,
    START,
    check_minimum_data,
    encode_with_recipe,
    load_cox_from_bundle,
    load_duration_csv,
    load_model_bundle,
    save_model_bundle,
)
from .pipeline import PipelineConfig, _run_core
from .plots import calibration_plot, fold_cindex_plot, km_by_group_plot
from .shap_analysis import write_shap_figures


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
    horizons_days: tuple[float, ...] = (90.0, 180.0, 365.0),
    min_train_frac: float = 0.4,
    out_dir: str | Path | None = None,
    n_bootstrap: int = 500,
    km_col: str | None = None,
) -> dict:
    """Full evaluation of a duration CSV; writes a run directory and returns
    the metrics dict. Raises ValueError on a malformed file or one too small
    to support the requested folds. `km_col` names a categorical column to
    draw a Kaplan-Meier by-group figure from; the report cites it in its Data
    section when present."""
    data = load_duration_csv(
        data_path, id_col, date_col, duration_col, event_col, drop_cols, categorical_cols
    )
    frame, x = data.frame, data.features

    if km_col is not None and km_col not in frame.columns:
        raise ValueError(
            f"--km-col {km_col!r} is not a feature column of this dataset; "
            f"declared categorical columns: {', '.join(categorical_cols) or '(none)'}"
        )

    refusal = check_minimum_data(len(frame), int(frame[EVENT].sum()), n_folds)
    if refusal is not None:
        raise ValueError(refusal)

    horizons = tuple(float(h) for h in horizons_days)
    cfg = PipelineConfig(
        n_folds=n_folds,
        min_train_frac=min_train_frac,
        horizons_days=horizons,
        # The calibration deep-dive happens at the middle requested horizon,
        # not a hardcoded 180 days: real datasets live on their own timescale.
        calibration_horizon_days=horizons[len(horizons) // 2],
        n_bootstrap=n_bootstrap,
    )
    core = _run_core(
        frame,
        x,
        cfg,
        date_col=START,
        dataset_block={
            "n_rows": len(frame),
            "event_rate": float(frame[EVENT].mean()),
            "median_observed_duration_days": float(frame[DURATION].median()),
            "date_min": str(frame[START].min().date()),
            "date_max": str(frame[START].max().date()),
            "n_features": x.shape[1],
        },
        fit_final_cox=True,
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
        "horizons_days": list(horizons),
        "calibration_horizon_days": cfg.calibration_horizon_days,
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
            core["oof_pred_days"],
            frame[km_col].iloc[test_idx],
        )
        if decomposition is not None:
            metrics["within_group"] = {"col": km_col, **decomposition}

    figures = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    fold_cindex_plot(core["fold_metrics"], figures / "fold_cindex.png")
    h_cal = core["h_cal"]
    calibration_plot(core["cal"], h_cal, figures / f"calibration_{int(h_cal)}d.png")
    write_shap_figures(core["x_sample"], core["shap_values"], core["mean_abs"], figures)
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
            # The synthetic defaults assume trading-strategy timescales and
            # vocabulary; real datasets set the axis from their own tail and
            # take dataset-neutral labels.
            max_days=float(np.quantile(frame[DURATION].to_numpy(dtype=float), 0.95)),
            xlabel="days since start",
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
    horizons_days: tuple[float, ...] = (90.0, 180.0, 365.0),
    model_type: str | None = None,
) -> pd.DataFrame:
    """Score new rows with a saved model bundle. The CSV must carry the id
    column and every feature column the model was trained on; outcome columns
    are not needed and are ignored if present.

    `model_type` picks 'aft' or 'cox'; the default follows the bundle's
    recommendation, which is whichever scored higher on out-of-time fold-mean
    concordance during the run that produced it.
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
    scored_note = f" (out-of-time C-index {score:.3f})" if score else ""
    print(f"scoring with the {chosen} model{scored_note}")

    raw = pd.read_csv(data_path)
    id_col = sidecar["id_col"]
    if id_col not in raw.columns:
        raise ValueError(f"id column {id_col!r} (from the saved model) not found in the input")
    x = encode_with_recipe(raw.drop(columns=[id_col]), recipe)

    horizons = np.asarray([float(h) for h in horizons_days])
    median_days = model.predict_median_days(x)
    survival = model.predict_survival(x, horizons)
    if np.isinf(median_days).any():
        n_inf = int(np.isinf(median_days).sum())
        print(
            f"{n_inf} rows have no finite median: their survival curve never reaches 0.5 "
            "inside the observed follow-up, so the data cannot say when half of them fail"
        )
    out = pd.DataFrame({id_col: raw[id_col], "model": chosen, "predicted_median_days": median_days})
    for j, h in enumerate(horizons):
        out[f"p_survive_{int(h)}d"] = survival[:, j]
    return out


def config_summary(cfg: PipelineConfig) -> dict:
    """Small helper for logging: the config as a plain dict without the
    synthetic generator block, which real runs never use."""
    d = dataclasses.asdict(cfg)
    d.pop("generator", None)
    d.pop("data_dir", None)
    d.pop("reports_dir", None)
    return d
