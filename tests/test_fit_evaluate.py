from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from survival_analysis_pipeline.aft_model import XGBoostAFT
from survival_analysis_pipeline.duration_csv import (
    EncodingRecipe,
    load_model_bundle,
    save_model_bundle,
)
from survival_analysis_pipeline.fit_evaluate import fit_evaluate, predict

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def exported_csv(small_csv):
    return small_csv


@pytest.fixture(scope="module")
def demo_run(exported_csv, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("run") / "synthetic-export"
    metrics = fit_evaluate(
        exported_csv,
        name="synthetic-export",
        id_col="strategy_id",
        date_col="discovery_date",
        duration_col="duration_days",
        event_col="event",
        n_folds=3,
        out_dir=run_dir,
        n_bootstrap=50,
    )
    return metrics, run_dir


def test_no_oracle_keys_without_latents(demo_run):
    metrics, _ = demo_run
    assert "c_oracle" not in metrics["pooled"]
    for fold in metrics["folds"]:
        assert "c_oracle" not in fold


def test_signal_recovered_from_exported_csv(demo_run):
    # The generated file carries signal the oracle check proves is there, and
    # this is the door that has to find it: nothing about the file tells the
    # loader it was generated.
    metrics, _ = demo_run
    assert metrics["pooled"]["c_xgb"] > 0.6


def test_run_directory_contents(demo_run):
    metrics, run_dir = demo_run
    assert json.loads((run_dir / "metrics.json").read_text()) == json.loads(json.dumps(metrics))
    for name in ("fold_cindex.png", "calibration_180d.png", "shap_bar.png", "cox_hr.png"):
        assert (run_dir / "figures" / name).exists(), name
    assert (run_dir / "model" / "booster.json").exists()
    assert (run_dir / "model" / "sidecar.json").exists()


def test_both_models_saved_and_winner_recorded(demo_run):
    """Saving only the boosted model would misrepresent any run the Cox
    baseline won, which on real data it can."""
    metrics, run_dir = demo_run
    assert (run_dir / "model" / "cox.pkl").exists()
    sidecar = json.loads((run_dir / "model" / "sidecar.json").read_text())
    assert set(sidecar["models"]) == {"aft", "cox"}
    scores = {
        model_type: record["c_index_fold_mean"] for model_type, record in sidecar["models"].items()
    }
    assert scores["aft"] == pytest.approx(metrics["pooled"]["c_xgb_by_fold_mean"])
    assert scores["cox"] == pytest.approx(metrics["pooled"]["c_cox_by_fold_mean"])
    assert sidecar["recommended"] == max(scores, key=lambda model_type: scores[model_type])


def test_cox_gets_the_same_dissection_as_the_boosted_model(demo_run):
    """The report recommends whichever model wins, so the metrics must carry
    the winner's drivers and calibration either way: cox_top ranked by |z|,
    and Cox calibration bins at the same horizon as the boosted model's."""
    metrics, _ = demo_run
    top = metrics["cox_top"]
    assert 0 < len(top) <= 12
    abs_z_scores = [abs(coefficient["z"]) for coefficient in top]
    assert abs_z_scores == sorted(abs_z_scores, reverse=True)
    for coefficient in top:
        assert coefficient["hr_lo"] <= coefficient["hr"] <= coefficient["hr_hi"]

    cal_cox = metrics["calibration_cox_180d"]
    assert len(cal_cox) == len(metrics["calibration_180d"])
    for calibration_bin in cal_cox:
        assert 0.0 <= calibration_bin["predicted"] <= 1.0
        assert 0.0 <= calibration_bin["observed_km"] <= 1.0


def test_predict_with_cox_model(demo_run, exported_csv, tmp_path):
    _, run_dir = demo_run
    new_rows = pd.read_csv(exported_csv).tail(20).drop(columns=["duration_days", "event"])
    path = tmp_path / "new_rows_cox.csv"
    new_rows.to_csv(path, index=False)

    predictions = predict(run_dir, path, horizons=(90.0, 180.0), model_type="cox")
    assert (predictions["model"] == "cox").all()
    survive_90, survive_180 = (
        predictions["p_survive_90d"].to_numpy(),
        predictions["p_survive_180d"].to_numpy(),
    )
    assert (survive_90 >= survive_180).all()
    assert ((survive_90 >= 0) & (survive_90 <= 1)).all()


def test_predict_rejects_unknown_model_type(demo_run, exported_csv, tmp_path):
    _, run_dir = demo_run
    path = tmp_path / "rows.csv"
    pd.read_csv(exported_csv).tail(5).to_csv(path, index=False)
    with pytest.raises(ValueError, match="not in this bundle"):
        predict(run_dir, path, model_type="randomforest")


def test_run_block_records_provenance(demo_run):
    metrics, _ = demo_run
    run = metrics["run"]
    assert run["name"] == "synthetic-export"
    assert run["columns"]["duration"] == "duration_days"
    assert run["calibration_horizon_days"] == 180.0
    assert run["time_unit"] == "days"
    assert metrics["config"]["time_unit"] == "days"


def test_sidecar_records_time_unit(demo_run):
    _metrics, run_dir = demo_run
    sidecar = json.loads((run_dir / "model" / "sidecar.json").read_text())
    assert sidecar["time_unit"] == "days"


def test_fit_evaluate_rejects_unknown_time_unit(exported_csv, tmp_path):
    # Checked before the CSV is even read, so a typo fails in milliseconds
    # rather than after a full load.
    with pytest.raises(ValueError, match="unknown time unit"):
        fit_evaluate(
            exported_csv,
            name="bad-unit",
            id_col="strategy_id",
            date_col="discovery_date",
            duration_col="duration_days",
            event_col="event",
            out_dir=tmp_path / "bad-unit",
            time_unit="fortnights",
        )


def _rescaled_csv(exported_csv, tmp_path, factor, name):
    strategies = pd.read_csv(exported_csv)
    strategies["duration_days"] = strategies["duration_days"] * factor
    path = tmp_path / name
    strategies.to_csv(path, index=False)
    return path


def test_refusal_when_median_duration_is_below_one_timestep(exported_csv, tmp_path):
    """Day-scale lifetimes restated in years put the median near 0.25, where
    the one-timestep re-censoring floor fabricates most training labels;
    measured with the unit correctly declared, c_xgb fell from 0.749 to
    0.589 before this guard existed."""
    path = _rescaled_csv(exported_csv, tmp_path, 1 / 365.25, "years.csv")
    with pytest.raises(ValueError, match="finer --time-unit"):
        fit_evaluate(
            path,
            name="too-coarse",
            id_col="strategy_id",
            date_col="discovery_date",
            duration_col="duration_days",
            event_col="event",
            out_dir=tmp_path / "too-coarse",
            time_unit="years",
        )


def test_predict_columns_carry_the_bundle_time_unit(small_data, small_features, tmp_path):
    """An hours-trained bundle must label predictions in hours. The bundle is
    assembled directly rather than through a full fit, since only the sidecar
    field and the naming are under test."""
    strategies, _ = small_data
    features = small_features
    model = XGBoostAFT().fit(
        features, strategies["duration_days"].to_numpy(), strategies["event"].to_numpy()
    )
    model.predictive_sigma = 0.7
    recipe = EncodingRecipe(
        numeric_columns=tuple(features.columns),
        categorical_levels={},
        dropped_columns=(),
        feature_names=tuple(features.columns),
    )
    save_model_bundle(
        tmp_path / "model",
        model,
        recipe,
        metadata={"id_col": "strategy_id", "time_unit": "hours"},
    )

    rows = features.tail(10).copy()
    rows.insert(0, "strategy_id", strategies["strategy_id"].tail(10).to_numpy())
    path = tmp_path / "rows.csv"
    rows.to_csv(path, index=False)

    predictions = predict(tmp_path, path, horizons=(24.0, 48.0))
    assert "predicted_median_hours" in predictions.columns
    assert {"p_survive_24h", "p_survive_48h"} <= set(predictions.columns)
    assert (predictions["p_survive_24h"] >= predictions["p_survive_48h"]).all()


def test_refusal_below_minimums(exported_csv, tmp_path):
    tiny = pd.read_csv(exported_csv).head(50)
    path = tmp_path / "tiny.csv"
    tiny.to_csv(path, index=False)
    with pytest.raises(ValueError, match="refusing to fit"):
        fit_evaluate(
            path,
            name="tiny",
            id_col="strategy_id",
            date_col="discovery_date",
            duration_col="duration_days",
            event_col="event",
            out_dir=tmp_path / "tiny-run",
        )


def test_model_bundle_round_trip(small_data, small_features, tmp_path):
    strategies, _ = small_data
    features = small_features
    duration = strategies["duration_days"].to_numpy()
    event = strategies["event"].to_numpy()
    model = XGBoostAFT().fit(features, duration, event)
    model.predictive_sigma = 0.7

    recipe = EncodingRecipe(
        numeric_columns=tuple(features.columns),
        categorical_levels={},
        dropped_columns=(),
        feature_names=tuple(features.columns),
    )
    save_model_bundle(tmp_path / "model", model, recipe, metadata={"id_col": "strategy_id"})
    loaded, loaded_recipe, sidecar = load_model_bundle(tmp_path / "model")

    held = features.tail(100)
    np.testing.assert_array_equal(model.predict_median_time(held), loaded.predict_median_time(held))
    horizons = np.array([90.0, 180.0])
    np.testing.assert_array_equal(
        model.predict_survival(held, horizons), loaded.predict_survival(held, horizons)
    )
    assert loaded.predictive_sigma == 0.7
    assert loaded_recipe == recipe
    assert sidecar["id_col"] == "strategy_id"


def test_predict_on_matching_csv(demo_run, exported_csv, tmp_path):
    _, run_dir = demo_run
    new_rows = pd.read_csv(exported_csv).tail(20).drop(columns=["duration_days", "event"])
    path = tmp_path / "new_rows.csv"
    new_rows.to_csv(path, index=False)

    predictions = predict(run_dir, path, horizons=(90.0, 180.0, 365.0))
    assert len(predictions) == 20
    assert (predictions["predicted_median_days"] > 0).all()
    # Survival probabilities must fall as the horizon grows.
    survive_90, survive_180, survive_365 = (
        predictions[f"p_survive_{horizon}d"].to_numpy() for horizon in (90, 180, 365)
    )
    assert (survive_90 >= survive_180).all() and (survive_180 >= survive_365).all()
    assert ((survive_90 >= 0) & (survive_90 <= 1)).all()


def test_predict_defaults_to_the_model_the_run_recommended(demo_run, exported_csv, tmp_path):
    """The bundle records which of the two models scored higher out of time and
    predict() is supposed to follow it. Only the explicit --model-type path was
    covered, so the default could have silently always used the boosted model,
    which is the bug the two-model saving was added to prevent."""
    import json

    _, run_dir = demo_run
    sidecar = json.loads((run_dir / "model" / "sidecar.json").read_text())
    new_rows = pd.read_csv(exported_csv).tail(5).drop(columns=["duration_days", "event"])
    path = tmp_path / "new_rows.csv"
    new_rows.to_csv(path, index=False)

    predictions = predict(run_dir, path)

    assert sidecar["recommended"] in {"aft", "cox"}
    assert (predictions["model"] == sidecar["recommended"]).all()


def test_predict_column_mismatch_names_the_missing_column(demo_run, exported_csv, tmp_path):
    _, run_dir = demo_run
    broken = pd.read_csv(exported_csv).tail(5).drop(columns=["val_sharpe", "duration_days"])
    path = tmp_path / "broken.csv"
    broken.to_csv(path, index=False)
    with pytest.raises(ValueError, match="'val_sharpe'"):
        predict(run_dir, path)


def _run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args], capture_output=True, text=True
    )


def test_fit_script_refusal_shows_traceback_then_plain_line(exported_csv, tmp_path):
    tiny = pd.read_csv(exported_csv).head(50)
    path = tmp_path / "tiny.csv"
    tiny.to_csv(path, index=False)
    completed_run = _run_script(
        "run_fit_evaluate.py",
        "--data", str(path),
        "--name", "tiny",
        "--id-col", "strategy_id",
        "--date-col", "discovery_date",
        "--duration-col", "duration_days",
        "--event-col", "event",
        "--out", str(tmp_path / "tiny-run"),
        "--no-report",
    )  # fmt: skip
    assert completed_run.returncode == 2
    assert "Traceback" in completed_run.stderr
    assert "ValueError: refusing to fit" in completed_run.stderr
    last = completed_run.stderr.strip().splitlines()[-1]
    assert "rerun this command" in last


def test_predict_script_refusal_shows_traceback_then_plain_line(demo_run, exported_csv, tmp_path):
    _, run_dir = demo_run
    path = tmp_path / "no_id.csv"
    pd.read_csv(exported_csv).head(5).drop(columns=["strategy_id"]).to_csv(path, index=False)
    completed_run = _run_script("run_predict.py", "--model", str(run_dir), "--data", str(path))
    assert completed_run.returncode == 2
    assert "Traceback" in completed_run.stderr
    assert "not found in the input" in completed_run.stderr
    last = completed_run.stderr.strip().splitlines()[-1]
    assert last.startswith("No predictions were written")
    assert not path.with_name("no_id_predictions.csv").exists()
