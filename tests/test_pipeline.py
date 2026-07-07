from __future__ import annotations

import json

import pytest

from strategy_survival.generate import GeneratorConfig
from strategy_survival.pipeline import PipelineConfig, run_pipeline


@pytest.fixture(scope="module")
def mini_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    cfg = PipelineConfig(
        generator=GeneratorConfig(n_strategies=800, seed=11),
        n_folds=3,
        n_bootstrap=50,
        shap_sample_n=300,
        data_dir=root / "data",
        reports_dir=root / "reports",
    )
    return run_pipeline(cfg), cfg


def test_pipeline_outputs_exist(mini_run):
    _, cfg = mini_run
    assert (cfg.data_dir / "strategies.csv").exists()
    assert (cfg.reports_dir / "metrics.json").exists()
    figures = cfg.reports_dir / "figures"
    for name in (
        "fold_cindex.png",
        "calibration_180d.png",
        "km_by_asset_class.png",
        "shap_beeswarm.png",
        "shap_bar.png",
        "shap_dependence.png",
    ):
        assert (figures / name).exists(), name


def test_metrics_json_round_trips(mini_run):
    _, cfg = mini_run
    metrics = json.loads((cfg.reports_dir / "metrics.json").read_text())
    assert len(metrics["folds"]) == 3
    assert metrics["pooled"]["n_test"] == sum(f["n_test"] for f in metrics["folds"])


def test_model_has_signal_and_orders_baselines(mini_run):
    metrics, _ = mini_run
    pooled = metrics["pooled"]
    assert pooled["c_xgb"] > 0.55
    assert pooled["c_oracle"] > pooled["c_xgb"]
    assert pooled["c_xgb"] > pooled["c_sharpe"]


def test_calibration_bins_present(mini_run):
    metrics, _ = mini_run
    total = sum(row["n"] for row in metrics["calibration_180d"])
    assert total == metrics["pooled"]["n_test"]
