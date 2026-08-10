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


def test_flipped_sharpe_complements_and_stays_below_model(mini_run):
    # val_sharpe is continuous, so flipping the ranking flips every comparable
    # pair and the two concordances sum to one up to tie noise. The flipped
    # ranking must also stay below the model, or the report's claim that the
    # reversal does not substitute for the model would be false on this draw.
    metrics, _ = mini_run
    pooled = metrics["pooled"]
    assert abs(pooled["c_sharpe"] + pooled["c_sharpe_flipped"] - 1.0) < 0.01
    assert pooled["c_sharpe_flipped"] < pooled["c_xgb"]


def test_calibration_bins_present(mini_run):
    metrics, _ = mini_run
    total = sum(row["n"] for row in metrics["calibration_180d"])
    assert total == metrics["pooled"]["n_test"]


def test_training_labels_are_recensored_at_each_split(mini_run):
    """The pipeline must train on labels as they stood at the split date, not
    on final ones. This pins the wiring, not the recensor function: deleting
    the recensor call from _evaluate_fold left every other test green, because
    the C-index barely moves and nothing else looked at the training labels.

    The observable signature is the training event rate. Re-censored, an early
    fold has seen only the deaths that had happened by its split date, so its
    rate sits well below the file's final rate. Training on final labels sends
    it to nearly 1.0, since almost every row in an early window has died by the
    end of the file.
    """
    metrics, _ = mini_run
    final_event_rate = metrics["dataset"]["event_rate"]
    rates = [f["train_event_rate"] for f in metrics["folds"]]

    assert rates[0] < final_event_rate - 0.05, (
        f"fold 1 trains at event rate {rates[0]:.3f} against a final rate of "
        f"{final_event_rate:.3f}; that is what training on unre-censored labels looks like"
    )
    # Later splits have watched longer, so more of their window has resolved.
    assert rates == sorted(rates), f"training event rate should rise with the split date: {rates}"
