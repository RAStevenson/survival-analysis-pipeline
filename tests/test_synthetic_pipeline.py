"""The synthetic study on the unified path: generate a CSV, run it through
the same public door a user's file goes through, then merge the ground-truth
extras. These are the checks that only ground truth makes possible."""

from __future__ import annotations

import json

import pytest

from survival_analysis_pipeline.fit_evaluate import fit_evaluate
from survival_analysis_pipeline.synthetic_extras import (
    DATE_COL,
    DURATION_COL,
    EVENT_COL,
    ID_COL,
    KM_COL,
    add_synthetic_extras,
)
from survival_analysis_pipeline.synthetic_generator import GeneratorConfig, generate


@pytest.fixture(scope="module")
def mini_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    cfg = GeneratorConfig(n_strategies=800, seed=11)
    df, latents = generate(cfg)
    data_path, latents_path = root / "strategies.csv", root / "latents.csv"
    df.to_csv(data_path, index=False)
    latents.to_csv(latents_path, index=False)

    out = root / "run"
    fit_evaluate(
        data_path,
        name="synthetic",
        id_col=ID_COL,
        date_col=DATE_COL,
        duration_col=DURATION_COL,
        event_col=EVENT_COL,
        km_col=KM_COL,
        n_folds=3,
        out_dir=out,
        n_bootstrap=50,
    )
    return add_synthetic_extras(out, data_path, latents_path, cfg), out


def test_run_directory_is_a_normal_run(mini_run):
    _, out = mini_run
    assert (out / "metrics.json").exists()
    for name in (
        "fold_cindex.png",
        "calibration_180d.png",
        "km_by_group.png",
        "shap_bar.png",
        "shap_beeswarm.png",
    ):
        assert (out / "figures" / name).exists(), name
    # The validation run must produce the same artifact a user run produces,
    # or it is not exercising the product.
    assert (out / "model" / "booster.json").exists()
    assert (out / "model" / "sidecar.json").exists()


def test_extras_are_written_to_the_metrics_file(mini_run):
    metrics, out = mini_run
    on_disk = json.loads((out / "metrics.json").read_text())
    assert on_disk["generator"]["seed"] == 11
    assert on_disk["pooled"]["c_oracle"] == metrics["pooled"]["c_oracle"]
    assert all("c_oracle" in f for f in on_disk["folds"])


def test_metrics_json_round_trips(mini_run):
    metrics, _ = mini_run
    assert len(metrics["folds"]) == 3
    assert metrics["pooled"]["n_test"] == sum(f["n_test"] for f in metrics["folds"])


def test_model_has_signal_and_orders_baselines(mini_run):
    metrics, _ = mini_run
    pooled = metrics["pooled"]
    assert pooled["c_xgb"] > 0.55
    assert pooled["c_oracle"] > pooled["c_xgb"]


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


def test_extras_refuse_a_frame_that_does_not_match_the_run(mini_run, tmp_path):
    """The oracle figures are joined onto a reloaded frame by
    position, so a reload that does not reproduce the scored frame row for row
    would silently misattribute every latent. The fold-size check is what makes
    that failure loud."""
    import pandas as pd

    _, out = mini_run
    metrics = json.loads((out / "metrics.json").read_text())
    source = metrics["run"]["source"]
    short = pd.read_csv(source).head(400)
    short_path = tmp_path / "short.csv"
    short.to_csv(short_path, index=False)

    with pytest.raises(AssertionError, match="does not match the one the run scored"):
        add_synthetic_extras(out, short_path, tmp_path / "latents.csv", GeneratorConfig(seed=11))
