"""SHAP attribution for the fitted AFT model.

SHAP values live on the model margin, which for survival:aft is log survival
time: positive pushes the prediction toward longer survival. That sign
convention is what the interpretation report relies on.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .aft_model import XGBoostAFT
from .report_plots import SURFACE, apply_style, shap_bar_plot, shap_dependence_grid, wrap_label


def compute_shap(
    model: XGBoostAFT, x: pd.DataFrame, sample_n: int = 2000, seed: int = 0
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Returns (sampled rows, shap values aligned to them, mean |SHAP| table
    sorted descending over the sample)."""
    if model.booster is None:
        raise RuntimeError("model not fitted")
    if len(x) > sample_n:
        x = x.sample(sample_n, random_state=seed)
    explainer = shap.TreeExplainer(model.booster)
    values = explainer.shap_values(x)
    mean_abs = pd.DataFrame(
        {"feature": x.columns, "mean_abs_shap": np.abs(values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    return x, values, mean_abs


def write_shap_figures(
    x_sample: pd.DataFrame,
    shap_values: np.ndarray,
    mean_abs: pd.DataFrame,
    figures_dir: Path,
    n_dependence: int = 4,
    time_unit: str = "days",
    numeric_only_dependence: bool = False,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    # The beeswarm spreads overlapping dots with random jitter. Unseeded it was
    # the one figure that changed every run, and because figures are embedded
    # in the report as base64 the committed HTML and PDF changed with it, which
    # reads as the numbers having moved when only pixels had.
    # Rows sit at a fixed pitch whatever their label height, so wrapped labels
    # need the figure to grow by the lines they add or they collide.
    labels = [wrap_label(c) for c in x_sample.columns]
    shown = min(12, len(labels))
    extra_lines = sum(label.count(chr(10)) for label in labels[:shown])
    shap.summary_plot(
        shap_values,
        x_sample,
        feature_names=labels,
        max_display=shown,
        show=False,
        plot_size=(9.0, 6.0 + 0.22 * extra_lines),
        rng=np.random.default_rng(0),
    )
    plt.gca().tick_params(axis="y", labelsize=10)
    fig = plt.gcf()
    fig.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(figures_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    shap_bar_plot(mean_abs, figures_dir / "shap_bar.png", time_unit=time_unit)
    # A 0/1 dummy has no shape for a dependence panel to show, so real-data
    # runs restrict the grid to numeric features (no "col=level" name). A run
    # short on numeric features falls back to the strongest flags so the
    # figure the report cites always exists.
    pool = mean_abs
    if numeric_only_dependence:
        numeric = mean_abs[~mean_abs["feature"].str.contains("=", regex=False)]
        if len(numeric) >= 2:
            pool = numeric
    top = pool["feature"].head(n_dependence).tolist()
    shap_dependence_grid(
        x_sample, shap_values, top, figures_dir / "shap_dependence.png", time_unit=time_unit
    )
