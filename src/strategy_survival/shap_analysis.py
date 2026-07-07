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

from .model import XGBoostAFT
from .plots import SURFACE, shap_bar_plot, shap_dependence_grid


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
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(shap_values, x_sample, max_display=12, show=False, plot_size=(9.0, 6.0))
    fig = plt.gcf()
    fig.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(figures_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    shap_bar_plot(mean_abs, figures_dir / "shap_bar.png")
    top = mean_abs["feature"].head(n_dependence).tolist()
    shap_dependence_grid(x_sample, shap_values, top, figures_dir / "shap_dependence.png")
