"""SHAP attribution for the fitted AFT model.

SHAP values live on the model margin, which for survival:aft is log survival
time: positive pushes the prediction toward longer survival. That sign
convention is what the interpretation report relies on.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .aft_model import XGBoostAFT
from .report_plots import (
    SURFACE,
    apply_style,
    shap_bar_plot,
    short_feature_labels,
    wrap_label,
)


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
    n_display: int = 12,
    time_unit: str = "days",
    keep_prefix: Collection[str] = (),
) -> None:
    """Write the mean-attribution bar chart and the per-row beeswarm to the figures folder, with the
    beeswarm jitter seeded so the files are reproducible.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    shap_bar_plot(
        mean_abs, figures_dir / "shap_bar.png", time_unit=time_unit, keep_prefix=keep_prefix
    )

    # The beeswarm spreads overlapping dots with random jitter. Unseeded it was
    # the one figure that changed every run, and because figures are embedded
    # in the report as base64 the committed HTML and PDF changed with it, which
    # reads as the numbers having moved when only pixels had.
    # Rows sit at a fixed pitch whatever their label height, so wrapped labels
    # need the figure to grow by the lines they add or they collide.
    # Shortening is decided over the rows this figure will actually draw, not
    # over every column. A wide one-hot frame almost always holds a collision
    # somewhere (ward=1 against community_area=1), which would veto the strip
    # for the whole figure even when the drawn rows are unambiguous. The
    # undrawn columns keep their full names; nothing reads them.
    shown = min(n_display, len(x_sample.columns))
    drawn = list(mean_abs["feature"].head(shown))
    short = dict(zip(drawn, short_feature_labels(drawn, keep_prefix), strict=True))
    labels = [wrap_label(short.get(c, c)) for c in x_sample.columns]
    extra_lines = sum(wrap_label(short[f]).count(chr(10)) for f in drawn)
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
