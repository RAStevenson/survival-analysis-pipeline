"""Report figures. Palette and chrome follow the validated reference palette
(categorical slots in fixed order; aqua and yellow sit below 3:1 contrast on
the light surface, so any series using them carries direct value labels).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from .units import unit_abbrev

SERIES = {"blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "green": "#008300"}
# The reference palette's full categorical order. The ordering is the
# colorblind-safety mechanism (adjacent pairs validated), so append-only.
# Magenta, yellow, and aqua sit below 3:1 contrast on the light surface,
# which is why every multi-series line chart carries direct labels.
KM_SERIES = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "text.color": INK,
            "axes.labelcolor": SECONDARY,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": SECONDARY,
            "ytick.labelcolor": SECONDARY,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fold_cindex_plot(fold_metrics: pd.DataFrame, path: Path) -> None:
    """Grouped bars of test C-index per temporal fold. Expects columns
    fold_label, c_xgb, c_cox; plots c_sharpe bars and the dashed c_oracle
    ceiling only when those columns exist (real-data runs have neither)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    n = len(fold_metrics)
    xpos = np.arange(n)
    width = 0.26
    series = [
        ("c_xgb", "XGBoost AFT", SERIES["blue"]),
        ("c_cox", "Cox PH", SERIES["aqua"]),
    ]
    if "c_sharpe" in fold_metrics.columns:
        series.append(("c_sharpe", "Rank by val. Sharpe", SERIES["yellow"]))
    # Bars start at a data-driven floor, not zero: differences of a few
    # hundredths are the story, and a clipped bar would silently vanish.
    floor = min(0.4, float(fold_metrics[[c for c, _, _ in series]].min().min()) - 0.03)
    floor = np.floor(floor * 20) / 20
    for i, (col, label, color) in enumerate(series):
        vals = fold_metrics[col].to_numpy()
        bars = ax.bar(
            xpos + (i - 1) * width,
            vals - floor,
            width * 0.92,
            bottom=floor,
            color=color,
            label=label,
            zorder=3,
        )
        for rect, v in zip(bars, vals, strict=True):
            ax.annotate(
                f"{v:.2f}",
                (rect.get_x() + rect.get_width() / 2, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=SECONDARY,
            )
    if "c_oracle" in fold_metrics.columns:
        for i, v in enumerate(fold_metrics["c_oracle"].to_numpy()):
            ax.hlines(
                v,
                xpos[i] - 1.7 * width,
                xpos[i] + 1.7 * width,
                color=INK,
                linestyle=(0, (4, 3)),
                linewidth=1.4,
                zorder=4,
                label="Oracle (latent) ceiling" if i == 0 else None,
            )
        top = fold_metrics["c_oracle"].max()
    else:
        top = max(fold_metrics[c].max() for c, _, _ in series)
    ax.axhline(0.5, color=MUTED, linewidth=1.0, zorder=2)
    ax.annotate("random = 0.50", (n - 0.55, 0.501), fontsize=8, color=MUTED, va="bottom")
    ax.set_xticks(xpos, fold_metrics["fold_label"])
    ax.set_ylim(floor, max(0.85, top + 0.06))
    ax.set_ylabel("Harrell C-index (test fold)")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=2, fontsize=9)
    _save(fig, path)


def calibration_plot(
    bins_df: pd.DataFrame, horizon: float, path: Path, time_unit: str = "days"
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lo = min(bins_df["predicted"].min(), bins_df["observed_km"].min()) - 0.05
    hi = max(bins_df["predicted"].max(), bins_df["observed_km"].max()) + 0.05
    lo, hi = max(0.0, lo), min(1.0, hi)
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=MUTED,
        linestyle=(0, (4, 3)),
        linewidth=1.2,
        label="perfect calibration",
        zorder=2,
    )
    ax.plot(
        bins_df["predicted"],
        bins_df["observed_km"],
        color=SERIES["blue"],
        linewidth=2.0,
        marker="o",
        markersize=7,
        zorder=3,
        label="model, by decile",
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ua = unit_abbrev(time_unit)
    ax.set_xlabel(f"predicted P(survive > {horizon:.0f}{ua})")
    ax.set_ylabel(f"observed (Kaplan-Meier) at {horizon:.0f}{ua}")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, path)


def km_by_group_plot(
    duration: np.ndarray,
    event: np.ndarray,
    group: pd.Series,
    path: Path,
    max_time: float = 720.0,
    xlabel: str = "days since deployment",
    ylabel: str = "fraction surviving",
) -> None:
    """Kaplan-Meier curves per group with direct labels at the line ends, so
    identity never rides on color alone. "Surviving" rather than "profitable"
    on the default y-axis: death in the synthetic report is a retention-rule
    event, not the end of profitability, and the label must not conflate the
    two."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    colors = KM_SERIES
    # Sorted levels keep each entity's color stable across regenerated data.
    levels = sorted(pd.Series(group).unique())
    for i, level in enumerate(levels):
        mask = (group == level).to_numpy()
        kmf = KaplanMeierFitter()
        kmf.fit(duration[mask], event_observed=event[mask])
        grid = np.linspace(0, max_time, 240)
        surv = kmf.predict(grid).to_numpy()
        color = colors[i % len(colors)]
        ax.plot(grid, surv, color=color, linewidth=2.0, zorder=3, label=str(level))
        # Curves converge near zero, so direct labels sit at staggered interior
        # x positions instead of the line ends. Spacing by group count keeps
        # every label on-axis however many groups there are.
        x_lab = max_time * (i + 1) / (len(levels) + 1)
        y_lab = float(kmf.predict(x_lab)) + 0.035
        ax.annotate(str(level), (x_lab, y_lab), fontsize=8.5, color=color, ha="center", va="bottom")
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def shap_bar_plot(
    mean_abs_shap: pd.DataFrame, path: Path, top_n: int = 12, time_unit: str = "days"
) -> None:
    """Expects columns feature, mean_abs_shap, sorted descending."""
    apply_style()
    top = mean_abs_shap.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 0.38 * len(top) + 1.2))
    bars = ax.barh(
        top["feature"], top["mean_abs_shap"], color=SERIES["blue"], height=0.62, zorder=3
    )
    for rect, v in zip(bars, top["mean_abs_shap"], strict=True):
        ax.annotate(
            f"{v:.3f}",
            (v, rect.get_y() + rect.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=SECONDARY,
        )
    # Two lines: long feature names squeeze the axes, and a one-line label
    # centred on the narrowed axes runs off the raster canvas.
    ax.set_xlabel(f"mean |SHAP|\n(log-{time_unit} of predicted survival)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)
    _save(fig, path)


def shap_dependence_grid(
    x: pd.DataFrame,
    shap_values: np.ndarray,
    features: list[str],
    path: Path,
    time_unit: str = "days",
) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    for ax, feature in zip(axes.ravel(), features, strict=True):
        col = list(x.columns).index(feature)
        ax.scatter(
            x[feature],
            shap_values[:, col],
            s=9,
            alpha=0.30,
            color=SERIES["blue"],
            edgecolors="none",
            zorder=3,
        )
        ax.axhline(0.0, color=MUTED, linewidth=1.0, zorder=2)
        ax.set_xlabel(feature)
        ax.set_ylabel(f"SHAP (log-{time_unit})")
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
    _save(fig, path)
