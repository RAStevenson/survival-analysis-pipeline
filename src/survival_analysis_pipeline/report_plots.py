"""Report figures. Palette and chrome follow the validated reference palette
(categorical slots in fixed order; aqua and yellow sit below 3:1 contrast on
the light surface, so any series using them carries direct value labels).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from .time_units import unit_abbrev

SERIES = {"blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "green": "#008300"}
# The reference palette's full categorical order. The ordering is the
# colorblind-safety mechanism (adjacent pairs validated), so append-only.
# Magenta, yellow, and aqua sit below 3:1 contrast on the light surface, so
# a multi-series line chart identifies its series through a text legend, never
# through color alone.
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


LABEL_WIDTH = 28


def short_feature_labels(features: list[str]) -> list[str]:
    """Feature names for a figure's axis, with the one-hot column prefix
    dropped where doing so stays unambiguous.

    Every row of a categorical-heavy figure otherwise opens with the same
    `column=` before the level that distinguishes it, which costs a wrapped
    line per row and forced Chicago's beeswarm to a full page. The prefix is
    dropped only if the level names it leaves behind are all distinct and none
    collides with a plain numeric feature, so `ward=1` and `community_area=1`
    keep their columns rather than both becoming `1`. The caption carries the
    column names either way.
    """
    short = [f.split("=", 1)[1] if "=" in f else f for f in features]
    if len(set(short)) == len(short):
        return short
    return list(features)


def wrap_label(name: str, width: int = LABEL_WIDTH) -> str:
    """Feature name as a tick or axis label. One-hot names from a text column
    with long levels (license_description=Consumption on Premises - Incidental
    Activity) squeeze a figure's data area into a strip, and nothing stops a
    user's column from doing the same. Names past the width break at the '='
    first, so the column and its level sit on separate lines, then wrap."""
    if len(name) <= width:
        return name
    if "=" in name:
        col, level = name.split("=", 1)
        return f"{col}=\n" + textwrap.fill(level, width)
    return textwrap.fill(name, width)


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fold_cindex_plot(fold_metrics: pd.DataFrame, path: Path) -> None:
    """Grouped bars of test C-index per temporal fold. Expects columns
    fold_label, c_xgb, c_cox; plots the dashed c_oracle ceiling only when
    that column exists (real-data runs have none)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    n = len(fold_metrics)
    xpos = np.arange(n)
    width = 0.26
    series = [
        ("c_xgb", "XGBoost AFT", SERIES["blue"]),
        ("c_cox", "Cox PH", SERIES["aqua"]),
    ]
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
                label="Oracle (latent) ceiling" if i == 0 else "_nolegend_",
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
    bins_df: pd.DataFrame,
    horizon: float,
    path: Path,
    time_unit: str = "days",
    cox_bins_df: pd.DataFrame | None = None,
) -> None:
    """Each model's series uses its own predicted deciles, so the two lines
    share axes but not bin edges. Series are identified by the legend, per
    the palette rule for multi-series line charts."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    frames = [bins_df] + ([cox_bins_df] if cox_bins_df is not None else [])
    lo = min(min(f["predicted"].min(), f["observed_km"].min()) for f in frames) - 0.05
    hi = max(max(f["predicted"].max(), f["observed_km"].max()) for f in frames) + 0.05
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
        label="boosted AFT, by decile",
    )
    if cox_bins_df is not None:
        ax.plot(
            cox_bins_df["predicted"],
            cox_bins_df["observed_km"],
            color=SERIES["aqua"],
            linewidth=2.0,
            marker="s",
            markersize=6,
            zorder=3,
            label="Cox PH, by decile",
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ua = unit_abbrev(time_unit)
    ax.set_xlabel(f"predicted P(survive > {horizon:.0f}{ua})")
    ax.set_ylabel(f"observed (Kaplan-Meier) at {horizon:.0f}{ua}")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, path)


def cox_hr_plot(coefficients: pd.DataFrame, path: Path) -> None:
    """Forest-style hazard ratios with 95% intervals on a log axis. Expects
    columns feature, hr, hr_lo, hr_hi, ordered strongest first; plots them
    top-down in that order. The log axis is what makes a doubling and a
    halving of risk the same visual distance from the no-effect line."""
    apply_style()
    names = short_feature_labels(list(coefficients["feature"]))
    fitted = _rows_that_fit(names, len(coefficients))
    top = coefficients.head(fitted).iloc[::-1]
    labels = [wrap_label(f) for f in names[:fitted][::-1]]
    extra_lines = sum(label.count("\n") for label in labels)
    fig, ax = plt.subplots(figsize=(6.8, 0.38 * len(top) + 0.16 * extra_lines + 1.2))
    ypos = np.arange(len(top))
    ax.hlines(
        ypos,
        top["hr_lo"].to_numpy(),
        top["hr_hi"].to_numpy(),
        color=SERIES["blue"],
        linewidth=1.8,
        zorder=3,
    )
    ax.plot(
        top["hr"].to_numpy(),
        ypos,
        linestyle="none",
        marker="o",
        markersize=6,
        color=SERIES["blue"],
        zorder=4,
    )
    ax.axvline(1.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.set_xscale("log")
    # Log-axis defaults label minor ticks as 7x10^-1; a reader comparing
    # hazard ratios needs plain decimals at round values instead.
    lo = float(top["hr_lo"].min()) * 0.9
    hi = float(top["hr_hi"].max()) * 1.1
    candidates = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0)
    ticks = [t for t in candidates if lo <= t <= hi] or [1.0]
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(ypos, labels)
    ax.set_xlabel("hazard ratio (log scale); above 1 shortens survival")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
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
    """Kaplan-Meier curves per group, identified by the legend. "Surviving"
    rather than "profitable" on the default y-axis: death in the synthetic
    report is a retention-rule event, not the end of profitability, and the
    label must not conflate the two."""
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
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def _rows_that_fit(features: list[str], limit: int, max_inches: float = 6.4) -> int:
    """How many of the strongest features to draw so the figure still fits a
    page beside its caption.

    Row height is fixed but label height is not, so a run whose categories
    carry long names produces a taller figure from the same row count. The
    Chicago demo hit this: twelve one-hot licence names wrapped to 8.2 inches,
    and a figure that tall cannot honour the stylesheet's page-break-inside
    rule, so it split across pages. Counting instead of capping the count
    keeps every short-labelled run at the full `limit`.
    """
    for n in range(min(limit, len(features)), 4, -1):
        extra = sum(wrap_label(f).count(chr(10)) for f in features[:n])
        if 0.38 * n + 0.16 * extra + 1.2 <= max_inches:
            return n
    return min(limit, len(features))


def shap_bar_plot(
    mean_abs_shap: pd.DataFrame, path: Path, top_n: int = 12, time_unit: str = "days"
) -> None:
    """Expects columns feature, mean_abs_shap, sorted descending."""
    apply_style()
    names = short_feature_labels(list(mean_abs_shap["feature"]))
    fitted = _rows_that_fit(names, top_n)
    top = mean_abs_shap.head(fitted).iloc[::-1]
    labels = [wrap_label(f) for f in names[:fitted][::-1]]
    extra_lines = sum(label.count("\n") for label in labels)
    fig, ax = plt.subplots(figsize=(6.8, 0.38 * len(top) + 0.16 * extra_lines + 1.2))
    bars = ax.barh(labels, top["mean_abs_shap"], color=SERIES["blue"], height=0.62, zorder=3)
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
