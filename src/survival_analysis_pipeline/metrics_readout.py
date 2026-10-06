"""Plain-text readout of a run's results for the console.

The main numbers from metrics.json, each block introduced by a few lines saying what the
statistic answers and how to read it. The full explanation lives in the report; this is the
version a user can read in the terminal and copy from. The sentences that judge a result (which
model scores higher, which horizons lose to the no-skill forecast, where within-group ranking
sits) come from report_generator, so the console and the report never state different results.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from .report_generator import (
    _fold_mean_comparison,
    _losing_horizons,
    _within_group_gloss,
    _within_group_lead,
)
from .time_units import horizon_label, unit_abbrev


def _largest_calibration_gap(bins: list[dict]) -> tuple[float, int]:
    """The largest gap between predicted and observed survival across the bins, and its decile."""
    gap, position = max(
        (abs(calibration_bin["predicted"] - calibration_bin["observed_km"]), i)
        for i, calibration_bin in enumerate(bins)
    )
    return gap, position + 1


def _guidance(text: str) -> list[str]:
    """A block's explanation wrapped to the terminal width the readout is laid out for."""
    return textwrap.wrap(text, width=80)


def _result(text: str) -> list[str]:
    """A result sentence, indented under its numbers and wrapped to the same width."""
    return textwrap.wrap(text, width=80, initial_indent="  ", subsequent_indent="  ")


def readout_lines(metrics: dict, run_dir: Path, report_path: Path | None = None) -> list[str]:
    """The readout as printable lines, ending with where the full report is when one was built."""
    pooled, folds, config = metrics["pooled"], metrics["folds"], metrics["config"]
    time_unit = config.get("time_unit", "days")
    unit_suffix = unit_abbrev(time_unit)
    name = (metrics.get("run") or {}).get("name", run_dir.name)
    lines = [f"Results for {name} ({run_dir.as_posix()})"]
    lines += _guidance(
        "The run fits two models, a boosted model (XGBoost) and a Cox proportional hazards"
        " baseline, the standard survival model."
    )
    lines.append("")

    # Folds come first because every score below is defined in terms of them.
    train_sizes = [fold["n_train"] for fold in folds]
    requested = config["n_folds"]
    merged = len(folds) != requested
    lines.append(
        f"Testing: {len(folds)} folds ({requested} requested)."
        if merged
        else f"Testing: {len(folds)} folds."
    )
    lines += _guidance(
        "Each fold trains both models on rows that started before a split date and tests"
        " them on the next block of rows, which started on or after it."
        + (" Folds that shared a split date were merged into one." if merged else "")
        + f" Training sets run from {min(train_sizes):,} to {max(train_sizes):,} rows."
    )

    winner_clause, recommended_model, recommendation_margin = _fold_mean_comparison(
        pooled["c_xgb_by_fold_mean"], pooled["c_cox_by_fold_mean"]
    )
    lines += ["", "Ranking: how often a model puts two rows in the right order (concordance)."]
    lines += _guidance(
        "The right order means the row that ends first is predicted to end first. Higher is"
        " better, and 0.500 is a coin flip. Pooled scores all test rows as one set, while"
        " fold mean scores each fold separately and averages them. Compare the two models"
        " on the fold mean."
    )
    lines += [
        f"  fold mean   boosted model {pooled['c_xgb_by_fold_mean']:.3f}"
        f"   Cox baseline {pooled['c_cox_by_fold_mean']:.3f}",
        f"  pooled      boosted model {pooled['c_xgb']:.3f},"
        f" 95% interval {pooled['c_xgb_ci'][0]:.3f} to {pooled['c_xgb_ci'][1]:.3f}",
    ]
    if "c_oracle" in pooled:
        lines += textwrap.wrap(
            f"oracle      {pooled['c_oracle']:.3f}, the best score any model could reach given"
            " the generator's hidden truth",
            width=80,
            initial_indent="  ",
            subsequent_indent=" " * 14,
        )
    lines += _result(f"{winner_clause}.")

    within_group = metrics.get("within_group")
    if within_group:
        column = within_group["col"]
        lines.append("")
        lines += _guidance(
            f"Groups by {column}: whether the boosted model does more than recognize which"
            " group a row belongs to."
        )
        lines += _guidance(
            "Ranking rows by their group's average prediction uses"
            " only group membership. An inside-group score clear of 0.500 means the model"
            " also orders rows within a group, which a table of group averages cannot do."
        )
        lines += [
            f"  by group average alone   {within_group['c_group_mean']:.3f}",
            f"  inside a group           {within_group['c_within']:.3f}"
            f" ({_within_group_gloss(within_group['c_within'])})",
        ]
        lines += _result(
            _within_group_lead(column, within_group["c_group_mean"], pooled["c_xgb"]) + "."
        )

    brier = metrics["ipcw_brier"]
    lines += ["", "Probabilities: how accurate each model's predicted chances are (Brier score)."]
    lines += _guidance(
        "At each horizon, a model predicts the chance that a row is still running. Lower is"
        " better, and a model above the no-skill forecast adds nothing at that horizon. The"
        " no-skill forecast gives every row the same chance, the share of all test rows"
        " still running at that horizon."
    )
    lines.append(f"  {'horizon':<14}{'boosted':>9}{'Cox':>9}{'no-skill':>10}")
    for horizon_key, scores in brier.items():
        horizon_text = f"{horizon_key.removesuffix(unit_suffix)} {time_unit}"
        lines.append(
            f"  {horizon_text:<14}{scores['xgb']:>9.3f}{scores['cox']:>9.3f}"
            f"{scores['km_marginal']:>10.3f}"
        )
    losing_text = _losing_horizons(brier, unit_suffix, time_unit)
    lines += _result(
        f"{losing_text}."
        if losing_text
        else "Both models beat the no-skill forecast at every horizon."
    )

    calibration_horizon = horizon_label(config["calibration_horizon_days"])
    aft_bins = metrics[f"calibration_{calibration_horizon}{unit_suffix}"]
    cox_bins = metrics.get(f"calibration_cox_{calibration_horizon}{unit_suffix}")
    aft_gap, aft_decile = _largest_calibration_gap(aft_bins)
    calibration_line = f"  boosted model {aft_gap:.3f} (decile {aft_decile})"
    if cox_bins:
        cox_gap, cox_decile = _largest_calibration_gap(cox_bins)
        calibration_line += f"   Cox baseline {cox_gap:.3f} (decile {cox_decile})"
    lines.append("")
    lines += _guidance(
        f"Calibration at {calibration_horizon} {time_unit}: whether predicted chances come true"
        " as often as they say."
    )
    lines += _guidance(
        "Rows are sorted by"
        " predicted chance into ten groups of equal size (deciles), 1 the lowest and 10 the"
        " highest. Each number is the largest gap in any decile between the predicted"
        " chance and the observed share still running. Lower is better."
    )
    lines.append(calibration_line)

    lines += [
        "",
        f"Recommended for scoring new rows: the {recommended_model}{recommendation_margin}.",
        "run_predict.py uses it by default.",
    ]
    if report_path is not None:
        lines.append(f"The report explains each result in full: {report_path.as_posix()}")
    return lines
