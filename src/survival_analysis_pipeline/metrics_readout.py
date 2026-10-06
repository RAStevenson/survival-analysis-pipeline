"""Plain-text readout of a run's results for the console.

The main numbers from metrics.json as labeled lines, each block headed by a short guide to
reading it, printed to the terminal and saved as readout.txt beside the report. The full
explanation lives in the report. Which model scores higher, where a model loses to the
no-skill forecast, how much group membership carries, and the worst calibration decile are
decided by the same functions the report uses, so the two never state different results; only
the wording differs.
"""

from __future__ import annotations

from pathlib import Path

from .report_generator import (
    _display_path,
    _fold_mean_outcome,
    _largest_calibration_gap,
    _loses_to_no_skill,
    _recommended_model,
    _within_group_gloss,
    _within_group_outcome,
)
from .time_units import horizon_label, unit_abbrev

_FOLD_MEAN_RESULTS = {
    "printed_tie": "tie at the printed precision",
    "near_tie": "effectively tied",
    "cox": "Cox higher",
    "boosted": "boosted higher",
}

_WITHIN_GROUP_RESULTS = {
    "mostly": "mostly group membership",
    "little": "little of it group membership",
    "split": "split between group membership and ranking within groups",
}


def readout_lines(metrics: dict, run_dir: Path, report_path: Path | None = None) -> list[str]:
    """The readout as printable lines, ending with where the full report is when one was built."""
    pooled, folds, config = metrics["pooled"], metrics["folds"], metrics["config"]
    time_unit = config.get("time_unit", "days")
    unit_suffix = unit_abbrev(time_unit)
    name = (metrics.get("run") or {}).get("name", run_dir.name)
    aft_fold_mean, cox_fold_mean = pooled["c_xgb_by_fold_mean"], pooled["c_cox_by_fold_mean"]
    outcome = _fold_mean_outcome(aft_fold_mean, cox_fold_mean)

    train_sizes = [fold["n_train"] for fold in folds]
    requested = config["n_folds"]
    fold_count = (
        f"{len(folds)} of {requested} requested (merged where split dates coincide)"
        if len(folds) != requested
        else f"{len(folds)}"
    )
    lines = [
        f"{name}: {_display_path(run_dir)}",
        "Models: boosted (XGBoost) and Cox baseline (Cox proportional hazards)",
        f"Folds: {fold_count}",
        "  each trains on rows before its split date, tests on the next block",
        f"  training rows {min(train_sizes):,} to {max(train_sizes):,}",
        "",
        "Ranking (concordance): higher is better, 0.500 = coin flip",
        f"  fold mean   boosted {aft_fold_mean:.3f}   Cox {cox_fold_mean:.3f}"
        "   (compare models here)",
        f"  pooled      boosted {pooled['c_xgb']:.3f}"
        f"   95% interval {pooled['c_xgb_ci'][0]:.3f} to {pooled['c_xgb_ci'][1]:.3f}",
    ]
    if "c_oracle" in pooled:
        lines.append(
            f"  oracle      {pooled['c_oracle']:.3f}   best possible, from the hidden truth"
        )
    lines.append(f"  result      {_FOLD_MEAN_RESULTS[outcome]}")

    within_group = metrics.get("within_group")
    if within_group:
        group_outcome = _within_group_outcome(within_group["c_group_mean"], pooled["c_xgb"])
        gloss = _within_group_gloss(within_group["c_within"]).replace("the ", "")
        lines += [
            "",
            f"Groups by {within_group['col']}",
            "  does the boosted model do more than tell the groups apart?",
            f"  group average alone   {within_group['c_group_mean']:.3f}",
            f"  inside a group        {within_group['c_within']:.3f}   {gloss}",
            f"  pooled score is       {_WITHIN_GROUP_RESULTS[group_outcome]}",
        ]

    brier = metrics["ipcw_brier"]
    lines += [
        "",
        "Brier score by horizon: lower is better; no-skill = same chance for every row",
        f"  {'horizon':<12}{'boosted':>10}{'Cox':>10}{'no-skill':>10}",
    ]
    any_loss = False
    for horizon_key, scores in brier.items():
        cells = []
        for model in ("xgb", "cox"):
            loses = _loses_to_no_skill(scores[model], scores["km_marginal"])
            any_loss = any_loss or loses
            cells.append(f"{scores[model]:.3f}{'*' if loses else ' '}")
        horizon_text = f"{horizon_key.removesuffix(unit_suffix)} {time_unit}"
        lines.append(
            f"  {horizon_text:<12}{cells[0]:>10}{cells[1]:>10}{scores['km_marginal']:>9.3f}"
        )
    if any_loss:
        lines.append("  * no better than no-skill at that horizon")

    calibration_horizon = horizon_label(config["calibration_horizon_days"])
    aft_bins = metrics[f"calibration_{calibration_horizon}{unit_suffix}"]
    cox_bins = metrics.get(f"calibration_cox_{calibration_horizon}{unit_suffix}")
    aft_gap, aft_decile = _largest_calibration_gap(aft_bins)
    calibration_line = f"  boosted {aft_gap:.3f} (decile {aft_decile})"
    if cox_bins:
        cox_gap, cox_decile = _largest_calibration_gap(cox_bins)
        calibration_line += f"   Cox {cox_gap:.3f} (decile {cox_decile})"
    lines += [
        "",
        f"Calibration at {calibration_horizon} {time_unit}: largest predicted vs observed gap"
        " in any decile",
        "  deciles of predicted survival, 1 = lowest; lower gap is better",
        calibration_line,
        "",
    ]

    margin = ", near-tie margin" if outcome.endswith("tie") else ""
    lines.append(
        f"Recommended: {_recommended_model(aft_fold_mean, cox_fold_mean)}{margin}"
        " (run_predict.py default)"
    )
    if report_path is not None:
        lines.append(f"Full report: {_display_path(report_path)}")
    return lines


def save_readout(metrics: dict, run_dir: Path, report_path: Path | None = None) -> list[str]:
    """Write the readout to readout.txt in the run folder and return its lines for printing."""
    lines = readout_lines(metrics, run_dir, report_path)
    # Fixed line endings, so a committed readout reads the same from any platform.
    text = "\n".join(lines) + "\n"
    (run_dir / "readout.txt").write_text(text, encoding="utf-8", newline="\n")
    return lines
