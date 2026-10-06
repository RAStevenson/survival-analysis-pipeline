"""Report templates shared by the synthetic study and real-data runs.

THE REPORT CONTRACT. A generated report is instrument output: it states
what was fit, on what data, under what scheme, with what results, in prose
that holds word-for-word on a stranger's dataset. Three kinds of content
are allowed. (1) Template prose: fixed sentences plus injected values, each
technical term defined inline exactly once, in one place in this module.
Template prose includes generic reading guidance, meaning the question a
statistic answers and what a high or low value would mean, because that
guidance holds on any dataset. (2) Presence-keyed measurement blocks,
wrapped in pk-comment markers and rendered only when the metrics carry the
measurement (the generator block, the oracle column, the
within-group decomposition); never a mode flag. A presence-keyed block
states or suppresses a measurement and never comments on one; the
synthetic run's commentary on its generator lives in its notes like any
other run's (ruled 2026-09-02). (3) Notes: authored
markdown per run (report_notes.py), inserted at fixed anchors, carrying
all run-specific interpretation, motivation, and dataset-specific claims,
citing metric values through @val tokens that fail the build when
unresolvable. Anything editorial about a specific run belongs in its
notes, not here. The template-invariance and word-budget tests enforce (1) and
(2); render-time numbering and the cited-or-fail figure rule live in
report_document.ReportDoc.

FINDING YOUR WAY BACK FROM A RENDERED REPORT. Section, figure, and table
NUMBERS exist only in the rendered output; the code carries slugs and
titles, so "Table 3" is not greppable and shifts whenever something above
it is added or cut. Go by the heading text, which each `_sec_*` docstring
names, or by the slug inside a caption's `@fig:`/`@tab:` token. The order
the sections appear in is the call order in `compose_report`.
"""

from __future__ import annotations

from pathlib import Path

from .report_document import (
    ReportDoc as ReportDoc,
)
from .report_document import (
    emit_pdf as emit_pdf,
)
from .report_document import (
    img_uri,
    pct,
)
from .report_notes import load_run_notes
from .time_units import horizon_label, unit_abbrev


def _pk(name: str, html: str) -> str:
    """Wrap a presence-keyed block in markers the invariance test strips by
    whitelist. A block that renders in one variant and not the other without
    these markers is a template divergence and fails that test."""
    return f"<!--pk:{name}-->{html}<!--/pk:{name}-->"


def _display_path(run_dir: Path) -> str:
    """The run directory as a reader could type it.

    A report is a published document, so it must never print the absolute path
    of whichever machine built it. Anything at or under the working directory
    renders relative to it; a run somewhere else keeps the path it was given,
    since there is nothing shorter that would still be true.
    """
    try:
        return run_dir.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return run_dir.as_posix()


def _losing_horizons(brier: dict, unit_suffix: str, time_unit: str) -> str:
    """Which models' probabilities lose to the no-skill forecast, and where.

    Both models are checked. Reporting only the boosted model's losses hid
    that the Cox baseline also loses at Chicago's one-year horizon, which a
    reader caught by comparing the prose against the report's own table.
    """
    aft = [
        horizon_key
        for horizon_key, scores in brier.items()
        if scores["xgb"] >= scores["km_marginal"]
    ]
    cox = [
        horizon_key
        for horizon_key, scores in brier.items()
        if scores["cox"] >= scores["km_marginal"]
    ]
    if not aft and not cox:
        return ""
    if aft == cox:
        return "Both models' probabilities lose to a no-skill forecast at " + _horizon_list(
            aft, unit_suffix, time_unit
        )
    parts = []
    if aft:
        parts.append(
            "The boosted model's probabilities lose to a no-skill forecast at "
            + _horizon_list(aft, unit_suffix, time_unit)
        )
    if cox:
        lead = (
            "the Cox baseline's at"
            if parts
            else "The Cox baseline's probabilities lose to a no-skill forecast at"
        )
        parts.append(f"{lead} {_horizon_list(cox, unit_suffix, time_unit)}")
    return ", and ".join(parts)


def _horizon_list(keys: list[str], unit_suffix: str, time_unit: str) -> str:
    """Horizon keys as prose: '365 days', '365 and 730 days', or
    '365, 730, and 1460 days'. Oxford comma on three or more."""
    horizon_numbers = [key.removesuffix(unit_suffix) for key in keys]
    if len(horizon_numbers) == 1:
        return f"{horizon_numbers[0]} {time_unit}"
    if len(horizon_numbers) == 2:
        return f"{horizon_numbers[0]} and {horizon_numbers[1]} {time_unit}"
    return ", ".join(horizon_numbers[:-1]) + f", and {horizon_numbers[-1]} {time_unit}"


def _fold_mean_comparison(aft_fold_mean: float, cox_fold_mean: float) -> tuple[str, str, str]:
    """The comparison sentence, the recommended model, and the margin note for two fold means.

    Shared with metrics_readout, so the console and the report always state the same result.
    """
    if f"{aft_fold_mean:.3f}" == f"{cox_fold_mean:.3f}":
        winner_clause = "The two models tie at the printed precision"
    elif abs(aft_fold_mean - cox_fold_mean) < 0.0015:
        # A gap the bootstrap interval swallows is not a winner.
        winner_clause = "The two models effectively tie"
    elif cox_fold_mean > aft_fold_mean:
        winner_clause = "The Cox baseline scores higher"
    else:
        winner_clause = "The boosted model scores higher"
    # Mirrors save_model_bundle's tie-break (aft on equality), so the report
    # names the same model the saved sidecar records as recommended. When the
    # winner clause above calls the run a tie, the bundle still records a
    # recommendation, and the sentence must say the margin is thin rather
    # than let the two statements read as a contradiction.
    recommended_model = "Cox baseline" if cox_fold_mean > aft_fold_mean else "boosted model"
    recommendation_margin = "" if "tie" not in winner_clause else ", on a near-tie margin"
    return winner_clause, recommended_model, recommendation_margin


def _within_group_lead(column_label: str, c_group_mean: float, c_pooled: float) -> str:
    """How much of the pooled score group membership carries, as a sentence opening.

    column_label arrives already formatted, as HTML code markup for the report or plain text
    for the console.
    """
    # Whether group membership dominates is a property of the run, so the
    # lead-in is computed, not asserted (same rule as the comparison). The
    # middle branch would misdescribe both ends: a run whose group means
    # sit near a coin flip has essentially no group effect to split with.
    if c_group_mean >= c_pooled:
        return f"Most of the pooled score reflects a row's {column_label} group"
    if c_group_mean - 0.5 < 0.25 * (c_pooled - 0.5):
        return f"Little of the pooled score is {column_label} group membership"
    return f"The pooled score splits between {column_label} group membership and ranking within it"


def _within_group_gloss(c_within: float) -> str:
    """Where the within-group concordance sits relative to a coin flip, in words."""
    # The characterization is computed like the number it describes. The
    # asserted version shipped "0.779, close to the coin flip" in the
    # synthetic report, caught in review on 2026-08-28.
    if abs(c_within - 0.5) < 0.02:
        return "close to the coin flip"
    if c_within > 0.5:
        return "clear of the coin flip"
    return "below the coin flip"


def _km(metrics: dict, run_dir: Path) -> dict | None:
    """The grouping column and figure file for the survival-curve figure, or None when the run drew
    none.
    """
    km_col = (metrics.get("run") or {}).get("km_col")
    if km_col and (run_dir / "figures" / "km_by_group.png").exists():
        return {"column": km_col, "filename": "km_by_group.png"}
    return None


def synthetic_context(metrics: dict, run_dir: Path, notes_dir: Path | None = None) -> dict:
    """The rendering context for the synthetic run: header rows, footer, reproduce command, and
    notes, with the seed as the run's provenance.
    """
    generator, dataset, run = metrics["generator"], metrics["dataset"], metrics["run"]
    relative_run_dir = _display_path(run_dir)

    # Same header shape as real_context, values only. The synthetic run's one
    # extra provenance fact, the seed, rides in the source-data cell.
    censored_overall = 1.0 - dataset["event_rate"]
    meta_rows = f"""    <tr><th>Source data</th><td><code>{run["source"]}</code>, drawn at
      seed {generator["seed"]}</td></tr>
    <tr><th>Rows</th><td>{dataset["n_rows"]:,} ({pct(dataset["event_rate"])} with the ending
      observed, {pct(censored_overall)} censored)</td></tr>
    <tr><th>Start dates</th><td>{dataset["date_min"]} to {dataset["date_max"]}</td></tr>
    <tr><th>Evaluation</th><td>{len(metrics["folds"])} expanding-window temporal folds</td></tr>
    <tr><th>Source of figures</th><td><code>{relative_run_dir}/metrics.json</code>,
      regenerated by the command in section @sec:repro</td></tr>"""

    footer = f"""<p>Generated from <code>{relative_run_dir}/metrics.json</code> by
<code>scripts/run_build_report.py</code>. {dataset["n_rows"]:,} rows drawn at
seed {generator["seed"]}, {metrics["pooled"]["n_test"]:,} out-of-time test rows.</p>"""

    # The synthetic run's input is regenerated from the seed rather than
    # committed, so the reproducing command is the runner that writes the CSV
    # and then calls the same fit_evaluate a user calls directly.
    command = (
        "pip install -r requirements.txt\n"
        "python -m pytest\n"
        "python scripts/run_synthetic_pipeline.py"
    )

    return {
        "metrics": metrics,
        "figures_dir": run_dir / "figures",
        "title": f"Survival Model Evaluation: {run['name']}",
        "subtitle": "Fitted with the survival-analysis-pipeline,\n"
        "  on synthetic data with known ground truth",
        "meta_rows": meta_rows,
        "footer": footer,
        "command": command,
        "source_description": f"synthetic data drawn at seed {generator['seed']}",
        "km": _km(metrics, run_dir),
        "notes": load_run_notes(notes_dir, metrics),
    }


def real_context(metrics: dict, run_dir: Path, notes_dir: Path | None = None) -> dict:
    """The rendering context for a real-data run: header rows, footer, the exact fit command that
    reproduces it, and notes.
    """
    run, dataset, pooled = metrics["run"], metrics["dataset"], metrics["pooled"]

    # One line on purpose. A cmd.exe caret continuation is a parse error in
    # PowerShell and a stray argument in bash, so a wrapped command is a
    # command the reader cannot paste. --out is included because without it
    # the run lands in the default runs/<name>/ and the rebuild step below
    # would re-render the old report instead of the one just produced.
    columns = run["columns"]
    drop_part = f" --drop-cols {','.join(columns['dropped'])}" if columns["dropped"] else ""
    categorical = columns.get("categorical")
    categorical_part = f" --categorical-cols {','.join(categorical)}" if categorical else ""
    km_part = f" --km-col {run['km_col']}" if run.get("km_col") else ""
    out_part = f" --out {run['out_dir']}" if run.get("out_dir") else ""
    # Days is the CLI default, so day-based commands stay exactly as the
    # committed reports print them.
    run_time_unit = run.get("time_unit", "days")
    unit_part = f" --time-unit {run_time_unit}" if run_time_unit != "days" else ""
    command = (
        f"python scripts/run_fit_evaluate.py --data {run['source']} --name {run['name']} "
        f"--id-col {columns['id']} --date-col {columns['date']} "
        f"--duration-col {columns['duration']} --event-col {columns['event']}"
        f"{drop_part}{categorical_part}{km_part}{unit_part} --folds {run['n_folds']} "
        f"--horizons {','.join(horizon_label(horizon) for horizon in run['horizons_days'])}"
        f"{out_part}"
        f"\npython scripts/run_build_report.py --run {_display_path(run_dir)}"
    )

    censored_overall = 1.0 - dataset["event_rate"]
    meta_rows = f"""    <tr><th>Source data</th><td><code>{run["source"]}</code></td></tr>
    <tr><th>Rows</th><td>{dataset["n_rows"]:,} ({pct(dataset["event_rate"])} with the ending
      observed, {pct(censored_overall)} censored)</td></tr>
    <tr><th>Start dates</th><td>{dataset["date_min"]} to {dataset["date_max"]}</td></tr>
    <tr><th>Evaluation</th><td>{len(metrics["folds"])} expanding-window temporal folds</td></tr>
    <tr><th>Source of figures</th><td><code>{_display_path(run_dir)}/metrics.json</code>,
      regenerated by the command in section @sec:repro</td></tr>"""

    footer = f"""<p>Generated from <code>{_display_path(run_dir)}/metrics.json</code> by
<code>scripts/run_build_report.py --run</code>. {dataset["n_rows"]:,} rows,
{pooled["n_test"]:,} out-of-time test rows.</p>"""

    if notes_dir is None:
        notes_dir = run_dir / "notes"

    return {
        "metrics": metrics,
        "figures_dir": run_dir / "figures",
        "title": f"Survival Model Evaluation: {run['name']}",
        "subtitle": "Fitted with the survival-analysis-pipeline.",
        "meta_rows": meta_rows,
        "footer": footer,
        "command": command,
        "source_description": f"<code>{Path(run['source']).name}</code>",
        "km": _km(metrics, run_dir),
        "notes": load_run_notes(notes_dir, metrics),
    }


# --------------------------------------------------------------------------
# Derived rendering state shared by the section builders.


def _derive(run_context: dict) -> dict:
    """Everything the section builders read, computed once from the context: the metrics, presence
    flags, unit labels, the winner clause, and the recommended model.
    """
    metrics = run_context["metrics"]
    dataset, pooled, folds, config = (
        metrics["dataset"],
        metrics["pooled"],
        metrics["folds"],
        metrics["config"],
    )
    generator, run = metrics.get("generator"), metrics.get("run")
    time_unit = config.get("time_unit", "days")
    calibration_horizon = horizon_label(config["calibration_horizon_days"])
    unit_suffix = unit_abbrev(time_unit)
    winner_clause, recommended_model, recommendation_margin = _fold_mean_comparison(
        pooled["c_xgb_by_fold_mean"], pooled["c_cox_by_fold_mean"]
    )
    return {
        "metrics": metrics,
        "notes": run_context.get("notes") or {},
        "figures_dir": run_context["figures_dir"],
        "source_description": run_context["source_description"],
        "command": run_context["command"],
        "km": run_context["km"],
        "generator": generator,
        "run": run,
        "params": metrics["params"],
        "dataset": dataset,
        "pooled": pooled,
        "folds": folds,
        "brier": metrics["ipcw_brier"],
        "config": config,
        "within_group": metrics.get("within_group"),
        "time_unit": time_unit,
        "unit_suffix": unit_suffix,
        "calibration_horizon": calibration_horizon,
        "calibration": metrics[f"calibration_{calibration_horizon}{unit_suffix}"],
        "cox_calibration": metrics.get(f"calibration_cox_{calibration_horizon}{unit_suffix}"),
        "cox_top": metrics.get("cox_top"),
        "has_oracle": "c_oracle" in pooled,
        "n_rows": dataset["n_rows"],
        "date_min": dataset["date_min"],
        "date_max": dataset["date_max"],
        "censored_overall": 1.0 - dataset["event_rate"],
        "max_fold_censored": max(1.0 - fold["test_event_rate"] for fold in folds),
        "c_fold_min": min(fold["c_xgb"] for fold in folds),
        "c_fold_max": max(fold["c_xgb"] for fold in folds),
        "n_train_min": min(fold["n_train"] for fold in folds),
        "n_train_max": max(fold["n_train"] for fold in folds),
        "winner_clause": winner_clause,
        "recommended_model": recommended_model,
        "recommendation_margin": recommendation_margin,
    }


def _sec_summary(derived: dict, doc: ReportDoc) -> None:
    """Emits "Summary". Registers no figures or tables; every number in it is
    repeated with its full treatment later, so nothing here is the only home
    of a fact."""
    pooled, folds = derived["pooled"], derived["folds"]
    source_description = derived["source_description"]
    body = f"""<p>This report evaluates two survival models fitted to {source_description}.
It holds {derived["n_rows"]:,} rows, each observed from its start
date. {pct(derived["dataset"]["event_rate"])} have observed endings and
{pct(derived["censored_overall"])} are censored, still running when observation
stopped. Median observed duration, censored
rows included, is
{derived["dataset"]["median_observed_duration_days"]:.0f} {derived["time_unit"]}, the scale on
which every horizon and prediction below sits. The models predict,
from what was on file at the start date, how long each row
survives.</p>

<p>When evaluating the two models' performance, the apples-to-apples
comparison is the fold-mean concordance, the
share of row pairs a ranking orders correctly
where 0.500 is a coin flip. The pipeline uses walkforward validation. Each of the {len(folds)}
temporal folds trains both models on one stretch of history and tests them
on the next, and the {len(folds)} scores average. On concordance the XGBoost
accelerated-failure-time (AFT) model scores
{pooled["c_xgb_by_fold_mean"]:.3f} and the Cox proportional hazards baseline
{pooled["c_cox_by_fold_mean"]:.3f}. {derived["winner_clause"]}.</p>

<p>To attach an uncertainty range, which says how far the score could
reasonably move, the AFT model is also scored with all
{pooled["n_test"]:,} out-of-time test rows pooled into one list.
That concordance is {pooled["c_xgb"]:.3f}, with a 95% bootstrap interval
of {pooled["c_xgb_ci"][0]:.3f} to {pooled["c_xgb_ci"][1]:.3f}. Section
@sec:results explains the two figures and how to read the
interval.</p>"""

    if derived["within_group"]:
        within_group = derived["within_group"]
        within_group_lead = _within_group_lead(
            f"<code>{within_group['col']}</code>", within_group["c_group_mean"], pooled["c_xgb"]
        )
        within_gloss = _within_group_gloss(within_group["c_within"])
        body += "\n" + _pk(
            "within-group",
            f"""<p>{within_group_lead}, where <code>{within_group["col"]}</code> is the grouping
column this run was given. Ranking rows by their group's average
prediction alone scores {within_group["c_group_mean"]:.3f}, and comparing only
rows inside the same group scores {within_group["c_within"]:.3f},
{within_gloss}. Section @sec:results gives the decomposition.</p>""",
        )
    if derived["has_oracle"]:
        body += "\n" + _pk(
            "oracle-summary",
            f"""<p>An oracle ranking, defined in section @sec:results, bounds every
model at {pooled["c_oracle"]:.3f}.</p>""",
        )
    losing_text_lead = _losing_horizons(
        derived["brier"], derived["unit_suffix"], derived["time_unit"]
    )
    if losing_text_lead:
        body += "\n" + _pk(
            "losing-horizons-summary",
            f"""<p>{losing_text_lead}. A no-skill forecast assigns every row the same
population-average probability. The limitations section says what remains
usable.</p>""",
        )
    body += (
        "\n<p>Both models are saved in one bundle, which records the"
        f" {derived['recommended_model']} as recommended for scoring new"
        f" rows{derived['recommendation_margin']}.</p>"
    )
    if derived["generator"]:
        body += "\n" + _pk(
            "synthetic-callout",
            '<p class="callout">All results are synthetic. The run validates the'
            " pipeline and asserts nothing about live data.</p>",
        )
    doc.section("summary", "Summary", body)


def _sec_data(derived: dict, doc: ReportDoc) -> None:
    """Emits "Data" and, when the run passed --km-col, figure `km`. Takes the
    `data` note as an append."""
    columns = (derived["run"] or {}).get("columns")
    extra_cols = ""
    if columns and columns.get("dropped"):
        extra_cols += f" Columns dropped before fitting: {', '.join(columns['dropped'])}."
    if columns and columns.get("categorical"):
        extra_cols += (
            " These columns hold numeric codes rather than quantities and were"
            f" forced to categorical: {', '.join(columns['categorical'])}."
        )
    if extra_cols:
        extra_cols = _pk("columns", extra_cols)
    body = f"""<p>Durations are measured in {derived["time_unit"]}.{extra_cols}</p>"""
    body += """
<p>Left truncation, where a row was already running when the source's
records begin, leaves a recorded start that is not the true start. This
pipeline has no delayed-entry handling, so it cannot take a row that
entered observation partway through its life, and those rows must be
excluded during preparation. Whether they were, and how, is recorded in
the dataset's own documentation.</p>"""
    if derived["notes"].get("data"):
        body += "\n\n" + _marked_note(derived["notes"]["data"])
    if derived["km"]:
        km_col = derived["km"]["column"]
        # The estimator is glossed here rather than in the prose above because
        # this is its first use in the document and a caption costs no template
        # words. Section @sec:results defines it again for runs that draw no
        # such figure, which is the only place it would otherwise appear.
        km_caption = (
            f"Kaplan-Meier survival curves by <code>{km_col}</code>: the"
            f" fraction of each group still running at each age, with censored"
            f" rows counted for as long as they were observed. Groups"
            " beyond the seven most frequent, where present, are collapsed"
            " into (other) for this plot only."
        )
        km_fig = doc.figure(
            "km",
            img_uri(derived["figures_dir"], derived["km"]["filename"]),
            f"Kaplan-Meier survival curves by {km_col}",
            km_caption,
        )
        body += "\n" + _pk(
            "km-figure",
            f"""<p>Figure @fig:km shows Kaplan-Meier survival curves by
<code>{km_col}</code>, the coarsest structure in the outcome before any
model is fitted.</p>

{km_fig}""",
        )
    doc.section("data", "Data", body)


def _sec_method(derived: dict, doc: ReportDoc) -> None:
    """Emits "Method" with subsections 1 to 3 (model class, temporal
    validation, selection and calibration). Registers nothing citable."""
    pooled, config, folds = derived["params"], derived["config"], derived["folds"]
    body = f"""<h3>@sec:method.1 Model class</h3>

<p>The pipeline fits two models on every run. The first is a boosted-tree
accelerated failure time (AFT) model, XGBoost with its
<code>survival:aft</code> objective, built for durations with incomplete
observations. An observed ending tells the model the exact lifetime, and a
censored row tells it only "at least this long", so every
row contributes. It predicts each row's median survival
time in {derived["time_unit"]}, and a log-normal curve around that median, whose width
is fitted once and shared by every row, gives the probability of
surviving any horizon. The second is a Cox proportional hazards baseline, the
standard linear survival model, fitted with lifelines'
<code>CoxPHFitter</code> on the same features to answer whether the
boosted model was necessary. Its coefficients are fitted under a penalty
that pulls them toward no effect, which holds the fit steady when features
move together but leaves the uncertainty intervals drawn around those
coefficients approximate. The run recommends whichever scores the higher fold-mean
concordance.</p>

<p>The two differ in what they assume. The AFT model assumes every
row follows the same survival curve run on a faster or slower
clock, so features change how fast that clock runs, never the curve's shape. The
Cox model makes no assumption about that shape at all,
estimating the curve from the data by counting who was still running at
each age. In exchange it assumes each feature multiplies risk by the same
factor at every age, which this report does not test. Which set of
assumptions suits a dataset cannot be known in advance, which is why both
are fitted and scored.</p>

<p>Numeric features pass through, missing values included (the Cox baseline
gets train-window median imputation). Text columns are one-hot encoded with
the vocabulary refit per training window, so a fold's features reflect only
what was on file by its split date.</p>

<h3>@sec:method.2 Temporal validation and label re-censoring</h3>

<p>Evaluation uses {len(folds)} expanding-window folds ordered by start
date. The earliest {pct(config["min_train_frac"], 0)} of rows is set aside
as burn-in and never tested. Each fold trains on every row started
before its split date and tests on the next block. A split date can only
fall on a date the data contains, so the first fold trains on
{derived["n_train_min"]:,} rows, the burn-in rounded to a date boundary (sizes
in Table @tab:folds).</p>

<p>Training labels are re-censored at each split date. A row
started long before a split may have died after it, and its label
contains that future, so every post-split death is rewritten as a censoring
at the split. Omitting this raises scores by importing the future.</p>

<h3>@sec:method.3 Selection and calibration</h3>

<p>The boosted model's hyperparameters are selected once on the first
fold's training window by an inner temporal split, the same
past-then-future cut made inside that window. Concordance grades only
whether rows are ordered correctly, and a model can order them well
while drawing survival curves far too wide or too narrow. So selection is
scored on held-out censored log-likelihood instead, which grades the whole
predicted distribution against what was observed.</p>

<p>The predictive scale is the width of the boosted model's log-normal
curve, one unitless number shared by every row. A larger scale
spreads the model's probability over a wider range of lifetimes. Two
values of it appear in this run. The first, {pooled["aft_sigma"]}, is the
setting used while training, where it shapes how the medians are fitted.
The second, {pooled["predictive_sigma_final"]:.2f}, is measured afterwards.
With the fitted medians held fixed, it is the width that best matches the
held-out outcomes on the training window's most recent stretch, and it is
carried to the model refitted on the full window. The training value
answers which width trains the best medians. The measured value answers
which width matches the outcomes those medians actually got. The two need
not agree, and every probability the boosted model reports here uses the
measured value.</p>"""
    doc.section("method", "Method", body)


def _sec_results(derived: dict, doc: ReportDoc) -> None:
    """Emits "Results" with subsections 1 (discrimination) and 2
    (calibration). Registers tables `concordance`, `folds`, `brier` and
    figures `fold-cindex`, `calibration`."""
    pooled, folds, config, brier, calibration = (
        derived["pooled"],
        derived["folds"],
        derived["config"],
        derived["brier"],
        derived["calibration"],
    )
    concordance_rows = ""
    if derived["has_oracle"]:
        concordance_rows += _pk(
            "oracle-row",
            f"""    <tr><td>Oracle ranking on latent log-time (ceiling)</td>
      <td>{pooled["c_oracle"]:.3f}</td><td>not resampled</td></tr>""",
        )
    concordance_rows += f"""
    <tr class="highlight"><td>XGBoost AFT (pooled)</td><td>{pooled["c_xgb"]:.3f}</td>
      <td>{pooled["c_xgb_ci"][0]:.3f} to {pooled["c_xgb_ci"][1]:.3f}</td></tr>
    <tr><td>XGBoost AFT (fold mean)</td>
      <td>{pooled["c_xgb_by_fold_mean"]:.3f}</td><td>not resampled</td></tr>
    <tr><td>Cox proportional hazards (fold mean)</td>
      <td>{pooled["c_cox_by_fold_mean"]:.3f}</td><td>not resampled</td></tr>"""
    table_concordance = doc.table(
        "concordance",
        f"Concordance index by model, pooled over {pooled['n_test']:,} test"
        f" rows with 95% percentile bootstrap intervals over"
        f" {config['n_bootstrap']} resamples. Higher is better, and 0.500 is a"
        f" coin flip, and Harrell's C is the standard estimator of it. Fold-mean"
        f" rows score each of the {len(folds)} folds"
        " separately and average. The interval resamples test rows with the"
        " fitted models held fixed, so it measures scoring precision, not"
        " stability across history. The fold spread is the guide to that.",
        "<tr><th>Method</th><th>Concordance (Harrell's C)</th><th>95% interval</th></tr>",
        concordance_rows,
    )

    fold_head = (
        "<tr><th>Fold</th><th>Split date</th><th>Train n</th><th>Test n</th>\n"
        "    <th>Censored</th><th>AFT</th><th>Cox</th>"
    )
    if derived["has_oracle"]:
        fold_head += "<th>Oracle</th>"
    fold_head += "</tr>"
    fold_rows = []
    for i, fold in enumerate(folds):
        row = (
            f"<tr><td>{i + 1}</td><td>{fold['split_date']}</td><td>{fold['n_train']:,}</td>"
            f"<td>{fold['n_test']:,}</td><td>{pct(1 - fold['test_event_rate'], 0)}</td>"
            f"<td>{fold['c_xgb']:.3f}</td><td>{fold['c_cox']:.3f}</td>"
        )
        if derived["has_oracle"]:
            row += f"<td>{fold['c_oracle']:.3f}</td>"
        fold_rows.append(row + "</tr>")
    table_folds = doc.table(
        "folds",
        "Per-fold results. Censoring is the share of each fold's test"
        " rows whose ending was not observed."
        + (
            f" The {derived['config']['n_folds']} requested folds merged to {len(folds)}"
            " where the start dates' granularity gave several the same split"
            " date."
            if len(folds) != derived["config"]["n_folds"]
            else ""
        ),
        fold_head,
        "\n".join(fold_rows),
    )
    fig_folds = doc.figure(
        "fold-cindex",
        img_uri(derived["figures_dir"], "fold_cindex.png"),
        "Concordance index by temporal fold",
        f"Concordance by fold. The boosted model spans {derived['c_fold_min']:.3f} to"
        f" {derived['c_fold_max']:.3f}. Folds differ in censoring mix, so cross-fold"
        " comparisons are indicative rather than exact.",
    )

    body = f"""<h3>@sec:results.1 Discrimination</h3>

{table_concordance}

<p>Each fold fits its own models. Because the AFT model predicts a
survival time in {derived["time_unit"]}, and {derived["time_unit"]} is a universal unit that exists
outside of an individual fold model, its predictions can be pooled into
one list while Cox proportional hazards models cannot. A Cox model
predicts a hazard score, the risk of ending at any given moment, relative
to the average row in the window it was trained on, not universal time
units. As a result, Cox scores from
different folds cannot be compared in one list. For a fair comparison,
the two models must be compared on fold mean concordance, which is each
fold scored on its own and the {len(folds)} scores averaged.</p>

<p>The pooled row answers a different question. It scores all test
rows in one list, which is enough data to attach an uncertainty
range as seen in Table @tab:concordance. The interval is the
range the score stayed inside for 95% of resamples of the test
rows. A narrow interval means the score is precise on this test
set. And an interval that sits wholly above 0.500 means the model beats a
coin flip even at its low end.</p>

"""
    if derived["within_group"]:
        within_group = derived["within_group"]
        body += "\n" + _pk(
            "within-group-results",
            f"""<p>It is important to determine whether the model does more
than recognize which group a row belongs to. In order to assess
this, we split the pooled concordance by <code>{within_group["col"]}</code>.
Re-ranking rows by their group's average replaces every
row in a group with the same score, so comparisons only ever run
between groups. Evaluating the pooled concordance on those re-ranked
rows gives a score of {pct(within_group["c_group_mean"])}. The boosted
model scores each row individually instead, which orders
rows inside a group. Inside a group this ranking is correct
{pct(within_group["c_within"])} of the
time, averaged over the {within_group["n_groups"]} groups large enough to score (at
least {within_group["min_n"]} rows and {within_group["min_events"]} observed endings),
each weighted by its number of comparable pairs. When the group-average
score matches or beats the model's own pooled score, the model's ranking
comes from telling groups apart. When the within-group score sits well
above 0.500, the model also orders rows inside a group, which is the part
a lookup table of group averages cannot do.</p>""",
        )
    if derived["has_oracle"]:
        body += "\n" + _pk(
            "oracle-results",
            f"""<p>The oracle ranking in Table @tab:concordance orders rows by the
latent log-time the generator actually used. Nothing is fitted and the
ranking predates the noise draw, so its {pooled["c_oracle"]:.3f} bounds
every model. The gap to a perfect score is installed noise. A suite test
asserts, on a generated run, that no model outscores it, since beating
perfect information indicates a leak.</p>""",
        )
    body += f"""

<p>Figure @fig:fold-cindex plots the per-fold concordance from
Table @tab:folds.</p>

{table_folds}

{fig_folds}

<h3>@sec:results.2 Calibration</h3>

<p>A majority of this report focuses on concordance. But concordance
grades only whether models produce the correct ordering when comparing
two rows. It cannot be used to assess the absolute accuracy of
the predicted probabilities. To assess the absolute accuracy, we use the
Brier score, the mean squared error of a probability forecast
(lower is better). Censored rows can bias the Brier score, so we
use the inverse probability of censoring weighting
(IPCW) to compensate. At each horizon in Table @tab:brier, the models
predict each row's chance of still running that length of time. 
The no-skill forecast predicts the same chance for every row, the
share of the whole population still running at that horizon, estimated
with the Kaplan-Meier method so censored rows count for as long
as they were observed.</p>"""
    brier_rows = "\n".join(
        f"<tr><td>{horizon_key.removesuffix(derived['unit_suffix'])} {derived['time_unit']}</td>"
        f"<td>{scores['xgb']:.3f}</td>"
        f"<td>{scores['cox']:.3f}</td><td>{scores['km_marginal']:.3f}</td></tr>"
        for horizon_key, scores in brier.items()
    )
    table_brier = doc.table(
        "brier",
        "IPCW Brier score by horizon. Lower is better, and the no-skill"
        " column is the score to beat, so a model above it adds nothing at"
        " that horizon.",
        "<tr><th>Horizon</th><th>AFT</th><th>Cox</th>\n    <th>No-skill forecast</th></tr>",
        brier_rows,
    )
    gaps = [
        (abs(calibration_bin["predicted"] - calibration_bin["observed_km"]), i)
        for i, calibration_bin in enumerate(calibration)
    ]
    worst_gap, worst_bin = max(gaps)
    if derived["cox_calibration"]:
        cox_gaps = [
            (abs(calibration_bin["predicted"] - calibration_bin["observed_km"]), i)
            for i, calibration_bin in enumerate(derived["cox_calibration"])
        ]
        cox_worst_gap, cox_worst_bin = max(cox_gaps)
        calibration_scope = "both models, each binned on its own predicted deciles"
        calibration_worst = (
            f"The largest deviation is {worst_gap:.3f} in the boosted model's"
            f" decile {worst_bin + 1} and {cox_worst_gap:.3f} in the Cox"
            f" baseline's decile {cox_worst_bin + 1}."
        )
    else:
        calibration_scope = "the boosted AFT model, by predicted decile"
        calibration_worst = f"The largest deviation is {worst_gap:.3f} in decile {worst_bin + 1}."
    smallest_bin, largest_bin = (
        min(calibration_bin["n"] for calibration_bin in calibration),
        max(calibration_bin["n"] for calibration_bin in calibration),
    )
    decile_size = (
        f"{smallest_bin:,} rows each"
        if smallest_bin == largest_bin
        else f"{smallest_bin:,} to {largest_bin:,} rows each"
    )
    calibration_horizon, time_unit = derived["calibration_horizon"], derived["time_unit"]
    fig_calibration = doc.figure(
        "calibration",
        img_uri(
            derived["figures_dir"],
            f"calibration_{calibration_horizon}{derived['unit_suffix']}.png",
        ),
        f"Decile calibration at {calibration_horizon} {time_unit}",
        f"Predicted against observed survival at {calibration_horizon} {time_unit} for"
        f" {calibration_scope}. Observed frequencies are Kaplan-Meier estimates"
        f" within each bin, so censored rows contribute correctly."
        f" {calibration_worst} Deviations are probabilities. Deciles are cut on"
        f" predicted value and hold {decile_size}. In small or heavily"
        " censored deciles the observed value carries more uncertainty than"
        " three decimals suggest.",
    )
    body += f"""

{table_brier}

<p>Figure @fig:calibration plots predicted against observed survival at
{calibration_horizon} {time_unit} for both models. Points falling on both sides of the
diagonal are noise. Points falling consistently on one side are bias in
that direction.</p>

{fig_calibration}"""
    doc.section("results", "Results", body)


def _sec_model_uses(derived: dict, doc: ReportDoc) -> None:
    """Emits "Feature analysis". Registers figures `shap-bar`, `beeswarm`
    and, when the metrics carry cox_top, figure `cox-hr` and table `cox-hr`
    inside the presence-keyed Cox subsection."""
    fig_bar = doc.figure(
        "shap-bar",
        img_uri(derived["figures_dir"], "shap_bar.png"),
        "Mean absolute SHAP by feature",
        "The strongest features, ranked by the average size of their"
        " attributions across the explanation sample. Size is taken without"
        " regard to sign, so this says how much each feature moves"
        " predictions and not which way it moves them; the direction is in"
        " Figure @fig:beeswarm. The scale is the same log scale of survival"
        " time, so a feature averaging 0.10 moves predicted survival time by"
        " about a tenth on a typical row. A yes/no flag from a text column keeps"
        " its column name in the label. Features below the strongest few are"
        " not drawn.",
    )
    fig_beeswarm = doc.figure(
        "beeswarm",
        img_uri(derived["figures_dir"], "shap_beeswarm.png"),
        "SHAP attributions per row and feature",
        "One row per feature, one dot per row of the explanation sample. The"
        " dot's position is the attribution that feature earned for that row,"
        " so dots left of zero shortened its predicted survival time and dots"
        " right of it lengthened. The dot's color is that row's value of the"
        " feature, red high and blue low, so a row whose reds sit left of its"
        " blues is a feature whose high values shorten survival. How far the"
        " dots spread says how much the feature's effect varies from row to"
        " row, and a feature pinned at zero did nothing. Single attributions run"
        " wider than the averages in Figure @fig:shap-bar, since averaging over"
        " rows shrinks them. A yes/no flag from a"
        " text column keeps its column name in the label and reads the same way,"
        " with red meaning the row has that value.",
    )
    body = f"""<h3>@sec:model-uses.1 The boosted model</h3>

<p>The scores above grade how well each model ranks rows and how accurate
its probabilities are, not which inputs it used. A feature attribution
answers that. For one row and one feature, it is that feature's
contribution to the difference between that row's prediction and the
model's average prediction. The method used here is SHAP (SHapley
Additive exPlanations).</p>

<p>Attributions sit on the log scale of survival time, so an attribution
is a multiplier on predicted time rather than a number of {derived["time_unit"]}. An
attribution of +0.3 multiplies that row's predicted survival time by
about 1.35, and a negative one shortens it. That conversion follows from
the scale and says nothing about this dataset.</p>

<p>Averaging a feature's attributions by size, ignoring sign, says how
much that feature moves predictions across the explanation sample, a
random subset of the final model's training rows drawn for the
attribution computation. A feature high in that average moves predictions a lot, and one low in it
barely moves them.</p>

<p>Attributions are descriptive and in-sample, computed on the final
boosted model's own training rows. Correlated features split credit by the model's
internal choices as much as by the data, so directions are more trustworthy
than magnitudes.</p>

<p>Figure @fig:shap-bar ranks the strongest features by that average, and
Figure @fig:beeswarm shows, for each of those features, which direction it
pushed and how much that varied from row to row.</p>

{fig_bar}

{fig_beeswarm}"""
    if derived["cox_top"]:
        # Older metrics files carry no cox_reference key, so the sentence
        # renders only when the references are known.
        references = dict(
            reference.split("=", 1)
            for reference in (derived["metrics"].get("cox_reference") or [])
            if "=" in reference
        )
        reference_columns: list[str] = []
        for row in derived["cox_top"][:12]:
            column = row["feature"].split("=", 1)[0]
            if "=" in row["feature"] and column in references and column not in reference_columns:
                reference_columns.append(column)
        ref_sentence = "".join(
            f" Ratios for <code>{column}</code> are relative to {references[column]}, the"
            " reference level, chosen alphabetically and given no bar of its own."
            for column in reference_columns
        )
        fig_hazard_ratios = doc.figure(
            "cox-hr",
            img_uri(derived["figures_dir"], "cox_hr.png"),
            "Cox hazard ratios with 95% intervals",
            "The Cox Proportional Hazards model's strongest covariates with"
            " their 95% intervals on a log axis. Dots right of the line"
            " shorten survival and left lengthen it. The covariate whose"
            " effect the data pins down most firmly sits at the top, and only"
            f" the strongest few are drawn.{ref_sentence}",
        )
        body += "\n" + _pk(
            "cox-uses",
            f"""<h3>@sec:model-uses.2 The Cox baseline</h3>

<p>The Cox baseline's drivers are its coefficients, reported as hazard
ratios. A hazard ratio is the factor by which one unit of a feature
multiplies the hazard and a hazard is the risk of an ending at a given age. The factor is
the same at every age, so a ratio above 1 shortens survival.
Figure @fig:cox-hr plots the strongest covariates.</p>

<p>Figure @fig:shap-bar describes the boosted model and Figure @fig:cox-hr
the Cox baseline, two different models read by two different methods, so
they need not name the same features, and neither is a conclusion about the
data.</p>

{fig_hazard_ratios}""",
        )
    doc.section("model-uses", "Feature analysis", body)


def _sec_limitations(derived: dict, doc: ReportDoc) -> None:
    """Emits "Limitations" as independent bolded paragraphs, each rendered
    only when it applies to the run. Takes the `limitations` note as an
    append, positioned mid-list so authored caveats sit among the generic
    ones rather than after them."""
    parts: list[str] = []
    losing = [
        horizon_key
        for horizon_key, scores in derived["brier"].items()
        if scores["xgb"] >= scores["km_marginal"]
    ]
    losing_cox = [
        horizon_key
        for horizon_key, scores in derived["brier"].items()
        if scores["cox"] >= scores["km_marginal"]
    ]
    if losing or losing_cox:
        losing_text = _horizon_list(
            losing or losing_cox, derived["unit_suffix"], derived["time_unit"]
        )
        lose_who, lose_whose = (
            ("Both models lose", "Both models' absolute probabilities")
            if losing and losing_cox
            else ("The boosted model loses", "The boosted model's absolute probabilities")
            if losing
            else ("The Cox baseline loses", "The Cox baseline's absolute probabilities")
        )
        parts.append(
            _pk(
                "losing-horizons",
                f"""<p><strong>{lose_whose} are not usable at
{losing_text}.</strong> {lose_who}
to the no-skill forecast on the censoring-weighted Brier score there, and
Table @tab:brier grades each model separately. Ranking and probability
quality are separate, so ordering rows is still supported even at
the horizons where the probabilities lose.</p>""",
            )
        )
    parts.append("""<p><strong>Censoring may be informative.</strong> The evaluation assumes
rows stop being observed for reasons unrelated to their risk. If the
rows that drop out of observation are the ones about to end, the
survival estimates come out too high.</p>""")
    if derived["notes"].get("limitations"):
        parts.append(_marked_note(derived["notes"]["limitations"]))
    parts.append("""<p><strong>The boosted model gives every row one curve
shape.</strong> The predictive scale is a single fitted number, so the
model runs a row's clock faster or slower but never changes the
curve's shape. Per-row width is future work.</p>""")
    parts.append(f"""<p><strong>Harrell's concordance is biased under heavy censoring,
usually upward.</strong> It scores only the pairs where the earlier
ending is known, and under heavy censoring those pairs over-represent
short observed lifetimes. The most censored test window is
{pct(derived["max_fold_censored"], 0)} censored, so read its rows in
Table @tab:folds with the most doubt.</p>""")
    doc.section("limitations", "Limitations", "\n\n".join(parts))


def _sec_repro(derived: dict, doc: ReportDoc) -> None:
    """Emits "Reproducing this run" with the run's own command."""
    body = f"""<p>Python 3.11 or later. From the repository root:</p>

<pre><code>{derived["command"]}</code></pre>

<p>Every number is read from the run's
<code>metrics.json</code> at build time, and a figure the prose never cites
fails the build. <code>requirements.txt</code> pins the exact versions the
committed numbers came from.</p>"""
    doc.section("repro", "Reproducing this run", body)


# The seam between generated template prose and authored prose must be
# visible: a reader otherwise credits the pipeline with the author's
# interpretation, or the author with the template's claims.
NOTEMARK = '<p class="notemark">Analyst notes:</p>\n'


def _marked_note(note_html: str) -> str:
    """Insert the authored-prose marker just inside the note wrapper, so the
    invariance and word-budget tests strip it together with the note."""
    return note_html.replace("-->", "-->\n" + NOTEMARK, 1)


def compose_report(run_context: dict) -> str:
    """Build the whole report from a context, section by section in reading order, and return the
    rendered HTML.
    """
    derived = _derive(run_context)
    doc = ReportDoc()
    _sec_summary(derived, doc)
    if derived["notes"].get("motivation"):
        doc.section(
            "motivation",
            "Motivation",
            _marked_note(derived["notes"]["motivation"]),
        )
    _sec_data(derived, doc)
    _sec_method(derived, doc)
    _sec_results(derived, doc)
    _sec_model_uses(derived, doc)
    if derived["notes"].get("interpretation"):
        doc.section(
            "interpretation",
            "Interpretation",
            _marked_note(derived["notes"]["interpretation"]),
        )
    _sec_limitations(derived, doc)
    _sec_repro(derived, doc)
    return doc.render(
        doctype="Technical Report",
        title=run_context["title"],
        subtitle=run_context["subtitle"],
        meta_rows=run_context["meta_rows"],
        footer=run_context["footer"],
    )
