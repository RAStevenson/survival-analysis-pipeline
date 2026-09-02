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
measurement (the generator block, the oracle and Sharpe columns, the
within-group decomposition); never a mode flag. (3) Notes: authored
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


def _losing_horizons(brier: dict, tua: str, tu: str) -> str:
    """Which models' probabilities lose to the no-skill forecast, and where.

    Both models are checked. Reporting only the boosted model's losses hid
    that the Cox baseline also loses at Chicago's one-year horizon, which a
    reader caught by comparing the prose against the report's own table.
    """
    aft = [h for h, v in brier.items() if v["xgb"] >= v["km_marginal"]]
    cox = [h for h, v in brier.items() if v["cox"] >= v["km_marginal"]]
    if not aft and not cox:
        return ""
    if aft == cox:
        return "Both models' probabilities lose to a no-skill forecast at " + _horizon_list(
            aft, tua, tu
        )
    parts = []
    if aft:
        parts.append(
            "The boosted model's probabilities lose to a no-skill forecast at "
            + _horizon_list(aft, tua, tu)
        )
    if cox:
        lead = (
            "the Cox baseline's at"
            if parts
            else "The Cox baseline's probabilities lose to a no-skill forecast at"
        )
        parts.append(f"{lead} {_horizon_list(cox, tua, tu)}")
    return ", and ".join(parts)


def _horizon_list(keys: list[str], tua: str, tu: str) -> str:
    """Horizon keys as prose: '365 days', '365 and 730 days', or
    '365, 730, and 1460 days'. Oxford comma on three or more."""
    xs = [k.removesuffix(tua) for k in keys]
    if len(xs) == 1:
        return f"{xs[0]} {tu}"
    if len(xs) == 2:
        return f"{xs[0]} and {xs[1]} {tu}"
    return ", ".join(xs[:-1]) + f", and {xs[-1]} {tu}"


def _km(m: dict, run_dir: Path) -> dict | None:
    km_col = (m.get("run") or {}).get("km_col")
    if km_col and (run_dir / "figures" / "km_by_group.png").exists():
        return {"col": km_col, "filename": "km_by_group.png"}
    return None


def synthetic_context(m: dict, run_dir: Path, notes_dir: Path | None = None) -> dict:
    g, d, run = m["generator"], m["dataset"], m["run"]
    rel = _display_path(run_dir)

    # Same header shape as real_context, values only. The synthetic run's one
    # extra provenance fact, the seed, rides in the source-data cell.
    censored_overall = 1.0 - d["event_rate"]
    meta_rows = f"""    <tr><th>Source data</th><td><code>{run["source"]}</code>, drawn at
      seed {g["seed"]}</td></tr>
    <tr><th>Rows</th><td>{d["n_rows"]:,} ({pct(d["event_rate"])} with the ending
      observed, {pct(censored_overall)} censored)</td></tr>
    <tr><th>Start dates</th><td>{d["date_min"]} to {d["date_max"]}</td></tr>
    <tr><th>Evaluation</th><td>{len(m["folds"])} expanding-window temporal folds</td></tr>
    <tr><th>Source of figures</th><td><code>{rel}/metrics.json</code>,
      regenerated by the command in section @sec:repro</td></tr>"""

    footer = f"""<p>Generated from <code>{rel}/metrics.json</code> by
<code>scripts/run_build_report.py</code>. {d["n_rows"]:,} rows drawn at
seed {g["seed"]}, {m["pooled"]["n_test"]:,} out-of-time test rows.</p>"""

    # The synthetic run's input is regenerated from the seed rather than
    # committed, so the reproducing command is the runner that writes the CSV
    # and then calls the same fit_evaluate a user calls directly.
    command = (
        "pip install -r requirements.txt\n"
        "python -m pytest\n"
        "python scripts/run_synthetic_pipeline.py"
    )

    return {
        "m": m,
        "figures_dir": run_dir / "figures",
        "title": f"Survival Model Evaluation: {run['name']}",
        "subtitle": "Fitted with the survival-analysis-pipeline,\n"
        "  on synthetic data with known ground truth",
        "meta_rows": meta_rows,
        "footer": footer,
        "command": command,
        "source_desc": f"synthetic data drawn at seed {g['seed']}",
        "km": _km(m, run_dir),
        "notes": load_run_notes(notes_dir, m),
    }


def real_context(m: dict, run_dir: Path, notes_dir: Path | None = None) -> dict:
    run, d, pool = m["run"], m["dataset"], m["pooled"]

    # One line on purpose. A cmd.exe caret continuation is a parse error in
    # PowerShell and a stray argument in bash, so a wrapped command is a
    # command the reader cannot paste. --out is included because without it
    # the run lands in the default runs/<name>/ and the rebuild step below
    # would re-render the old report instead of the one just produced.
    cols = run["columns"]
    drop_part = f" --drop-cols {','.join(cols['dropped'])}" if cols["dropped"] else ""
    categorical = cols.get("categorical")
    cat_part = f" --categorical-cols {','.join(categorical)}" if categorical else ""
    km_part = f" --km-col {run['km_col']}" if run.get("km_col") else ""
    out_part = f" --out {run['out_dir']}" if run.get("out_dir") else ""
    # Days is the CLI default, so day-based commands stay exactly as the
    # committed reports print them.
    run_unit = run.get("time_unit", "days")
    unit_part = f" --time-unit {run_unit}" if run_unit != "days" else ""
    command = (
        f"python scripts/run_fit_evaluate.py --data {run['source']} --name {run['name']} "
        f"--id-col {cols['id']} --date-col {cols['date']} "
        f"--duration-col {cols['duration']} --event-col {cols['event']}"
        f"{drop_part}{cat_part}{km_part}{unit_part} --folds {run['n_folds']} "
        f"--horizons {','.join(horizon_label(h) for h in run['horizons_days'])}{out_part}"
        f"\npython scripts/run_build_report.py --run {_display_path(run_dir)}"
    )

    censored_overall = 1.0 - d["event_rate"]
    meta_rows = f"""    <tr><th>Source data</th><td><code>{run["source"]}</code></td></tr>
    <tr><th>Rows</th><td>{d["n_rows"]:,} ({pct(d["event_rate"])} with the ending
      observed, {pct(censored_overall)} censored)</td></tr>
    <tr><th>Start dates</th><td>{d["date_min"]} to {d["date_max"]}</td></tr>
    <tr><th>Evaluation</th><td>{len(m["folds"])} expanding-window temporal folds</td></tr>
    <tr><th>Source of figures</th><td><code>{_display_path(run_dir)}/metrics.json</code>,
      regenerated by the command in section @sec:repro</td></tr>"""

    footer = f"""<p>Generated from <code>{_display_path(run_dir)}/metrics.json</code> by
<code>scripts/run_build_report.py --run</code>. {d["n_rows"]:,} rows,
{pool["n_test"]:,} out-of-time test rows.</p>"""

    if notes_dir is None:
        notes_dir = run_dir / "notes"

    return {
        "m": m,
        "figures_dir": run_dir / "figures",
        "title": f"Survival Model Evaluation: {run['name']}",
        "subtitle": "Fitted with the survival-analysis-pipeline.",
        "meta_rows": meta_rows,
        "footer": footer,
        "command": command,
        "source_desc": f"<code>{Path(run['source']).name}</code>",
        "km": _km(m, run_dir),
        "notes": load_run_notes(notes_dir, m),
    }


# --------------------------------------------------------------------------
# Derived rendering state shared by the section builders.


def _derive(ctx: dict) -> dict:
    m = ctx["m"]
    d, pool, folds, cfg = m["dataset"], m["pooled"], m["folds"], m["config"]
    g, run = m.get("generator"), m.get("run")
    tu = cfg.get("time_unit", "days")
    hs = horizon_label(cfg["calibration_horizon_days"])
    tua = unit_abbrev(tu)
    fm_x, fm_c = pool["c_xgb_by_fold_mean"], pool["c_cox_by_fold_mean"]
    if f"{fm_x:.3f}" == f"{fm_c:.3f}":
        winner_clause = "The two models tie at the printed precision"
    elif abs(fm_x - fm_c) < 0.0015:
        # A gap the bootstrap interval swallows is not a winner.
        winner_clause = "The two models effectively tie"
    elif fm_c > fm_x:
        winner_clause = "The Cox baseline scores higher"
    else:
        winner_clause = "The boosted model scores higher"
    # Mirrors save_model_bundle's tie-break (aft on equality), so the report
    # names the same model the saved sidecar records as recommended. When the
    # winner clause above calls the run a tie, the bundle still records a
    # recommendation, and the sentence must say the margin is thin rather
    # than let the two statements read as a contradiction.
    rec_model = "Cox baseline" if fm_c > fm_x else "boosted model"
    rec_margin = "" if "tie" not in winner_clause else ", on a near-tie margin"
    return {
        "m": m,
        "notes": ctx.get("notes") or {},
        "figures_dir": ctx["figures_dir"],
        "source_desc": ctx["source_desc"],
        "command": ctx["command"],
        "km": ctx["km"],
        "g": g,
        "run": run,
        "p": m["params"],
        "d": d,
        "pool": pool,
        "folds": folds,
        "brier": m["ipcw_brier"],
        "cfg": cfg,
        "wg": m.get("within_group"),
        "tu": tu,
        "tua": tua,
        "hs": hs,
        "cal": m[f"calibration_{hs}{tua}"],
        "cal_cox": m.get(f"calibration_cox_{hs}{tua}"),
        "cox_top": m.get("cox_top"),
        "has_oracle": "c_oracle" in pool,
        "has_sharpe": "c_sharpe" in pool,
        "n_rows": d["n_rows"],
        "date_min": d["date_min"],
        "date_max": d["date_max"],
        "censored_overall": 1.0 - d["event_rate"],
        "max_fold_censored": max(1.0 - f["test_event_rate"] for f in folds),
        "c_fold_min": min(f["c_xgb"] for f in folds),
        "c_fold_max": max(f["c_xgb"] for f in folds),
        "n_train_min": min(f["n_train"] for f in folds),
        "n_train_max": max(f["n_train"] for f in folds),
        "winner_clause": winner_clause,
        "rec_model": rec_model,
        "rec_margin": rec_margin,
    }


def _sec_summary(c: dict, doc: ReportDoc) -> None:
    """Emits "Summary". Registers no figures or tables; every number in it is
    repeated with its full treatment later, so nothing here is the only home
    of a fact."""
    pool, folds = c["pool"], c["folds"]
    body = f"""<p>This report evaluates two survival models fitted to {c["source_desc"]}.
It holds {c["n_rows"]:,} rows, each observed from its start
date. {pct(c["d"]["event_rate"])} have observed endings and
{pct(c["censored_overall"])} are censored, still running when observation
stopped. Median observed duration, censored
rows included, is
{c["d"]["median_observed_duration_days"]:.0f} {c["tu"]}, the scale on
which every horizon and prediction below sits. The models predict,
from what was on file at the start date, how long each row
survives.</p>

<p>When evaluating the two models' performance, the apples-to-apples
comparison is the fold-mean concordance, the
share of pairs a ranking orders correctly
where 0.500 is a coin flip. The pipeline uses walkforward validation. Each of the {len(folds)}
temporal folds trains both models on one stretch of history and tests them
on the next, and the {len(folds)} scores average. On concordance XGBoost AFT scores
{pool["c_xgb_by_fold_mean"]:.3f} and the Cox proportional hazards baseline
{pool["c_cox_by_fold_mean"]:.3f}. {c["winner_clause"]}.</p>

<p>To attach an uncertainty range, which says how far the score could
reasonably move, the AFT model is also scored with all
{pool["n_test"]:,} out-of-time test rows pooled into one list.
That concordance is {pool["c_xgb"]:.3f}, with a 95% bootstrap interval
of {pool["c_xgb_ci"][0]:.3f} to {pool["c_xgb_ci"][1]:.3f}. Section
@sec:results explains the two figures and how to read the
interval.</p>"""

    if c["wg"]:
        wg = c["wg"]
        # Whether group membership dominates is a property of the run, so the
        # lead-in is computed, not asserted (same rule as winner_clause). The
        # middle branch would misdescribe both ends: a run whose group means
        # sit near a coin flip has essentially no group effect to split with.
        if wg["c_group_mean"] >= pool["c_xgb"]:
            wg_lead = f"Most of the pooled score reflects a row's <code>{wg['col']}</code> group"
        elif wg["c_group_mean"] - 0.5 < 0.25 * (pool["c_xgb"] - 0.5):
            wg_lead = f"Little of the pooled score is <code>{wg['col']}</code> group membership"
        else:
            wg_lead = (
                f"The pooled score splits between <code>{wg['col']}</code>"
                " group membership and ranking within it"
            )
        # The characterization is computed like the number it describes. The
        # asserted version shipped "0.779, close to the coin flip" in the
        # synthetic report, caught by the 2026-08-28 panel's technical seat.
        if abs(wg["c_within"] - 0.5) < 0.02:
            within_gloss = "close to the coin flip"
        elif wg["c_within"] > 0.5:
            within_gloss = "clear of the coin flip"
        else:
            within_gloss = "below the coin flip"
        body += "\n" + _pk(
            "within-group",
            f"""<p>{wg_lead}. Ranking rows by their group's average
prediction alone scores {wg["c_group_mean"]:.3f}, and comparing only
rows inside the same group scores {wg["c_within"]:.3f},
{within_gloss}. Section @sec:results gives the decomposition.</p>""",
        )
    if c["g"] and c["has_oracle"] and c["has_sharpe"]:
        body += "\n" + _pk(
            "oracle-summary",
            f"""<p>An oracle ranking (Addendum A) bounds every model at
{pool["c_oracle"]:.3f}. Ranking by validation Sharpe scores
{pool["c_sharpe"]:.3f}, below the coin flip in every fold.</p>""",
        )
    lose_text = _losing_horizons(c["brier"], c["tua"], c["tu"])
    if lose_text:
        body += "\n" + _pk(
            "losing-horizons-summary",
            f"""<p>{lose_text}. A no-skill forecast assigns every row the same
population-average probability. The limitations section says what remains
usable.</p>""",
        )
    if c["run"]:
        body += "\n" + _pk(
            "bundle",
            f"<p>Both models are saved in one bundle, which records the"
            f" {c['rec_model']} as recommended for scoring new"
            f" rows{c['rec_margin']}.</p>",
        )
    if c["g"]:
        body += "\n" + _pk(
            "synthetic-callout",
            '<p class="callout">All results are synthetic. The run validates the'
            " pipeline and asserts nothing about live data.</p>",
        )
    doc.section("summary", "Summary", body)


def _sec_data(c: dict, doc: ReportDoc) -> None:
    """Emits "Data" and, when the run passed --km-col, figure `km`. Takes the
    `data` note as an append."""
    cols = (c["run"] or {}).get("columns")
    extra_cols = ""
    if cols and cols.get("dropped"):
        extra_cols += f" Columns dropped before fitting: {', '.join(cols['dropped'])}."
    if cols and cols.get("categorical"):
        extra_cols += (
            " These columns hold numeric codes rather than quantities and were"
            f" forced to categorical: {', '.join(cols['categorical'])}."
        )
    if extra_cols:
        extra_cols = _pk("columns", extra_cols)
    body = f"""<p>Durations are measured in {c["tu"]}.{extra_cols}</p>"""
    if not c["g"]:
        # A generated dataset cannot be left-truncated; the guidance is for
        # real sources only, or it reads as accusing the generator of a fault.
        body += "\n" + _pk(
            "left-truncation",
            """<p>Left truncation, where a row was already running when the source's
records begin, leaves a recorded start that is not the true start.
Nothing in the row marks it. This pipeline has no delayed-entry handling, so those
rows must be excluded during preparation. Whether they were
excluded is recorded in the dataset's own documentation.</p>""",
        )
    if c["g"]:
        g = c["g"]
        body += "\n" + _pk(
            "generator-data",
            f"""<p>Two generator settings matter for interpreting the results.
{pct(g["admin_censor_rate"], 0)} of rows are administratively
retired independent of performance, which installs the censoring the
evaluation must handle correctly. And lifetimes are drawn log-normally
around what their drivers predict, at noise scale {g["log_time_sigma"]},
the irreducible noise that keeps even the oracle in Addendum A short of
a perfect score.</p>""",
        )
    if c["notes"].get("data"):
        body += "\n\n" + _marked_note(c["notes"]["data"])
    if c["km"]:
        km_col = c["km"]["col"]
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
            img_uri(c["figures_dir"], c["km"]["filename"]),
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


def _sec_method(c: dict, doc: ReportDoc) -> None:
    """Emits "Method" with subsections 1 to 3 (model class, temporal
    validation, selection and calibration). Registers nothing citable."""
    p, cfg, folds = c["p"], c["cfg"], c["folds"]
    body = f"""<h3>@sec:method.1 Model class</h3>

<p>The pipeline fits two models on every run. The first is a boosted-tree
accelerated failure time (AFT) model, XGBoost with its
<code>survival:aft</code> objective, built for durations with incomplete
observations. An observed ending tells the model the exact lifetime, and a
censored row tells it only "at least this long", so every
row contributes. It predicts each row's median survival
time in {c["tu"]}, and a log-normal curve around that median, whose width
is fitted once and shared by every row, gives the probability of
surviving any horizon. The second is a Cox proportional hazards baseline, the
standard linear survival model, fitted with lifelines'
<code>CoxPHFitter</code> on the same features to answer whether the
boosted model was necessary. Its coefficients are fitted under a penalty
that pulls them toward no effect, which holds the fit steady when features
move together but leaves the intervals around those coefficients
approximate. The run recommends whichever scores the higher fold-mean
concordance.</p>

<p>The two differ in what they assume. The AFT model assumes every
row follows the same survival curve run on a faster or slower
clock, so features change when things happen, never the curve's shape. The
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
date: the earliest {pct(cfg["min_train_frac"], 0)} of rows is
burn-in, trained on, never tested, and each fold trains on every
row started before its split date and tests on the next block
(sizes in Table @tab:folds).</p>

<p>Training labels are re-censored at each split date. A row
started long before a split may have died after it, and its label
contains that future, so every post-split death is rewritten as a censoring
at the split. Omitting this raises scores by importing the future, which
makes it the most consequential detail in the pipeline. It is a standalone
function (<code>temporal_folds.recensor</code>) with dedicated tests.</p>

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
spreads the model's probability over a wider range of lifetimes. Training
used {p["aft_sigma"]}, a setting of the loss that shapes how the medians
are fitted. Then, with the medians held fixed, the width that best matches
held-out outcomes on the training window's most recent stretch was
measured at {p["predictive_sigma_final"]:.2f} and carried to the model
refitted on the full window. The training value answers which curve
width trains the best medians. The measured value answers which width
matches the outcomes the trained medians actually got. The two need not
agree. Every probability the boosted model reports here
uses the measured value.</p>"""
    if not c["g"]:
        body += "\n" + _pk(
            "validated-elsewhere",
            "<p>The methodology is validated separately against synthetic data"
            " with known ground truth.</p>",
        )
    doc.section("method", "Method", body)


def _sec_results(c: dict, doc: ReportDoc) -> None:
    """Emits "Results" with subsections 1 (discrimination) and 2
    (calibration). Registers tables `concordance`, `folds`, `brier` and
    figures `fold-cindex`, `calibration`."""
    pool, folds, cfg, brier, cal = c["pool"], c["folds"], c["cfg"], c["brier"], c["cal"]
    conc_rows = ""
    if c["has_oracle"]:
        conc_rows += _pk(
            "oracle-row",
            f"""    <tr><td>Oracle ranking on latent log-time (ceiling)</td>
      <td>{pool["c_oracle"]:.3f}</td><td>not resampled</td></tr>""",
        )
    conc_rows += f"""
    <tr class="highlight"><td>XGBoost AFT (pooled)</td><td>{pool["c_xgb"]:.3f}</td>
      <td>{pool["c_xgb_ci"][0]:.3f} to {pool["c_xgb_ci"][1]:.3f}</td></tr>
    <tr><td>XGBoost AFT (fold mean)</td>
      <td>{pool["c_xgb_by_fold_mean"]:.3f}</td><td>not resampled</td></tr>
    <tr><td>Cox proportional hazards (fold mean)</td>
      <td>{pool["c_cox_by_fold_mean"]:.3f}</td><td>not resampled</td></tr>"""
    if c["has_sharpe"]:
        conc_rows += _pk(
            "sharpe-row",
            f"""
    <tr><td>Rank by validation Sharpe</td><td>{pool["c_sharpe"]:.3f}</td>
      <td>{pool["c_sharpe_ci"][0]:.3f} to {pool["c_sharpe_ci"][1]:.3f}</td></tr>""",
        )
    tab_conc = doc.table(
        "concordance",
        f"Concordance index by model, pooled over {pool['n_test']:,} test"
        f" rows with 95% percentile bootstrap intervals over"
        f" {cfg['n_bootstrap']} resamples. Higher is better, and 0.500 is a"
        f" coin flip. Fold-mean rows score each of the {len(folds)} folds"
        " separately and average. The interval resamples test rows with the"
        " fitted models held fixed, so it measures scoring precision, not"
        " stability across history. The fold spread is the guide to that.",
        "<tr><th>Method</th><th>Concordance (Harrell's C)</th><th>95% interval</th></tr>",
        conc_rows,
    )

    fold_head = (
        "<tr><th>Fold</th><th>Split date</th><th>Train n</th><th>Test n</th>\n"
        "    <th>Censored</th><th>AFT</th><th>Cox</th>"
    )
    if c["has_oracle"] and c["has_sharpe"]:
        fold_head += "<th>Sharpe</th><th>Oracle</th>"
    fold_head += "</tr>"
    fold_rows = []
    for i, f in enumerate(folds):
        row = (
            f"<tr><td>{i + 1}</td><td>{f['split_date']}</td><td>{f['n_train']:,}</td>"
            f"<td>{f['n_test']:,}</td><td>{pct(1 - f['test_event_rate'], 0)}</td>"
            f"<td>{f['c_xgb']:.3f}</td><td>{f['c_cox']:.3f}</td>"
        )
        if c["has_oracle"] and c["has_sharpe"]:
            row += f"<td>{f['c_sharpe']:.3f}</td><td>{f['c_oracle']:.3f}</td>"
        fold_rows.append(row + "</tr>")
    tab_folds = doc.table(
        "folds",
        "Per-fold results. Censoring is the share of each fold's test"
        " rows whose ending was not observed."
        + (
            f" The {c['cfg']['n_folds']} requested folds merged to {len(folds)}"
            " where the start dates' granularity gave several the same split"
            " date."
            if len(folds) != c["cfg"]["n_folds"]
            else ""
        ),
        fold_head,
        "\n".join(fold_rows),
    )
    fig_folds = doc.figure(
        "fold-cindex",
        img_uri(c["figures_dir"], "fold_cindex.png"),
        "Concordance index by temporal fold",
        f"Concordance by fold. The boosted model spans {c['c_fold_min']:.3f} to"
        f" {c['c_fold_max']:.3f}. Folds differ in censoring mix, so cross-fold"
        " comparisons are indicative rather than exact.",
    )

    body = f"""<h3>@sec:results.1 Discrimination</h3>

{tab_conc}

<p>Each fold fits its own models. Because the AFT model predicts a
survival time in {c["tu"]}, and {c["tu"]} is a universal unit that exists
outside of an individual fold model, its predictions can be pooled into
one list while Cox proportional hazards models cannot. A Cox model
predicts a hazard risk score in relation to the average row in the window
it was trained on, not universal time units. As a result, Cox scores from
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
    if c["wg"]:
        wg = c["wg"]
        body += "\n" + _pk(
            "within-group-results",
            f"""<p>It is important to determine whether the model does more
than recognize which group a row belongs to. In order to assess
this, we split the pooled concordance by <code>{wg["col"]}</code>.
Re-ranking rows by their group's average replaces every
row in a group with the same score, so comparisons only ever run
between groups. Evaluating the pooled concordance on those re-ranked
rows gives a score of {pct(wg["c_group_mean"])}. The boosted
model scores each row individually instead, which orders
rows inside a group. Inside a group this ranking is correct
{pct(wg["c_within"])} of the
time, averaged over the {wg["n_groups"]} groups large enough to score (at
least {wg["min_n"]} rows and {wg["min_events"]} observed endings),
each weighted by its number of comparable pairs. If these values are
greater than or equal to the original rankings, the model's performance
is dominated by the within-group ranking.</p>""",
        )
    if c["has_sharpe"]:
        smin = min(f["c_sharpe"] for f in folds)
        smax = max(f["c_sharpe"] for f in folds)
        flip = pool.get("c_sharpe_flipped")
        flip_sentence = (
            f" Reversed, it scores {flip:.3f}, its arithmetic complement, short"
            f" of the model's {pool['c_xgb']:.3f}."
            if flip is not None
            else ""
        )
        body += "\n" + _pk(
            "sharpe-results",
            f"""<p>Validation-Sharpe ranking scores {pool["c_sharpe"]:.3f} pooled,
{smin:.3f} to {smax:.3f} across folds, never above 0.500.{flip_sentence}</p>""",
        )
    body += f"""

<p>Figure @fig:fold-cindex plots the per-fold concordance from
Table @tab:folds.</p>

{tab_folds}

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
        f"<tr><td>{h.removesuffix(c['tua'])} {c['tu']}</td><td>{v['xgb']:.3f}</td>"
        f"<td>{v['cox']:.3f}</td><td>{v['km_marginal']:.3f}</td></tr>"
        for h, v in brier.items()
    )
    tab_brier = doc.table(
        "brier",
        "IPCW Brier score by horizon. Lower is better, and the no-skill"
        " column is the score to beat, so a model above it adds nothing at"
        " that horizon.",
        "<tr><th>Horizon</th><th>AFT</th><th>Cox</th>\n    <th>No-skill forecast</th></tr>",
        brier_rows,
    )
    gaps = [(abs(b["predicted"] - b["observed_km"]), i) for i, b in enumerate(cal)]
    worst_gap, worst_bin = max(gaps)
    if c["cal_cox"]:
        cox_gaps = [(abs(b["predicted"] - b["observed_km"]), i) for i, b in enumerate(c["cal_cox"])]
        cox_worst_gap, cox_worst_bin = max(cox_gaps)
        cal_who = "both models, each binned on its own predicted deciles"
        cal_worst = (
            f"The largest deviation is {worst_gap:.3f} in the boosted model's"
            f" decile {worst_bin + 1} and {cox_worst_gap:.3f} in the Cox"
            f" baseline's decile {cox_worst_bin + 1}."
        )
    else:
        cal_who = "the boosted AFT model, by predicted decile"
        cal_worst = f"The largest deviation is {worst_gap:.3f} in decile {worst_bin + 1}."
    fig_cal = doc.figure(
        "calibration",
        img_uri(c["figures_dir"], f"calibration_{c['hs']}{c['tua']}.png"),
        f"Decile calibration at {c['hs']} {c['tu']}",
        f"Predicted against observed survival at {c['hs']} {c['tu']} for"
        f" {cal_who}. Observed frequencies are Kaplan-Meier estimates"
        f" within each bin, so censored rows contribute correctly."
        f" {cal_worst} Deviations are probabilities. Deciles are cut on"
        f" predicted value and hold {min(b['n'] for b in cal):,} to"
        f" {max(b['n'] for b in cal):,} rows each. In small or heavily"
        " censored deciles the observed value carries more uncertainty than"
        " three decimals suggest.",
    )
    body += f"""

{tab_brier}

<p>Figure @fig:calibration plots predicted against observed survival at
{c["hs"]} {c["tu"]} for both models. Points falling on both sides of the
diagonal are noise. Points falling consistently on one side are bias in
that direction.</p>

{fig_cal}"""
    doc.section("results", "Results", body)


def _sec_model_uses(c: dict, doc: ReportDoc) -> None:
    """Emits "Feature analysis". Registers figures `shap-bar`, `beeswarm`
    and, when the metrics carry cox_top, figure `cox-hr` and table `cox-hr`
    inside the presence-keyed Cox subsection."""
    fig_bar = doc.figure(
        "shap-bar",
        img_uri(c["figures_dir"], "shap_bar.png"),
        "Mean absolute SHAP by feature",
        "The strongest features, ranked by the average size of their"
        " attributions across the explanation sample. Size is taken without"
        " regard to sign, so this says how much each feature moves"
        " predictions and not which way it moves them; the direction is in"
        " Figure @fig:beeswarm. The scale is the same log scale of survival"
        " time, so a feature averaging 0.10 moves predicted survival time by"
        " about a tenth on a typical row. Features below the strongest few"
        " are not drawn.",
    )
    fig_bee = doc.figure(
        "beeswarm",
        img_uri(c["figures_dir"], "shap_beeswarm.png"),
        "SHAP attributions per row and feature",
        "One row per feature, one dot per row of the explanation sample. The"
        " dot's position is the attribution that feature earned for that row,"
        " so dots left of zero shortened its predicted survival time and dots"
        " right of it lengthened. The dot's colour is that row's value of the"
        " feature, red high and blue low, so a row whose reds sit left of its"
        " blues is a feature whose high values shorten survival. How far the"
        " dots spread says how much the feature's effect varies from row to"
        " row, and a feature pinned at zero did nothing. A yes/no feature"
        " reads the same way, with red meaning the row is in that category.",
    )
    body = f"""<h3>@sec:model-uses.1 The boosted model</h3>

<p>The scores above grade how well each model ranks rows and how accurate
its probabilities are, not which inputs it used. A feature attribution
answers that. For one row and one feature, it is that feature's
contribution to the difference between that row's prediction and the
model's average prediction. The method used here is SHAP (SHapley
Additive exPlanations).</p>

<p>Attributions sit on the log scale of survival time, so an attribution
is a multiplier on predicted time rather than a number of {c["tu"]}. An
attribution of +0.3 multiplies that row's predicted survival time by
about 1.35, and a negative one shortens it. That conversion follows from
the scale and says nothing about this dataset.</p>

<p>Averaging a feature's attributions by size, ignoring sign, says how
much that feature moves predictions across the explanation sample. A
feature high in that average moves predictions a lot, and one low in it
barely moves them.</p>

<p>Attributions are descriptive and in-sample, computed on the final
boosted model's own training rows. Correlated features split credit by the model's
internal choices as much as by the data, so directions are more trustworthy
than magnitudes.</p>

<p>Figure @fig:shap-bar ranks the strongest features by that average, and
Figure @fig:beeswarm shows, for each of those features, which direction it
pushed and how much that varied from row to row.</p>

{fig_bar}

{fig_bee}"""
    if c["cox_top"]:
        # Older metrics files carry no cox_reference key, so the sentence
        # renders only when the references are known.
        refs = dict(r.split("=", 1) for r in (c["m"].get("cox_reference") or []) if "=" in r)
        ref_cols: list[str] = []
        for row in c["cox_top"][:12]:
            col = row["feature"].split("=", 1)[0]
            if "=" in row["feature"] and col in refs and col not in ref_cols:
                ref_cols.append(col)
        ref_sentence = "".join(
            f" Ratios for <code>{col}</code> compare against {refs[col]}"
            " (automatically chosen alphabetically) and excluded from the fit."
            for col in ref_cols
        )
        fig_hr = doc.figure(
            "cox-hr",
            img_uri(c["figures_dir"], "cox_hr.png"),
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

{fig_hr}""",
        )
    doc.section("model-uses", "Feature analysis", body)


def _sec_limitations(c: dict, doc: ReportDoc) -> None:
    """Emits "Limitations" as independent bolded paragraphs, each rendered
    only when it applies to the run. Takes the `limitations` note as an
    append, positioned mid-list so authored caveats sit among the generic
    ones rather than after them."""
    parts: list[str] = []
    if c["g"]:
        parts.append(
            _pk(
                "generator-limits",
                """<p><strong>The generator encodes a prior, not evidence,</strong> so
the concordance here is an upper bound on production performance.
Lifetimes are drawn and resampled independently, understating real-book
uncertainty, and administrative retirement is modeled as independent
censoring, with competing risks left as future work. Every number here
comes from one generator seed, so data variance and model variance cannot
be separated.</p>""",
            )
        )
    losing = [h for h, v in c["brier"].items() if v["xgb"] >= v["km_marginal"]]
    losing_cox = [h for h, v in c["brier"].items() if v["cox"] >= v["km_marginal"]]
    if losing or losing_cox:
        losing_text = _horizon_list(losing or losing_cox, c["tua"], c["tu"])
        lose_who = (
            "Both models lose"
            if losing and losing_cox
            else "The boosted model loses"
            if losing
            else "The Cox baseline loses"
        )
        parts.append(
            _pk(
                "losing-horizons",
                f"""<p><strong>Absolute probabilities are not usable at
{losing_text}.</strong> {lose_who}
to the no-skill forecast on the censoring-weighted Brier score there, and
Table @tab:brier grades each model separately. Ranking and probability
quality are separate, so ordering rows is still supported even at
the horizons where the probabilities lose.</p>""",
            )
        )
    parts.append("""<p><strong>Censoring may be informative.</strong> The evaluation assumes
rows stop being observed for reasons unrelated to their risk. Early
exits of rows about to end anyway bias survival estimates
upward.</p>""")
    if c["notes"].get("limitations"):
        parts.append(_marked_note(c["notes"]["limitations"]))
    parts.append("""<p><strong>The boosted model gives every row one curve
shape.</strong> The predictive scale is a single fitted number, so the
model runs a row's clock faster or slower but never changes the
curve's shape. Per-row width is future work.</p>""")
    parts.append(f"""<p><strong>Harrell's concordance is biased under heavy censoring,
usually upward.</strong> It scores only the pairs censoring leaves
comparable, which under heavy censoring over-represent short observed
lifetimes. The most censored test window is
{pct(c["max_fold_censored"], 0)} censored, so read its rows in
Table @tab:folds with the most doubt.</p>""")
    doc.section("limitations", "Limitations", "\n\n".join(parts))


def _sec_repro(c: dict, doc: ReportDoc) -> None:
    """Emits "Reproducing this run" with the run's own command."""
    body = f"""<p>Python 3.11 or later. From the repository root:</p>

<pre><code>{c["command"]}</code></pre>

<p>Every number is read from the run's
<code>metrics.json</code> at build time, and a figure the prose never cites
fails the build. <code>requirements.txt</code> pins the exact versions the
committed numbers came from.</p>"""
    doc.section("repro", "Reproducing this run", body)


def _sec_addendum(c: dict, doc: ReportDoc) -> None:
    """Emits addendum "Synthetic ground truth". Synthetic runs only; called
    from compose_report behind the generator-block check."""
    pool = c["pool"]
    body = f"""<p>Effects are installed as fixed shifts on log survival time.
Nothing in the generator is proprietary, and the constants are in the run's
notes.</p>

<p>The oracle ranking orders rows by the latent log-time the
generator actually used. Nothing is fitted and the ranking predates the
noise draw, so its {pool["c_oracle"]:.3f} bounds every model. The gap to a
perfect score is installed noise. A suite test asserts the model never
outscores it, since beating perfect information indicates a leak.</p>"""
    doc.addendum("synthetic-truth", "Synthetic ground truth", body)


# The seam between generated template prose and authored prose must be
# visible: a reader otherwise credits the pipeline with the author's
# interpretation, or the author with the template's claims.
NOTEMARK = '<p class="notemark">Analyst notes:</p>\n'


def _marked_note(note_html: str) -> str:
    """Insert the authored-prose marker just inside the note wrapper, so the
    invariance and word-budget tests strip it together with the note."""
    return note_html.replace("-->", "-->\n" + NOTEMARK, 1)


def compose_report(ctx: dict) -> str:
    c = _derive(ctx)
    doc = ReportDoc()
    _sec_summary(c, doc)
    if c["notes"].get("motivation"):
        doc.section(
            "motivation",
            "Motivation",
            _marked_note(c["notes"]["motivation"]),
        )
    _sec_data(c, doc)
    _sec_method(c, doc)
    _sec_results(c, doc)
    _sec_model_uses(c, doc)
    if c["notes"].get("interpretation"):
        doc.section(
            "interpretation",
            "Interpretation",
            _marked_note(c["notes"]["interpretation"]),
        )
    _sec_limitations(c, doc)
    _sec_repro(c, doc)
    if c["g"]:
        _sec_addendum(c, doc)
    return doc.render(
        doctype="Technical Report",
        title=ctx["title"],
        subtitle=ctx["subtitle"],
        meta_rows=ctx["meta_rows"],
        footer=ctx["footer"],
    )
