"""Report templates shared by the synthetic study and real-data runs.

THE REPORT CONTRACT. A generated report is instrument output: it states
what was fit, on what data, under what scheme, with what results, in prose
that holds word-for-word on a stranger's dataset. Three kinds of content
are allowed. (1) Template prose: fixed sentences plus injected values, each
technical term defined inline exactly once, in one place in this module.
(2) Presence-keyed measurement blocks, wrapped in pk-comment markers and
rendered only when the metrics carry the measurement (the generator block,
the oracle and Sharpe columns, the within-group decomposition); never a mode
flag. (3) Notes: authored markdown per run (report_notes.py),
inserted at fixed anchors, carrying all interpretation, motivation, and
dataset-specific claims, citing metric values through @val tokens that fail
the build when unresolvable. Anything editorial belongs in a run's notes,
not here. The template-invariance and word-budget tests enforce (1) and
(2); render-time numbering and the cited-or-fail figure rule live in
report_document.ReportDoc.
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
from .time_units import horizon_label, unit_abbrev, unit_singular


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
<code>scripts/run_build_report.py</code>. {d["n_rows"]:,} strategies drawn at
seed {g["seed"]}, {m["pooled"]["n_test"]:,} out-of-time test strategies.</p>"""

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
        # A gap the bootstrap interval swallows is not a winner; the same
        # threshold the weakest-fold callout uses for a near-tie.
        winner_clause = "The two models effectively tie"
    elif fm_c > fm_x:
        winner_clause = "The Cox baseline scores higher"
    else:
        winner_clause = "The boosted model scores higher"
    # Whether pooled reads low is a property of the run, not of the method,
    # so the sentence is computed rather than asserted (the synthetic run's
    # pooled figure sits a hair below its fold mean, equal at the printed
    # precision).
    if f"{pool['c_xgb']:.3f}" == f"{fm_x:.3f}":
        pooled_clause = "On this run it matches the fold mean at the printed precision"
    elif pool["c_xgb"] < fm_x:
        pooled_clause = "On this run it sits below the fold mean"
    else:
        pooled_clause = "On this run it sits above the fold mean"
    # Mirrors save_model_bundle's tie-break (aft on equality), so the report
    # names the same model the saved sidecar records as recommended.
    rec_model = "Cox baseline" if fm_c > fm_x else "boosted model"
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
        "shap": m["shap_top"],
        "cfg": cfg,
        "wg": m.get("within_group"),
        "tu": tu,
        "tu1": unit_singular(tu),
        "tua": tua,
        "hs": hs,
        "cal": m[f"calibration_{hs}{tua}"],
        "cal_cox": m.get(f"calibration_cox_{hs}{tua}"),
        "cox_top": m.get("cox_top"),
        "has_oracle": "c_oracle" in pool,
        "has_sharpe": "c_sharpe" in pool,
        "unit": "strategy" if g else "row",
        "units": "strategies" if g else "rows",
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
        "pooled_clause": pooled_clause,
        "rec_model": rec_model,
    }


def _sec_summary(c: dict, doc: ReportDoc) -> None:
    pool, folds = c["pool"], c["folds"]
    body = f"""<p>This report evaluates two survival models fitted to {c["source_desc"]}.
It holds {c["n_rows"]:,} {c["units"]}, each observed from its start
date. {pct(c["d"]["event_rate"])} have observed endings and
{pct(c["censored_overall"])} are censored, still running when observation
stopped. Median observed duration, censored
{c["units"]} included, is
{c["d"]["median_observed_duration_days"]:.0f} {c["tu"]}. The models predict,
from what was on file at the start date, how long each {c["unit"]}
survives.</p>

<p>The like-for-like comparison is the fold mean. Each of the {len(folds)}
temporal folds trains both models on one stretch of history and tests them
on the next, and the {len(folds)} scores average. On concordance, the
share of pairs a ranking orders correctly
where 0.500 is a coin flip, XGBoost AFT scores
{pool["c_xgb_by_fold_mean"]:.3f} and the Cox proportional hazards baseline
{pool["c_cox_by_fold_mean"]:.3f}. {c["winner_clause"]}.</p>

<p>Pooled over {pool["n_test"]:,} out-of-time test {c["units"]}, the AFT
model scores {pool["c_xgb"]:.3f} (95% bootstrap interval
{pool["c_xgb_ci"][0]:.3f} to {pool["c_xgb_ci"][1]:.3f}). Section
@sec:results explains how the two figures differ.</p>"""
    if c["run"]:
        body += "\n" + _pk(
            "bundle",
            f"<p>Both models are saved in one bundle, which records the"
            f" {c['rec_model']} as recommended for scoring new rows.</p>",
        )
    if c["wg"]:
        wg = c["wg"]
        # Whether group membership dominates is a property of the run, so the
        # lead-in is computed, not asserted (same rule as pooled_clause). The
        # middle branch would misdescribe both ends: a run whose group means
        # sit near a coin flip has essentially no group effect to split with.
        if wg["c_group_mean"] >= pool["c_xgb"]:
            wg_lead = (
                f"Most of the pooled figure reflects a {c['unit']}'s <code>{wg['col']}</code> group"
            )
        elif wg["c_group_mean"] - 0.5 < 0.25 * (pool["c_xgb"] - 0.5):
            wg_lead = f"Little of the pooled figure is <code>{wg['col']}</code> group membership"
        else:
            wg_lead = (
                f"The pooled figure splits between <code>{wg['col']}</code>"
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
            f"""<p>{wg_lead}. Ranking {c["units"]} by their group's average
prediction alone scores {wg["c_group_mean"]:.3f}, and comparing only
{c["units"]} inside the same group scores {wg["c_within"]:.3f},
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
            f"""<p>{lose_text}. A no-skill forecast assigns every {c["unit"]} the same
population-average probability. The limitations section says what remains
usable.</p>""",
        )
    if c["g"]:
        body += "\n" + _pk(
            "synthetic-callout",
            '<p class="callout">All results are synthetic. The run validates the'
            " pipeline and asserts nothing about live data.</p>",
        )
    else:
        body += "\n" + _pk(
            "no-oracle-callout",
            '<p class="callout">This dataset has no known generating process, so'
            " there is no known best-achievable score to compare these against,"
            " and feature attributions cannot be checked against a true"
            " mechanism.</p>",
        )
    doc.section("summary", "Summary", body)


def _sec_data(c: dict, doc: ReportDoc) -> None:
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
            f"""<p>Left truncation, where a {c["unit"]} was already running when the source's
records begin, is a data fault. Its recorded start is not its true start and
nothing marks it. This pipeline has no delayed-entry handling, so those
{c["units"]} must be excluded during preparation. Whether they were
excluded is recorded in the dataset's own documentation.</p>""",
        )
    if c["g"]:
        g = c["g"]
        body += "\n" + _pk(
            "generator-data",
            f"""<p>{pct(g["admin_censor_rate"], 0)} of {c["units"]} are
administratively retired independent of performance, and survival time is
log-normal in the latents at scale {g["log_time_sigma"]}.</p>""",
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
            f" {c['units']} counted for as long as they were observed. Groups"
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
    p, cfg, folds = c["p"], c["cfg"], c["folds"]
    body = f"""<h3>@sec:method.1 Model class</h3>

<p>The pipeline fits two models on every run. The first is a boosted-tree
accelerated failure time (AFT) model, XGBoost with its
<code>survival:aft</code> objective, built for durations with incomplete
observations. An observed ending tells the model the exact lifetime, and a
censored {c["unit"]} tells it only "at least this long", so every
{c["unit"]} contributes. It predicts each {c["unit"]}'s median survival
time in {c["tu"]}, and a log-normal curve around that median, whose width
is fitted once and shared by every {c["unit"]}, gives the probability of
surviving any horizon. The second is a Cox proportional hazards baseline, the
standard linear survival model, fitted with lifelines'
<code>CoxPHFitter</code> on the same features. It answers whether the
boosted model was necessary. The run's recommended model is whichever
scores the higher fold-mean concordance.</p>

<p>The two differ in what they assume. The AFT model assumes every
{c["unit"]} follows the same survival curve run on a faster or slower
clock, so features change when things happen, never the curve's shape. The
Cox model makes no assumption about that shape at all,
reading the curve from the data by counting who was still running at each
age. In exchange it assumes each feature multiplies risk by the same
factor at every age, which this report does not test. Which set of
assumptions suits a dataset cannot be known in advance, which is why both
are fitted and scored.</p>

<p>Numeric features pass through, missing values included (the Cox baseline
gets train-window median imputation). Text columns are one-hot encoded with
the vocabulary refit per training window, so a fold's features reflect only
what was on file by its split date.</p>

<h3>@sec:method.2 Temporal validation and label re-censoring</h3>

<p>Evaluation uses {len(folds)} expanding-window folds ordered by start
date: the earliest {pct(cfg["min_train_frac"], 0)} of {c["units"]} is
burn-in, trained on but never tested, and each fold trains on every
{c["unit"]} started before its split date and tests on the next block
(sizes in Table @tab:folds).</p>

<p>Training labels are re-censored at each split date. A {c["unit"]}
started long before a split may have died after it, and its label
contains that future, so every post-split death is rewritten as a censoring
at the split. Omitting this raises scores by importing the future, which
makes it the most consequential detail in the pipeline. It is a standalone
function (<code>temporal_folds.recensor</code>) with dedicated tests.</p>

<h3>@sec:method.3 Selection and calibration</h3>

<p>The boosted model's hyperparameters are selected once on the first
fold's training window by an inner temporal split, the same
past-then-future cut made inside that window. Concordance grades only
whether {c["units"]} are ordered correctly, and a model can order them well
while drawing survival curves far too wide or too narrow. So selection is
scored on held-out censored log-likelihood instead, which grades the whole
predicted distribution against what was observed.</p>

<p>The predictive scale is the width of the boosted model's log-normal
curve, one unitless number shared by every {c["unit"]}. A larger scale
spreads the model's probability over a wider range of lifetimes. Training
used {p["aft_sigma"]}, a setting of the loss that shapes how the medians
are fitted. Then, with the medians held fixed, the width that best matches
held-out outcomes on the training window's most recent stretch was
measured at {p["predictive_sigma_final"]:.2f} and carried to the model
refitted on the full window. The two answer different questions, so they
need not agree. Every probability the boosted model reports here
uses the measured value.</p>"""
    if not c["g"]:
        body += "\n" + _pk(
            "validated-elsewhere",
            "<p>The methodology is validated separately against synthetic data"
            " with known ground truth.</p>",
        )
    doc.section("method", "Method", body)


def _sec_results(c: dict, doc: ReportDoc) -> None:
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
        f" {c['units']} with 95% percentile bootstrap intervals over"
        f" {cfg['n_bootstrap']} resamples. Higher is better, and 0.500 is a"
        f" coin flip. Fold-mean rows score each of the {len(folds)} folds"
        " separately and average. The interval resamples test rows with the"
        " fitted models held fixed, so it measures scoring precision, not"
        " stability across history. The fold spread is the guide to that.",
        "<tr><th>Method</th><th>Concordance (Harrell's C)</th><th>95% interval</th></tr>",
        conc_rows,
    )

    # Naming one window the weakest is only worth doing when it is separated
    # from the next one at the precision this report prints. On a near-tie the
    # choice of which to name is arbitrary, and pointing at a cause for it
    # invites an explanation of noise.
    ranked = sorted(folds, key=lambda f: f["c_xgb"])
    worst = ranked[0]
    worst_idx = folds.index(worst) + 1
    near_tie = len(ranked) > 1 and ranked[1]["c_xgb"] - worst["c_xgb"] < 0.0015
    if near_tie:
        # Listed in fold order, values paired to match, so the pair does not
        # read as a transposition.
        pair = sorted(
            ((worst_idx, worst), (folds.index(ranked[1]) + 1, ranked[1])), key=lambda p: p[0]
        )
        weakest_sentence = (
            f"Folds {pair[0][0]} and {pair[1][0]} are the boosted model's weakest, at "
            f"{pair[0][1]['c_xgb']:.3f} and {pair[1][1]['c_xgb']:.3f}, too close to separate."
        )
    else:
        weakest_sentence = (
            f"The boosted model's weakest window is fold {worst_idx}, starting "
            f"{worst['split_date']}, at {worst['c_xgb']:.3f}. When a cause has been "
            "established, the run's notes say so."
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
        f" {c['units']} whose ending was not observed."
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

<p>Each fold fits its own models. The boosted model predicts a survival
time in {c["tu"]}, and {c["tu"]} mean the same thing in every fold, so its
predictions can be pooled into one list. A Cox score is a risk relative to
the other {c["units"]} in its own fit, so Cox scores from different folds
cannot go in one list. The two models are therefore compared on fold
means, each fold scored on its own and the {len(folds)} scores
averaged.</p>

<p>The pooled row answers a different question. It scores all test
{c["units"]} in one list, which is enough data to attach an uncertainty
range, and that range is the interval beside it. It is not the number for
comparing models, because its {c["units"]} were scored by {len(folds)}
different fitted models. {c["pooled_clause"]}.</p>

<p>{weakest_sentence}</p>"""
    if c["wg"]:
        wg = c["wg"]
        body += "\n" + _pk(
            "within-group-results",
            f"""<p>The pooled concordance decomposes by <code>{wg["col"]}</code>.
Ranking {c["units"]} by their group's average gives every {c["unit"]} in a
group the same score, so comparisons only ever run between groups, and
those come out right {pct(wg["c_group_mean"])} of the time. The boosted
model scores each {c["unit"]} individually instead, which orders
{c["units"]} inside a group and can also move one past {c["units"]} in
other groups. Inside a group it is right {pct(wg["c_within"])} of the
time, averaged over the {wg["n_groups"]} groups big enough to score (at
least {wg["min_n"]} {c["units"]} and {wg["min_events"]} observed endings),
each weighted by its number of comparable pairs.</p>""",
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

<p>The Brier score, the mean squared error of a probability forecast
(lower is better), is weighted by the inverse probability of censoring
(IPCW) so censored {c["units"]} do not bias it. The no-skill reference
assigns every {c["unit"]} one population-wide probability, the
Kaplan-Meier marginal, the fraction of the whole population surviving at
each age with censored {c["units"]} counted while observed.</p>"""
    brier_rows = "\n".join(
        f"<tr><td>{h.removesuffix(c['tua'])} {c['tu']}</td><td>{v['xgb']:.3f}</td>"
        f"<td>{v['cox']:.3f}</td><td>{v['km_marginal']:.3f}</td></tr>"
        for h, v in brier.items()
    )
    tab_brier = doc.table(
        "brier",
        "IPCW Brier score by horizon. Lower is better. The weighting works"
        f" by counting each {c['unit']} still observed at a horizon, weighted"
        " up by one over the probability that observation lasted that long,"
        f" so the {c['units']} censoring removed are represented by those it"
        " spared.",
        "<tr><th>Horizon</th><th>AFT</th><th>Cox</th>\n    <th>No-skill marginal</th></tr>",
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
        f" within each bin, so censored {c['units']} contribute correctly."
        f" {cal_worst}",
    )
    cal_rows = "\n".join(
        f"<tr><td>{i + 1}</td><td>{b['n']}</td><td>{b['predicted']:.3f}</td>"
        f"<td>{b['observed_km']:.3f}</td>"
        f"<td>{b['observed_km'] - b['predicted']:+.3f}</td></tr>"
        for i, b in enumerate(cal)
    )
    tab_cal = doc.table(
        "calibration",
        f"Decile calibration at {c['hs']} {c['tu']} for the boosted AFT model,"
        + (" the values plotted as its series in" if c["cal_cox"] else " the values plotted in")
        + " Figure @fig:calibration. Deciles are cut on predicted value, so tied"
        " predictions make the counts uneven. The observed value in each is a"
        f" Kaplan-Meier estimate. When a decile's last observed {c['unit']}"
        f" falls short of the {c['hs']}-{c['tu1']} horizon the estimate"
        " carries its last value forward, so in small or heavily censored"
        " deciles the observed column carries more uncertainty than its"
        " three decimals suggest.",
        "<tr><th>Decile</th><th>n</th><th>Predicted</th><th>Observed (KM)</th>\n"
        "    <th>Deviation</th></tr>",
        cal_rows,
    )
    body += f"""

{tab_brier}

{fig_cal}

{tab_cal}"""
    doc.section("results", "Results", body)


def _sec_model_uses(c: dict, doc: ReportDoc) -> None:
    shap = c["shap"]
    shap_rows = "\n".join(
        f"<tr><td>{s['feature']}</td><td>{s['mean_abs_shap']:.3f}</td></tr>" for s in shap[:8]
    )
    tab_shap = doc.table(
        "shap-top",
        "Top eight features by mean absolute attribution.",
        "<tr><th>Feature</th><th>Mean |attribution|</th></tr>",
        shap_rows,
    )
    fig_bar = doc.figure(
        "shap-bar",
        img_uri(c["figures_dir"], "shap_bar.png"),
        "Mean absolute SHAP by feature",
        "Mean absolute attribution by feature.",
    )
    fig_bee = doc.figure(
        "beeswarm",
        img_uri(c["figures_dir"], "shap_beeswarm.png"),
        "SHAP beeswarm across the explanation sample",
        "Per-row attributions across the explanation sample. For a yes/no"
        " feature, such as a one-hot category flag, the high (red) end simply"
        " means the row is in that category.",
    )
    fig_dep = doc.figure(
        "dependence",
        img_uri(c["figures_dir"], "shap_dependence.png"),
        "SHAP dependence plots",
        "Attribution against feature value for the strongest numeric"
        " features. Category flags carry no shape and stay in the ranking"
        " above. A run short on numeric features fills the grid with flags"
        " instead.",
    )
    body = f"""<h3>@sec:model-uses.1 The boosted model</h3>

<p>Feature attributions (SHAP values, for SHapley Additive exPlanations)
are computed for the boosted model on the log scale of survival time. A
+0.3 attribution multiplies predicted survival by about 1.35, and negative
values shorten it.</p>

<p>Attributions are descriptive and in-sample, computed on the final
boosted model's own training rows. Correlated features split credit by the model's
internal choices as much as by the data, so directions are more trustworthy
than magnitudes.</p>

<p>Figure @fig:shap-bar ranks features, Table @tab:shap-top lists the top
eight, Figure @fig:beeswarm shows the per-{c["unit"]} spread, and
Figure @fig:dependence traces attribution against value.</p>

{fig_bar}

{tab_shap}

{fig_bee}

{fig_dep}"""
    # A generator run needs no pointer here: its notes check the ranking
    # against the installed mechanism in the section directly below this one.
    if not (c["g"] and c["has_oracle"]):
        body += "\n" + _pk(
            "no-mechanism",
            "<p>On this data there is no known mechanism to check the ranking"
            ' against, so read it as "what this model leaned on", not as'
            " importance in the world.</p>",
        )
    if c["cox_top"]:
        hr_rows = "\n".join(
            f"<tr><td>{r['feature']}</td><td>{r['hr']:.3f}</td>"
            f"<td>{r['hr_lo']:.3f} to {r['hr_hi']:.3f}</td><td>{r['z']:+.1f}</td></tr>"
            for r in c["cox_top"][:8]
        )
        # The two caveats are glossed here rather than in the prose above for
        # the same reason as the KM caption: this is where a reader meets the
        # one-hot ratios, and a caption costs no template words.
        tab_hr = doc.table(
            "cox-hr",
            "Top eight Cox covariates by |z|. A ratio above 1 raises the"
            " hazard and shortens survival. A one-hot flag's ratio compares"
            " its category against the column's reference level, the one"
            " category dropped from the Cox fit so the remaining flags stay"
            " independent. The ridge penalty that stabilizes the fit shrinks"
            " coefficients toward zero, so the 95% intervals are"
            " approximate.",
            "<tr><th>Feature</th><th>Hazard ratio</th><th>95% interval</th>\n    <th>z</th></tr>",
            hr_rows,
        )
        fig_hr = doc.figure(
            "cox-hr",
            img_uri(c["figures_dir"], "cox_hr.png"),
            "Cox hazard ratios with 95% intervals",
            "Hazard ratios for the strongest Cox covariates on a log axis, so"
            " a doubling and a halving of risk sit the same distance from the"
            " dashed no-effect line at 1.0.",
        )
        body += "\n" + _pk(
            "cox-uses",
            f"""<h3>@sec:model-uses.2 The Cox baseline</h3>

<p>The Cox baseline's drivers are its coefficients, reported as hazard
ratios. A hazard ratio is the factor by which one unit of a feature
multiplies the hazard, the risk of ending at a given age. The factor is
the same at every age, so a ratio above 1 shortens survival, the opposite
reading from the attributions above. Figure @fig:cox-hr plots the
strongest covariates, ranked by |z|, the coefficient over its standard
error, and Table @tab:cox-hr lists the top eight.</p>

{fig_hr}

{tab_hr}""",
        )
    doc.section("model-uses", "What the models use", body)


def _sec_limitations(c: dict, doc: ReportDoc) -> None:
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
quality are separate, so ordering {c["units"]} is still supported at those
horizons.</p>""",
            )
        )
    parts.append(f"""<p><strong>Censoring may be informative.</strong> The evaluation assumes
{c["units"]} stop being observed for reasons unrelated to their risk. Early
exits of {c["units"]} about to end anyway bias survival estimates
upward.</p>""")
    if c["notes"].get("limitations"):
        parts.append(_marked_note(c["notes"]["limitations"]))
    parts.append(f"""<p><strong>The boosted model gives every {c["unit"]} one curve
shape.</strong> The predictive scale is a single fitted number, so the
model runs a {c["unit"]}'s clock faster or slower but never changes the
curve's shape. Per-{c["unit"]} width is future work.</p>""")
    parts.append(f"""<p><strong>Harrell's concordance is biased under heavy censoring,
usually upward.</strong> It scores only the pairs censoring leaves
comparable, which under heavy censoring over-represent short observed
lifetimes. The most censored test window is
{pct(c["max_fold_censored"], 0)} censored, so read its rows in
Table @tab:folds with the most doubt.</p>""")
    doc.section("limitations", "Limitations", "\n\n".join(parts))


def _sec_repro(c: dict, doc: ReportDoc) -> None:
    body = f"""<p>Python 3.11 or later. From the repository root:</p>

<pre><code>{c["command"]}</code></pre>

<p>Every number is read from the run's
<code>metrics.json</code> at build time, and a figure the prose never cites
fails the build. <code>requirements.txt</code> pins the exact versions the
committed numbers came from.</p>"""
    doc.section("repro", "Reproducing this run", body)


def _sec_addendum(c: dict, doc: ReportDoc) -> None:
    pool = c["pool"]
    body = f"""<p>Effects are installed as fixed shifts on log survival time.
Nothing in the generator is proprietary, and the constants are in the run's
notes.</p>

<p>The oracle ranking orders strategies by the latent log-time the
generator actually used. Nothing is fitted and the ranking predates the
noise draw, so its {pool["c_oracle"]:.3f} bounds every model. The gap to a
perfect score is installed noise. A suite test asserts the model never
outscores it, since beating perfect information indicates a leak.</p>"""
    doc.addendum("synthetic-truth", "Synthetic ground truth", body)


# The seam between generated template prose and authored prose must be
# visible: a reader otherwise credits the pipeline with the author's
# interpretation, or the author with the template's claims.
NOTEMARK = '<p class="notemark">From the run\'s notes, written by the analyst.</p>\n'


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
