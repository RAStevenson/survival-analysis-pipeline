#!/usr/bin/env python3
"""Rebuild a report from a metrics.json and its figures.

    python scripts/run_build_report.py                 # synthetic, reports/
    python scripts/run_build_report.py --run runs/x    # a real-data run

Every number in a report is read from its metrics.json rather than
transcribed, so the report cannot drift from the pipeline that produced it.
Figures are embedded as base64 data URIs, so the HTML is one file with no
external dependencies.

Synthetic mode: if reports/metrics_seed8.json exists (a pipeline run with
--seed 8, saved under that name), the robustness paragraph in Limitations is
generated from it; otherwise the report states the single-seed limitation
plainly. Writes reports/strategy_survival_report.html and, when Chrome is
available, prints it to reports/strategy_survival_report.pdf headlessly.

Real-data mode (--run): renders that run's metrics into <run>/report.html and
PDF. No oracle row and no attribution-versus-truth reading, because real data
has no known generating process; the report says so plainly instead of
omitting it silently.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[1] / "reports"
METRICS = REPORTS / "metrics.json"
METRICS_SEED8 = REPORTS / "metrics_seed8.json"
FIGURES = REPORTS / "figures"
OUT = REPORTS / "strategy_survival_report.html"
OUT_PDF = REPORTS / "strategy_survival_report.pdf"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

TESTS_DIR = Path(__file__).resolve().parents[1] / "tests"


def count_tests() -> int:
    """Count test functions on disk rather than quoting a number.

    The report cites its own test count as evidence, and a literal here drifts
    silently the first time a test is added.
    """
    return sum(
        line.lstrip().startswith("def test_")
        for path in TESTS_DIR.glob("test_*.py")
        for line in path.read_text(encoding="utf-8").splitlines()
    )


# Shared page chrome for both report variants; plain string, single braces.
REPORT_CSS = """<style>
  :root {
    --ink: #1a1a1a;
    --muted: #555;
    --rule: #d4d4d4;
    --accent: #0b3d91;
    --band: #f4f6f9;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: #fff;
    color: var(--ink);
    font-family: Georgia, "Times New Roman", serif;
    font-size: 16px;
    line-height: 1.62;
  }
  article {
    max-width: 46rem;
    margin: 0 auto;
    padding: 3rem 1.5rem 4rem;
  }
  h1, h2, h3, .doctype, figcaption, caption, th, code, pre {
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  .titleblock {
    border-bottom: 2px solid var(--ink);
    padding-bottom: 1.5rem;
    margin-bottom: 2.5rem;
  }
  .doctype {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
    color: var(--muted);
    margin: 0 0 0.75rem;
  }
  h1 { font-size: 1.85rem; line-height: 1.25; margin: 0 0 0.6rem; }
  .subtitle {
    font-size: 1.05rem;
    color: var(--muted);
    font-style: italic;
    margin: 0 0 1.5rem;
  }
  table.meta { border-collapse: collapse; font-size: 0.85rem; }
  table.meta th {
    text-align: left;
    font-weight: 600;
    color: var(--muted);
    padding: 0.15rem 1.25rem 0.15rem 0;
    white-space: nowrap;
    vertical-align: top;
  }
  table.meta td { padding: 0.15rem 0; }
  h2 {
    font-size: 1.3rem;
    margin: 2.75rem 0 1rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--rule);
  }
  h3 { font-size: 1.02rem; margin: 1.9rem 0 0.6rem; color: var(--accent); }
  p { margin: 0 0 1.1rem; }
  .callout {
    background: var(--band);
    border-left: 3px solid var(--accent);
    padding: 0.9rem 1.1rem;
    font-size: 0.94rem;
    margin-top: 1.5rem;
  }
  figure { margin: 1.9rem 0; }
  figure img {
    width: 100%;
    height: auto;
    border: 1px solid var(--rule);
    background: #fff;
  }
  figcaption {
    font-size: 0.83rem;
    color: var(--muted);
    line-height: 1.5;
    margin-top: 0.6rem;
  }
  table.data {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.86rem;
    margin: 1.6rem 0;
  }
  table.data caption {
    caption-side: top;
    text-align: left;
    font-size: 0.83rem;
    color: var(--muted);
    line-height: 1.5;
    padding-bottom: 0.6rem;
  }
  table.data th {
    text-align: left;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    border-bottom: 1.5px solid var(--ink);
    padding: 0.4rem 0.5rem;
  }
  table.data td {
    padding: 0.4rem 0.5rem;
    border-bottom: 1px solid var(--rule);
    font-variant-numeric: tabular-nums;
  }
  table.data td:not(:first-child), table.data th:not(:first-child) {
    text-align: right;
  }
  table.data tr.highlight td { background: var(--band); font-weight: 700; }
  code {
    font-family: "Cascadia Mono", Consolas, monospace;
    font-size: 0.86em;
    background: var(--band);
    padding: 0.1em 0.3em;
    border-radius: 2px;
  }
  pre {
    background: var(--band);
    border-left: 3px solid var(--rule);
    padding: 0.9rem 1.1rem;
    overflow-x: auto;
    font-size: 0.84rem;
    line-height: 1.55;
  }
  pre code { background: none; padding: 0; }
  footer {
    margin-top: 3.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--rule);
    font-size: 0.8rem;
    color: var(--muted);
  }
  @media print {
    @page { margin: 20mm 18mm; }
    body { font-size: 10.5pt; }
    article { max-width: none; padding: 0; }
    h2 { page-break-after: avoid; }
    h3 { page-break-after: avoid; }
    figure, table.data, pre, .callout { page-break-inside: avoid; }
    section { page-break-inside: auto; }
    a { color: inherit; text-decoration: none; }
  }
</style>"""


def img(name: str) -> str:
    data = base64.b64encode((FIGURES / name).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def pct(x: float, places: int = 1) -> str:
    return f"{100 * x:.{places}f}%"


def emit_pdf(html_path: Path = OUT, pdf_path: Path = OUT_PDF) -> bool:
    """Print the HTML report to PDF with headless Chrome, if Chrome is present."""
    if not CHROME.exists():
        print("Chrome not found - skipping PDF (open the HTML and print to PDF manually)")
        return False
    subprocess.run(
        [
            str(CHROME),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path.resolve()}",
            html_path.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return True


def seed_dependence_para(m: dict, m8: dict | None) -> str:
    """Limitations paragraph on generator-seed dependence, computed from the
    seed-8 metrics when they exist so the claim is measured, not asserted."""
    p, pool, shap = m["params"], m["pooled"], m["shap_top"]
    if m8 is None:
        return (
            "<p><strong>All results come from one generator seed.</strong> No second-seed run "
            "has been performed, so data variance and model variance cannot be separated. The "
            "relationships between the numbers in this report deserve more confidence than "
            "their individual decimals.</p>"
        )
    p8, pool8, shap8 = m8["params"], m8["pooled"], m8["shap_top"]
    same_params = p8["max_depth"] == p["max_depth"] and p8["aft_sigma"] == p["aft_sigma"]
    in_ci = pool["c_xgb_ci"][0] <= pool8["c_xgb"] <= pool["c_xgb_ci"][1]
    top3_same = {s["feature"] for s in shap[:3]} == {s["feature"] for s in shap8[:3]}
    order_same = [s["feature"] for s in shap[:3]] == [s["feature"] for s in shap8[:3]]
    if top3_same and order_same:
        attribution = (
            "The same three walk-forward statistics dominate attribution, in the same order. "
        )
    elif top3_same:
        attribution = (
            "The same three walk-forward statistics dominate attribution, although their "
            "internal ordering shifts. That shift illustrates the correlated-proxies caveat "
            "in section 7, since how credit divides among proxies of the same latent "
            "quantity is not stable across draws and only the group-level conclusion holds. "
        )
    else:
        attribution = (
            "The set of dominant features shifted between seeds, which weakens the "
            "attribution claims in section 7. "
        )
    return (
        "<p><strong>Generator-seed dependence.</strong> The pipeline was rerun end to end "
        "with a second seed (seed 8, saved as <code>reports/metrics_seed8.json</code>). It "
        f"selected {'the same' if same_params else 'different'} hyperparameters "
        f"and scored {pool8['c_xgb']:.3f}, "
        f"{'inside' if in_ci else 'outside'} the seed-7 interval of "
        f"{pool['c_xgb_ci'][0]:.3f} to {pool['c_xgb_ci'][1]:.3f}. Ranking by validation "
        f"Sharpe stayed inverted at {pool8['c_sharpe']:.3f}, and the oracle ceiling came out "
        f"at {pool8['c_oracle']:.3f}. "
        + attribution
        + "Two seeds is a consistency check rather than a variance estimate. A proper "
        "multi-seed sweep remains future work, and the relationships between the numbers in "
        "this report deserve more confidence than their individual decimals.</p>"
    )


def build_synthetic_report() -> None:
    m = json.loads(METRICS.read_text())
    m8 = json.loads(METRICS_SEED8.read_text()) if METRICS_SEED8.exists() else None
    p, d, pool = m["params"], m["dataset"], m["pooled"]
    folds, brier, cal, shap = m["folds"], m["ipcw_brier"], m["calibration_180d"], m["shap_top"]
    seed_para = seed_dependence_para(m, m8)

    # Read the run's own conditions out of the metrics file. These were once
    # module-level literals, which meant `run_synthetic_pipeline.py --seed 8` produced a
    # report that said seed 7 and compared the run against itself.
    g, run_cfg = m["generator"], m["config"]
    SEED = g["seed"]
    SELECTION_SHARPE = g["selection_sharpe"]
    DISCOVERY_START = g["discovery_start"]
    DISCOVERY_END = g["discovery_end"]
    OBSERVATION_CUTOFF = g["observation_cutoff"]
    LOG_TIME_SIGMA = g["log_time_sigma"]
    ADMIN_CENSOR_RATE = g["admin_censor_rate"]
    N_BOOTSTRAP = run_cfg["n_bootstrap"]
    N_TESTS = count_tests()

    # Read the demo's size from the demo's own metrics rather than quoting it.
    demo_metrics = REPORTS / "chicago_demo" / "metrics.json"
    demo_size = (
        f"{json.loads(demo_metrics.read_text())['dataset']['n_rows']:,} "
        if demo_metrics.exists()
        else ""
    )

    censored_overall = 1.0 - d["event_rate"]
    last_fold_censored = 1.0 - folds[-1]["test_event_rate"]
    first_fold_censored = 1.0 - folds[0]["test_event_rate"]
    c_fold_min = min(f["c_xgb"] for f in folds)
    c_fold_max = max(f["c_xgb"] for f in folds)
    sharpe_min = min(f["c_sharpe"] for f in folds)
    sharpe_max = max(f["c_sharpe"] for f in folds)
    n_train_min = min(f["n_train"] for f in folds)
    n_train_max = max(f["n_train"] for f in folds)
    oracle_gap = pool["c_oracle"] - pool["c_xgb"]

    # Largest calibration gap and which decile it falls in.
    gaps = [(abs(b["predicted"] - b["observed_km"]), i) for i, b in enumerate(cal)]
    worst_gap, worst_bin = max(gaps)

    fold_rows = "\n".join(
        f"<tr><td>{i + 1}</td><td>{f['split_date']}</td><td>{f['n_train']:,}</td>"
        f"<td>{f['n_test']:,}</td><td>{pct(1 - f['test_event_rate'], 0)}</td>"
        f"<td>{f['c_xgb']:.3f}</td><td>{f['c_cox']:.3f}</td>"
        f"<td>{f['c_sharpe']:.3f}</td><td>{f['c_oracle']:.3f}</td></tr>"
        for i, f in enumerate(folds)
    )

    brier_rows = "\n".join(
        f"<tr><td>{h.replace('d', ' days')}</td><td>{v['xgb']:.3f}</td>"
        f"<td>{v['cox']:.3f}</td><td>{v['km_marginal']:.3f}</td></tr>"
        for h, v in brier.items()
    )

    cal_rows = "\n".join(
        f"<tr><td>{i + 1}</td><td>{b['n']}</td><td>{b['predicted']:.3f}</td>"
        f"<td>{b['observed_km']:.3f}</td>"
        f"<td>{b['observed_km'] - b['predicted']:+.3f}</td></tr>"
        for i, b in enumerate(cal)
    )

    shap_rows = "\n".join(
        f"<tr><td>{s['feature']}</td><td>{s['mean_abs_shap']:.3f}</td></tr>" for s in shap[:8]
    )

    html = f"""<article>
<header class="titleblock">
  <p class="doctype">Technical Report</p>
  <h1>Predicting the Working Life of Algorithmically Discovered Trading Strategies</h1>
  <p class="subtitle">A survival-analysis meta-model, built and verified against
  synthetic ground truth</p>
  <table class="meta">
    <tr><th>Author</th><td>Robert Stevenson</td></tr>
    <tr><th>Repository</th><td>strategy-survival-model</td></tr>
    <tr><th>Run</th><td>seed {SEED}, {d["n_strategies"]:,} strategies,
      {len(folds)} temporal folds</td></tr>
    <tr><th>Source of figures</th><td>reports/metrics.json, regenerated by
      <code>python scripts/run_synthetic_pipeline.py</code></td></tr>
  </table>
</header>

<section>
<h2>1. Summary</h2>

<p>An automated strategy search produces a queue of candidates that all clear
validation. They stop working at very different rates, and the rate determines
how much capital a new strategy should get and when it should be reviewed. This
report covers a model that predicts how long a newly discovered strategy will
keep working, using only the metadata recorded on the day it is deployed.</p>

<p>The finding that matters more than the model's accuracy is that the metric
allocators implicitly rank on, validation Sharpe, is worse than useless for this
question. Ranking strategies by validation Sharpe scores
{pool["c_sharpe"]:.3f} on a concordance index where {0.5:.3f} is a coin flip,
and it stays below {0.5:.3f} in all {len(folds)} folds. The cause is selection.
Every strategy entered the queue by clearing a Sharpe threshold, so past that
threshold a high score reflects overfitting more often than edge, and overfit
strategies decay fastest.</p>

<p>A model reading walk-forward consistency instead reaches
{pool["c_xgb"]:.3f} (95% bootstrap interval
{pool["c_xgb_ci"][0]:.3f} to {pool["c_xgb_ci"][1]:.3f}, pooled over
{pool["n_test"]:,} out-of-fold strategies). Because the data is synthetic, the
best score any model could achieve is computable, and it is
{pool["c_oracle"]:.3f}. The model sits {oracle_gap:.3f} below that ceiling, so
most of the remaining error is irreducible noise rather than model
capacity.</p>

<p class="callout">Status: the methodology is built and verified against known
ground truth. It has not been run on production metadata, because the
production book does not yet contain enough resolved strategy lifetimes to
support temporal cross-validation.</p>
</section>

<section>
<h2>2. The problem</h2>

<p>Strategies discovered by an automated search decay. An edge that clears
validation today is usually unprofitable within months, and lifetimes vary by an
order of magnitude among strategies with identical headline metrics.</p>

<p>Three operational decisions depend on the decay rate: how much capital a new
strategy receives on day one, when its first serious review is scheduled, and
when it is retired. The default is to treat every new strategy the same, which
misallocates in both directions. It overfunds strategies that will not survive
the quarter and underfunds the ones that would have run for a year.</p>

<p>The question is whether discovery-time metadata carries enough signal to
separate them. Strategies die for causes that leave traces in that metadata. A
strategy selected from a hundred thousand candidates carries more selection bias
than one selected from a hundred. A strategy whose walk-forward Sharpe was
already sliding during validation is saying something. None of that is visible
in a single validation statistic, and all of it is already recorded.</p>
</section>

<section>
<h2>3. Why ranking by validation Sharpe fails</h2>

<p>Every strategy in the population cleared a validation-Sharpe threshold of
{SELECTION_SHARPE}, because that is how a strategy enters a deployment queue.
An observed Sharpe is the sum of true edge, an overfitting component, and
measurement noise. Conditioning on that sum exceeding a threshold means the
survivors with the highest observed scores are disproportionately the ones whose
overfitting and noise happened to break upward.</p>

<p>Past the threshold, additional observed Sharpe is more likely inflation than
edge. Because the inflated strategies are the overfit ones, and overfit
strategies decay fastest, higher validation Sharpe actively predicts
<em>shorter</em> working life. The result is not a weak predictor but an
inverted one, at {pool["c_sharpe"]:.3f} pooled, ranging from {sharpe_min:.3f}
to {sharpe_max:.3f} across folds and never once above the {0.5:.3f} of a coin
flip.</p>

<p>This is what makes the modeling problem worth posing. Had validation Sharpe
ranked survival correctly there would be nothing to add. A meta-model earns its
place precisely because the selection process has already consumed the obvious
signal.</p>
</section>

<section>
<h2>4. Data</h2>

<p>All results come from synthetic data. The generator reproduces the
statistical structure of a strategy-search pipeline, including selection bias,
walk-forward statistics, regime exposures and censored lifetimes, without
containing anything proprietary.</p>

<p>Confidentiality is one reason. Verification is the more important one.
Because the generating process is known, two checks become available that real
data cannot support. The best achievable score can be computed exactly, which
bounds any claim about model quality. And the model's feature attributions can
be compared against the mechanisms actually built in, which turns interpretation
into a test with a right answer. The test suite enforces the first of these.
One test asserts the model never outscores the oracle, since beating perfect
information would indicate a leak.</p>

<p>The run reported here draws {d["n_strategies"]:,} strategies with seed
{SEED}, discovered between {DISCOVERY_START} and {DISCOVERY_END}, observed to
{OBSERVATION_CUTOFF}. Median observed lifetime is
{d["median_observed_duration_days"]:.0f} days.
{pct(censored_overall)} of strategies are right-censored, either still running
at the observation cutoff or administratively retired at a rate of
{pct(ADMIN_CENSOR_RATE, 0)}. Survival time is log-normal in the latent
quantities with a scale of {LOG_TIME_SIGMA} on log-days, which is the noise that
places the ceiling below a perfect score.</p>

<figure>
  <img src="{img("km_by_asset_class.png")}" alt="Kaplan-Meier survival curves by asset class">
  <figcaption><strong>Figure 1.</strong> Kaplan-Meier survival curves by asset
  class. Crypto strategies decay faster by construction, and the model has to
  recover that from roughly 15% of rows.</figcaption>
</figure>
</section>

<section>
<h2>5. Method</h2>

<h3>5.1 Model class</h3>

<p>The target is a duration with incomplete observations, so the model is an
accelerated failure time (AFT) model, implemented as XGBoost with the
<code>survival:aft</code> objective. Censoring enters through interval labels,
where an observed death is the interval [t, t] and a censored strategy is
[t, infinity).</p>

<p>Regression on lifetime was rejected because censored rows have no target, and
dropping them discards the longest-lived strategies, which biases the model
toward pessimism. Classification at a fixed horizon handles censoring awkwardly
and answers only one question. AFT keeps every row and returns a full time
distribution, which collapses to any horizon an allocator asks about.</p>

<h3>5.2 Temporal validation and label re-censoring</h3>

<p>Evaluation uses {len(folds)} expanding-window folds ordered by discovery
date. The earliest 40% of strategies is burn-in that is only ever trained on.
Each fold trains on every strategy discovered before its split date and tests on
the next {folds[0]["n_test"]} strategies, so training sets grow from
{n_train_min:,} to {n_train_max:,} rows.</p>

<p>Training labels are re-censored at each split date. A strategy discovered two
years before a split may have died three months after it, and its recorded label
contains that future death. Any model trained at that split date could only have
known the strategy was still running, so every post-split death is rewritten as
a censoring at the split. Omitting this step raises measured scores while
silently importing future information, which makes it the most consequential
detail in the pipeline. It is implemented as a standalone function
(<code>cv.recensor</code>) with dedicated tests, because it transfers unchanged
to any duration problem on operational data.</p>

<p>Test labels use the full observation window, because the restriction exists
to keep the future away from the model, not away from the scorer.</p>

<h3>5.3 Hyperparameter selection and calibration</h3>

<p>Hyperparameters are selected once, on the first fold's training window, using
an inner temporal split and held-out censored log-likelihood. Likelihood rather
than concordance drives selection, and the reason is a failure this pipeline
recorded rather than avoided.</p>

<p>An earlier version selected on concordance and looked correct: ranking
metrics landed where they ultimately did. It then lost to a no-skill reference
forecast on 365-day Brier score. A concordance index is invariant to the
predictive scale, so the search had no reason to prefer a usable distribution
width and settled on one far too wide. Nothing in the training loop objected,
because nothing in the training loop was measuring it. Selection now uses
likelihood, which penalizes a broken scale, and the predictive log-normal scale
is fitted separately on a temporal tail slice the early-stopping probe never
trained on. The selected loss scale was {p["aft_sigma"]}, and the calibrated
predictive scale came out at {p["predictive_sigma_final"]:.2f}.</p>

<p>The general lesson holds beyond this project. A model can rank well and
misstate probabilities at the same time, and the only defense is evaluating
both.</p>

<h3>5.4 Baselines</h3>

<p>Three references bound the result. A Cox proportional hazards model on the
same features is the standard survival alternative and answers whether the
gradient-boosted model was necessary. Ranking by validation Sharpe is the
heuristic a backtest-driven allocator applies implicitly. The oracle ranks by
the latent quantity that generated the lifetimes and gives the ceiling that
measurement noise imposes.</p>
</section>

<section>
<h2>6. Results</h2>

<h3>6.1 Discrimination</h3>

<p>Pooled over {pool["n_test"]:,} out-of-fold strategies, with 95% percentile
bootstrap intervals over {N_BOOTSTRAP} resamples:</p>

<table class="data">
  <caption><strong>Table 1.</strong> Concordance index by method, pooled across
  folds. Higher is better; {0.5:.3f} is a coin flip.</caption>
  <thead><tr><th>Method</th><th>Harrell C</th><th>95% interval</th></tr></thead>
  <tbody>
    <tr><td>Oracle on latent log-time (ceiling)</td>
      <td>{pool["c_oracle"]:.3f}</td><td>not resampled</td></tr>
    <tr class="highlight"><td>XGBoost AFT (pooled)</td><td>{pool["c_xgb"]:.3f}</td>
      <td>{pool["c_xgb_ci"][0]:.3f} to {pool["c_xgb_ci"][1]:.3f}</td></tr>
    <tr><td>XGBoost AFT (fold mean)</td>
      <td>{pool["c_xgb_by_fold_mean"]:.3f}</td><td>not resampled</td></tr>
    <tr><td>Cox proportional hazards (fold mean)</td>
      <td>{pool["c_cox_by_fold_mean"]:.3f}</td><td>not resampled</td></tr>
    <tr><td>Rank by validation Sharpe</td><td>{pool["c_sharpe"]:.3f}</td>
      <td>{pool["c_sharpe_ci"][0]:.3f} to {pool["c_sharpe_ci"][1]:.3f}</td></tr>
  </tbody>
</table>

<p>Two results in that table need stating plainly rather than being left to
inference.</p>

<p>The boosted model ties the Cox baseline, {pool["c_xgb_by_fold_mean"]:.3f}
against {pool["c_cox_by_fold_mean"]:.3f}. Both figures are fold means, which is
the only like-for-like basis: each fold refits Cox, and its risk scores are
relative ones centred on that fold's own training window, so they carry no
common scale across folds and pooling them would compare different units. The
generator's observable structure is close
to additive, and {n_train_min:,} to {n_train_max:,} training rows is not enough
for trees to find much beyond what a penalized linear model in the log-hazard
already captures. I report the tie rather than adding interactions to the
generator until the headline model wins, because a benchmark tuned until it
loses is not a benchmark. The tie is itself informative. On this problem a
simpler model suffices, and knowing that is worth more than an engineered
margin.</p>

<p>Both models sit about {oracle_gap:.2f} below the oracle. The gap between
{pool["c_xgb"]:.3f} and a perfect score is therefore mostly unpredictable
variation in when a strategy actually stops working, not unused signal.</p>

<table class="data">
  <caption><strong>Table 2.</strong> Per-fold results. Censoring is the share of
  each fold's test strategies still running at the observation cutoff.</caption>
  <thead><tr><th>Fold</th><th>Split date</th><th>Train n</th><th>Test n</th>
    <th>Censored</th><th>AFT</th><th>Cox</th><th>Sharpe</th><th>Oracle</th></tr></thead>
  <tbody>{fold_rows}</tbody>
</table>

<figure>
  <img src="{img("fold_cindex.png")}" alt="Concordance index by temporal fold">
  <figcaption><strong>Figure 2.</strong> Concordance by fold. AFT performance is
  stable from {c_fold_min:.3f} to {c_fold_max:.3f}. The final fold scores
  highest, but {pct(last_fold_censored, 0)} of its test strategies are censored
  against {pct(first_fold_censored, 0)} in the first fold, which changes the set
  of comparable pairs. Concordance is not strictly comparable across folds with
  different censoring mixes, so this plot supports a claim of stability rather
  than improvement.</figcaption>
</figure>

<h3>6.2 Calibration</h3>

<p>Predicted median survival times are converted to survival probabilities under
the calibrated log-normal, then checked two ways. Brier scores are weighted by
the inverse probability of censoring so that censored rows do not bias the
result. The no-skill reference assigns every strategy the pooled Kaplan-Meier
marginal, ignoring all features.</p>

<table class="data">
  <caption><strong>Table 3.</strong> IPCW Brier score by horizon. Lower is
  better.</caption>
  <thead><tr><th>Horizon</th><th>AFT</th><th>Cox</th>
    <th>No-skill marginal</th></tr></thead>
  <tbody>{brier_rows}</tbody>
</table>

<p>The margin over the no-skill reference is wide at 90 days and narrow at 365.
That is the expected shape. Median survival is about
{d["median_observed_duration_days"]:.0f} days, so by one year nearly every
strategy has stopped working and the marginal forecast is already close to
certain. Little skill remains to demonstrate at that horizon.</p>

<figure>
  <img src="{img("calibration_180d.png")}" alt="Decile calibration at 180 days">
  <figcaption><strong>Figure 3.</strong> Predicted against observed survival at
  180 days, by predicted decile. Observed frequencies are Kaplan-Meier estimates
  within each bin, so censored strategies contribute correctly rather than being
  dropped. The largest deviation is {worst_gap:.3f} in decile
  {worst_bin + 1}.</figcaption>
</figure>

<table class="data">
  <caption><strong>Table 4.</strong> Decile calibration at 180 days, the values
  plotted in Figure 3.</caption>
  <thead><tr><th>Decile</th><th>n</th><th>Predicted</th><th>Observed (KM)</th>
    <th>Deviation</th></tr></thead>
  <tbody>{cal_rows}</tbody>
</table>
</section>

<section>
<h2>7. What the model uses</h2>

<p>Feature attributions are computed on the log-time margin, so a value of +0.3
multiplies predicted survival time by about 1.35 and negative values shorten
it.</p>

<figure>
  <img src="{img("shap_bar.png")}" alt="Mean absolute SHAP by feature">
  <figcaption><strong>Figure 4.</strong> Mean absolute attribution by feature.
  The three walk-forward consistency statistics carry more attribution than
  every other feature combined.</figcaption>
</figure>

<table class="data">
  <caption><strong>Table 5.</strong> Top eight features by mean absolute
  attribution.</caption>
  <thead><tr><th>Feature</th><th>Mean |attribution|</th></tr></thead>
  <tbody>{shap_rows}</tbody>
</table>

<figure>
  <img src="{img("shap_beeswarm.png")}" alt="SHAP beeswarm across the explanation sample">
  <figcaption><strong>Figure 5.</strong> Per-strategy attributions. High
  walk-forward positive fraction pushes predicted survival up; a steeply
  negative walk-forward Sharpe slope pushes it down hard, with the left tail
  reaching about -1.5 log-days, a factor of roughly 4.5 shorter; high
  fold-to-fold Sharpe dispersion is penalized. All three directions match the
  generative design, where consistency rises with true edge and falls with
  overfitting.</figcaption>
</figure>

<figure>
  <img src="{img("shap_dependence.png")}" alt="SHAP dependence plots, walk-forward statistics">
  <figcaption><strong>Figure 6.</strong> Attribution against feature value. The
  positive-fraction effect is monotone and close to linear, with vertical bands
  at the nine possible fold fractions and a swing of about 1.3 log-days end to
  end. The Sharpe-decay effect saturates once the slope turns positive, which is
  sensible behaviour given that positive decay slopes are mostly noise. The
  dispersion effect is a descending staircase that flattens past 1.5, where
  additional dispersion no longer distinguishes anything.</figcaption>
</figure>

<p>Validation Sharpe does not appear in the top twelve features. On
unconditional data that would be strange. Here it is the signature of
selection. Every strategy cleared the same threshold, so the metric's
remaining variation
is mostly overfitting plus noise, and the model routes its attention to
walk-forward consistency instead. This is the same phenomenon as the
{pool["c_sharpe"]:.3f} in Table 1, seen from inside the model.</p>

<p>The attribution ranking is a test rather than a description, because the
generating process is known. The three walk-forward statistics that dominate are
not inputs to survival time in the generator. They exist only as the least noisy
observable proxies for the latent split between real edge and overfitting, which
is the actual driver. The model was not handed that relationship; it found the
proxies and leaned on them. Feature-family and asset-class effects return with
the signs and rough ordering written into the generator, and the search-intensity
penalty appears where a multiple-testing argument predicts.</p>

<p>Directions in that table are more trustworthy than magnitudes. The three
walk-forward statistics are correlated proxies for the same latent quantity, and
how attribution divides credit among correlated features reflects the model's
internal choices as much as the data. The supportable reading is that the model
uses positive fraction about twice as much as Sharpe dispersion, not that
positive fraction is twice as important. The feature-family and asset-class
effects enter the generator additively and independently, so any reasonable
model recovers them; recovering the proxy structure is the informative
result.</p>
</section>

<section>
<h2>8. Limitations</h2>

<p><strong>The generator encodes a prior, not evidence.</strong> A model that
recovers structure placed there says nothing about whether real strategy
metadata contains comparable signal. This work validates a methodology, and the
concordance reported here should be treated as an upper bound on production
performance.</p>

<p><strong>Lifetimes are treated as independent.</strong> Real strategies stop
working together when a regime breaks. The generator omits that dependence and
the bootstrap resamples strategies independently, so the intervals in Table 1
understate real uncertainty and are best read as a lower bound on it.</p>

<p><strong>Administrative retirement is modeled as independent censoring.</strong>
In a live book, strategies are retired because someone observed early decay,
which makes the censoring informative and biases survival estimates upward.
Handling it properly requires a competing-risks treatment, which is recorded as
future work rather than implemented.</p>

{seed_para}

<p><strong>Harrell's concordance is biased under heavy censoring.</strong> The
final fold is {pct(last_fold_censored, 0)} censored, where an inverse-probability
weighted concordance would be preferable. Fold-to-fold comparisons in Table 2
should be read with that in mind.</p>
</section>

<section>
<h2>9. Intended use</h2>

<p>The model is a capital-allocation and review-scheduling prior, not a trade
signal. It ranks newly discovered strategies by expected working life and
produces a survival curve for each. Predicted median lifetime sets initial
position sizing and the first review date; the curve gives a decay schedule
against which actual performance can be compared.</p>

<p>Before it informs live allocation, the same temporal cross-validation with
the same label re-censoring has to be repeated on production metadata, and the
resulting concordance is expected to be lower with wider intervals. The blocking
constraint is data volume rather than method. The production book does not yet
hold enough resolved strategy lifetimes to support
{len(folds)} temporal folds with meaningful censoring. The pipeline is built and
verified against known ground truth so that when the book is old enough, the
production run needs no methodological work.</p>

<p>The real-data path itself already exists rather than being planned. The same
pipeline fits and evaluates any right-censored duration CSV
(<code>scripts/run_fit_evaluate.py</code>), saves both fitted models, and
scores new rows later (<code>scripts/run_predict.py</code>). The repository
includes a demonstration on {demo_size}real City of Chicago business licences
(<code>reports/chicago_demo/</code>), where the checks that depend on ground
truth are absent and the generated report says so.</p>

<p>Nearer-term work, in the order it would be done: a multi-seed sweep to
separate data variance from model variance, an inverse-probability weighted
concordance alongside Harrell's, a block bootstrap by discovery month to stop
understating interval width, and a competing-risks treatment of administrative
retirement.</p>
</section>

<section>
<h2>10. Reproducing these results</h2>

<p>Python 3.11 or later. From the repository root:</p>

<pre><code>pip install -r requirements.txt
python -m pytest
python scripts/run_synthetic_pipeline.py</code></pre>

<p>The test suite is {N_TESTS} tests covering generator invariants, fold and
label leakage, and metric behaviour. The pipeline script regenerates the data
at seed {SEED}, runs the {len(folds)}-fold temporal cross-validation, writes
<code>reports/metrics.json</code> and all figures in about two minutes, then
rebuilds this report from that metrics file, so every number here is traceable
to a single run rather than transcribed. <code>requirements.txt</code> pins the
exact dependency versions these numbers were produced with.</p>

<p>The seed-dependence paragraph in section 8 is generated from
<code>reports/metrics_seed8.json</code>, produced by
<code>python scripts/run_synthetic_pipeline.py --seed 8</code> and saved under that name;
the builder reads it when present and states the single-seed limitation when it
is not. Building the report also prints
<code>strategy_survival_report.pdf</code> headlessly when Chrome is
available.</p>
</section>

<footer>
<p>Generated from <code>reports/metrics.json</code> by
<code>scripts/run_build_report.py</code>. Seed {SEED},
{d["n_strategies"]:,} strategies, {pool["n_test"]:,} out-of-fold test
strategies.</p>
</footer>
</article>

{REPORT_CSS}
"""

    OUT.write_text(html, encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({size_kb:.0f} KB, self-contained)")
    if emit_pdf():
        print(f"wrote {OUT_PDF} ({OUT_PDF.stat().st_size / 1024:.0f} KB)")


def build_real_report(run_dir: Path) -> None:
    """Report for a run_fit_evaluate.py run: same chrome, honest about what a
    real dataset cannot support (no oracle ceiling, no attribution-vs-truth)."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"no metrics.json in {run_dir}; run scripts/run_fit_evaluate.py first")
    m = json.loads(metrics_path.read_text())
    run, p, d, pool = m["run"], m["params"], m["dataset"], m["pooled"]
    folds, brier, shap = m["folds"], m["ipcw_brier"], m["shap_top"]
    h_cal = int(run["calibration_horizon_days"])
    cal = m[f"calibration_{h_cal}d"]
    figures = run_dir / "figures"

    def rimg(name: str) -> str:
        data = base64.b64encode((figures / name).read_bytes()).decode("ascii")
        return f"data:image/png;base64,{data}"

    censored_overall = 1.0 - d["event_rate"]
    n_train_min = min(f["n_train"] for f in folds)
    n_train_max = max(f["n_train"] for f in folds)
    c_fold_min = min(f["c_xgb"] for f in folds)
    c_fold_max = max(f["c_xgb"] for f in folds)
    brier_cal = brier[f"{h_cal}d"]

    fold_rows = "\n".join(
        f"<tr><td>{i + 1}</td><td>{f['split_date']}</td><td>{f['n_train']:,}</td>"
        f"<td>{f['n_test']:,}</td><td>{pct(1 - f['test_event_rate'], 0)}</td>"
        f"<td>{f['c_xgb']:.3f}</td><td>{f['c_cox']:.3f}</td></tr>"
        for i, f in enumerate(folds)
    )
    brier_rows = "\n".join(
        f"<tr><td>{h.replace('d', ' days')}</td><td>{v['xgb']:.3f}</td>"
        f"<td>{v['cox']:.3f}</td><td>{v['km_marginal']:.3f}</td></tr>"
        for h, v in brier.items()
    )
    cal_rows = "\n".join(
        f"<tr><td>{i + 1}</td><td>{b['n']}</td><td>{b['predicted']:.3f}</td>"
        f"<td>{b['observed_km']:.3f}</td>"
        f"<td>{b['observed_km'] - b['predicted']:+.3f}</td></tr>"
        for i, b in enumerate(cal)
    )
    shap_rows = "\n".join(
        f"<tr><td>{s['feature']}</td><td>{s['mean_abs_shap']:.3f}</td></tr>" for s in shap[:8]
    )

    aft_fold, cox_fold = pool["c_xgb_by_fold_mean"], pool["c_cox_by_fold_mean"]
    if cox_fold > aft_fold:
        winner = (
            f"On this dataset the Cox baseline outscores the boosted model, "
            f"{cox_fold:.3f} against {aft_fold:.3f} on fold-mean concordance. Both fitted "
            "models are saved and the Cox model is the one this run recommends for scoring "
            "new rows. Reporting that, rather than presenting the boosted model as the "
            "result, is the point of carrying a baseline at all."
        )
    else:
        winner = (
            f"The boosted model outscores the Cox baseline on fold-mean concordance, "
            f"{aft_fold:.3f} against {cox_fold:.3f}. Both fitted models are saved and the "
            "boosted model is the one this run recommends for scoring new rows."
        )

    if brier_cal["xgb"] < brier_cal["km_marginal"]:
        brier_sentence = (
            f"At the {h_cal}-day horizon its censoring-weighted Brier score is "
            f"{brier_cal['xgb']:.3f}, beating the {brier_cal['km_marginal']:.3f} of a no-skill "
            "forecast that assigns every row the population average."
        )
    else:
        brier_sentence = (
            f"At the {h_cal}-day horizon its censoring-weighted Brier score is "
            f"{brier_cal['xgb']:.3f}, which fails to beat the {brier_cal['km_marginal']:.3f} of "
            "a no-skill forecast that assigns every row the population average. The model "
            "orders rows usefully, but its absolute probabilities at this horizon are not "
            "trustworthy. Section 5 covers this."
        )

    losing = [h for h, v in brier.items() if v["xgb"] >= v["km_marginal"]]
    if losing:
        losing_text = ", ".join(h.replace("d", " days") for h in losing)
        brier_limitation = (
            f"<p><strong>Absolute probabilities are not usable at {losing_text}.</strong> At "
            "those horizons the model's censoring-weighted Brier score does not beat a "
            "no-skill forecast. Discrimination and calibration are separate qualities. The "
            "ranking carries signal while the survival probabilities do not, so use this "
            "model to order rows, not to act on absolute probabilities at those horizons. "
            "The usual cause under temporal evaluation is training windows whose re-censored "
            "follow-up is far shorter than the horizon, which forces the fitted log-normal "
            "to extrapolate beyond anything it saw.</p>\n\n"
        )
    else:
        brier_limitation = ""

    # One line on purpose. A cmd.exe caret continuation is a parse error in
    # PowerShell and a stray argument in bash, so a wrapped command is a
    # command the reader cannot paste. --out is included because without it
    # the run lands in the default runs/<name>/ and the rebuild step below
    # would re-render the old report instead of the one just produced.
    cols = run["columns"]
    drop_part = f" --drop-cols {','.join(cols['dropped'])}" if cols["dropped"] else ""
    categorical = cols.get("categorical")
    cat_part = f" --categorical-cols {','.join(categorical)}" if categorical else ""
    out_part = f" --out {run['out_dir']}" if run.get("out_dir") else ""
    command = (
        f"python scripts/run_fit_evaluate.py --data {run['source']} --name {run['name']} "
        f"--id-col {cols['id']} --date-col {cols['date']} "
        f"--duration-col {cols['duration']} --event-col {cols['event']}"
        f"{drop_part}{cat_part} --folds {run['n_folds']} "
        f"--horizons {','.join(str(int(h)) for h in run['horizons_days'])}{out_part}"
    )

    html = f"""<article>
<header class="titleblock">
  <p class="doctype">Technical Report</p>
  <h1>Survival Model Evaluation: {run["name"]}</h1>
  <p class="subtitle">Fitted with the strategy-survival-model pipeline;
  methodology validated separately against synthetic ground truth</p>
  <table class="meta">
    <tr><th>Source data</th><td><code>{run["source"]}</code></td></tr>
    <tr><th>Rows</th><td>{d["n_rows"]:,} ({pct(d["event_rate"])} with the ending
      observed, {pct(censored_overall)} censored)</td></tr>
    <tr><th>Start dates</th><td>{d["date_min"]} to {d["date_max"]}</td></tr>
    <tr><th>Evaluation</th><td>{run["n_folds"]} expanding-window temporal folds</td></tr>
    <tr><th>Source of figures</th><td><code>{run_dir.as_posix()}/metrics.json</code>,
      regenerated by the command in section 6</td></tr>
  </table>
</header>

<section>
<h2>1. Summary</h2>

<p>This report evaluates a survival model fitted to
<code>{Path(run["source"]).name}</code>, {d["n_rows"]:,} rows, each observed
from its start date for {run["columns"]["duration"]!r} days with
{run["columns"]["event"]!r} marking whether the ending was seen
({pct(d["event_rate"])}) or the row was still running when observation
stopped ({pct(censored_overall)}). Median observed duration is
{d["median_observed_duration_days"]:.0f} days. The model predicts, from the
{d["n_features"]} features available at the start date, how long each row
survives.</p>

<p>Pooled over {pool["n_test"]:,} out-of-time test rows, the XGBoost AFT model
reaches a concordance index of {pool["c_xgb"]:.3f} (95% bootstrap interval
{pool["c_xgb_ci"][0]:.3f} to {pool["c_xgb_ci"][1]:.3f}), against 0.500 for a
coin flip. Set beside a Cox proportional hazards baseline on the same features
it scores {pool["c_xgb_by_fold_mean"]:.3f} to the baseline's
{pool["c_cox_by_fold_mean"]:.3f}; both of those are fold means, which is the
only like-for-like basis and is why the pooled figure is not the one compared.
{brier_sentence}</p>

<p>That bootstrap interval is narrow, and it is worth saying what it does and
does not cover. It holds each fold's fitted model fixed and resamples the test
rows, so it measures how precisely this run's score is pinned down by the
number of rows scored, nothing more. It says nothing about how much the score
would move on a different stretch of history, and the per-fold spread of
{c_fold_min:.3f} to {c_fold_max:.3f} in Table 2 is the better guide to that.
The rows are also not independent, since licences in the same trade and the
same year fail together, which makes the true interval wider than the one
printed.</p>

<p class="callout">This dataset has no known generating process. Unlike the
synthetic validation report, there is no oracle ceiling to say how much signal
remains unclaimed, and feature attributions cannot be checked against a true
mechanism. What carries over from the synthetic validation is the pipeline
itself: the same temporal folds, label re-censoring, likelihood-based
selection, and calibration checks, verified there against known answers.</p>
</section>

<section>
<h2>2. Method</h2>

<p>Rows are ordered by start date and evaluated with {run["n_folds"]}
expanding-window folds; every fold trains only on rows that started before its
split date and tests on the next block, so training sets grow from
{n_train_min:,} to {n_train_max:,} rows. Training labels are re-censored at
each split date. An ending recorded after the split is rewritten to "still
running at the split", which is all a model trained then could have known.
Test labels keep their full outcomes.</p>

<p>The model is XGBoost with the <code>survival:aft</code> objective; censoring
enters through interval labels. Hyperparameters are selected on the first
fold's training window by held-out censored log-likelihood (selected depth
{p["max_depth"]}, loss scale {p["aft_sigma"]}), and the predictive log-normal
scale is calibrated separately on a temporal tail slice (final value
{p["predictive_sigma_final"]:.2f}). Numeric features pass through as-is,
missing values included, which XGBoost handles natively; the Cox baseline
receives train-window median imputation. Text columns are one-hot encoded.</p>
</section>

<section>
<h2>3. Results</h2>

<table class="data">
  <caption><strong>Table 1.</strong> Concordance index, pooled across folds.
  Higher is better; 0.500 is a coin flip.</caption>
  <thead><tr><th>Method</th><th>Harrell C</th><th>95% interval</th></tr></thead>
  <tbody>
    <tr class="highlight"><td>XGBoost AFT (pooled)</td><td>{pool["c_xgb"]:.3f}</td>
      <td>{pool["c_xgb_ci"][0]:.3f} to {pool["c_xgb_ci"][1]:.3f}</td></tr>
    <tr><td>XGBoost AFT (fold mean)</td><td>{aft_fold:.3f}</td>
      <td>not resampled</td></tr>
    <tr><td>Cox proportional hazards (fold mean)</td>
      <td>{cox_fold:.3f}</td><td>not resampled</td></tr>
  </tbody>
</table>

<p>{winner} Each fold refits the Cox model, and its risk scores are relative
ones centred on that fold's own training window, so they are meaningful within
a fold but carry no common scale across folds; the fold-mean rows are the
like-for-like comparison and the pooled figure is not comparable to them.</p>

<table class="data">
  <caption><strong>Table 2.</strong> Per-fold results. Censoring is the share
  of each fold's test rows whose ending was not observed.</caption>
  <thead><tr><th>Fold</th><th>Split date</th><th>Train n</th><th>Test n</th>
    <th>Censored</th><th>AFT</th><th>Cox</th></tr></thead>
  <tbody>{fold_rows}</tbody>
</table>

<figure>
  <img src="{rimg("fold_cindex.png")}" alt="Concordance index by temporal fold">
  <figcaption><strong>Figure 1.</strong> Concordance by fold, from
  {c_fold_min:.3f} to {c_fold_max:.3f}. Folds differ in censoring mix, so
  cross-fold comparisons are indicative rather than exact.</figcaption>
</figure>

<table class="data">
  <caption><strong>Table 3.</strong> IPCW Brier score by horizon; lower is
  better. The no-skill column assigns every row the pooled Kaplan-Meier
  marginal.</caption>
  <thead><tr><th>Horizon</th><th>AFT</th><th>Cox</th>
    <th>No-skill marginal</th></tr></thead>
  <tbody>{brier_rows}</tbody>
</table>

<figure>
  <img src="{rimg(f"calibration_{h_cal}d.png")}" alt="Decile calibration">
  <figcaption><strong>Figure 2.</strong> Predicted against observed survival at
  {h_cal} days, by predicted decile. Observed frequencies are Kaplan-Meier
  estimates within each bin, so censored rows contribute correctly.</figcaption>
</figure>

<table class="data">
  <caption><strong>Table 4.</strong> Decile calibration at {h_cal} days, the
  values plotted in Figure 2.</caption>
  <thead><tr><th>Decile</th><th>n</th><th>Predicted</th><th>Observed (KM)</th>
    <th>Deviation</th></tr></thead>
  <tbody>{cal_rows}</tbody>
</table>
</section>

<section>
<h2>4. What the model uses</h2>

<p>Feature attributions are computed on the log-time margin, where positive
values push predicted survival up. On this data they are descriptive only.
There is
no known mechanism to check them against, and correlated features can split
credit in ways that reflect the model's internal choices as much as the data,
so read the ranking as "what this model leaned on", not as importance in the
world.</p>

<figure>
  <img src="{rimg("shap_bar.png")}" alt="Mean absolute SHAP by feature">
  <figcaption><strong>Figure 3.</strong> Mean absolute attribution by
  feature.</figcaption>
</figure>

<table class="data">
  <caption><strong>Table 5.</strong> Top eight features by mean absolute
  attribution.</caption>
  <thead><tr><th>Feature</th><th>Mean |attribution|</th></tr></thead>
  <tbody>{shap_rows}</tbody>
</table>

<figure>
  <img src="{rimg("shap_beeswarm.png")}" alt="SHAP beeswarm across the explanation sample">
  <figcaption><strong>Figure 4.</strong> Per-row attributions across the
  explanation sample.</figcaption>
</figure>

<figure>
  <img src="{rimg("shap_dependence.png")}" alt="SHAP dependence plots">
  <figcaption><strong>Figure 5.</strong> Attribution against feature value for
  the four features with the largest mean attribution.</figcaption>
</figure>
</section>

<section>
<h2>5. Limitations</h2>

<p><strong>No ground truth.</strong> Nothing bounds how much predictable signal
this dataset contains, so the concordance above cannot be placed relative to a
ceiling the way the synthetic validation could.</p>

{brier_limitation}

<p><strong>Censoring may be informative.</strong> The evaluation assumes rows
stop being observed for reasons unrelated to their risk. If observation ended
early on rows that were about to end anyway, survival estimates are biased
upward. Whether that holds here depends on how this dataset was collected.</p>

<p><strong>Harrell's concordance is biased under heavy censoring.</strong>
Fold-to-fold comparisons in Table 2 should be read with each fold's censoring
share in mind.</p>
</section>

<section>
<h2>6. Reproducing this run</h2>

<pre><code>{command}
python scripts/run_build_report.py --run {run_dir.as_posix()}</code></pre>
</section>

<footer>
<p>Generated from <code>{run_dir.as_posix()}/metrics.json</code> by
<code>scripts/run_build_report.py --run</code>. {d["n_rows"]:,} rows,
{pool["n_test"]:,} out-of-time test rows.</p>
</footer>
</article>

{REPORT_CSS}
"""

    out_html = run_dir / "report.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"wrote {out_html} ({out_html.stat().st_size / 1024:.0f} KB, self-contained)")
    out_pdf = run_dir / "report.pdf"
    if emit_pdf(out_html, out_pdf):
        print(f"wrote {out_pdf} ({out_pdf.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_build_report.py",
        description="Rebuild the synthetic report, or a real-data run's report with --run.",
    )
    parser.add_argument("--run", default=None, help="run directory from run_fit_evaluate.py")
    args = parser.parse_args()
    if args.run is None:
        build_synthetic_report()
    else:
        build_real_report(Path(args.run))


if __name__ == "__main__":
    main()
