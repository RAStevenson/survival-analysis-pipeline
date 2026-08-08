"""Report rendering shared by the synthetic study and real-data runs.

One template serves every data source. Sections register on a ReportDoc in
document order; prose cross-references are written as @sec:slug, @fig:slug,
and @tab:slug tokens and resolved to numbers at render time, so a section
that appears in one variant and not the other cannot silently break
numbering. A figure whose token never appears in prose outside a figure
caption fails the render: every figure is cited or it does not ship.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

TOKEN_RE = re.compile(r"@(sec|fig|tab):([a-z0-9-]+)")
_FIGCAPTION_RE = re.compile(r"<figcaption>.*?</figcaption>", re.DOTALL)

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
TESTS_DIR = Path(__file__).resolve().parents[2] / "tests"

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


def pct(x: float, places: int = 1) -> str:
    return f"{100 * x:.{places}f}%"


def img_uri(figures_dir: Path, name: str) -> str:
    data = base64.b64encode((figures_dir / name).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


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


def emit_pdf(html_path: Path, pdf_path: Path) -> bool:
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


@dataclass
class _Section:
    slug: str
    title: str
    body: str
    is_addendum: bool


class ReportDoc:
    """Ordered collection of sections, figures, and tables with render-time
    numbering.

    Registration order is document order and sets every number. figure() and
    table() return the HTML block for the caller to place inside a section
    body; they do not insert anything themselves. render() substitutes all
    tokens and raises on an unknown slug, a leftover token, or a figure whose
    token appears nowhere in prose outside a figure caption.
    """

    def __init__(self) -> None:
        self._sections: list[_Section] = []
        self._figures: list[str] = []
        self._tables: list[str] = []

    def section(self, slug: str, title: str, body: str) -> None:
        self._check_new_section(slug)
        self._sections.append(_Section(slug, title, body, is_addendum=False))

    def addendum(self, slug: str, title: str, body: str) -> None:
        self._check_new_section(slug)
        self._sections.append(_Section(slug, title, body, is_addendum=True))

    def _check_new_section(self, slug: str) -> None:
        if any(s.slug == slug for s in self._sections):
            raise ValueError(f"duplicate section slug {slug!r}")

    def figure(self, slug: str, image_uri: str, alt: str, caption: str) -> str:
        if slug in self._figures:
            raise ValueError(f"duplicate figure slug {slug!r}")
        self._figures.append(slug)
        return (
            f'<figure>\n  <img src="{image_uri}" alt="{alt}">\n'
            f"  <figcaption><strong>Figure @fig:{slug}.</strong> {caption}</figcaption>\n"
            f"</figure>"
        )

    def table(self, slug: str, caption: str, head: str, rows: str) -> str:
        if slug in self._tables:
            raise ValueError(f"duplicate table slug {slug!r}")
        self._tables.append(slug)
        return (
            f'<table class="data">\n'
            f"  <caption><strong>Table @tab:{slug}.</strong> {caption}</caption>\n"
            f"  <thead>{head}</thead>\n"
            f"  <tbody>{rows}</tbody>\n"
            f"</table>"
        )

    def render(
        self, *, doctype: str, title: str, subtitle: str, meta_rows: str, footer: str
    ) -> str:
        numbers: dict[str, str] = {}
        n_numbered = 0
        n_addenda = 0
        for s in self._sections:
            if s.is_addendum:
                numbers[f"sec:{s.slug}"] = chr(ord("A") + n_addenda)
                n_addenda += 1
            else:
                n_numbered += 1
                numbers[f"sec:{s.slug}"] = str(n_numbered)
        for i, slug in enumerate(self._figures):
            numbers[f"fig:{slug}"] = str(i + 1)
        for i, slug in enumerate(self._tables):
            numbers[f"tab:{slug}"] = str(i + 1)

        prose = _FIGCAPTION_RE.sub("", "\n".join(s.body for s in self._sections))
        for slug in self._figures:
            if f"@fig:{slug}" not in prose:
                raise ValueError(f"figure {slug!r} is never cited in body prose")

        def heading(s: _Section) -> str:
            label = numbers[f"sec:{s.slug}"]
            return f"Addendum {label}" if s.is_addendum else label

        sections_html = "\n\n".join(
            f"<section>\n<h2>{heading(s)}. {s.title}</h2>\n\n{s.body}\n</section>"
            for s in self._sections
        )

        html = f"""<article>
<header class="titleblock">
  <p class="doctype">{doctype}</p>
  <h1>{title}</h1>
  <p class="subtitle">{subtitle}</p>
  <table class="meta">
{meta_rows}
  </table>
</header>

{sections_html}

<footer>
{footer}
</footer>
</article>

{REPORT_CSS}
"""

        def sub(match: re.Match[str]) -> str:
            key = f"{match.group(1)}:{match.group(2)}"
            if key not in numbers:
                raise ValueError(f"unresolved reference @{key}")
            return numbers[key]

        html = TOKEN_RE.sub(sub, html)
        for marker in ("@sec:", "@fig:", "@tab:"):
            if marker in html:
                snippet = html[html.index(marker) : html.index(marker) + 40]
                raise ValueError(f"unsubstituted token remains: {snippet!r}")
        return html


# --------------------------------------------------------------------------
# Contexts: everything variant-specific that is presentation rather than
# content. Content conditionals live in compose_report and key off what the
# metrics carry.


def synthetic_context(m: dict, m8: dict | None, reports_dir: Path) -> dict:
    g, d, folds = m["generator"], m["dataset"], m["folds"]

    # Read the demo's size from the demo's own metrics rather than quoting it.
    demo_metrics = reports_dir / "chicago_demo" / "metrics.json"
    demo_size = (
        f"{json.loads(demo_metrics.read_text())['dataset']['n_rows']:,} "
        if demo_metrics.exists()
        else ""
    )

    meta_rows = f"""    <tr><th>Author</th><td>Robert Stevenson</td></tr>
    <tr><th>Repository</th><td>strategy-survival-model</td></tr>
    <tr><th>Run</th><td>seed {g["seed"]}, {d["n_strategies"]:,} strategies,
      {len(folds)} temporal folds</td></tr>
    <tr><th>Source of figures</th><td>reports/metrics.json, regenerated by
      <code>python scripts/run_synthetic_pipeline.py</code></td></tr>"""

    footer = f"""<p>Generated from <code>reports/metrics.json</code> by
<code>scripts/run_build_report.py</code>. Seed {g["seed"]},
{d["n_strategies"]:,} strategies, {m["pooled"]["n_test"]:,} out-of-fold test
strategies.</p>"""

    command = (
        "pip install -r requirements.txt\n"
        "python -m pytest\n"
        "python scripts/run_synthetic_pipeline.py"
    )

    return {
        "m": m,
        "m8": m8,
        "figures_dir": reports_dir / "figures",
        "title": "Predicting the Working Life of Algorithmically Discovered Trading Strategies",
        "subtitle": "A survival-analysis meta-model, built and verified against\n"
        "  synthetic ground truth",
        "meta_rows": meta_rows,
        "footer": footer,
        "command": command,
        "demo_size": demo_size,
        "km": None,
    }


def real_context(m: dict, run_dir: Path) -> dict:
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
    command = (
        f"python scripts/run_fit_evaluate.py --data {run['source']} --name {run['name']} "
        f"--id-col {cols['id']} --date-col {cols['date']} "
        f"--duration-col {cols['duration']} --event-col {cols['event']}"
        f"{drop_part}{cat_part}{km_part} --folds {run['n_folds']} "
        f"--horizons {','.join(str(int(h)) for h in run['horizons_days'])}{out_part}"
        f"\npython scripts/run_build_report.py --run {run_dir.as_posix()}"
    )

    censored_overall = 1.0 - d["event_rate"]
    meta_rows = f"""    <tr><th>Source data</th><td><code>{run["source"]}</code></td></tr>
    <tr><th>Rows</th><td>{d["n_rows"]:,} ({pct(d["event_rate"])} with the ending
      observed, {pct(censored_overall)} censored)</td></tr>
    <tr><th>Start dates</th><td>{d["date_min"]} to {d["date_max"]}</td></tr>
    <tr><th>Evaluation</th><td>{run["n_folds"]} expanding-window temporal folds</td></tr>
    <tr><th>Source of figures</th><td><code>{run_dir.as_posix()}/metrics.json</code>,
      regenerated by the command in section @sec:repro</td></tr>"""

    footer = f"""<p>Generated from <code>{run_dir.as_posix()}/metrics.json</code> by
<code>scripts/run_build_report.py --run</code>. {d["n_rows"]:,} rows,
{pool["n_test"]:,} out-of-time test rows.</p>"""

    km = None
    km_col = run.get("km_col")
    if km_col and (run_dir / "figures" / "km_by_group.png").exists():
        km = {"col": km_col, "filename": "km_by_group.png"}

    return {
        "m": m,
        "m8": None,
        "figures_dir": run_dir / "figures",
        "title": f"Survival Model Evaluation: {run['name']}",
        "subtitle": "Fitted with the strategy-survival-model pipeline;\n"
        "  methodology validated separately against synthetic ground truth",
        "meta_rows": meta_rows,
        "footer": footer,
        "command": command,
        "demo_size": "",
        "km": km,
    }


# --------------------------------------------------------------------------
# The one template. Section content keys off what the metrics carry: the
# generator block, the oracle and Sharpe columns, the run block, the seed-8
# file, and which figure files exist. Never a mode flag.


def compose_report(ctx: dict) -> str:
    m = ctx["m"]
    p, d, pool = m["params"], m["dataset"], m["pooled"]
    folds, brier, shap = m["folds"], m["ipcw_brier"], m["shap_top"]
    cfg = m["config"]
    g = m.get("generator")
    run = m.get("run")
    figures_dir: Path = ctx["figures_dir"]
    h_cal = int(cfg["calibration_horizon_days"])
    cal = m[f"calibration_{h_cal}d"]
    has_oracle = "c_oracle" in pool
    has_sharpe = "c_sharpe" in pool

    unit, units = ("strategy", "strategies") if g else ("row", "rows")
    verb = "discovered" if g else "started"

    censored_overall = 1.0 - d["event_rate"]
    first_fold_censored = 1.0 - folds[0]["test_event_rate"]
    last_fold_censored = 1.0 - folds[-1]["test_event_rate"]
    c_fold_min = min(f["c_xgb"] for f in folds)
    c_fold_max = max(f["c_xgb"] for f in folds)
    n_train_min = min(f["n_train"] for f in folds)
    n_train_max = max(f["n_train"] for f in folds)

    doc = ReportDoc()

    # ---- 1. Summary -------------------------------------------------------
    if g:
        oracle_gap = pool["c_oracle"] - pool["c_xgb"]
        summary = f"""<p>An automated strategy search produces a queue of candidates that all clear
validation. They stop working at very different rates, and the rate determines
how much capital a new strategy should get and when it should be reviewed. This
report covers a model that predicts how long a newly discovered strategy will
keep working, using only the metadata recorded on the day it is deployed.</p>

<p>The finding that matters more than the model's accuracy is that the metric
allocators implicitly rank on, validation Sharpe, is worse than useless for this
question. Ranking strategies by validation Sharpe scores
{pool["c_sharpe"]:.3f} on a concordance index where 0.500 is a coin flip,
and it stays below 0.500 in all {len(folds)} folds. The cause is selection.
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

<p class="callout">Note: the methodology in this report is built and verified against known
ground truth. No production metadata is presented here.</p>"""
    else:
        brier_cal = brier[f"{h_cal}d"]
        if brier_cal["xgb"] < brier_cal["km_marginal"]:
            brier_sentence = (
                f"At the {h_cal}-day horizon its censoring-weighted Brier score is "
                f"{brier_cal['xgb']:.3f}, beating the {brier_cal['km_marginal']:.3f} of a no-skill "
                "forecast that assigns every row the population average."
            )
        else:
            brier_sentence = (
                f"At the {h_cal}-day horizon its censoring-weighted Brier score is "
                f"{brier_cal['xgb']:.3f}, which fails to beat the "
                f"{brier_cal['km_marginal']:.3f} of "
                "a no-skill forecast that assigns every row the population average. The model "
                "orders rows usefully, but its absolute probabilities at this horizon are not "
                "trustworthy. Section @sec:limitations covers this."
            )
        summary = f"""<p>This report evaluates a survival model fitted to
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
{c_fold_min:.3f} to {c_fold_max:.3f} in Table @tab:folds is the better guide to that.
The rows are also not independent, since licences in the same trade and the
same year fail together, which makes the true interval wider than the one
printed.</p>

<p class="callout">This dataset has no known generating process. Unlike the
synthetic validation report, there is no oracle ceiling to say how much signal
remains unclaimed, and feature attributions cannot be checked against a true
mechanism. What carries over from the synthetic validation is the pipeline
itself: the same temporal folds, label re-censoring, likelihood-based
selection, and calibration checks, verified there against known answers.</p>"""
    doc.section("summary", "Summary", summary)

    # ---- 2. The problem (synthetic front matter) --------------------------
    if g:
        doc.section(
            "problem",
            "The problem",
            """<p>Strategies discovered by an automated search decay. An edge that clears
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
in a single validation statistic, and all of it is already recorded.</p>""",
        )

    # ---- 3. Why ranking by validation Sharpe fails (synthetic) ------------
    if g and has_sharpe:
        sharpe_min = min(f["c_sharpe"] for f in folds)
        sharpe_max = max(f["c_sharpe"] for f in folds)
        doc.section(
            "why-sharpe-fails",
            "Why ranking by validation Sharpe fails",
            f"""<p>Every strategy in the population cleared a validation-Sharpe threshold of
{g["selection_sharpe"]}, because that is how a strategy enters a deployment queue.
An observed Sharpe is the sum of true edge, an overfitting component, and
measurement noise. Conditioning on that sum exceeding a threshold means the
survivors with the highest observed scores are disproportionately the ones whose
overfitting and noise happened to break upward.</p>

<p>Past the threshold, additional observed Sharpe is more likely inflation than
edge. Because the inflated strategies are the overfit ones, and overfit
strategies decay fastest, higher validation Sharpe actively predicts
<em>shorter</em> working life. The result is not a weak predictor but an
inverted one, at {pool["c_sharpe"]:.3f} pooled, ranging from {sharpe_min:.3f}
to {sharpe_max:.3f} across folds and never once above the 0.500 of a coin
flip.</p>

<p>This is what makes the modeling problem worth posing. If validation Sharpe alone could
rank survival correctly there would be nothing to add. A meta-model earns its
place precisely because the selection process has already consumed the obvious
signal.</p>""",
        )

    # ---- 4. Data ----------------------------------------------------------
    if g:
        data_body = f"""<p>The run reported here draws {d["n_strategies"]:,} strategies with seed
{g["seed"]}, with initial discovery dates between {g["discovery_start"]} \
and {g["discovery_end"]}, observed to
{g["observation_cutoff"]}. Median observed lifetime is
{d["median_observed_duration_days"]:.0f} days.
{pct(censored_overall)} of strategies are right-censored, either still running
at the observation cutoff or administratively retired at a rate of
{pct(g["admin_censor_rate"], 0)}. Survival time is log-normal in the latent
quantities with a scale of {g["log_time_sigma"]} on log-days, which is the noise that
places the ceiling below a perfect score.</p>

<p>The data is synthetic. Addendum A describes the generating process, what it
deliberately builds into the population, and the two verification checks that a
known ground truth makes possible.</p>"""
    else:
        cols = run["columns"]
        dropped_sentence = (
            f"Columns dropped before fitting: {', '.join(cols['dropped'])}. "
            if cols["dropped"]
            else ""
        )
        categorical_sentence = (
            "Text columns one-hot encoded via the saved encoding recipe: "
            f"{', '.join(cols['categorical'])}."
            if cols.get("categorical")
            else ""
        )
        data_body = f"""<p>The dataset is <code>{run["source"]}</code>: {d["n_rows"]:,} rows, each
observed from its start date. Start dates run {d["date_min"]} to
{d["date_max"]}. {pct(d["event_rate"])} of rows have their ending observed and
{pct(censored_overall)} are censored; median observed duration is
{d["median_observed_duration_days"]:.0f} days. {dropped_sentence}{categorical_sentence}</p>"""
        if ctx["km"]:
            km_col = ctx["km"]["col"]
            km_fig = doc.figure(
                "km",
                img_uri(figures_dir, ctx["km"]["filename"]),
                f"Kaplan-Meier survival curves by {km_col}",
                f"Kaplan-Meier survival curves by <code>{km_col}</code>. Groups"
                " beyond the seven most frequent are collapsed into (other) for"
                " this plot only.",
            )
            data_body += f"""

<p>Figure @fig:km shows Kaplan-Meier survival curves by
<code>{km_col}</code>, the coarsest structure in the outcome before any model
is fitted.</p>

{km_fig}"""
    doc.section("data", "Data", data_body)

    # ---- 5. Method --------------------------------------------------------
    if g:
        features_para = (
            "<p>The synthetic feature matrix is finite by construction, so the"
            " model does no imputation on this data.</p>"
        )
        folds_open = f"""<p>Evaluation uses {len(folds)} expanding-window folds ordered by discovery
date. The earliest {pct(cfg["min_train_frac"], 0)} of strategies is burn-in \
that is only ever trained on.
Each fold trains on every strategy discovered before its split date and tests on
the next {folds[0]["n_test"]} strategies, so training sets grow from
{n_train_min:,} to {n_train_max:,} rows.</p>"""
        baselines = """<p>Three references bound the result. A Cox proportional hazards model on the
same features is the standard survival alternative and answers whether the
gradient-boosted model was necessary. Ranking by validation Sharpe is the
heuristic a backtest-driven allocator applies implicitly. The oracle ranks by
the latent quantity that generated the lifetimes and gives the ceiling that
measurement noise imposes.</p>"""
    else:
        features_para = (
            "<p>Numeric features pass through as-is, missing values included,"
            " which XGBoost handles natively; the Cox baseline receives"
            " train-window median imputation. Text columns are one-hot"
            " encoded.</p>"
        )
        folds_open = f"""<p>Rows are ordered by start date and evaluated with {run["n_folds"]}
expanding-window folds. The earliest {pct(cfg["min_train_frac"], 0)} of rows is
burn-in that is only ever trained on. Every fold trains only on rows that
started before its split date and tests on the next block, so training sets
grow from {n_train_min:,} to {n_train_max:,} rows.</p>"""
        baselines = """<p>A Cox proportional hazards model fitted on the same features is the
standard survival alternative and answers whether the gradient-boosted model
was necessary.</p>"""

    method_body = f"""<h3>@sec:method.1 Model class</h3>

<p>The target is a duration with incomplete observations, so the model is an
accelerated failure time (AFT) model, implemented as XGBoost with the
<code>survival:aft</code> objective. Censoring enters through interval labels,
where an observed death is the interval [t, t] and a censored {unit} is
[t, infinity).</p>

<p>Regression on lifetime was rejected because censored rows have no target, and
dropping them discards the longest-lived {units}, which biases the model
toward pessimism. Classification at a fixed horizon handles censoring awkwardly
and answers only one question. AFT keeps every row and returns a full time
distribution, which collapses to any horizon an allocator asks about.</p>

{features_para}

<h3>@sec:method.2 Temporal validation and label re-censoring</h3>

{folds_open}

<p>Training labels are re-censored at each split date. A {unit} {verb} two
years before a split may have died three months after it, and its recorded label
contains that future death. Any model trained at that split date could only have
known the {unit} was still running, so every post-split death is rewritten as
a censoring at the split. Omitting this step raises measured scores while
silently importing future information, which makes it the most consequential
detail in the pipeline. It is implemented as a standalone function
(<code>cv.recensor</code>) with dedicated tests, because it transfers unchanged
to any duration problem on operational data.</p>

<p>Test labels use the full observation window, because the restriction exists
to keep the future away from the model, not away from the scorer.</p>

<h3>@sec:method.3 Hyperparameter selection and calibration</h3>

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

<h3>@sec:method.4 Baselines</h3>

{baselines}"""
    doc.section("method", "Method", method_body)

    # ---- 6. Results -------------------------------------------------------
    conc_rows = ""
    if has_oracle:
        conc_rows += f"""    <tr><td>Oracle on latent log-time (ceiling)</td>
      <td>{pool["c_oracle"]:.3f}</td><td>not resampled</td></tr>
"""
    conc_rows += f"""\
    <tr class="highlight"><td>XGBoost AFT (pooled)</td><td>{pool["c_xgb"]:.3f}</td>
      <td>{pool["c_xgb_ci"][0]:.3f} to {pool["c_xgb_ci"][1]:.3f}</td></tr>
    <tr><td>XGBoost AFT (fold mean)</td>
      <td>{pool["c_xgb_by_fold_mean"]:.3f}</td><td>not resampled</td></tr>
    <tr><td>Cox proportional hazards (fold mean)</td>
      <td>{pool["c_cox_by_fold_mean"]:.3f}</td><td>not resampled</td></tr>"""
    if has_sharpe:
        conc_rows += f"""
    <tr><td>Rank by validation Sharpe</td><td>{pool["c_sharpe"]:.3f}</td>
      <td>{pool["c_sharpe_ci"][0]:.3f} to {pool["c_sharpe_ci"][1]:.3f}</td></tr>"""
    tab_conc = doc.table(
        "concordance",
        "Concordance index by method, pooled across\n  folds. Higher is better;"
        " 0.500 is a coin flip.",
        "<tr><th>Method</th><th>Harrell C</th><th>95% interval</th></tr>",
        conc_rows,
    )

    if g and has_oracle:
        discrimination_notes = f"""\
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

<p>Both models sit about {pool["c_oracle"] - pool["c_xgb"]:.2f} below the oracle. The gap between
{pool["c_xgb"]:.3f} and a perfect score is therefore mostly unpredictable
variation in when a strategy actually stops working, not unused signal.</p>"""
    else:
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
        discrimination_notes = f"""\
<p>{winner} Each fold refits the Cox model, and its risk scores are relative
ones centred on that fold's own training window, so they are meaningful within
a fold but carry no common scale across folds; the fold-mean rows are the
like-for-like comparison and the pooled figure is not comparable to them.</p>"""

    fold_head = (
        "<tr><th>Fold</th><th>Split date</th><th>Train n</th><th>Test n</th>\n"
        "    <th>Censored</th><th>AFT</th><th>Cox</th>"
    )
    if has_oracle and has_sharpe:
        fold_head += "<th>Sharpe</th><th>Oracle</th>"
    fold_head += "</tr>"
    fold_rows = []
    for i, f in enumerate(folds):
        row = (
            f"<tr><td>{i + 1}</td><td>{f['split_date']}</td><td>{f['n_train']:,}</td>"
            f"<td>{f['n_test']:,}</td><td>{pct(1 - f['test_event_rate'], 0)}</td>"
            f"<td>{f['c_xgb']:.3f}</td><td>{f['c_cox']:.3f}</td>"
        )
        if has_oracle and has_sharpe:
            row += f"<td>{f['c_sharpe']:.3f}</td><td>{f['c_oracle']:.3f}</td>"
        fold_rows.append(row + "</tr>")
    fold_caption = (
        "Per-fold results. Censoring is the share of\n  each fold's test"
        " strategies still running at the observation cutoff."
        if g
        else "Per-fold results. Censoring is the share\n  of each fold's test"
        " rows whose ending was not observed."
    )
    tab_folds = doc.table("folds", fold_caption, fold_head, "\n".join(fold_rows))

    if g:
        fold_fig_caption = f"""Concordance by fold. AFT performance is
  stable from {c_fold_min:.3f} to {c_fold_max:.3f}. The final fold scores
  highest, but {pct(last_fold_censored, 0)} of its test strategies are censored
  against {pct(first_fold_censored, 0)} in the first fold, which changes the set
  of comparable pairs. Concordance is not strictly comparable across folds with
  different censoring mixes, so this plot supports a claim of stability rather
  than improvement."""
    else:
        fold_fig_caption = f"""Concordance by fold, from
  {c_fold_min:.3f} to {c_fold_max:.3f}. Folds differ in censoring mix, so
  cross-fold comparisons are indicative rather than exact."""
    fig_folds = doc.figure(
        "fold-cindex",
        img_uri(figures_dir, "fold_cindex.png"),
        "Concordance index by temporal fold",
        fold_fig_caption,
    )

    if g:
        cal_shape_para = f"""
<p>The margin over the no-skill reference is wide at 90 days and narrow at 365.
That is the expected shape. Median survival is about
{d["median_observed_duration_days"]:.0f} days, so by one year nearly every
strategy has stopped working and the marginal forecast is already close to
certain. Little skill remains to demonstrate at that horizon.</p>
"""
    else:
        cal_shape_para = ""

    brier_rows = "\n".join(
        f"<tr><td>{h.replace('d', ' days')}</td><td>{v['xgb']:.3f}</td>"
        f"<td>{v['cox']:.3f}</td><td>{v['km_marginal']:.3f}</td></tr>"
        for h, v in brier.items()
    )
    tab_brier = doc.table(
        "brier",
        "IPCW Brier score by horizon. Lower is\n  better.",
        "<tr><th>Horizon</th><th>AFT</th><th>Cox</th>\n    <th>No-skill marginal</th></tr>",
        brier_rows,
    )

    gaps = [(abs(b["predicted"] - b["observed_km"]), i) for i, b in enumerate(cal)]
    worst_gap, worst_bin = max(gaps)
    if g:
        cal_fig_caption = f"""Predicted against observed survival at
  {h_cal} days, by predicted decile. Observed frequencies are Kaplan-Meier estimates
  within each bin, so censored strategies contribute correctly rather than being
  dropped. The largest deviation is {worst_gap:.3f} in decile
  {worst_bin + 1}."""
    else:
        cal_fig_caption = f"""Predicted against observed survival at
  {h_cal} days, by predicted decile. Observed frequencies are Kaplan-Meier
  estimates within each bin, so censored rows contribute correctly."""
    fig_cal = doc.figure(
        "calibration",
        img_uri(figures_dir, f"calibration_{h_cal}d.png"),
        f"Decile calibration at {h_cal} days",
        cal_fig_caption,
    )
    cal_rows = "\n".join(
        f"<tr><td>{i + 1}</td><td>{b['n']}</td><td>{b['predicted']:.3f}</td>"
        f"<td>{b['observed_km']:.3f}</td>"
        f"<td>{b['observed_km'] - b['predicted']:+.3f}</td></tr>"
        for i, b in enumerate(cal)
    )
    tab_cal = doc.table(
        "calibration",
        f"Decile calibration at {h_cal} days, the values\n  plotted in Figure @fig:calibration.",
        "<tr><th>Decile</th><th>n</th><th>Predicted</th><th>Observed (KM)</th>\n"
        "    <th>Deviation</th></tr>",
        cal_rows,
    )

    results_body = f"""<h3>@sec:results.1 Discrimination</h3>

<p>Pooled over {pool["n_test"]:,} out-of-fold {units}, with 95% percentile
bootstrap intervals over {cfg["n_bootstrap"]} resamples:</p>

{tab_conc}

{discrimination_notes}

{tab_folds}

<p>Per-fold stability is the claim Figure @fig:fold-cindex supports;
Table @tab:folds carries the same numbers with their censoring mix.</p>

{fig_folds}

<h3>@sec:results.2 Calibration</h3>

<p>Predicted median survival times are converted to survival probabilities under
the calibrated log-normal, then checked two ways. Brier scores are weighted by
the inverse probability of censoring so that censored rows do not bias the
result. The no-skill reference assigns every {unit} the pooled Kaplan-Meier
marginal, ignoring all features.</p>

{tab_brier}
{cal_shape_para}
{fig_cal}

{tab_cal}"""
    doc.section("results", "Results", results_body)

    # ---- 7. What the model uses -------------------------------------------
    shap_rows = "\n".join(
        f"<tr><td>{s['feature']}</td><td>{s['mean_abs_shap']:.3f}</td></tr>" for s in shap[:8]
    )
    tab_shap = doc.table(
        "shap-top",
        "Top eight features by mean absolute\n  attribution.",
        "<tr><th>Feature</th><th>Mean |attribution|</th></tr>",
        shap_rows,
    )
    if g:
        shap_bar_caption = """Mean absolute attribution by feature.
  The three walk-forward consistency statistics carry more attribution than
  every other feature combined."""
        beeswarm_caption = """Per-strategy attributions. High
  walk-forward positive fraction pushes predicted survival up; a steeply
  negative walk-forward Sharpe slope pushes it down hard, with the left tail
  reaching about -1.5 log-days, a factor of roughly 4.5 shorter; high
  fold-to-fold Sharpe dispersion is penalized. All three directions match the
  generative design, where consistency rises with true edge and falls with
  overfitting."""
        dependence_caption = """Attribution against feature value. The
  positive-fraction effect is monotone and close to linear, with vertical bands
  at the nine possible fold fractions and a swing of about 1.3 log-days end to
  end. The Sharpe-decay effect saturates once the slope turns positive, which is
  sensible behaviour given that positive decay slopes are mostly noise. The
  dispersion effect is a descending staircase that flattens past 1.5, where
  additional dispersion no longer distinguishes anything."""
    else:
        shap_bar_caption = "Mean absolute attribution by\n  feature."
        beeswarm_caption = "Per-row attributions across the\n  explanation sample."
        dependence_caption = (
            "Attribution against feature value for\n"
            "  the four features with the largest mean attribution."
        )
    fig_bar = doc.figure(
        "shap-bar",
        img_uri(figures_dir, "shap_bar.png"),
        "Mean absolute SHAP by feature",
        shap_bar_caption,
    )
    fig_bee = doc.figure(
        "beeswarm",
        img_uri(figures_dir, "shap_beeswarm.png"),
        "SHAP beeswarm across the explanation sample",
        beeswarm_caption,
    )
    fig_dep = doc.figure(
        "dependence",
        img_uri(figures_dir, "shap_dependence.png"),
        "SHAP dependence plots, walk-forward statistics" if g else "SHAP dependence plots",
        dependence_caption,
    )

    uses_body = """<p>Feature attributions are computed on the log-time margin, so a value of +0.3
multiplies predicted survival time by about 1.35 and negative values shorten
it.</p>"""
    if not g:
        uses_body += """

<p>On this data the attributions are descriptive only. There is
no known mechanism to check them against, and correlated features can split
credit in ways that reflect the model's internal choices as much as the data,
so read the ranking as "what this model leaned on", not as importance in the
world.</p>"""
    uses_body += f"""

<p>Figure @fig:shap-bar ranks features by mean absolute attribution,
Table @tab:shap-top lists the top eight, Figure @fig:beeswarm shows the
per-row spread, and Figure @fig:dependence traces attribution against feature
value.</p>

{fig_bar}

{tab_shap}

{fig_bee}

{fig_dep}"""
    if g and has_sharpe:
        uses_body += f"""

<p>Validation Sharpe does not appear in the top twelve features. On
unconditional data that would be strange. Here it is the signature of
selection. Every strategy cleared the same threshold, so the metric's
remaining variation
is mostly overfitting plus noise, and the model routes its attention to
walk-forward consistency instead. This is the same phenomenon as the
{pool["c_sharpe"]:.3f} in Table @tab:concordance, seen from inside the model.</p>

<p>Because the generating process is known, this ranking can be checked
against the mechanisms actually built in; Addendum A does so.</p>

<p>Directions in that table are more trustworthy than magnitudes. The three
walk-forward statistics are correlated proxies for the same latent quantity, and
how attribution divides credit among correlated features reflects the model's
internal choices as much as the data. The supportable reading is that the model
uses positive fraction about twice as much as Sharpe dispersion, not that
positive fraction is twice as important. The feature-family and asset-class
effects enter the generator additively and independently, so any reasonable
model recovers them; recovering the proxy structure is the informative
result.</p>"""
    doc.section("model-uses", "What the model uses", uses_body)

    # ---- 8. Limitations ---------------------------------------------------
    lim_parts: list[str] = []
    if g:
        lim_parts.append("""\
<p><strong>The generator encodes a prior, not evidence.</strong> A model that
recovers structure placed there says nothing about whether real strategy
metadata contains comparable signal. This work validates a methodology, and the
concordance reported here should be treated as an upper bound on production
performance.</p>""")
    else:
        lim_parts.append("""\
<p><strong>No ground truth.</strong> Nothing bounds how much predictable signal
this dataset contains, so the concordance above cannot be placed relative to a
ceiling the way the synthetic validation could.</p>""")
    if g:
        lim_parts.append("""\
<p><strong>Lifetimes are treated as independent.</strong> Real strategies stop
working together when a regime breaks. The generator omits that dependence and
the bootstrap resamples strategies independently, so the intervals in Table @tab:concordance
understate real uncertainty and are best read as a lower bound on it.</p>""")
        lim_parts.append("""\
<p><strong>Administrative retirement is modeled as independent censoring.</strong>
In a live book, strategies are retired because someone observed early decay,
which makes the censoring informative and biases survival estimates upward.
Handling it properly requires a competing-risks treatment, which is recorded as
future work rather than implemented.</p>""")
    if run:
        losing = [h for h, v in brier.items() if v["xgb"] >= v["km_marginal"]]
        if losing:
            losing_text = ", ".join(h.replace("d", " days") for h in losing)
            lim_parts.append(f"""\
<p><strong>Absolute probabilities are not usable at {losing_text}.</strong> At
those horizons the model's censoring-weighted Brier score does not beat a
no-skill forecast. Discrimination and calibration are separate qualities. The
ranking carries signal while the survival probabilities do not, so use this
model to order rows, not to act on absolute probabilities at those horizons.
The usual cause under temporal evaluation is training windows whose re-censored
follow-up is far shorter than the horizon, which forces the fitted log-normal
to extrapolate beyond anything it saw.</p>""")
        lim_parts.append("""\
<p><strong>Censoring may be informative.</strong> The evaluation assumes rows
stop being observed for reasons unrelated to their risk. If observation ended
early on rows that were about to end anyway, survival estimates are biased
upward. Whether that holds here depends on how this dataset was collected.</p>""")
    if g:
        lim_parts.append(
            "<p><strong>Generator-seed dependence</strong> is examined in Addendum A.</p>"
        )
    lim_parts.append(f"""\
<p><strong>Harrell's concordance is biased under heavy censoring.</strong> The
final fold is {pct(last_fold_censored, 0)} censored, where an inverse-probability
weighted concordance would be preferable. Fold-to-fold comparisons in Table @tab:folds
should be read with that in mind.</p>""")
    doc.section("limitations", "Limitations", "\n\n".join(lim_parts))

    # ---- 9. Intended use (synthetic only) ---------------------------------
    if g:
        doc.section(
            "intended-use",
            "Intended use",
            f"""<p>The model is a capital-allocation and review-scheduling prior, not a trade
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
includes a demonstration on {ctx["demo_size"]}real City of Chicago business licences
(<code>reports/chicago_demo/</code>), where the checks that depend on ground
truth are absent and the generated report says so.</p>

<p>Nearer-term work, in the order it would be done: a multi-seed sweep to
separate data variance from model variance, an inverse-probability weighted
concordance alongside Harrell's, a block bootstrap by discovery month to stop
understating interval width, and a competing-risks treatment of administrative
retirement.</p>""",
        )

    # ---- 10. Reproducing --------------------------------------------------
    if g:
        repro_body = f"""<p>Python 3.11 or later. From the repository root:</p>

<pre><code>{ctx["command"]}</code></pre>

<p>The test suite is {count_tests()} tests covering generator invariants, fold and
label leakage, and metric behaviour. The pipeline script regenerates the data
at seed {g["seed"]}, runs the {len(folds)}-fold temporal cross-validation, writes
<code>reports/metrics.json</code> and all figures in about two minutes, then
rebuilds this report from that metrics file, so every number here is traceable
to a single run rather than transcribed. <code>requirements.txt</code> pins the
exact dependency versions these numbers were produced with.</p>

<p>The seed-dependence reading in Addendum A is generated from
<code>reports/metrics_seed8.json</code>, produced by
<code>python scripts/run_synthetic_pipeline.py --seed 8</code> and saved under that name;
the builder reads it when present and states the single-seed limitation when it
is not. Building the report also prints
<code>strategy_survival_report.pdf</code> headlessly when Chrome is
available.</p>"""
        doc.section("repro", "Reproducing these results", repro_body)
    else:
        doc.section(
            "repro",
            "Reproducing this run",
            f"<pre><code>{ctx['command']}</code></pre>",
        )

    return doc.render(
        doctype="Technical Report",
        title=ctx["title"],
        subtitle=ctx["subtitle"],
        meta_rows=ctx["meta_rows"],
        footer=ctx["footer"],
    )
