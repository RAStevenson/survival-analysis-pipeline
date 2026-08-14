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

from .units import horizon_label, unit_abbrev, unit_singular

TOKEN_RE = re.compile(r"@(sec|fig|tab):([a-z0-9-]+)")
_FIGCAPTION_RE = re.compile(r"<figcaption>.*?</figcaption>", re.DOTALL)

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

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

    # Read the demo's size and headline score from the demo's own metrics
    # rather than quoting them.
    demo_metrics = reports_dir / "chicago_demo" / "metrics.json"
    if demo_metrics.exists():
        demo_m = json.loads(demo_metrics.read_text())
        demo_size = f"{demo_m['dataset']['n_rows']:,} "
        demo_c = demo_m["pooled"]["c_xgb"]
        demo_within = (demo_m.get("within_group") or {}).get("c_within")
    else:
        demo_size = ""
        demo_c = None
        demo_within = None

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
        "demo_c": demo_c,
        "demo_within": demo_within,
        "km": None,
    }


def _load_dataset_notes(source: str) -> dict:
    """Prose the dataset's preparer recorded beside the data file, in
    <name>.notes.json next to it, keys "data", "limitations", "km_caption",
    and "worst_fold" (an investigated cause for the weakest fold, which
    replaces the template's default no-cause-established sentence).
    The generic template never asserts facts about a specific dataset;
    anything dataset-specific a report says enters through this file."""
    p = Path(source)
    notes_path = p.parent / (p.name.split(".")[0] + ".notes.json")
    if notes_path.exists():
        return json.loads(notes_path.read_text())
    return {}


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
        "notes": _load_dataset_notes(run["source"]),
    }


def seed_dependence_para(m: dict, m8: dict | None) -> str:
    """Addendum paragraph on generator-seed dependence, computed from the
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
    # An order claim needs a margin that survives platform noise. requirements.txt
    # records cross-platform attribution drift near 1e-3; below twice that, a
    # matching order is a coincidence of decimals, not a confirmation.
    order_margin = min(shap8[i]["mean_abs_shap"] - shap8[i + 1]["mean_abs_shap"] for i in range(2))
    if top3_same and order_same and order_margin >= 0.002:
        attribution = (
            "The same three walk-forward statistics dominate attribution, in the same order. "
        )
    elif top3_same and order_same:
        attribution = (
            "The same three walk-forward statistics dominate attribution. Their order "
            "matches as well, but the closest pair is separated by a gap smaller than the "
            "attribution drift this pipeline has measured across platforms (the "
            "cross-platform run recorded in requirements.txt), so the matching "
            "order reads as a tie rather than a confirmation. "
        )
    elif top3_same:
        attribution = (
            "The same three walk-forward statistics dominate attribution, although their "
            "internal ordering shifts. That shift illustrates the correlated-proxies caveat "
            "in section @sec:model-uses, since how credit divides among proxies of the same "
            "latent quantity is not stable across draws and only the group-level conclusion "
            "holds. "
        )
    else:
        attribution = (
            "The set of dominant features shifted between seeds, which weakens the "
            "attribution claims in section @sec:model-uses. "
        )
    return (
        "<p>The pipeline was rerun end to end "
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


# --------------------------------------------------------------------------
# The one template. Section content keys off what the metrics carry: the
# generator block, the oracle and Sharpe columns, the run block, the seed-8
# file, and which figure files exist. Never a mode flag.


def compose_report(ctx: dict) -> str:
    m = ctx["m"]
    notes = ctx.get("notes") or {}
    p, d, pool = m["params"], m["dataset"], m["pooled"]
    folds, brier, shap = m["folds"], m["ipcw_brier"], m["shap_top"]
    cfg = m["config"]
    g = m.get("generator")
    run = m.get("run")
    figures_dir: Path = ctx["figures_dir"]
    # The dataset's time unit, worded three ways: plural for prose ("180
    # days"), singular for hyphenated adjectives ("180-day horizon"), and
    # the abbreviation used in metrics keys and figure file names ("180d").
    # Metrics written before multi-unit support carry no field and are days.
    tu = cfg.get("time_unit", "days")
    tu1 = unit_singular(tu)
    tua = unit_abbrev(tu)
    # hs is the calibration horizon's display spelling ("180", "0.25");
    # int() would collapse fractional horizons.
    hs = horizon_label(cfg["calibration_horizon_days"])
    cal = m[f"calibration_{hs}{tua}"]
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
        if ctx.get("demo_c") is not None:
            demo_within = ctx.get("demo_within")
            within_clause = (
                f" Most of that figure reflects licence-category membership rather than"
                f" ranking within a category, where comparisons score {demo_within:.3f};"
                f" the companion report gives the decomposition."
                if demo_within is not None
                else ""
            )
            companion_para = f"""
<p>The same pipeline also runs on real data. A companion report in this
repository applies it to {ctx["demo_size"]}public City of Chicago business
licences and reaches a pooled out-of-fold concordance of
{ctx["demo_c"]:.3f} (<code>reports/chicago_demo/</code>).{within_clause}</p>
"""
        else:
            companion_para = ""
        summary = f"""<p>An automated strategy search produces a queue of candidates that all clear
validation. They stop working at very different rates, and the rate determines
how much capital a new strategy should get and when it should be reviewed. This
report covers a model that predicts how long a newly discovered strategy will
keep working, using only the metadata recorded on the day it is deployed.</p>

<p>The finding that matters more than the model's accuracy is that the primary metric
allocators implicitly rank on, validation Sharpe, the risk-adjusted return score
a strategy's backtest reports, is worse than useless for this
question. Ranking strategies by validation Sharpe scores
{pool["c_sharpe"]:.3f} on a concordance index, the share of pairs a ranking
puts in the right order, where 0.500 is a coin flip, and it stays below
0.500 in all {len(folds)} folds. The cause is selection.
Every strategy entered the queue by clearing a Sharpe threshold, so beyond that
threshold a high score reflects overfitting more often than edge, and overfit
strategies decay fastest. In this report the inversion is demonstrated on
synthetic data whose generator installs that selection step by construction;
section @sec:why-sharpe-fails walks the mechanism, and whether it holds on any
particular real book is an empirical question this report does not settle.</p>

<p>A model reading walk-forward consistency, how a strategy's returns held up
across repeated refit-and-test steps through its own validation history,
instead reaches
{pool["c_xgb"]:.3f} (95% bootstrap interval
{pool["c_xgb_ci"][0]:.3f} to {pool["c_xgb_ci"][1]:.3f}, pooled over
{pool["n_test"]:,} out-of-fold strategies). Because the data is synthetic but
includes built-in baseline noise, the best score a model with the generator's
hidden variables in hand could achieve is computable via an oracle model, and it is
{pool["c_oracle"]:.3f}. The winning valid model sits {oracle_gap:.3f} below that ceiling,
and a model reading only the observable metadata faces a ceiling somewhat below
the oracle's, so most of the remaining error is irreducible noise rather than
model capacity.</p>
{companion_para}
<p class="callout">Note: the methodology in this report is built and verified against known
ground truth derived from the synthetic data. No production metadata is
presented here. Its purpose is to demonstrate methodology, not strategy
edge.</p>"""
    else:
        brier_cal = brier[f"{hs}{tua}"]
        losing = [k for k in brier if brier[k]["xgb"] >= brier[k]["km_marginal"]]

        def _horizon_name(key: str) -> str:
            return f"{key.removesuffix(tua)}-{tu1}"

        if not losing:
            brier_sentence = (
                f"At the {hs}-{tu1} horizon its censoring-weighted Brier score is "
                f"{brier_cal['xgb']:.3f}, beating the {brier_cal['km_marginal']:.3f} of a no-skill "
                "forecast that assigns every row the population average, and it beats that "
                "reference at every requested horizon."
            )
        elif f"{hs}{tua}" not in losing:
            lose_names = " and ".join(_horizon_name(k) for k in losing)
            plural = "s" if len(losing) > 1 else ""
            brier_sentence = (
                f"At the {hs}-{tu1} horizon its censoring-weighted Brier score is "
                f"{brier_cal['xgb']:.3f}, beating the {brier_cal['km_marginal']:.3f} of a no-skill "
                f"forecast that assigns every row the population average, but at the "
                f"{lose_names} horizon{plural} it loses to that reference; section "
                "@sec:limitations covers the loss."
            )
        else:
            brier_sentence = (
                f"At the {hs}-{tu1} horizon its censoring-weighted Brier score is "
                f"{brier_cal['xgb']:.3f}, which fails to beat the "
                f"{brier_cal['km_marginal']:.3f} of "
                "a no-skill forecast that assigns every row the population average. The model "
                "orders rows usefully, but its absolute probabilities at this horizon are not "
                "trustworthy. Section @sec:limitations covers this."
            )
        wg_s = m.get("within_group")
        wg_summary = (
            f"Most of the pooled figure reflects a row's <code>{wg_s['col']}</code> group"
            f" rather than ranking within it. Ranking every row by its group's mean"
            f" prediction alone scores {wg_s['c_group_mean']:.3f}, and comparisons within"
            f" a group score {wg_s['c_within']:.3f}; the results section gives the"
            f" decomposition. "
            if wg_s
            else ""
        )
        fm_x, fm_c = pool["c_xgb_by_fold_mean"], pool["c_cox_by_fold_mean"]
        if fm_c > fm_x:
            fm_verdict = (
                " The simpler baseline wins, and scoring new rows defaults to it,"
                " because the saved bundle recommends whichever model scored higher"
                " on this comparison."
            )
        elif fm_x > fm_c:
            fm_verdict = (
                " The boosted model wins, and scoring new rows defaults to it,"
                " because the saved bundle recommends whichever model scored higher"
                " on this comparison."
            )
        else:
            fm_verdict = " The two models tie on this comparison."
        summary = f"""<p>This report evaluates a survival model fitted to
<code>{Path(run["source"]).name}</code>, {d["n_rows"]:,} rows, each observed
from its start date for {run["columns"]["duration"]!r} {tu} with
{run["columns"]["event"]!r} marking whether the ending was seen
({pct(d["event_rate"])}) or the row was still running when observation
stopped ({pct(censored_overall)}). Median observed duration is
{d["median_observed_duration_days"]:.0f} {tu}. The model predicts, from the
{d["n_features"]} features the deployed encoding builds out of the start-date
columns (each evaluation fold refits a smaller vocabulary of its own), how
long each row survives.</p>

<p>The like-for-like comparison between models is the fold mean: each of the
{run["n_folds"]} folds fits both models on its own training window and the
fold scores average. On concordance, the share of pairs a ranking puts in the
right order where 0.500 is a coin flip, the XGBoost AFT model scores
{pool["c_xgb_by_fold_mean"]:.3f} and a Cox proportional hazards baseline on
the same features {pool["c_cox_by_fold_mean"]:.3f}.{fm_verdict}</p>

<p>Pooled over {pool["n_test"]:,} out-of-time test rows, the AFT model scores
{pool["c_xgb"]:.3f} (95% bootstrap interval
{pool["c_xgb_ci"][0]:.3f} to {pool["c_xgb_ci"][1]:.3f}). The pooled figure
exists for the decomposition below, which needs every test row scored on one
axis, rather than as a headline. Pooling concatenates predictions from
{run["n_folds"]} separately fitted models, and although AFT medians share the
time scale, each fold's model carries its own level, so cross-fold pairs blur
and the pooled figure reads conservative next to the fold means. The Cox
baseline's fold-relative risks cannot be pooled at all, which is why the model
comparison above uses fold means for both.
{wg_summary}{brier_sentence}</p>

<p>That bootstrap interval is narrow, and it is worth saying what it does and
does not cover. It holds each fold's fitted model fixed and resamples the test
rows, so it measures how precisely this run's score is pinned down by the
number of rows scored, nothing more. It says nothing about how much the score
would move on a different stretch of history, and the per-fold spread of
{c_fold_min:.3f} to {c_fold_max:.3f} in Table @tab:folds is the better guide to that.
The rows are also unlikely to be independent, since rows that share a
category or a stretch of time tend to fail together, which makes the true
interval wider than the one printed.</p>

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
order of magnitude among strategies with identical headline metrics. Death here
is a bookkeeping event, the date a strategy stops clearing the book's retention
rule, so the lifetimes any model learns are partly a property of that rule.</p>

<p>Three operational decisions depend on the decay rate: how much capital a new
strategy receives on day one, when its first serious review is scheduled, and
when it is retired. The default is to treat every new strategy the same, which
misallocates in both directions. It overfunds strategies that will not survive
the quarter and underfunds the ones that would have run for a year.</p>

<p>The question is whether discovery-time metadata carries enough signal to
separate them. Strategies die for causes that leave traces in that metadata. A
strategy selected from a hundred thousand candidates carries more selection bias
than one selected from a hundred. A strategy whose walk-forward Sharpe was
already sliding during validation had begun decaying before it was deployed.
None of that is visible
in a single validation statistic, and all of it is already recorded.</p>""",
        )

    # ---- 3. Why ranking by validation Sharpe fails (synthetic) ------------
    if g and has_sharpe:
        sharpe_min = min(f["c_sharpe"] for f in folds)
        sharpe_max = max(f["c_sharpe"] for f in folds)
        flip = pool.get("c_sharpe_flipped")
        flip_sentence = (
            f" An inverted ranking is still information, since it can be read backwards."
            f" Reversing the ranking scores {flip:.3f} pooled, the arithmetic complement"
            f" of the {pool['c_sharpe']:.3f}, and still sits well short of the"
            f" {pool['c_xgb']:.3f} the model reaches, so knowing the direction of the"
            f" bias does not substitute for the model."
            if flip is not None
            else ""
        )
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
flip.{flip_sentence}</p>

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
{pct(censored_overall)} of strategies are right-censored, meaning the end of
their working life was never observed. They are either still running
at the observation cutoff or administratively retired, a retirement the
generator draws for {pct(g["admin_censor_rate"], 0)} of strategies
independent of performance. Censoring concentrates among recently discovered
strategies, which have had the least time to fail before the cutoff, so the
final fold's test block is far more censored than the population overall.
Survival time is log-normal in the latent
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
            "Text columns are one-hot encoded automatically. These columns hold "
            "numeric codes rather than quantities and were forced to categorical "
            "as well: "
            f"{', '.join(cols['categorical'])}."
            if cols.get("categorical")
            else ""
        )
        data_body = f"""<p>The dataset is <code>{run["source"]}</code>: {d["n_rows"]:,} rows, each
observed from its start date. Start dates run {d["date_min"]} to
{d["date_max"]}. {pct(d["event_rate"])} of rows have their ending observed and
{pct(censored_overall)} are censored, meaning the row was still open when
observation stopped, so its full duration is unknown; median observed duration
is {d["median_observed_duration_days"]:.0f} {tu}. \
{dropped_sentence}{categorical_sentence}</p>

<p>The pipeline treats every row as observed from its own start date. It has
no support for left truncation, rows whose life began before the source
window opened and that would enter the data mid-life, so any such rows have
to be excluded when the dataset is prepared. Whether they were, and how many,
is a property of the preparation step recorded with the dataset.</p>"""
        if notes.get("data"):
            data_body += f"""

<p>{notes["data"]}</p>"""
        if ctx["km"]:
            km_col = ctx["km"]["col"]
            km_caption = (
                f"Kaplan-Meier survival curves by <code>{km_col}</code>. Groups"
                " beyond the seven most frequent are collapsed into (other) for"
                " this plot only."
            )
            if notes.get("km_caption"):
                km_caption += " " + notes["km_caption"]
            km_fig = doc.figure(
                "km",
                img_uri(figures_dir, ctx["km"]["filename"]),
                f"Kaplan-Meier survival curves by {km_col}",
                km_caption,
            )
            data_body += f"""

<p>Figure @fig:km shows Kaplan-Meier survival curves by
<code>{km_col}</code>, the coarsest structure in the outcome before any model
is fitted. A Kaplan-Meier curve estimates the fraction of rows still
surviving at each age, with censored rows counted for as long as they were
observed.</p>

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
            " encoded, and the encoding vocabulary (which levels exist, which"
            " rare levels collapse) is refit on each fold's training window,"
            " so a fold's features reflect only what was on file by its split"
            " date. The deployed model's recipe is fit on the full window,"
            " whose past is legitimately the whole file.</p>"
        )
        folds_open = f"""<p>Rows are ordered by start date and evaluated with {run["n_folds"]}
expanding-window folds. The earliest {pct(cfg["min_train_frac"], 0)} of rows is
burn-in that is only ever trained on. Every fold trains only on rows that
started before its split date and tests on the next block, so training sets
grow from {n_train_min:,} to {n_train_max:,} rows.</p>"""
        baselines = """<p>A Cox proportional hazards model fitted on the same features is the
standard survival alternative and answers whether the gradient-boosted model
was necessary.</p>"""

    # "An allocator" belongs to the trading report; a generic dataset has no
    # allocator to ask, so the shared sentence scopes its tail by variant.
    horizon_tail = "an allocator asks about" if g else "of interest"

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
distribution, which collapses to any horizon {horizon_tail}.</p>

<p>Concretely, the fitted model predicts one number per row, the median
survival time in {tu}. The full curve comes from wrapping a log-normal
distribution of fitted width around that median, so the probability of
surviving any horizon is a point read off that curve. The width is a single
value shared by every row; how it is selected and calibrated is covered in
section @sec:method.3.</p>

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

<p>An earlier version selected on concordance and looked correct, with ranking
metrics landing where they ultimately did. It then lost to a no-skill reference
forecast on 365-day Brier score, the mean squared error of a probability
forecast. A concordance index is invariant to the
predictive scale, so the search had no reason to prefer a usable distribution
width and settled on one far too wide. Nothing in the training loop objected,
because nothing in it measured the width. Selection now uses
likelihood, which penalizes a broken scale, and the predictive log-normal scale
is calibrated on a temporal tail slice by a probe model that never trained on
those rows; the final model is then refit on the full window, including that
slice, and carries the probe-calibrated scale. The two scales do different
jobs. The loss scale is a setting inside the training objective, while the
predictive scale is the width of the probability curves the model actually
reports, measured afterward on rows the fit never used. The selected loss scale was
{p["aft_sigma"]}, and the calibrated
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
        "Concordance index by method (Harrell's C is its\n  standard estimator)."
        " Higher is better; 0.500 is a coin flip. Cox appears as a fold\n  mean only, because its"
        " per-fold risk scores share no scale to pool.",
        "<tr><th>Method</th><th>Concordance (Harrell's C)</th><th>95% interval</th></tr>",
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
for trees to find much beyond what a penalized linear model in the log-hazard,
the logarithm of the instantaneous failure rate, already captures. I report
the tie rather than adding interactions to the
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
                f"On this dataset the Cox baseline scores {cox_fold:.4f} against the boosted "
                f"model's {aft_fold:.4f} on fold-mean concordance, a margin that sets the "
                "default for scoring new rows without proving a real difference. Both fitted "
                "models are saved and the Cox model is the one this run recommends. Reporting "
                "the near-tie, rather than presenting the boosted model as the result, is the "
                "point of carrying a baseline at all."
            )
        else:
            winner = (
                f"The boosted model outscores the Cox baseline on fold-mean concordance, "
                f"{aft_fold:.4f} against {cox_fold:.4f}. Both fitted models are saved and the "
                "boosted model is the one this run recommends for scoring new rows."
            )
        worst_fold = min(folds, key=lambda f: f["c_xgb"])
        worst_fold_idx = folds.index(worst_fold) + 1
        worst_cause = notes.get("worst_fold")
        worst_sentence = (
            f" {worst_cause}" if worst_cause else " No cause is established for it in this report."
        )
        discrimination_notes = f"""\
<p>{winner} Each fold refits the Cox model, and its risk scores are relative
ones centred on that fold's own training window, so they are meaningful within
a fold but carry no common scale across folds; the fold-mean rows are the
like-for-like comparison and the pooled figure is not comparable to them.</p>

<p>The per-fold spread deserves as much attention as the mean. The weakest
window is fold {worst_fold_idx}, starting {worst_fold["split_date"]}, at
{worst_fold["c_xgb"]:.3f}.{worst_sentence}
Any use of the pooled figure should carry that range: there are stretches of
history where the model ranks these rows substantially worse than its
headline number suggests.</p>"""
        wg = m.get("within_group")
        if wg:
            discrimination_notes += f"""

<p>The pooled concordance also decomposes by <code>{wg["col"]}</code>.
Ranking every row by its group's mean prediction alone scores
{wg["c_group_mean"]:.3f}, and comparisons restricted to rows in the same
group score {wg["c_within"]:.3f}, pair-weighted across {wg["n_groups"]}
groups with at least {wg["min_n"]} rows and {wg["min_events"]} observed
endings. The distance of the within-group figure above 0.500 is the
row-level skill; the rest of the pooled figure is group membership. Both
numbers come from the same out-of-fold predictions as the pooled
figure. Stated plainly, the group means alone outscore the full model's
pooled figure; the within-group variation adds ranking skill inside a
group and costs a little ordering across groups.</p>"""

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
        f"<tr><td>{h.removesuffix(tua)} {tu}</td><td>{v['xgb']:.3f}</td>"
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
  {hs} days, by predicted decile. Observed frequencies are Kaplan-Meier estimates
  within each bin, so censored strategies contribute correctly rather than being
  dropped. The largest deviation is {worst_gap:.3f} in decile
  {worst_bin + 1}."""
    else:
        cal_fig_caption = f"""Predicted against observed survival at
  {hs} {tu}, by predicted decile. Observed frequencies are Kaplan-Meier
  estimates within each bin, so censored rows contribute correctly. The
  largest deviation is {worst_gap:.3f} in decile {worst_bin + 1}, and it is
  a real miss, not display noise."""
    fig_cal = doc.figure(
        "calibration",
        img_uri(figures_dir, f"calibration_{hs}{tua}.png"),
        f"Decile calibration at {hs} {tu}",
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
        f"Decile calibration at {hs} {tu}, the values\n  plotted in Figure @fig:calibration.",
        "<tr><th>Decile</th><th>n</th><th>Predicted</th><th>Observed (KM)</th>\n"
        "    <th>Deviation</th></tr>",
        cal_rows,
    )

    # First use of "Kaplan-Meier" in the synthetic report is this paragraph,
    # so it carries the definition there; the real-data report defines the
    # term at its own first use in the Data section.
    km_marginal_gloss = (
        ", the population's own survival curve with censored strategies"
        " counted for as long as they were observed,"
        if g
        else ","
    )

    results_body = f"""<h3>@sec:results.1 Discrimination</h3>

<p>Pooled over {pool["n_test"]:,} out-of-fold {units}, with 95% percentile
bootstrap intervals over {cfg["n_bootstrap"]} resamples:</p>

{tab_conc}

{discrimination_notes}

{tab_folds}

<p>Figure @fig:fold-cindex plots the per-fold concordance from
Table @tab:folds; each fold's censoring mix is in the table.</p>

{fig_folds}

<h3>@sec:results.2 Calibration</h3>

<p>Predicted median survival times are converted to survival probabilities under
the calibrated log-normal, then checked two ways. The Brier score is the mean
squared error of a probability forecast, lower being better, and here it is
weighted by the inverse probability of censoring (IPCW), so that {units}
whose outcome was censored away do not bias the result. The no-skill
reference assigns every {unit} the pooled Kaplan-Meier
marginal{km_marginal_gloss} ignoring all features.</p>

{tab_brier}
{cal_shape_para}
{fig_cal}

{tab_cal}

<p>One caveat on the decile table. The observed value inside each decile is a
Kaplan-Meier estimate, and when a decile's last observed {unit} falls short
of the {hs}-{tu1} horizon the estimate carries its last value forward. In
small or heavily censored deciles the observed column is softer than its
three decimals look, and deviations there should be read accordingly.</p>"""
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
        beeswarm_caption = (
            "Per-row attributions across the\n"
            "  explanation sample. For a yes/no feature, such as a one-hot\n"
            "  category flag, the high (red) end simply means the row is in\n"
            "  that category."
        )
        dependence_caption = (
            "Attribution against feature value for\n"
            "  the strongest numeric features; category flags carry no shape\n"
            "  and stay in the ranking above. A run short on numeric features\n"
            "  fills the grid with flags instead, two columns of dots per\n"
            "  panel."
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

    uses_body = """<p>Feature attributions (SHAP values, for SHapley Additive
exPlanations) are computed on the log scale of
survival time, so a value of +0.3
multiplies predicted survival time by about 1.35 and negative values shorten
it.</p>"""
    if not g:
        uses_body += """

<p>On this data the attributions are descriptive only. There is
no known mechanism to check them against, and correlated features can split
credit in ways that reflect the model's internal choices as much as the data,
so read the ranking as "what this model leaned on", not as importance in the
world. They are also computed on the final model's own training rows, an
in-sample reading rather than out-of-fold evidence.</p>"""
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
unconditional data that would be strange; here it is exactly what
selection predicts. Every strategy cleared the same threshold, so the metric's
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
future work rather than implemented. If the model's own predictions schedule
the reviews that trigger retirements, future training labels partly echo past
model output. That loop is not modeled either.</p>""")
    if run:
        losing = [h for h, v in brier.items() if v["xgb"] >= v["km_marginal"]]
        if losing:
            losing_text = ", ".join(h.removesuffix(tua) + f" {tu}" for h in losing)
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
        if notes.get("limitations"):
            lim_parts.append(f"<p>{notes['limitations']}</p>")
    if g:
        lim_parts.append(
            "<p><strong>Generator-seed dependence</strong> is examined in Addendum A.</p>"
        )
    lim_parts.append("""\
<p><strong>Every row shares one curve shape.</strong> The predictive scale is
a single fitted number, so the model shifts each row's survival curve earlier
or later but never widens or narrows it, and two rows with the same predicted
median receive identical probabilities at every horizon. Letting the width
vary by row would need a model that predicts it, which is future work.</p>""")
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
produces a survival curve for each. Predicted median lifetime sets the first
review date and is one input to initial position sizing, alongside expected
return, costs, and capacity. The full curve gives a decay schedule
against which actual performance can be compared. The model prices day-one
information only. Once a strategy is live, realized performance carries
information no discovery-time metadata can, so this is the pre-live prior,
and a version that updates on live returns is future work.</p>

<p>Before it informs live allocation, the same temporal cross-validation with
the same label re-censoring has to be repeated on production metadata, and the
resulting concordance is expected to be lower with wider intervals. No
production metadata is presented in this report. The pipeline is built and
verified against known ground truth so that a production run needs no
methodological work.</p>

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

<p>The test suite covers generator invariants, fold and
label leakage, and metric behaviour. The pipeline script regenerates the data
at seed {g["seed"]}, runs the {len(folds)}-fold temporal cross-validation, writes
<code>reports/metrics.json</code> and all figures in about two minutes, then
rebuilds this report from that metrics file, so every number here is traceable
to a single run rather than transcribed. <code>requirements.txt</code> pins the
exact dependency versions these numbers were produced with.</p>

<p>The seed-dependence reading in Addendum A is generated from
<code>reports/metrics_seed8.json</code>, produced by
<code>python scripts/run_synthetic_pipeline.py --seed 8 --no-report --no-figures
--metrics-name metrics_seed8.json</code>. The extra flags keep the committed
seed-7 metrics, figures, and report in place; without them the rerun replaces
them all with seed-8 output. The builder reads the file when present and states
the single-seed limitation when it is not. Building the report also prints
<code>strategy_survival_report.pdf</code> headlessly when Chrome is
available.</p>"""
        doc.section("repro", "Reproducing these results", repro_body)
    else:
        doc.section(
            "repro",
            "Reproducing this run",
            f"<pre><code>{ctx['command']}</code></pre>",
        )

    # ---- Addendum A: synthetic ground truth (synthetic only) ---------------
    if g:
        km_fig = doc.figure(
            "km",
            img_uri(figures_dir, "km_by_asset_class.png"),
            "Kaplan-Meier survival curves by asset class",
            """Kaplan-Meier survival curves by asset
  class. Crypto strategies decay faster by construction, and the model has to
  recover that from roughly 15% of rows.""",
        )
        addendum_body = f"""\
<p>In this report, all results come from synthetic data. The generator reproduces the
statistical structure of a strategy-search pipeline, including selection bias,
walk-forward statistics, regime exposures, and censored lifetimes, without
containing anything proprietary.</p>

<p>Confidentiality is one reason. Verification is the more important one.
Because the generating process is known, two checks become available that real
data cannot support. The best achievable score can be computed exactly, which
bounds any claim about model quality. And the model's feature attributions can
be compared against the mechanisms actually built in, which turns interpretation
into a test with a right answer. The test suite enforces the first of these.
One test asserts the model never outscores the oracle, since beating perfect
information would indicate a leak.</p>

<h3>@sec:synthetic-truth.1 What the generator builds</h3>

<p>The generator writes structure into the population on purpose, so that
recovering it is a test rather than an interpretation. Each strategy uses one
to three of six feature families, and each family shifts log survival time by a
fixed amount, with microstructure the most fragile and value-carry the most
durable. Each strategy belongs to one of four asset classes. Crypto is built to
decay fastest, a shift of -0.30 on log time against the fx-majors baseline, and
it is drawn for roughly 15 percent of rows, so the model has to recover the
class effect from a minority slice. Survival time is log-normal around these
effects with a scale of {g["log_time_sigma"]} on log-days. Figure @fig:km shows
the resulting survival curves. The crypto curve separates early and stays
separated, which is the coarsest piece of built-in structure a fitted model
should find.</p>

{km_fig}

<h3>@sec:synthetic-truth.2 The oracle ceiling</h3>

<p>The oracle ranks strategies by the latent log-time quantity the generator
actually used, so its concordance of {pool["c_oracle"]:.3f} bounds every
model; the distance between it and a perfect score is measurement noise, not
missing skill. A model reading only the observable metadata faces a ceiling
somewhat below the oracle's, since the observables are noisy proxies for the
latent quantity.</p>

<h3>@sec:synthetic-truth.3 Attribution against the mechanism</h3>

<p>The attribution ranking is a test rather than a description, because the
generating process is known. The three walk-forward statistics that dominate are
not inputs to survival time in the generator. They exist only as the least noisy
observable proxies for the latent split between real edge and overfitting, which
is the actual driver. The model was not handed that relationship; it found the
proxies and leaned on them. Feature-family and asset-class effects return with
the signs and rough ordering written into the generator, and the search-intensity
penalty appears where a multiple-testing argument predicts.</p>

<p>Two qualifiers keep that reading honest. The walk-forward statistics were
assigned generous signal-to-noise when the generator was written, so their
dominance was likely before any model ran. And the attributions are computed on
the final model's own training rows, so the check is a qualitative match
against the mechanism, not out-of-fold evidence.</p>

<h3>@sec:synthetic-truth.4 Generator-seed dependence</h3>

{seed_dependence_para(m, ctx["m8"])}"""
        doc.addendum("synthetic-truth", "Synthetic ground truth", addendum_body)

    return doc.render(
        doctype="Technical Report",
        title=ctx["title"],
        subtitle=ctx["subtitle"],
        meta_rows=ctx["meta_rows"],
        footer=ctx["footer"],
    )
