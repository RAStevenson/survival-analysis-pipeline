"""Rendering engine for generated reports: the stylesheet, the ReportDoc
section/figure/table registry with render-time numbering and the
cited-or-fail figure rule, and the headless-Chrome PDF printing step.
Templates and contexts live in report.py; this module knows nothing about
metrics."""

from __future__ import annotations

import base64
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

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
  .notemark { font-size: 0.8rem; color: var(--muted); font-style: italic;
    margin: -0.4rem 0 1rem; }
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
