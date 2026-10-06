"""The template-invariance and word-budget contract.

The report template must render the same prose on any dataset. This test
builds the three committed variants, strips everything legitimately
allowed to differ (injected values, presence-keyed blocks on the declared
whitelist, notes, figures, tables, and command blocks), and asserts the
remaining prose is byte-identical. A sentence
that appears in one variant and not the other is a template fork and fails
here instead of waiting for a reader to notice.

The word-budget test is a flat ceiling of 2,200 words on template prose,
excluding tables, figure captions, command blocks, and notes. Set on
2026-09-02, replacing the derived content-point-plus-band budget after the
review fixes of that week outgrew it. The earlier derivation is in this
file's git log.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from survival_analysis_pipeline.report_generator import (
    compose_report,
    real_context,
    synthetic_context,
)
from survival_analysis_pipeline.report_notes import load_run_notes

REPO = Path(__file__).resolve().parents[1]

# Every presence-keyed block the template may render. A pk marker not on
# this list fails the test, so adding a new block is a deliberate act here,
# not a silent fork.
PK_WHITELIST = {
    "within-group",
    "losing-horizons-summary",
    "oracle-summary",
    "synthetic-callout",
    "columns",
    "km-figure",
    "oracle-row",
    "oracle-results",
    "within-group-results",
    "cox-uses",
    "losing-horizons",
}

_PK_RE = re.compile(r"<!--pk:([a-z0-9-]+)-->.*?<!--/pk:\1-->", re.DOTALL)
_NOTE_RE = re.compile(r"<!--note:([a-z_]+)-->.*?<!--/note:\1-->", re.DOTALL)


def _synthetic() -> tuple[str, dict, dict]:
    run_dir = REPO / "reports" / "synthetic"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    notes = load_run_notes(run_dir / "notes", metrics)
    html = compose_report(synthetic_context(metrics, run_dir))
    return html, metrics, notes


def _real() -> tuple[str, dict, dict]:
    run_dir = REPO / "reports" / "chicago_demo"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    notes = load_run_notes(run_dir / "notes", metrics)
    html = compose_report(real_context(metrics, run_dir))
    return html, metrics, notes


def _flchain() -> tuple[str, dict, dict]:
    run_dir = REPO / "reports" / "flchain_demo"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    notes = load_run_notes(run_dir / "notes", metrics)
    html = compose_report(real_context(metrics, run_dir))
    return html, metrics, notes


def _body(html: str) -> str:
    return html[html.index("</header>") : html.index("<footer>")]


def _template_skeleton(html: str, metrics: dict) -> str:
    """Reduce a rendered report to its template skeleton: what remains must
    be identical across variants."""
    body = _body(html)
    # Whole sections that exist only as notes.
    body = re.sub(
        r"<section>\s*<h2>[^<]*(?:Motivation|Interpretation)</h2>"
        r".*?</section>",
        "",
        body,
        flags=re.DOTALL,
    )
    seen = {match.group(1) for match in _PK_RE.finditer(body)}
    unknown = seen - PK_WHITELIST
    assert not unknown, f"presence-keyed blocks not on the whitelist: {sorted(unknown)}"
    body = _PK_RE.sub("", body)
    body = _NOTE_RE.sub("", body)
    body = re.sub(r"<figure>.*?</figure>", "", body, flags=re.DOTALL)
    body = re.sub(r"<table class=\"data\">.*?</table>", "", body, flags=re.DOTALL)
    body = re.sub(r"<pre>.*?</pre>", "", body, flags=re.DOTALL)
    # Injected values the template legitimately varies on.
    generator = metrics.get("generator")
    if generator:
        source_desc = f"synthetic data drawn at seed {generator['seed']}"
    else:
        source_desc = f"<code>{Path(metrics['run']['source']).name}</code>"
    body = body.replace(source_desc, "SOURCE")
    for clause in (
        "The two models tie at the printed precision",
        "The two models effectively tie",
        "The Cox baseline scores higher",
        "The boosted model scores higher",
    ):
        body = body.replace(clause, "WINNER")
    # The bundle sentence names the recommended model and, on a tie, the
    # margin; both are injected values.
    body = body.replace(", on a near-tie margin", "")
    for model in ("the Cox baseline as recommended", "the boosted model as recommended"):
        body = body.replace(model, "MODEL as recommended")
    body = re.sub(r"[0-9][0-9,.%+-]*", "#", body)
    return " ".join(body.split())


def _budget_text(html: str) -> str:
    """Template prose only: no tables, figures, captions, command blocks, or
    notes; presence-keyed blocks count, since they are template."""
    body = _body(html)
    body = re.sub(
        r"<section>\s*<h2>[^<]*(?:Motivation|Interpretation)</h2>.*?</section>",
        "",
        body,
        flags=re.DOTALL,
    )
    body = _NOTE_RE.sub("", body)
    body = re.sub(r"<figure>.*?</figure>", "", body, flags=re.DOTALL)
    body = re.sub(r"<table class=\"data\">.*?</table>", "", body, flags=re.DOTALL)
    body = re.sub(r"<pre>.*?</pre>", "", body, flags=re.DOTALL)
    body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    return re.sub(r"<[^>]+>", " ", body)


def test_template_is_invariant_across_variants() -> None:
    synthetic_html, synthetic_metrics, _ = _synthetic()
    real_html, real_metrics, _ = _real()
    flchain_html, flchain_metrics, _ = _flchain()
    assert _template_skeleton(flchain_html, flchain_metrics) == _template_skeleton(
        real_html, real_metrics
    ), "the two real-shaped runs disagree on template prose"
    synthetic_skeleton = _template_skeleton(synthetic_html, synthetic_metrics)
    real_skeleton = _template_skeleton(real_html, real_metrics)
    if synthetic_skeleton != real_skeleton:
        # Point at the first divergence rather than dumping both skeletons.
        i = next(
            (
                position
                for position, (synthetic_char, real_char) in enumerate(
                    zip(synthetic_skeleton, real_skeleton, strict=False)
                )
                if synthetic_char != real_char
            ),
            min(len(synthetic_skeleton), len(real_skeleton)),
        )
        window_start = max(0, i - 80)
        pytest.fail(
            "template prose diverges between variants:\n"
            f"  synthetic: ...{synthetic_skeleton[window_start : i + 80]}...\n"
            f"  real:      ...{real_skeleton[window_start : i + 80]}..."
        )


def test_invariance_checker_catches_a_divergence() -> None:
    # The checker itself must fail on a one-word template fork; otherwise a
    # green invariance test proves nothing.
    synthetic_html, synthetic_metrics, _ = _synthetic()
    doctored = synthetic_html.replace(
        "This report evaluates two survival models", "This report evaluates 2 survival models", 1
    )
    assert _template_skeleton(doctored, synthetic_metrics) != _template_skeleton(
        synthetic_html, synthetic_metrics
    )


@pytest.mark.parametrize("variant", ["synthetic", "real", "flchain"])
def test_template_word_budget(variant: str) -> None:
    builders = {"synthetic": _synthetic, "real": _real, "flchain": _flchain}
    html = builders[variant]()[0]
    words = _budget_text(html).split()
    assert len(words) <= 2200, (
        f"{variant} template prose is {len(words)} words against the ceiling of 2,200"
    )
