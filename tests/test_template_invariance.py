"""The template-invariance and word-budget contract.

The report template must render the same prose on any dataset. This test
builds both committed variants (the synthetic-shaped and the real-shaped
metrics), strips everything legitimately allowed to differ (injected values,
presence-keyed blocks on the declared whitelist, notes, figures, tables,
command blocks, and the synthetic addendum), and asserts the remaining prose
is byte-identical. A sentence that appears in one variant and not the other
is a template fork and fails here instead of waiting for a reader to notice.

The word-budget test keeps the template honest about length: at most 1,200
words of template prose per report, excluding tables, figure captions,
command blocks, and notes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from survival_analysis_pipeline.notes import load_run_notes
from survival_analysis_pipeline.report import compose_report, real_context, synthetic_context

REPO = Path(__file__).resolve().parents[1]

# Every presence-keyed block the template may render. A pk marker not on
# this list fails the test, so adding a new block is a deliberate act here,
# not a silent fork.
PK_WHITELIST = {
    "bundle",
    "within-group",
    "left-truncation",
    "oracle-summary",
    "synthetic-callout",
    "no-oracle-callout",
    "columns",
    "generator-data",
    "km-figure",
    "oracle-method",
    "validated-elsewhere",
    "oracle-row",
    "sharpe-row",
    "within-group-results",
    "sharpe-results",
    "oracle-results",
    "no-mechanism",
    "generator-limits",
    "no-ground-truth",
    "losing-horizons",
}

_PK_RE = re.compile(r"<!--pk:([a-z0-9-]+)-->.*?<!--/pk:\1-->", re.DOTALL)
_NOTE_RE = re.compile(r"<!--note:([a-z_]+)-->.*?<!--/note:\1-->", re.DOTALL)


def _synthetic() -> tuple[str, dict, dict]:
    run_dir = REPO / "reports" / "synthetic"
    m = json.loads((run_dir / "metrics.json").read_text())
    notes = load_run_notes(run_dir / "notes", m)
    html = compose_report(synthetic_context(m, run_dir))
    return html, m, notes


def _real() -> tuple[str, dict, dict]:
    run_dir = REPO / "reports" / "chicago_demo"
    m = json.loads((run_dir / "metrics.json").read_text())
    notes = load_run_notes(run_dir / "notes", m)
    html = compose_report(real_context(m, run_dir))
    return html, m, notes


def _body(html: str) -> str:
    return html[html.index("</header>") : html.index("<footer>")]


def _template_skeleton(html: str, m: dict) -> str:
    """Reduce a rendered report to its template skeleton: what remains must
    be identical across variants."""
    body = _body(html)
    # Whole sections that exist only as notes or only for the generator.
    body = re.sub(
        r"<section>\s*<h2>[^<]*(?:Motivation|Interpretation|Synthetic ground truth)</h2>"
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
    g = m.get("generator")
    if g:
        source_desc = f"synthetic data drawn at seed {g['seed']}"
    else:
        source_desc = f"<code>{Path(m['run']['source']).name}</code>"
    body = body.replace(source_desc, "SOURCE")
    for clause in (
        "The two models tie at the printed precision",
        "The two models effectively tie",
        "The Cox baseline scores higher",
        "The boosted model scores higher",
    ):
        body = body.replace(clause, "WINNER")
    for clause in (
        "On this run it matches the fold mean at the printed precision",
        "On this run it reads conservative next to the fold mean",
        "On this run it reads slightly high next to the fold mean",
    ):
        body = body.replace(clause, "POOLEDNOTE")
    # The weakest-fold sentence takes one of two computed shapes depending on
    # whether the two lowest folds separate at the printed precision.
    body = re.sub(
        r"Folds \d+ and \d+ are the weakest,.*?too close to separate\."
        r"|The weakest window is fold \d+,.*?where\s+it would appear\.",
        "WEAKESTFOLD",
        body,
        flags=re.DOTALL,
    )
    for word, token in (
        ("strategies", "UNITS"),
        ("strategy", "UNIT"),
        ("rows", "UNITS"),
        ("row", "UNIT"),
    ):
        body = re.sub(rf"\b{word}\b", token, body)
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
    syn_html, syn_m, _ = _synthetic()
    real_html, real_m, _ = _real()
    syn_skel = _template_skeleton(syn_html, syn_m)
    real_skel = _template_skeleton(real_html, real_m)
    if syn_skel != real_skel:
        # Point at the first divergence rather than dumping both skeletons.
        i = next(
            (k for k, (a, b) in enumerate(zip(syn_skel, real_skel, strict=False)) if a != b),
            min(len(syn_skel), len(real_skel)),
        )
        lo = max(0, i - 80)
        pytest.fail(
            "template prose diverges between variants:\n"
            f"  synthetic: ...{syn_skel[lo : i + 80]}...\n"
            f"  real:      ...{real_skel[lo : i + 80]}..."
        )


def test_invariance_checker_catches_a_divergence() -> None:
    # The checker itself must fail on a one-word template fork; otherwise a
    # green invariance test proves nothing.
    syn_html, syn_m, _ = _synthetic()
    doctored = syn_html.replace("The like-for-like comparison", "A like-for-like comparison", 1)
    assert _template_skeleton(doctored, syn_m) != _template_skeleton(syn_html, syn_m)


@pytest.mark.parametrize("variant", ["synthetic", "real"])
def test_template_word_budget(variant: str) -> None:
    html = (_synthetic() if variant == "synthetic" else _real())[0]
    words = _budget_text(html).split()
    assert len(words) <= 1200, f"{variant} template prose is {len(words)} words (budget 1,200)"
