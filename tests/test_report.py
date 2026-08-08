"""Engine tests for the report renderer, plus contract tests over the two
built report variants."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from strategy_survival.report import (
    ReportDoc,
    compose_report,
    real_context,
    synthetic_context,
)


def _render(doc: ReportDoc) -> str:
    return doc.render(doctype="Test", title="T", subtitle="s", meta_rows="", footer="f")


def test_section_and_addendum_numbering() -> None:
    doc = ReportDoc()
    doc.section("intro", "Intro", "<p>See section @sec:method and Addendum @sec:extra.</p>")
    doc.section("method", "Method", "<p>body</p>")
    doc.addendum("extra", "Extra", "<p>body</p>")
    html = _render(doc)
    assert "<h2>1. Intro</h2>" in html
    assert "<h2>2. Method</h2>" in html
    assert "<h2>Addendum A. Extra</h2>" in html
    assert "See section 2 and Addendum A." in html


def test_subsection_numbering_via_parent_token() -> None:
    doc = ReportDoc()
    doc.section("intro", "Intro", "<p>x</p>")
    doc.section("method", "Method", "<h3>@sec:method.1 Model class</h3>")
    html = _render(doc)
    assert "<h3>2.1 Model class</h3>" in html


def test_figure_and_table_numbers_follow_registration_order() -> None:
    doc = ReportDoc()
    fig_a = doc.figure("alpha", "data:x", "alt a", "cap a")
    fig_b = doc.figure("beta", "data:y", "alt b", "cap b")
    tab = doc.table("t-one", "cap t", "<tr><th>h</th></tr>", "<tr><td>1</td></tr>")
    doc.section(
        "s",
        "S",
        f"<p>Figure @fig:alpha, Figure @fig:beta, Table @tab:t-one.</p>{fig_a}{fig_b}{tab}",
    )
    html = _render(doc)
    assert "Figure 1, Figure 2, Table 1." in html
    assert "<strong>Figure 1.</strong> cap a" in html
    assert "<strong>Figure 2.</strong> cap b" in html
    assert "<strong>Table 1.</strong> cap t" in html


def test_uncited_figure_fails_render() -> None:
    doc = ReportDoc()
    block = doc.figure("orphan", "data:x", "alt", "caption that cites nothing")
    doc.section("s", "S", f"<p>prose with no citation</p>{block}")
    with pytest.raises(ValueError, match=r"orphan.*never cited"):
        _render(doc)


def test_citation_inside_own_figcaption_does_not_count() -> None:
    # A caption necessarily contains its own token; that must not satisfy
    # the citation rule.
    doc = ReportDoc()
    block = doc.figure("selfie", "data:x", "alt", "this is Figure @fig:selfie itself")
    doc.section("s", "S", f"<p>no real citation</p>{block}")
    with pytest.raises(ValueError, match="selfie"):
        _render(doc)


def test_citation_from_table_caption_counts() -> None:
    doc = ReportDoc()
    block = doc.figure("plotted", "data:x", "alt", "cap")
    tab = doc.table("vals", "the values plotted in Figure @fig:plotted", "<tr></tr>", "<tr></tr>")
    doc.section("s", "S", f"{block}{tab}")
    html = _render(doc)
    assert "the values plotted in Figure 1" in html


def test_unknown_token_raises() -> None:
    doc = ReportDoc()
    doc.section("s", "S", "<p>see Figure @fig:nonexistent</p>")
    with pytest.raises(ValueError, match="unresolved reference @fig:nonexistent"):
        _render(doc)


def test_duplicate_slugs_raise() -> None:
    doc = ReportDoc()
    doc.section("s", "S", "<p>x</p>")
    with pytest.raises(ValueError, match="duplicate section"):
        doc.addendum("s", "S again", "<p>y</p>")
    doc.figure("f", "data:x", "a", "c")
    with pytest.raises(ValueError, match="duplicate figure"):
        doc.figure("f", "data:x", "a", "c")


# --------------------------------------------------------------------------
# Contract tests over the two variants, rendered from the committed metrics.

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synthetic_html() -> str:
    reports = REPO / "reports"
    m = json.loads((reports / "metrics.json").read_text())
    seed8 = reports / "metrics_seed8.json"
    m8 = json.loads(seed8.read_text()) if seed8.exists() else None
    return compose_report(synthetic_context(m, m8, reports))


@pytest.fixture(scope="module")
def real_html() -> str:
    run_dir = REPO / "reports" / "chicago_demo"
    m = json.loads((run_dir / "metrics.json").read_text())
    return compose_report(real_context(m, run_dir))


def _assert_every_figure_cited(html: str) -> None:
    numbers = re.findall(r"<figcaption><strong>Figure (\d+)\.</strong>", html)
    assert numbers, "report has no figures"
    prose = re.sub(r"<figcaption>.*?</figcaption>", "", html, flags=re.DOTALL)
    for n in numbers:
        assert f"Figure {n}" in prose, f"Figure {n} is never cited outside its caption"


def test_every_figure_cited_synthetic(synthetic_html: str) -> None:
    _assert_every_figure_cited(synthetic_html)


def test_every_figure_cited_real(real_html: str) -> None:
    _assert_every_figure_cited(real_html)


def test_no_unresolved_tokens(synthetic_html: str, real_html: str) -> None:
    for html in (synthetic_html, real_html):
        for marker in ("@sec:", "@fig:", "@tab:"):
            assert marker not in html


def test_variant_shape(synthetic_html: str, real_html: str) -> None:
    assert "Addendum A. Synthetic ground truth" in synthetic_html
    assert "Oracle on latent log-time (ceiling)" in synthetic_html
    assert "Why ranking by validation Sharpe fails" in synthetic_html
    assert "Addendum" not in real_html
    assert "Oracle on latent log-time" not in real_html
    assert "Why ranking by validation Sharpe fails" not in real_html
    assert "This dataset has no known generating process." in real_html


def test_shared_skeleton(synthetic_html: str, real_html: str) -> None:
    def titles(html: str) -> list[str]:
        found = re.findall(r"<h2>(?:Addendum )?[A-Z0-9]+\. ([^<]+)</h2>", html)
        return [t.strip() for t in found]

    syn, real = titles(synthetic_html), titles(real_html)
    shared = ["Summary", "Data", "Method", "Results", "What the model uses", "Limitations"]
    assert [t for t in syn if t in real] == shared
    assert [t for t in real if t in syn] == shared
    assert syn[-2].startswith("Reproducing") or syn[-1].startswith("Reproducing")
    assert real[-1].startswith("Reproducing")


def test_generic_prose_carries_no_dataset_specific_claims(real_html: str) -> None:
    # The generic template must not assert facts about a particular dataset.
    # Pins the Chicago clause that once leaked into every real-data report
    # ("licences in the same trade and the same year fail together").
    assert "in the same trade" not in real_html
