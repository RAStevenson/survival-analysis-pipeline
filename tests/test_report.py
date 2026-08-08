"""Engine tests for the report renderer, plus (later tasks) contract tests
over the two built report variants."""

from __future__ import annotations

import pytest

from strategy_survival.report import ReportDoc


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
