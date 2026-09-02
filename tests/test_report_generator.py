"""Engine tests for the report renderer, plus contract tests over the two
built report variants."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from survival_analysis_pipeline.report_generator import (
    ReportDoc,
    compose_report,
    real_context,
    synthetic_context,
)


def _render(doc: ReportDoc) -> str:
    return doc.render(doctype="Test", title="T", subtitle="s", meta_rows="", footer="f")


def test_section_numbering() -> None:
    doc = ReportDoc()
    doc.section("intro", "Intro", "<p>See section @sec:method and section @sec:extra.</p>")
    doc.section("method", "Method", "<p>body</p>")
    doc.section("extra", "Extra", "<p>body</p>")
    html = _render(doc)
    assert "<h2>1. Intro</h2>" in html
    assert "<h2>2. Method</h2>" in html
    assert "<h2>3. Extra</h2>" in html
    assert "See section 2 and section 3." in html


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


def test_uncited_table_fails_render() -> None:
    # Tables answer to the same cited-or-fail rule as figures; an uncited
    # decile table shipped before the rule covered them.
    doc = ReportDoc()
    tab = doc.table("orphan", "a caption", "<tr><th>h</th></tr>", "<tr><td>1</td></tr>")
    doc.section("s", "S", f"<p>prose with no citation</p>{tab}")
    with pytest.raises(ValueError, match=r"orphan.*never cited"):
        _render(doc)


def test_citation_inside_own_table_caption_does_not_count() -> None:
    doc = ReportDoc()
    tab = doc.table("selfie", "this is Table @tab:selfie itself", "<tr></tr>", "<tr></tr>")
    doc.section("s", "S", f"<p>no real citation</p>{tab}")
    with pytest.raises(ValueError, match="selfie"):
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
    doc.section("s", "S", f"<p>See Table @tab:vals.</p>{block}{tab}")
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
        doc.section("s", "S again", "<p>y</p>")
    doc.figure("f", "data:x", "a", "c")
    with pytest.raises(ValueError, match="duplicate figure"):
        doc.figure("f", "data:x", "a", "c")


# --------------------------------------------------------------------------
# Contract tests over the two variants, rendered from the committed metrics.

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synthetic_html() -> str:
    run_dir = REPO / "reports" / "synthetic"
    m = json.loads((run_dir / "metrics.json").read_text())
    return compose_report(synthetic_context(m, run_dir))


@pytest.fixture(scope="module")
def real_html() -> str:
    run_dir = REPO / "reports" / "chicago_demo"
    m = json.loads((run_dir / "metrics.json").read_text())
    return compose_report(real_context(m, run_dir))


@pytest.fixture(scope="module")
def flchain_html() -> str:
    run_dir = REPO / "reports" / "flchain_demo"
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


def test_every_figure_cited_flchain(flchain_html: str) -> None:
    _assert_every_figure_cited(flchain_html)


def test_no_unresolved_tokens(synthetic_html: str, real_html: str, flchain_html: str) -> None:
    for html in (synthetic_html, real_html, flchain_html):
        for marker in ("@sec:", "@fig:", "@tab:"):
            assert marker not in html


def test_reports_carry_no_machine_specific_paths(
    synthetic_html: str, real_html: str, flchain_html: str
) -> None:
    """A report prints its run directory in the header and footer. Built from an
    absolute run path it printed the author's home directory on page one of a
    repository headed for publication, which a reader cannot use and should not
    see. The builder makes the path repo-relative; this pins that."""
    for name, html in (
        ("synthetic", synthetic_html),
        ("real", real_html),
        ("flchain", flchain_html),
    ):
        assert str(REPO) not in html, f"{name} report contains the checkout's absolute path"
        assert str(REPO.as_posix()) not in html, f"{name} report contains the checkout path"
        for pattern in ("C:/Users", "C:\\Users", "/home/", "/Users/"):
            assert pattern not in html, f"{name} report contains a home-directory path: {pattern}"


def test_variant_shape(synthetic_html: str, real_html: str, flchain_html: str) -> None:
    assert "Oracle ranking on latent log-time (ceiling)" in synthetic_html
    assert "ranking predates the noise draw" in synthetic_html
    for html in (synthetic_html, real_html, flchain_html):
        assert "Addendum" not in html
    for html in (real_html, flchain_html):
        assert "Oracle ranking" not in html


def test_cox_dissection_renders_from_committed_metrics(synthetic_html: str, real_html: str) -> None:
    """Every committed run's metrics carry cox_top, so every committed report
    dissects both models, not only the boosted one."""
    for html in (synthetic_html, real_html):
        assert "The Cox baseline" in html
        assert "hazard ratio" in html.lower()


def test_shared_skeleton(synthetic_html: str, real_html: str) -> None:
    def titles(html: str) -> list[str]:
        found = re.findall(r"<h2>[0-9]+\. ([^<]+)</h2>", html)
        return [t.strip() for t in found]

    syn, real = titles(synthetic_html), titles(real_html)
    # Notes-driven sections sit outside the shared template skeleton;
    # everything else must match exactly, in order.
    note_sections = {"Motivation", "Interpretation"}

    def template(ts: list[str]) -> list[str]:
        return [t for t in ts if t not in note_sections]

    shared = [
        "Summary",
        "Data",
        "Method",
        "Results",
        "Feature analysis",
        "Limitations",
        "Reproducing this run",
    ]
    assert template(syn) == shared
    assert template(real) == shared
    # Notes sections land at their fixed anchors: motivation directly after
    # the summary, interpretation directly after the attribution section.
    if "Motivation" in syn:
        assert syn.index("Motivation") == syn.index("Summary") + 1
    for ts in (syn, real):
        if "Interpretation" in ts:
            assert ts.index("Interpretation") == ts.index("Feature analysis") + 1


def test_real_report_prose_follows_the_time_unit(tmp_path_factory) -> None:
    """The Chicago metrics recast as an hours run must render with hours
    wording everywhere the unit is data-driven. Horizon keys and the
    calibration figure name carry the unit abbreviation, so the fixture
    renames them the way an hours run would have written them."""
    import shutil

    src = REPO / "reports" / "chicago_demo"
    run_dir = tmp_path_factory.mktemp("hours") / "run"
    shutil.copytree(src / "figures", run_dir / "figures")

    m = json.loads((src / "metrics.json").read_text())
    m["config"]["time_unit"] = "hours"
    m["run"]["time_unit"] = "hours"
    m["ipcw_brier"] = {k.removesuffix("d") + "h": v for k, v in m["ipcw_brier"].items()}
    h_cal = int(m["config"]["calibration_horizon_days"])
    m[f"calibration_{h_cal}h"] = m.pop(f"calibration_{h_cal}d")
    fig = run_dir / "figures" / f"calibration_{h_cal}d.png"
    fig.rename(run_dir / "figures" / f"calibration_{h_cal}h.png")

    html = compose_report(real_context(m, run_dir))
    assert f"Decile calibration at {h_cal} hours" in html
    assert "365 hours" in html  # the Brier table's horizon column
    assert "--time-unit hours" in html  # the reproduce command
    # The one remaining "days" is the schema-named --duration-col value
    # (licensed_days) inside code spans; no prose sentence may assert days.
    assert " days." not in html and " days," not in html


def test_generic_prose_carries_no_dataset_specific_claims(real_html: str) -> None:
    # The generic template must not assert facts about a particular dataset.
    # Pins the Chicago clause that once leaked into every real-data report
    # ("licences in the same trade and the same year fail together").
    assert "in the same trade" not in real_html


def test_dataset_notes_render_in_real_report(real_html: str) -> None:
    # Dataset-specific prose enters only through the run's notes/ directory.
    # The committed Chicago notes carry these anchors; each must surface in
    # its section.
    assert "61,351" in real_html  # data note: the left-truncation exclusion
    assert "Endings near the cutoff are provisional." in real_html  # limitation
    assert "terms expiring" in real_html  # data note, above the KM figure
    # the fold-3 cause now reads as a finding in the interpretation note
    assert "worth investigating against" in real_html
    assert "No cause is established" not in real_html
    # interpretation note tokens resolved to numbers, not left as @val
    assert "@val" not in real_html


def test_within_group_hedge_travels_with_pooled_figure(
    synthetic_html: str, real_html: str, flchain_html: str
) -> None:
    # The decomposition must appear in the summary, not only in the results
    # section, on every run that computes one.
    hedge = "gives the decomposition"
    for html in (synthetic_html, real_html, flchain_html):
        assert hedge in html
        assert html.index(hedge) < html.index(". Data</h2>")


def test_flchain_notes_render(flchain_html: str) -> None:
    # The benchmark reconciliation and the metric-naming rule, pinned so a
    # notes regeneration cannot drop either silently.
    assert "around 0.794" in flchain_html
    assert "Brier rows in the results section" in flchain_html
    # All four anchors carry the authored-prose marker.
    assert flchain_html.count("Analyst notes:") == 4
