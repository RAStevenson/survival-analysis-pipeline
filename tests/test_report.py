"""Engine tests for the report renderer, plus contract tests over the two
built report variants."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from survival_analysis_pipeline.report import (
    ReportDoc,
    compose_report,
    real_context,
    seed_dependence_para,
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
    return compose_report(synthetic_context(m, m8, reports, notes_dir=REPO / "notes" / "synthetic"))


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
    assert "Oracle ranking on latent log-time (ceiling)" in synthetic_html
    assert "Addendum" not in real_html
    assert "Oracle ranking" not in real_html
    assert "validation Sharpe" not in real_html
    assert "This dataset has no known generating process" in real_html


def test_shared_skeleton(synthetic_html: str, real_html: str) -> None:
    def titles(html: str) -> list[str]:
        found = re.findall(r"<h2>(?:Addendum )?[A-Z0-9]+\. ([^<]+)</h2>", html)
        return [t.strip() for t in found]

    syn, real = titles(synthetic_html), titles(real_html)
    # Notes-driven sections and the synthetic addendum sit outside the shared
    # template skeleton; everything else must match exactly, in order.
    note_sections = {"Motivation", "Interpretation"}

    def template(ts: list[str]) -> list[str]:
        return [t for t in ts if t not in note_sections and t != "Synthetic ground truth"]

    shared = [
        "Summary",
        "Data",
        "Method",
        "Results",
        "What the model uses",
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
            assert ts.index("Interpretation") == ts.index("What the model uses") + 1


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
    assert "-hour horizon" in html
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
    assert "licence terms expiring" in real_html  # km figure caption note
    # worst_fold note replaces the template's default no-cause sentence
    assert "category structure that had just moved" in real_html
    assert "No cause is established" not in real_html
    # interpretation note tokens resolved to numbers, not left as @val
    assert "@val" not in real_html


def test_within_group_hedge_travels_with_pooled_figure(synthetic_html: str, real_html: str) -> None:
    # The decomposition must appear in the Chicago summary, not only in its
    # results section; the synthetic run has no within-group split.
    hedge = "gives the decomposition"
    assert hedge in real_html
    assert real_html.index(hedge) < real_html.index(". Data</h2>")
    assert hedge not in synthetic_html


def test_flipped_sharpe_sentence_in_synthetic(synthetic_html: str) -> None:
    # Renders only when metrics.json carries c_sharpe_flipped; pins the R5
    # control sentence so a metrics regeneration cannot drop it silently.
    assert "arithmetic complement" in synthetic_html


def _seed_para_inputs(gap: float) -> tuple[dict, dict]:
    shap = [
        {"feature": "wf_positive_fraction", "mean_abs_shap": 0.30},
        {"feature": "wf_sharpe_decay", "mean_abs_shap": 0.20},
        {"feature": "wf_sharpe_std", "mean_abs_shap": 0.10},
    ]
    shap8 = [
        {"feature": "wf_positive_fraction", "mean_abs_shap": 0.25},
        {"feature": "wf_sharpe_decay", "mean_abs_shap": 0.25 - gap},
        {"feature": "wf_sharpe_std", "mean_abs_shap": 0.10},
    ]
    m = {
        "params": {"max_depth": 3, "aft_sigma": 0.6},
        "pooled": {"c_xgb": 0.78, "c_xgb_ci": [0.77, 0.79], "c_sharpe": 0.41},
        "shap_top": shap,
    }
    m8 = {
        "params": {"max_depth": 3, "aft_sigma": 0.6},
        "pooled": {"c_xgb": 0.785, "c_sharpe": 0.40, "c_oracle": 0.81},
        "shap_top": shap8,
    }
    return m, m8


def test_seed_order_claim_requires_margin() -> None:
    # A matching order separated by less than cross-platform attribution
    # drift must not be printed as a confirmation.
    m, m8 = _seed_para_inputs(gap=0.0008)
    para = seed_dependence_para(m, m8)
    assert "in the same order" not in para
    assert "a tie rather than a confirmation" in para


def test_seed_order_claim_stated_when_margin_is_real() -> None:
    m, m8 = _seed_para_inputs(gap=0.05)
    para = seed_dependence_para(m, m8)
    assert "in the same order" in para
