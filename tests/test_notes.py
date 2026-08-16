"""Tests for the notes mechanism: anchor discovery, token resolution, the
minimal markdown subset, and the failure contract (unknown anchors and
unresolvable tokens fail the build rather than rendering silently)."""

from __future__ import annotations

from pathlib import Path

import pytest

from survival_analysis_pipeline.notes import load_run_notes, resolve_tokens

VALUES = {
    "pooled": {"c_xgb": 0.5732853, "n_test": 191777},
    "folds": [{"c_xgb": 0.554}, {"c_xgb": 0.601}],
    "run": {"name": "chicago_licences"},
}


def test_token_resolves_with_format() -> None:
    out = resolve_tokens("scores @val{pooled.c_xgb:.3f} pooled.", VALUES)
    assert out == "scores 0.573 pooled."


def test_token_resolves_list_index_and_int_default() -> None:
    out = resolve_tokens("fold one at @val{folds.0.c_xgb:.3f}, n @val{pooled.n_test:,}.", VALUES)
    assert out == "fold one at 0.554, n 191,777."


def test_string_token_needs_no_format() -> None:
    assert resolve_tokens("run @val{run.name}.", VALUES) == "run chicago_licences."


def test_unknown_token_path_raises() -> None:
    with pytest.raises(ValueError, match="no key 'c_orcale'"):
        resolve_tokens("@val{pooled.c_orcale:.3f}", VALUES)


def test_float_without_format_raises() -> None:
    with pytest.raises(ValueError, match="needs an explicit format"):
        resolve_tokens("@val{pooled.c_xgb}", VALUES)


def test_unknown_anchor_filename_fails(tmp_path: Path) -> None:
    (tmp_path / "limitation.md").write_text("typo anchor", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown notes anchor 'limitation'"):
        load_run_notes(tmp_path, VALUES)


def test_missing_dir_and_empty_file_insert_nothing(tmp_path: Path) -> None:
    assert load_run_notes(None, VALUES) == {}
    assert load_run_notes(tmp_path / "absent", VALUES) == {}
    (tmp_path / "data.md").write_text("   \n\n  ", encoding="utf-8")
    assert load_run_notes(tmp_path, VALUES) == {}


def test_paragraphs_render_with_markers_and_inline_markdown(tmp_path: Path) -> None:
    (tmp_path / "data.md").write_text(
        "First paragraph with `code` and **bold**.\n\nSecond, scoring @val{folds.1.c_xgb:.3f}.",
        encoding="utf-8",
    )
    notes = load_run_notes(tmp_path, VALUES)
    assert notes["data"].startswith("<!--note:data-->")
    assert notes["data"].endswith("<!--/note:data-->")
    expected = "<p>First paragraph with <code>code</code> and <strong>bold</strong>.</p>"
    assert expected in notes["data"]
    assert "<p>Second, scoring 0.601.</p>" in notes["data"]


def test_heading_in_note_raises(tmp_path: Path) -> None:
    (tmp_path / "motivation.md").write_text("# My heading\n\nbody", encoding="utf-8")
    with pytest.raises(ValueError, match="template owns the headings"):
        load_run_notes(tmp_path, VALUES)


def test_retired_anchors_are_refused_by_name(tmp_path: Path) -> None:
    """The two micro-slots retired with the four-anchor rule. A notes folder
    still carrying one is a stale run, and the build says so rather than
    silently dropping prose its author expected to see."""
    for retired in ("worst_fold", "km_caption"):
        path = tmp_path / f"{retired}.md"
        path.write_text("Prose from an older layout.", encoding="utf-8")
        with pytest.raises(ValueError, match=f"unknown notes anchor '{retired}'"):
            load_run_notes(tmp_path, VALUES)
        path.unlink()


def test_html_in_note_source_is_escaped(tmp_path: Path) -> None:
    (tmp_path / "data.md").write_text("a < b and x > y stay text", encoding="utf-8")
    notes = load_run_notes(tmp_path, VALUES)
    assert "a &lt; b and x &gt; y stay text" in notes["data"]
