"""The console readout states each main number with a short guide, and judges each result
with the same outcome functions the report uses, so the two never disagree."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from survival_analysis_pipeline.metrics_readout import (
    _FOLD_MEAN_RESULTS,
    _WITHIN_GROUP_RESULTS,
    readout_lines,
    save_readout,
)
from survival_analysis_pipeline.report_generator import (
    _FOLD_MEAN_SENTENCES,
    _fold_mean_outcome,
    _loses_to_no_skill,
    _within_group_lead,
    _within_group_outcome,
)

REPO = Path(__file__).resolve().parents[1]
COMMITTED_RUNS = ("synthetic", "chicago_demo", "flchain_demo")


def _metrics(run: str) -> dict:
    return json.loads((REPO / "reports" / run / "metrics.json").read_text(encoding="utf-8"))


def _readout(metrics: dict, report_path: Path | None = None) -> str:
    return "\n".join(readout_lines(metrics, Path("runs/example"), report_path))


def _visible_text(text: str) -> str:
    """Text with HTML tags dropped and all line wrapping collapsed to single spaces."""
    return " ".join(re.sub(r"<[^>]+>", "", text).split())


@pytest.mark.parametrize("run", COMMITTED_RUNS)
def test_readout_and_report_state_the_same_results(run: str) -> None:
    """Each judgment comes from one shared outcome, worded in the readout and in the report."""
    metrics = _metrics(run)
    pooled = metrics["pooled"]
    readout = _readout(metrics)
    report = _visible_text((REPO / "reports" / run / "report.html").read_text(encoding="utf-8"))

    outcome = _fold_mean_outcome(pooled["c_xgb_by_fold_mean"], pooled["c_cox_by_fold_mean"])
    assert _FOLD_MEAN_RESULTS[outcome] in readout
    assert _FOLD_MEAN_SENTENCES[outcome] in report

    losses = [
        (horizon_key, model)
        for horizon_key, scores in metrics["ipcw_brier"].items()
        for model in ("xgb", "cox")
        if _loses_to_no_skill(scores[model], scores["km_marginal"])
    ]
    assert readout.count("*") == len(losses) + (1 if losses else 0)
    assert ("lose to a no-skill forecast" in report) == bool(losses)

    within_group = metrics.get("within_group")
    if within_group:
        group_outcome = _within_group_outcome(within_group["c_group_mean"], pooled["c_xgb"])
        assert _WITHIN_GROUP_RESULTS[group_outcome] in readout
        lead = _within_group_lead(
            within_group["col"], within_group["c_group_mean"], pooled["c_xgb"]
        )
        assert lead in report


def test_near_tie_is_called_a_tie_and_the_recommendation_says_so() -> None:
    readout = _readout(_metrics("synthetic"))
    assert "result      tie at the printed precision" in readout
    assert "Cox higher" not in readout
    assert "Recommended: Cox baseline, near-tie margin" in readout


def test_clear_boosted_win_names_the_boosted_model() -> None:
    metrics = copy.deepcopy(_metrics("flchain_demo"))
    metrics["pooled"]["c_xgb_by_fold_mean"] = 0.85
    readout = _readout(metrics)
    assert "result      boosted higher" in readout
    assert "Recommended: boosted model (run_predict.py default)" in readout


def test_oracle_and_group_blocks_appear_only_when_measured() -> None:
    synthetic = _readout(_metrics("synthetic"))
    assert "oracle" in synthetic
    bare = copy.deepcopy(_metrics("flchain_demo"))
    bare.pop("within_group")
    readout = _readout(bare)
    assert "oracle" not in readout
    assert "Groups by" not in readout


def test_no_losing_horizon_prints_no_marks() -> None:
    readout = _readout(_metrics("synthetic"))
    assert "*" not in readout


def test_merged_folds_are_reported_against_the_request() -> None:
    readout = _readout(_metrics("flchain_demo"))
    assert "Folds: 3 of 5 requested (merged where split dates coincide)" in readout


def test_report_location_is_printed_only_when_given() -> None:
    metrics = _metrics("flchain_demo")
    assert "Full report" not in _readout(metrics)
    assert "Full report: runs/example/report.html" in _readout(
        metrics, Path("runs/example/report.html")
    )


def test_saved_readout_matches_what_is_printed(tmp_path: Path) -> None:
    metrics = _metrics("flchain_demo")
    lines = save_readout(metrics, tmp_path, tmp_path / "report.html")
    assert (tmp_path / "readout.txt").read_bytes() == ("\n".join(lines) + "\n").encode("utf-8")


def test_readout_never_prints_an_absolute_path_under_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "example"
    lines = readout_lines(_metrics("flchain_demo"), run_dir, run_dir / "report.html")
    assert lines[0] == "flchain: runs/example"
    assert lines[-1] == "Full report: runs/example/report.html"
