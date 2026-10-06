"""The console readout states each main number with its reading guidance, and judges each
result with the same sentences the report uses, so the two never disagree."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from survival_analysis_pipeline.metrics_readout import readout_lines
from survival_analysis_pipeline.report_generator import (
    _fold_mean_comparison,
    _losing_horizons,
    _within_group_lead,
)
from survival_analysis_pipeline.time_units import unit_abbrev

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
def test_readout_judgments_match_the_committed_report(run: str) -> None:
    """Every judgment sentence the readout prints appears word for word in that run's report."""
    metrics = _metrics(run)
    pooled, time_unit = metrics["pooled"], metrics["config"].get("time_unit", "days")
    judgments = [
        _fold_mean_comparison(pooled["c_xgb_by_fold_mean"], pooled["c_cox_by_fold_mean"])[0]
    ]
    losing_text = _losing_horizons(metrics["ipcw_brier"], unit_abbrev(time_unit), time_unit)
    if losing_text:
        judgments.append(losing_text)
    within_group = metrics.get("within_group")
    if within_group:
        judgments.append(
            _within_group_lead(within_group["col"], within_group["c_group_mean"], pooled["c_xgb"])
        )
    readout = _visible_text(" ".join(readout_lines(metrics, REPO / "reports" / run)))
    report = _visible_text((REPO / "reports" / run / "report.html").read_text(encoding="utf-8"))
    for judgment in judgments:
        assert judgment in readout, judgment
        assert judgment in report, judgment


def test_near_tie_is_called_a_tie_and_the_recommendation_says_so() -> None:
    readout = _readout(_metrics("synthetic"))
    assert "tie at the printed precision" in readout
    assert "Cox baseline scores higher" not in readout
    assert "on a near-tie margin" in readout


def test_clear_boosted_win_names_the_boosted_model() -> None:
    metrics = copy.deepcopy(_metrics("flchain_demo"))
    metrics["pooled"]["c_xgb_by_fold_mean"] = 0.85
    readout = _readout(metrics)
    assert "The boosted model scores higher." in readout
    assert "Recommended for scoring new rows: the boosted model." in readout


def test_oracle_and_group_blocks_appear_only_when_measured() -> None:
    synthetic = _readout(_metrics("synthetic"))
    assert "oracle" in synthetic
    bare = copy.deepcopy(_metrics("flchain_demo"))
    bare.pop("within_group")
    readout = _readout(bare)
    assert "oracle" not in readout
    assert "Groups by" not in readout


def test_no_losing_horizon_gets_its_own_sentence() -> None:
    readout = _readout(_metrics("synthetic"))
    assert "Both models beat the no-skill forecast at every horizon." in _visible_text(readout)


def test_merged_folds_are_reported_against_the_request() -> None:
    readout = _readout(_metrics("flchain_demo"))
    assert "Testing: 3 folds (5 requested)." in readout
    assert "merged into one" in _visible_text(readout)


def test_report_location_is_printed_only_when_given() -> None:
    metrics = _metrics("flchain_demo")
    assert "report.html" not in _readout(metrics)
    assert "runs/example/report.html" in _readout(metrics, Path("runs/example/report.html"))
