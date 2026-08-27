from __future__ import annotations

import pytest

from survival_analysis_pipeline.time_units import (
    TIME_UNITS,
    check_time_unit,
    horizon_label,
    unit_abbrev,
    unit_seconds,
    unit_singular,
)


def test_days_wordings_match_the_original_day_based_artifacts():
    # "d", "day", 86400: anything else would rename every existing metrics
    # key, figure file, and prediction column produced by day-based runs.
    assert unit_abbrev("days") == "d"
    assert unit_singular("days") == "day"
    assert unit_seconds("days") == 86400.0


def test_unknown_unit_is_refused_naming_the_options():
    with pytest.raises(ValueError, match=r"fortnights.*seconds.*years"):
        check_time_unit("fortnights")


def test_abbreviations_are_unique():
    # Horizon keys like "90mo" must parse back to exactly one unit.
    abbrevs = [unit_abbrev(u) for u in TIME_UNITS]
    assert len(abbrevs) == len(set(abbrevs))


def test_month_and_year_use_consistent_averages():
    assert unit_seconds("years") == 365.25 * 86400.0
    assert unit_seconds("months") * 12 == pytest.approx(unit_seconds("years"))


def test_horizon_label_keeps_whole_numbers_and_distinguishes_fractions():
    """int() formatting collapsed the year-unit horizons 0.25, 0.49, and
    0.99 into one colliding '0' key; whole-number spellings must stay
    exactly as the day-based artifacts had them."""
    assert horizon_label(90.0) == "90"
    assert horizon_label(1825) == "1825"
    assert horizon_label(0.25) == "0.25"
    labels = {horizon_label(h) for h in (0.25, 0.49, 0.99)}
    assert len(labels) == 3
