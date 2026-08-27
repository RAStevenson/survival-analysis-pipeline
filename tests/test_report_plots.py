from __future__ import annotations

from survival_analysis_pipeline.report_plots import wrap_label


def test_short_names_pass_through_unchanged():
    assert wrap_label("latitude") == "latitude"
    assert wrap_label("ward=43") == "ward=43"


def test_long_one_hot_name_breaks_at_the_equals_then_wraps():
    name = "license_description=Consumption on Premises - Incidental Activity"
    lines = wrap_label(name).split("\n")
    assert lines[0] == "license_description="
    assert all(len(line) <= 28 for line in lines)
    assert " ".join(lines[1:]) == "Consumption on Premises - Incidental Activity"


def test_long_plain_name_wraps_without_an_equals():
    name = "walk_forward_positive_fraction_over_the_validation_window"
    wrapped = wrap_label(name)
    assert "\n" in wrapped or len(wrapped) <= 28
