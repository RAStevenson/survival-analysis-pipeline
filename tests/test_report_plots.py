from __future__ import annotations

import pandas as pd

from survival_analysis_pipeline.report_plots import cox_hr_plot, wrap_label


def test_cox_hr_plot_writes_a_figure(tmp_path):
    coefficients = pd.DataFrame(
        [
            {"feature": "age", "hr": 2.4, "hr_lo": 2.1, "hr_hi": 2.7},
            {
                "feature": "license_description=Consumption on Premises",
                "hr": 0.6,
                "hr_lo": 0.4,
                "hr_hi": 0.9,
            },
            {"feature": "monthly_fee", "hr": 1.1, "hr_lo": 0.9, "hr_hi": 1.3},
        ]
    )
    path = tmp_path / "cox_hr.png"
    cox_hr_plot(coefficients, path)
    assert path.exists() and path.stat().st_size > 0


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
