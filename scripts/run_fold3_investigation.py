#!/usr/bin/env python3
"""Where does the Chicago demo's weakest fold lose to the Cox baseline?

    python scripts/run_fold3_investigation.py

This is the provenance for the fold-3 paragraph in
reports/chicago_demo/notes/interpretation.md. That paragraph's numbers,
unlike the note's other figures, cannot come from @val tokens, because
nothing in metrics.json holds them: they are properties of the dataset's
category timeline and one fold's composition, not of the run's results. So
the script that measured them is committed here, and it prints each claim
beside the value it recomputes.

It refits one fold, which takes a couple of minutes and touches nothing.
No file is written and no committed artifact changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
from dataclasses import fields

from survival_analysis_pipeline.aft_model import AFTParams
from survival_analysis_pipeline.cox_model import CoxBaseline
from survival_analysis_pipeline.duration_csv import (
    DURATION,
    EVENT,
    ROW_ID,
    START,
    load_duration_csv,
    make_fold_encoder,
)
from survival_analysis_pipeline.evaluate_model import harrell_c, within_group_concordance
from survival_analysis_pipeline.fit_evaluate import _fit_aft, _select_params
from survival_analysis_pipeline.temporal_folds import recensor, temporal_folds

ROOT = Path(__file__).resolve().parents[1]
CATEGORICAL = ("ward", "community_area", "police_district", "zip_code")
GROUP_COL = "license_description"
# The two categories the note says vanish from the test block, and the one it
# says grows. Named here so the script checks the note's actual claims rather
# than whatever happens to top the table.
VANISHED = ("Home Occupation", "Home Repair")
GREW = "Regulated Business License"


def main() -> None:
    """Refit Chicago's fold 3 and print each claim in its note beside the value recomputed."""
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    loaded = load_duration_csv(
        ROOT / "datasets" / "chicago_licences.csv.gz",
        "licence_id",
        "first_issued",
        "licensed_days",
        "closed",
        (),
        CATEGORICAL,
    )
    dataset = loaded.frame
    feature_cols = [
        column for column in dataset.columns if column not in (ROW_ID, START, DURATION, EVENT)
    ]
    fold_encoder = make_fold_encoder(dataset[feature_cols], CATEGORICAL)

    committed = json.loads((ROOT / "reports" / "chicago_demo" / "metrics.json").read_text())
    folds = temporal_folds(dataset[START], committed["config"]["n_folds"], 0.4)
    weakest_position = min(range(len(folds)), key=lambda i: committed["folds"][i]["c_xgb"])
    fold = folds[weakest_position]
    reported = committed["folds"][weakest_position]
    print(
        f"weakest fold is {weakest_position + 1}, split {fold.split_date.date()}, "
        f"train {len(fold.train_rows):,}, test {len(fold.test_rows):,}"
    )

    field_names = {field.name for field in fields(AFTParams)}
    params = AFTParams(
        **{name: value for name, value in committed["params"].items() if name in field_names}
    )

    dates = dataset[START]
    train_durations, train_events = recensor(
        dataset[DURATION].to_numpy()[fold.train_rows],
        dataset[EVENT].to_numpy()[fold.train_rows],
        dates.iloc[fold.train_rows],
        fold.split_date,
        committed["config"]["time_unit"],
    )
    train_features, test_features, cox_drop_columns = fold_encoder(fold.train_rows, fold.test_rows)
    test_durations = dataset[DURATION].to_numpy()[fold.test_rows]
    test_events = dataset[EVENT].to_numpy()[fold.test_rows]
    groups_test = dataset[GROUP_COL].iloc[fold.test_rows]
    groups_train = dataset[GROUP_COL].iloc[fold.train_rows]

    aft = _fit_aft(
        params, train_features, train_durations, train_events, dates.iloc[fold.train_rows]
    )
    cox = CoxBaseline(drop_columns=cox_drop_columns).fit(
        train_features, train_durations, train_events
    )
    aft_predicted_median = aft.predict_median_time(test_features)
    c_aft = harrell_c(test_durations, test_events, aft_predicted_median)
    c_cox = harrell_c(test_durations, test_events, cox.predict_neg_risk(test_features))
    print(
        f"reproduced: c_aft {c_aft:.4f} (committed {reported['c_xgb']:.4f}), "
        f"c_cox {c_cox:.4f} (committed {reported['c_cox']:.4f})"
    )

    for name, predicted_score in (
        ("AFT", aft_predicted_median),
        ("Cox", cox.predict_neg_risk(test_features)),
    ):
        decomposition = within_group_concordance(
            test_durations, test_events, predicted_score, groups_test
        )
        assert decomposition is not None, (
            "decomposition unavailable: no group met the size thresholds"
        )
        print(
            f"  {name} decomposition: group_mean {decomposition['c_group_mean']:.4f}, "
            f"within {decomposition['c_within']:.4f}, groups {decomposition['n_groups']}"
        )

    print("\n--- the note's claims, recomputed ---")

    years = dataset[START].dt.year
    groups_all = dataset[GROUP_COL]
    last_issued = {category: int(years[groups_all == category].max()) for category in VANISHED}
    grew_years = years[groups_all == GREW]
    grew_first = int(grew_years.min())
    grew_first_n = int((grew_years == grew_first).sum())
    grew_later_peak = int(grew_years[grew_years > grew_first].value_counts().max())
    print(
        f'"The city stopped issuing the {" and ".join(VANISHED)} license types in 2012, '
        f'the same year {GREW} first appears with a one-year spike of issues"'
        f"\n    -> last issued {last_issued}; {GREW} first appears {grew_first} "
        f"with {grew_first_n:,} issues against a later-year peak of {grew_later_peak:,}"
    )

    train_share = groups_train.value_counts(normalize=True)
    test_share = groups_test.value_counts(normalize=True)
    vanished_share = sum(float(train_share.get(category, 0.0)) for category in VANISHED)
    still_present = [category for category in VANISHED if float(test_share.get(category, 0.0)) > 0]
    print(
        f'"{" and ".join(VANISHED)} carry 11 percent of its training rows and stop '
        f"appearing in the test block entirely"
        f"\n    -> {vanished_share:.1%} of training rows; "
        f"still present in test: {still_present or 'none'}"
    )

    grew_from, grew_to = float(train_share.get(GREW, 0.0)), float(test_share.get(GREW, 0.0))
    growth_factor = grew_to / grew_from if grew_from else float("inf")
    print(
        f'"{GREW} nearly triples its share"'
        f"\n    -> {grew_from:.1%} of train to {grew_to:.1%} of test, "
        f"a factor of {growth_factor:.2f}"
    )

    unseen = ~groups_test.isin(set(groups_train.unique()))
    print(
        '"Another 2 percent of test rows fall in categories the training window never saw"'
        f"\n    -> {int(unseen.sum()):,} rows, {unseen.mean():.2%}"
    )

    reselected_params = _select_params(
        train_features, train_durations, train_events, dates.iloc[fold.train_rows]
    )
    if reselected_params == params:
        print(
            "\"Re-selecting hyperparameters on that fold's own window closes about a third "
            'of the gap"\n    -> re-selection picked the same grid point; no gap closed'
        )
    else:
        reselected_aft = _fit_aft(
            reselected_params,
            train_features,
            train_durations,
            train_events,
            dates.iloc[fold.train_rows],
        )
        reselected_c_aft = harrell_c(
            test_durations, test_events, reselected_aft.predict_median_time(test_features)
        )
        gap, gap_closed = c_cox - c_aft, reselected_c_aft - c_aft
        print(
            "\"Re-selecting hyperparameters on that fold's own window closes about a third "
            f'of the gap"\n    -> {reselected_params}: c_aft {reselected_c_aft:.4f}, '
            f"closing {gap_closed:.4f} "
            f"of a {gap:.4f} gap ({gap_closed / gap:.0%})"
        )


if __name__ == "__main__":
    main()
