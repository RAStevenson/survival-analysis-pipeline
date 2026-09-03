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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
from dataclasses import fields

from survival_analysis_pipeline.aft_model import AFTParams
from survival_analysis_pipeline.cox_model import CoxBaseline
from survival_analysis_pipeline.duration_csv import (
    DURATION,
    EVENT,
    ID,
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
    data = load_duration_csv(
        ROOT / "datasets" / "chicago_licences.csv.gz",
        "licence_id",
        "first_issued",
        "licensed_days",
        "closed",
        (),
        CATEGORICAL,
    )
    frame = data.frame
    feature_cols = [c for c in frame.columns if c not in (ID, START, DURATION, EVENT)]
    encode = make_fold_encoder(frame[feature_cols], CATEGORICAL)

    committed = json.loads((ROOT / "reports" / "chicago_demo" / "metrics.json").read_text())
    folds = temporal_folds(frame[START], committed["config"]["n_folds"], 0.4)
    worst_idx = min(range(len(folds)), key=lambda i: committed["folds"][i]["c_xgb"])
    fold = folds[worst_idx]
    reported = committed["folds"][worst_idx]
    print(
        f"weakest fold is {worst_idx + 1}, split {fold.split_date.date()}, "
        f"train {len(fold.train_idx):,}, test {len(fold.test_idx):,}"
    )

    field_names = {f.name for f in fields(AFTParams)}
    params = AFTParams(**{k: v for k, v in committed["params"].items() if k in field_names})

    dates = frame[START]
    train_dur, train_ev = recensor(
        frame[DURATION].to_numpy()[fold.train_idx],
        frame[EVENT].to_numpy()[fold.train_idx],
        dates.iloc[fold.train_idx],
        fold.split_date,
        committed["config"]["time_unit"],
    )
    x_train, x_test, cox_drop = encode(fold.train_idx, fold.test_idx)
    test_dur = frame[DURATION].to_numpy()[fold.test_idx]
    test_ev = frame[EVENT].to_numpy()[fold.test_idx]
    groups_test = frame[GROUP_COL].iloc[fold.test_idx]
    groups_train = frame[GROUP_COL].iloc[fold.train_idx]

    aft = _fit_aft(params, x_train, train_dur, train_ev, dates.iloc[fold.train_idx])
    cox = CoxBaseline(drop_columns=cox_drop).fit(x_train, train_dur, train_ev)
    pred_aft = aft.predict_median_time(x_test)
    c_aft = harrell_c(test_dur, test_ev, pred_aft)
    c_cox = harrell_c(test_dur, test_ev, cox.predict_neg_risk(x_test))
    print(
        f"reproduced: c_aft {c_aft:.4f} (committed {reported['c_xgb']:.4f}), "
        f"c_cox {c_cox:.4f} (committed {reported['c_cox']:.4f})"
    )

    for name, pred in (("AFT", pred_aft), ("Cox", cox.predict_neg_risk(x_test))):
        d = within_group_concordance(test_dur, test_ev, pred, groups_test)
        assert d is not None, "decomposition unavailable: no group met the size thresholds"
        print(
            f"  {name} decomposition: group_mean {d['c_group_mean']:.4f}, "
            f"within {d['c_within']:.4f}, groups {d['n_groups']}"
        )

    print("\n--- the note's claims, recomputed ---")

    years = frame[START].dt.year
    groups_all = frame[GROUP_COL]
    last_issued = {c: int(years[groups_all == c].max()) for c in VANISHED}
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

    tr_share = groups_train.value_counts(normalize=True)
    te_share = groups_test.value_counts(normalize=True)
    vanished_share = sum(float(tr_share.get(c, 0.0)) for c in VANISHED)
    still_present = [c for c in VANISHED if float(te_share.get(c, 0.0)) > 0]
    print(
        f'"{" and ".join(VANISHED)} carry 11 percent of its training rows and stop '
        f"appearing in the test block entirely"
        f"\n    -> {vanished_share:.1%} of training rows; "
        f"still present in test: {still_present or 'none'}"
    )

    grew_from, grew_to = float(tr_share.get(GREW, 0.0)), float(te_share.get(GREW, 0.0))
    ratio = grew_to / grew_from if grew_from else float("inf")
    print(
        f'"{GREW} nearly triples its share"'
        f"\n    -> {grew_from:.1%} of train to {grew_to:.1%} of test, a factor of {ratio:.2f}"
    )

    unseen = ~groups_test.isin(set(groups_train.unique()))
    print(
        '"Another 2 percent of test rows fall in categories the training window never saw"'
        f"\n    -> {int(unseen.sum()):,} rows, {unseen.mean():.2%}"
    )

    params3 = _select_params(x_train, train_dur, train_ev, dates.iloc[fold.train_idx])
    if params3 == params:
        print(
            "\"Re-selecting hyperparameters on that fold's own window closes about a third "
            'of the gap"\n    -> re-selection picked the same grid point; no gap closed'
        )
    else:
        aft3 = _fit_aft(params3, x_train, train_dur, train_ev, dates.iloc[fold.train_idx])
        c_aft3 = harrell_c(test_dur, test_ev, aft3.predict_median_time(x_test))
        gap, closed = c_cox - c_aft, c_aft3 - c_aft
        print(
            "\"Re-selecting hyperparameters on that fold's own window closes about a third "
            f'of the gap"\n    -> {params3}: c_aft {c_aft3:.4f}, closing {closed:.4f} '
            f"of a {gap:.4f} gap ({closed / gap:.0%})"
        )


if __name__ == "__main__":
    main()
