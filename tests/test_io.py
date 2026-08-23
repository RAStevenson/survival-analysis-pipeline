from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survival_analysis_pipeline.io import (
    DURATION,
    EVENT,
    START,
    EncodingRecipe,
    check_minimum_data,
    encode_with_recipe,
    load_duration_csv,
    make_fold_encoder,
)


def write_csv(tmp_path, frame: pd.DataFrame, name: str = "data.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


@pytest.fixture()
def good_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uid": [f"r{i}" for i in range(8)],
            "signup": ["2021-01-15"] * 4 + ["2022-06-01"] * 4,
            "days": [10.0, 20, 30, 40, 50, 60, 70, 80],
            "died": [1, 0, 1, 1, 0, 1, 0, 1],
            "age": [30, 40, 50, 60, 35, 45, 55, 65],
            "score": [1.5, 2.5, np.nan, 4.5, 5.5, 6.5, 7.5, 8.5],
        }
    )


def load(path):
    return load_duration_csv(
        path, id_col="uid", date_col="signup", duration_col="days", event_col="died"
    )


def test_good_csv_loads(tmp_path, good_frame):
    data = load(write_csv(tmp_path, good_frame))
    assert len(data.frame) == 8
    assert data.frame[DURATION].tolist() == [10, 20, 30, 40, 50, 60, 70, 80]
    assert data.frame[EVENT].sum() == 5
    assert data.frame[START].dtype.kind == "M"
    assert list(data.features.columns) == ["age", "score"]
    # NaN passes through for XGBoost's native missing handling.
    assert np.isnan(data.features["score"].iloc[2])


def test_categorical_one_hot(tmp_path, good_frame):
    good_frame["region"] = ["north", "south", "north", "east", "south", "north", "east", "north"]
    data = load(write_csv(tmp_path, good_frame))
    assert "region=east" in data.features.columns
    assert data.features["region=north"].sum() == 4
    assert data.recipe.categorical_levels["region"] == ("east", "north", "south")


def test_constant_and_all_null_columns_dropped(tmp_path, good_frame, capsys):
    good_frame["constant"] = 7
    good_frame["empty"] = np.nan
    data = load(write_csv(tmp_path, good_frame))
    assert "constant" not in data.features.columns
    assert "empty" not in data.features.columns
    assert set(data.recipe.dropped_columns) == {"constant", "empty"}
    assert "dropped 2 constant or all-null" in capsys.readouterr().out


def test_all_problems_reported_at_once(tmp_path, good_frame):
    bad = good_frame.copy()
    bad.loc[1, "uid"] = "r0"  # duplicate id
    bad.loc[2, "signup"] = "not-a-date"
    bad.loc[3, "days"] = -5
    bad.loc[4, "died"] = 2
    with pytest.raises(ValueError) as err:
        load(write_csv(tmp_path, bad))
    message = str(err.value)
    assert "4 problems" in message
    assert "duplicate" in message
    assert "not-a-date" in message
    assert "<= 0" in message
    assert "0 (censored) or 1" in message


def test_missing_columns_reported_together(tmp_path, good_frame):
    path = write_csv(tmp_path, good_frame.drop(columns=["uid", "died"]))
    with pytest.raises(ValueError) as err:
        load(path)
    message = str(err.value)
    assert "'uid'" in message and "'died'" in message


def test_missing_drop_column_is_an_error(tmp_path, good_frame):
    path = write_csv(tmp_path, good_frame)
    with pytest.raises(ValueError, match="drop columns not found"):
        load_duration_csv(
            path,
            id_col="uid",
            date_col="signup",
            duration_col="days",
            event_col="died",
            drop_cols=("no_such_column",),
        )


def test_drop_cols_excluded_from_features(tmp_path, good_frame):
    good_frame["leaky"] = range(8)
    path = write_csv(tmp_path, good_frame)
    data = load_duration_csv(
        path,
        id_col="uid",
        date_col="signup",
        duration_col="days",
        event_col="died",
        drop_cols=("leaky",),
    )
    assert "leaky" not in data.features.columns


def test_recipe_round_trips_through_dict(tmp_path, good_frame):
    good_frame["region"] = ["a", "b"] * 4
    data = load(write_csv(tmp_path, good_frame))
    rebuilt = EncodingRecipe.from_dict(data.recipe.to_dict())
    assert rebuilt == data.recipe


def test_encode_with_recipe_matches_training_encoding(tmp_path, good_frame):
    good_frame["region"] = ["north", "south"] * 4
    data = load(write_csv(tmp_path, good_frame))
    re_encoded = encode_with_recipe(good_frame[["age", "score", "region"]], data.recipe)
    pd.testing.assert_frame_equal(re_encoded, data.features)


def test_encode_with_recipe_missing_column_fails(tmp_path, good_frame):
    data = load(write_csv(tmp_path, good_frame))
    with pytest.raises(ValueError, match=r"missing feature columns.*'score'"):
        encode_with_recipe(good_frame[["age"]], data.recipe)


def test_encode_with_recipe_unseen_level_is_zero_flags(tmp_path, good_frame, capsys):
    good_frame["region"] = ["north", "south"] * 4
    data = load(write_csv(tmp_path, good_frame))
    new = pd.DataFrame({"age": [33], "score": [2.0], "region": ["west"]})
    encoded = encode_with_recipe(new, data.recipe)
    assert encoded[["region=north", "region=south"]].to_numpy().sum() == 0
    assert "unseen in training" in capsys.readouterr().out


def test_numeric_code_column_can_be_forced_categorical(tmp_path, good_frame):
    """Nothing in a file distinguishes a ward number from a quantity. Left to
    dtype inference these become one continuous axis, and a linear model fits
    them a single monotonic slope: the Chicago demo shipped a Cox model
    asserting hazard rises steadily with ward number."""
    good_frame["ward"] = [1, 1, 12, 12, 43, 43, 7, 7]
    path = write_csv(tmp_path, good_frame)

    inferred = load(path)
    assert "ward" in inferred.recipe.numeric_columns

    forced = load_duration_csv(
        path,
        id_col="uid",
        date_col="signup",
        duration_col="days",
        event_col="died",
        categorical_cols=("ward",),
    )
    assert "ward" not in forced.recipe.numeric_columns
    assert forced.recipe.categorical_levels["ward"] == ("1", "12", "43", "7")
    assert forced.features["ward=43"].sum() == 2


def test_forced_categorical_must_name_a_real_feature_column(tmp_path, good_frame):
    path = write_csv(tmp_path, good_frame)
    with pytest.raises(ValueError, match="categorical columns are not feature columns"):
        load_duration_csv(
            path,
            id_col="uid",
            date_col="signup",
            duration_col="days",
            event_col="died",
            categorical_cols=("no_such_column",),
        )


def test_code_levels_do_not_depend_on_the_inferred_dtype(tmp_path, good_frame):
    """The same column is float64 when every value parses as a number and object
    when one row does not, and astype(str) renders those as '42.0' and '42'. A
    model trained through one path and scored through the other would treat
    every level as unseen."""
    good_frame["ward"] = [1, 1, 12, 12, 43, 43, 7, 7]
    data = load_duration_csv(
        write_csv(tmp_path, good_frame),
        id_col="uid",
        date_col="signup",
        duration_col="days",
        event_col="died",
        categorical_cols=("ward",),
    )
    assert "ward=12" in data.features.columns
    assert "ward=12.0" not in data.features.columns

    as_text = pd.DataFrame({"age": [33], "score": [2.0], "ward": ["12"]})
    as_float = pd.DataFrame({"age": [33], "score": [2.0], "ward": [12.0]})
    pd.testing.assert_frame_equal(
        encode_with_recipe(as_text, data.recipe), encode_with_recipe(as_float, data.recipe)
    )
    assert encode_with_recipe(as_float, data.recipe)["ward=12"].iloc[0] == 1.0


def test_missing_category_gets_its_own_level(tmp_path, good_frame):
    """All-zero dummies is exactly how the dropped reference level encodes, so
    without a (missing) level a row with no ward scores as whichever ward sorted
    first. On the Chicago file that would be 11 percent of rows."""
    good_frame["region"] = ["north", "south", None, "north", "south", "north", None, "south"]
    data = load(write_csv(tmp_path, good_frame))

    assert "(missing)" in data.recipe.categorical_levels["region"]
    assert data.features["region=(missing)"].sum() == 2
    # The reference level a linear model drops must be a real level, not the gap.
    assert data.recipe.reference_columns == ("region=north",)
    # Every row lands in exactly one level, which is what makes the gap visible.
    flags = [c for c in data.features.columns if c.startswith("region=")]
    assert (data.features[flags].sum(axis=1) == 1).all()


def test_rare_levels_collapse_into_other(tmp_path):
    """Never exercised before, yet it produced three of the four categorical
    encodings in the committed Chicago demo."""
    n = 400
    frame = pd.DataFrame(
        {
            "uid": [f"r{i}" for i in range(n)],
            "signup": ["2021-01-15"] * n,
            "days": np.arange(1, n + 1, dtype=float),
            "died": [1, 0] * (n // 2),
            "kind": ["common"] * (n - 3) + ["rare_a", "rare_b", "rare_c"],
        }
    )
    data = load(write_csv(tmp_path, frame))
    levels = data.recipe.categorical_levels["kind"]

    assert "(other)" in levels
    assert "rare_a" not in levels
    assert data.features["kind=(other)"].sum() == 3


def test_day_first_dates_are_refused_rather_than_silently_reordered(tmp_path, good_frame):
    """Per-element format inference splits a day-first column: 01/02/2021 reads
    as 2 January while 13/02/2021 reads as 13 February, because only the second
    is impossible to read month-first. This column sets every fold's split date,
    so a reordering would quietly stop the evaluation being out of time."""
    good_frame["signup"] = [
        "01/02/2021",
        "13/02/2021",
        "02/03/2021",
        "14/03/2021",
        "03/04/2021",
        "15/04/2021",
        "04/05/2021",
        "16/05/2021",
    ]
    with pytest.raises(ValueError, match="do not match the format the rest of the column uses"):
        load(write_csv(tmp_path, good_frame))


def test_durations_that_outrun_the_calendar_are_refused(tmp_path, good_frame):
    """The one unit mismatch the loader can prove: 2,400 hours recorded in
    the duration column but declared days puts every implied end years past
    today, which observed data cannot do."""
    good_frame["days"] = [2400.0] * 8
    with pytest.raises(ValueError, match="end in the future"):
        load(write_csv(tmp_path, good_frame))


def test_same_durations_pass_when_declared_in_their_real_unit(tmp_path, good_frame):
    good_frame["days"] = [2400.0] * 8
    data = load_duration_csv(
        write_csv(tmp_path, good_frame),
        id_col="uid",
        date_col="signup",
        duration_col="days",
        event_col="died",
        time_unit="hours",
    )
    assert len(data.frame) == 8


def test_future_end_check_survives_absurd_magnitudes(tmp_path, good_frame):
    """Month-scale lifetimes in seconds declared as days imply ends tens of
    thousands of years out, past what a pandas timestamp can represent; the
    check must refuse, not overflow."""
    good_frame["days"] = [7.9e6] * 8
    with pytest.raises(ValueError, match="end in the future"):
        load(write_csv(tmp_path, good_frame))


def test_timestamp_dates_with_seconds_parse_and_keep_time_of_day(tmp_path, good_frame):
    """Start columns finer than a date are legitimate input: sub-day
    precision now reaches the re-censoring step, so it must survive the
    load rather than being truncated or refused."""
    good_frame["signup"] = [f"2021-01-15 0{i}:30:0{i}" for i in range(4)] + [
        f"2022-06-01 1{i}:45:3{i}" for i in range(4)
    ]
    data = load(write_csv(tmp_path, good_frame))
    assert data.frame[START].dt.second.tolist() == [0, 1, 2, 3, 30, 31, 32, 33]


def test_year_only_dates_parse_as_years(tmp_path, good_frame):
    """An integer year column must read as the year, not as nanoseconds
    after 1970; the loader stringifies before parsing for exactly this."""
    good_frame["signup"] = [1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004]
    data = load(write_csv(tmp_path, good_frame))
    assert data.frame[START].dt.year.tolist() == list(range(1997, 2005))


def test_minimum_data_guards():
    assert check_minimum_data(n_rows=5000, n_events=1000, n_folds=5) is None
    warning = check_minimum_data(50, 40, 5)
    assert warning is not None and "below the minimum of 300" in warning
    message = check_minimum_data(400, 90, 5)
    assert message is not None
    assert "need 200" in message and "fewer folds" in message


def test_fold_encoder_vocabulary_is_train_window_only():
    """A level that first appears after a fold's split date must not shape
    that fold's training matrix. The vocabulary was once fit on the full
    file, which let future level frequencies leak into early folds'
    features; the fold encoder pins the fix."""
    raw = pd.DataFrame(
        {
            "kind": ["a"] * 4 + ["b"] * 2 + ["late_only"] * 4,
            "size": [float(i) for i in range(10)],
        }
    )
    encode = make_fold_encoder(raw, ())
    x_train, x_eval, cox_drop = encode(np.arange(6), np.arange(6, 10))

    assert not any("late_only" in c for c in x_train.columns)
    assert list(x_train.columns) == list(x_eval.columns)
    # Training created no (other) level, so the unseen level encodes as
    # all-zero flags, exactly the scoring-time rule for new rows.
    kind_cols = [c for c in x_eval.columns if c.startswith("kind=")]
    assert kind_cols and (x_eval[kind_cols].to_numpy() == 0.0).all()
    assert cox_drop == ("kind=a",)
