from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_survival.io import (
    DURATION,
    EVENT,
    START,
    EncodingRecipe,
    check_minimum_data,
    encode_with_recipe,
    load_duration_csv,
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


def test_minimum_data_guards():
    assert check_minimum_data(n_rows=5000, n_events=1000, n_folds=5) is None
    assert "below the minimum of 300" in check_minimum_data(50, 40, 5)
    message = check_minimum_data(400, 90, 5)
    assert "need 200" in message and "fewer folds" in message
