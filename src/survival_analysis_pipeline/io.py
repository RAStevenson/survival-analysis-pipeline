"""Loading, validating, and encoding right-censored duration data.

This is the door, and every run comes through it, the synthetic study
included: it maps user column names onto the canonical (id, start date,
duration, event) contract, collects every validation problem before failing so
one pass reports them all, and turns the remaining columns into a model-ready
feature table. It does no feature engineering, so a file arrives carrying
whatever features its author prepared. Numeric columns pass through (NaN allowed;
XGBoost treats missing values natively, and the Cox baseline imputes train
medians downstream). Text columns are one-hot encoded. Constant and all-null
columns are dropped with a printed notice. The encoding recipe is kept so a
saved model can re-apply the identical encoding to new rows at prediction
time; a recipe mismatch is an error, never a silent zero-fill of a whole
column.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .units import unit_seconds

# Canonical internal column names after mapping. Neutral wording on purpose:
# the contract covers churn and equipment failure as well as strategies, and
# the duration is unit-neutral, in whatever timestep the run declared.
ID = "row_id"
START = "start_date"
DURATION = "duration"
EVENT = "event"

_MAX_EXAMPLES = 5  # bad values quoted per problem before "and N more"


@dataclass(frozen=True)
class EncodingRecipe:
    """Everything needed to rebuild the exact training feature matrix from a
    raw frame: which columns were numeric, which were categorical and with
    which levels, what was dropped, and the final column order.

    `reference_columns` names one dummy per categorical column. Tree models
    use the full set, but a linear model must not: a categorical's dummies
    sum to 1, so the full set is collinear with the intercept and the design
    matrix is singular. Linear baselines drop these.
    """

    numeric_columns: tuple[str, ...]
    categorical_levels: dict[str, tuple[str, ...]]
    dropped_columns: tuple[str, ...]
    feature_names: tuple[str, ...]
    reference_columns: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "numeric_columns": list(self.numeric_columns),
            "categorical_levels": {k: list(v) for k, v in self.categorical_levels.items()},
            "dropped_columns": list(self.dropped_columns),
            "feature_names": list(self.feature_names),
            "reference_columns": list(self.reference_columns),
        }

    @classmethod
    def from_dict(cls, d: dict) -> EncodingRecipe:
        return cls(
            numeric_columns=tuple(d["numeric_columns"]),
            categorical_levels={k: tuple(v) for k, v in d["categorical_levels"].items()},
            dropped_columns=tuple(d["dropped_columns"]),
            feature_names=tuple(d["feature_names"]),
            reference_columns=tuple(d.get("reference_columns", ())),
        )


@dataclass(frozen=True)
class LoadedData:
    """frame carries the canonical columns plus the raw feature columns;
    features is the encoded matrix aligned to frame's rows."""

    frame: pd.DataFrame
    features: pd.DataFrame
    recipe: EncodingRecipe


def _examples(values: pd.Series) -> str:
    shown = [repr(v) for v in values.head(_MAX_EXAMPLES).tolist()]
    extra = len(values) - len(shown)
    tail = f" and {extra} more" if extra > 0 else ""
    return ", ".join(shown) + tail


def load_duration_csv(
    path: str | Path,
    id_col: str,
    date_col: str,
    duration_col: str,
    event_col: str,
    drop_cols: tuple[str, ...] = (),
    categorical_cols: tuple[str, ...] = (),
    time_unit: str = "days",
) -> LoadedData:
    """Read a duration CSV, validate the contract, encode the features.

    `time_unit` is the declared unit of the duration column. The loader
    cannot verify a unit in general, since a number carries none, but one
    direction is checkable: a row's start date plus its observed duration
    cannot land in the future, so durations that outrun the calendar mean
    the declared unit is coarser than the data (hours read as days behave
    exactly like this) or the column holds planned rather than observed
    durations. Both are refused; neither has a legitimate override.

    `categorical_cols` forces columns to be treated as labels regardless of
    their dtype. Nothing in a file distinguishes a ward number from a
    quantity, so a column of administrative codes is otherwise read as a
    continuous axis and a linear model fits it one monotonic slope, asserting
    that ward 50 carries five times whatever ward 10 carries. Only the person
    who knows the data can make that call, so the tool takes it as input
    rather than guessing.

    Raises ValueError listing every problem found, not just the first.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"no such file: {path}")
    # low_memory=False so a column's dtype is decided from the whole file rather
    # than per chunk. Chunked inference can hand back object for one read and
    # float64 for another on the same file, which changes how a column is
    # encoded.
    raw = pd.read_csv(path, low_memory=False)
    problems: list[str] = []

    for role, col in (
        ("id", id_col),
        ("date", date_col),
        ("duration", duration_col),
        ("event", event_col),
    ):
        if col not in raw.columns:
            problems.append(f"{role} column {col!r} not found in {path.name}")
    missing_drops = [c for c in drop_cols if c not in raw.columns]
    if missing_drops:
        problems.append(
            "drop columns not found (typo would silently keep a column you "
            f"meant to exclude): {', '.join(repr(c) for c in missing_drops)}"
        )
    if problems:
        raise ValueError(_format_problems(path, problems))

    ids = raw[id_col].astype(str)
    if raw[id_col].isna().any():
        problems.append(f"id column {id_col!r} has {int(raw[id_col].isna().sum())} null values")
    dupes = ids[ids.duplicated()]
    if len(dupes) > 0:
        problems.append(
            f"id column {id_col!r} has {len(dupes)} duplicate values: {_examples(dupes)}"
        )

    # Parse dates from their text form: an integer year column like 1997 must
    # mean the year 1997, not (as pandas would read a raw int) a moment 1997
    # nanoseconds after 1970.
    #
    # One format is inferred for the whole column, deliberately. The permissive
    # alternative, format="mixed", infers per element, so a day-first column
    # silently splits: 01/02/2021 reads as 2 January and 13/02/2021 as 13
    # February, because only the second is impossible to read month-first. That
    # reorders the column, and this column sets every fold's split date, so the
    # out-of-time evaluation would quietly stop being out of time. Refusing a
    # column that cannot be read one way is the safe failure.
    date_text = raw[date_col].astype(str).where(raw[date_col].notna())
    dates = pd.to_datetime(date_text, errors="coerce")
    bad_dates = raw[date_col][dates.isna() & raw[date_col].notna()]
    if len(bad_dates) > 0:
        problems.append(
            f"date column {date_col!r} has {len(bad_dates)} values that do not match the "
            f"format the rest of the column uses: {_examples(bad_dates)} (every row must "
            f"parse the same way; a day-first file needs converting to ISO first)"
        )
    if raw[date_col].isna().any():
        problems.append(
            f"date column {date_col!r} has {int(raw[date_col].isna().sum())} null values "
            "(a row with no start date cannot be placed in a temporal fold)"
        )

    durations = pd.to_numeric(raw[duration_col], errors="coerce")
    bad_dur = raw[duration_col][durations.isna()]
    if len(bad_dur) > 0:
        problems.append(
            f"duration column {duration_col!r} has {len(bad_dur)} non-numeric or "
            f"null values: {_examples(bad_dur)}"
        )
    nonpositive = raw[duration_col][durations <= 0]
    if len(nonpositive) > 0:
        problems.append(
            f"duration column {duration_col!r} has {len(nonpositive)} values <= 0 "
            f"(a survival time must be positive; drop or correct these rows): "
            f"{_examples(nonpositive)}"
        )

    # The future-end tripwire described in the docstring. Epoch-second floats
    # rather than date arithmetic, because a badly mismatched unit implies
    # end dates thousands of years out, past what pandas timestamps can hold.
    ok = dates.notna() & durations.notna() & (durations > 0)
    if ok.any():
        start_s = dates[ok].astype("int64").to_numpy(dtype=float) / 1e9
        implied_end_s = start_s + durations[ok].to_numpy(dtype=float) * unit_seconds(time_unit)
        # One timestep of slack for durations rounded up to a coarse unit,
        # plus three days for clocks and timezones.
        limit_s = pd.Timestamp.now().timestamp() + unit_seconds(time_unit) + 3 * 86400.0
        future = implied_end_s > limit_s
        if future.any():
            offenders = raw.loc[ok].loc[future.tolist()]
            ex = ", ".join(
                f"{r[date_col]} + {float(r[duration_col]):g} {time_unit}"
                for _, r in offenders.head(_MAX_EXAMPLES).iterrows()
            )
            extra = len(offenders) - min(len(offenders), _MAX_EXAMPLES)
            tail = f" and {extra} more" if extra > 0 else ""
            problems.append(
                f"{len(offenders)} rows end in the future: start date plus duration, "
                f"read as {time_unit}, lands past today ({ex}{tail}). An observed "
                "duration cannot outrun the calendar, so either the duration column "
                f"is in a finer unit than the declared {time_unit!r}, or it holds "
                "planned rather than observed durations; fix the unit or the data"
            )

    events = pd.to_numeric(raw[event_col], errors="coerce")
    bad_event = raw[event_col][~events.isin([0, 1])]
    if len(bad_event) > 0:
        problems.append(
            f"event column {event_col!r} must be 0 (censored) or 1 (event observed); "
            f"{len(bad_event)} other values: {_examples(bad_event)}"
        )

    reserved = {id_col, date_col, duration_col, event_col, *drop_cols}
    feature_cols = [c for c in raw.columns if c not in reserved]
    clashes = [c for c in feature_cols if c in {ID, START, DURATION, EVENT}]
    if clashes:
        problems.append(
            "feature columns clash with the canonical internal names "
            f"({', '.join(repr(c) for c in clashes)}); rename them or map them "
            "with the column flags"
        )

    unknown_categorical = [c for c in categorical_cols if c not in feature_cols]
    if unknown_categorical:
        problems.append(
            "categorical columns are not feature columns in this file (check for a typo, "
            "or for a column also named in --drop-cols): "
            + ", ".join(repr(c) for c in unknown_categorical)
        )

    if problems:
        raise ValueError(_format_problems(path, problems))
    features, recipe = _encode_features(raw[feature_cols], tuple(categorical_cols))

    frame = pd.DataFrame(
        {
            ID: ids.to_numpy(),
            START: dates.to_numpy(),
            DURATION: durations.to_numpy(dtype=float),
            EVENT: events.to_numpy(dtype=int),
        }
    )
    frame = pd.concat([frame, raw[feature_cols].reset_index(drop=True)], axis=1)
    return LoadedData(frame=frame, features=features, recipe=recipe)


def _format_problems(path: Path, problems: list[str]) -> str:
    lines = "\n".join(f"  - {p}" for p in problems)
    plural = "problem" if len(problems) == 1 else "problems"
    return f"{path.name}: {len(problems)} {plural} found:\n{lines}"


OTHER = "(other)"
# A missing category gets its own level rather than all-zero flags. All-zero is
# exactly how the dropped reference level encodes, so without this a row with no
# ward scores as whichever ward happened to sort first, silently and with no
# notice. The Cox median imputation cannot catch it either, because a dummy is
# 0.0 and never NaN. On the Chicago file that is 11 percent of rows.
MISSING = "(missing)"


def _label_strings(series: pd.Series) -> pd.Series:
    """Stringify a column for one-hot encoding without depending on the dtype
    pandas happened to infer.

    The same code column is float64 when every value parses as a number and
    object when one row does not, and `astype(str)` renders those as '42.0' and
    '42'. A model trained through one path and scored through the other then
    sees every level as unseen and dumps the whole column into `(other)`, under
    a notice that names levels which were in fact in training. Rendering
    whole-valued numbers without the decimal tail makes both paths agree.
    """
    if pd.api.types.is_float_dtype(series):
        present = series.dropna()
        if len(present) > 0 and (present == present.round()).all():
            return series.map(lambda v: str(int(v)) if pd.notna(v) else None).astype(object)
    return series.astype(str).where(series.notna())


def _encode_features(
    raw: pd.DataFrame, force_categorical: tuple[str, ...] = (), verbose: bool = True
) -> tuple[pd.DataFrame, EncodingRecipe]:
    numeric: list[str] = []
    categorical: dict[str, tuple[str, ...]] = {}
    dropped: list[str] = []

    # Levels seen only a handful of times cannot support an estimate and are a
    # common source of complete separation in a linear model, where one rare
    # level perfectly predicts the outcome and the fit fails to converge.
    # Scaled to the dataset but capped: any level with 200 observations can
    # support an estimate however large the file, and on a small file the
    # threshold has to fall to 1 or it would delete every categorical column.
    min_level_count = max(1, min(200, round(0.005 * len(raw))))
    collapsed_total = 0

    for col in raw.columns:
        series = raw[col]
        forced = col in force_categorical
        if series.isna().all() or series.nunique(dropna=True) <= 1:
            dropped.append(col)
        elif not forced and (
            pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
        ):
            numeric.append(col)
        else:
            counts = _label_strings(series).dropna().value_counts()
            kept = sorted(counts[counts >= min_level_count].index)
            n_collapsed = len(counts) - len(kept)
            if n_collapsed > 0:
                collapsed_total += n_collapsed
                kept = [*kept, OTHER]
            if series.isna().any():
                kept = [*kept, MISSING]
            if len(kept) <= 1:
                dropped.append(col)
            else:
                categorical[col] = tuple(kept)

    if dropped and verbose:
        print(f"dropped {len(dropped)} constant or all-null feature columns: {', '.join(dropped)}")
    if collapsed_total and verbose:
        print(
            f"collapsed {collapsed_total} categorical levels seen fewer than "
            f"{min_level_count} times into {OTHER}"
        )

    recipe = EncodingRecipe(
        numeric_columns=tuple(numeric),
        categorical_levels=categorical,
        dropped_columns=tuple(dropped),
        feature_names=_feature_names(numeric, categorical),
        reference_columns=tuple(f"{col}={levels[0]}" for col, levels in categorical.items()),
    )
    return _apply_encoding(raw, recipe), recipe


def _feature_names(numeric: list[str], categorical: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    names = list(numeric)
    for col, levels in categorical.items():
        names.extend(f"{col}={level}" for level in levels)
    return tuple(names)


def _apply_encoding(raw: pd.DataFrame, recipe: EncodingRecipe) -> pd.DataFrame:
    # Assembled as a dict and concatenated once. Assigning a few hundred
    # one-hot columns individually reallocates the frame each time.
    columns: dict[str, np.ndarray] = {}
    for col in recipe.numeric_columns:
        columns[col] = pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype=float)
    for col, levels in recipe.categorical_levels.items():
        as_str = _label_strings(raw[col])
        if OTHER in levels:
            known = set(levels) - {OTHER, MISSING}
            as_str = as_str.where(as_str.isin(known) | as_str.isna(), OTHER)
        if MISSING in levels:
            as_str = as_str.fillna(MISSING)
        values = as_str.to_numpy()
        for level in levels:
            columns[f"{col}={level}"] = (values == level).astype(float)
    x = pd.DataFrame(columns, index=pd.RangeIndex(len(raw)))
    return x[list(recipe.feature_names)]


def encode_with_recipe(raw: pd.DataFrame, recipe: EncodingRecipe) -> pd.DataFrame:
    """Encode new rows exactly as the training data was encoded.

    Missing feature columns are an error (a model scoring rows that lack a
    training feature is silently wrong). Unseen categorical levels join the
    training column's `(other)` level when one exists, otherwise they encode
    as all-zeros for that column's flags; a printed notice says which.
    Extra columns are ignored with a printed notice.
    """
    required = list(recipe.numeric_columns) + list(recipe.categorical_levels)
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(
            "input is missing feature columns the model was trained on: "
            + ", ".join(repr(c) for c in missing)
        )
    extra = [c for c in raw.columns if c not in required]
    if extra:
        print(f"ignoring {len(extra)} columns the model was not trained on: {', '.join(extra)}")

    for col, levels in recipe.categorical_levels.items():
        seen = set(levels)
        values = _label_strings(raw[col]).dropna()
        unseen = sorted(set(values.unique()) - seen)
        if unseen:
            n_rows = int(values.isin(unseen).sum())
            landing = (
                f"they join the {OTHER} bucket"
                if OTHER in levels
                else "they encode as all-zero flags"
            )
            print(
                f"column {col!r}: {n_rows} rows have levels unseen in training "
                f"({', '.join(unseen)}); {landing}"
            )
    return _apply_encoding(raw, recipe)


def make_fold_encoder(
    raw_features: pd.DataFrame, categorical_cols: tuple[str, ...] = ()
) -> Callable[[np.ndarray, np.ndarray], tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]]:
    """Per-window encoding for temporal cross-validation.

    The returned encode(train_idx, eval_idx) learns the vocabulary (levels,
    rare-level collapse, constant drops) from the training rows alone, so
    level frequencies from after a fold's split date cannot shape the
    features that fold trains on. Eval rows are encoded with the training
    recipe, unseen levels landing per the same rules as scoring new rows.
    The third element is the fold recipe's reference columns for the Cox
    baseline's collinearity drop.
    """

    def encode(
        train_idx: np.ndarray, eval_idx: np.ndarray
    ) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
        x_train, recipe = _encode_features(
            raw_features.iloc[train_idx].reset_index(drop=True),
            tuple(categorical_cols),
            verbose=False,
        )
        x_eval = _apply_encoding(raw_features.iloc[eval_idx].reset_index(drop=True), recipe)
        return x_train, x_eval, recipe.reference_columns

    return encode


def save_model_bundle(
    dir_path: str | Path,
    model,
    recipe: EncodingRecipe,
    meta: dict,
    cox=None,
    scores: dict | None = None,
) -> None:
    """Save both fitted models plus one sidecar carrying what the model files
    cannot: the predictive scale, the encoding recipe, the out-of-time score
    each model earned, and which of the two to prefer.

    Saving only the boosted model would be a quiet lie whenever the Cox
    baseline scored higher, which on real data it may well do.
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    if model.booster is None:
        raise RuntimeError("model not fitted")
    model.booster.save_model(str(dir_path / "booster.json"))

    models = {"aft": {"file": "booster.json", "c_index_fold_mean": None}}
    if cox is not None:
        cox.save(dir_path / "cox.pkl")
        models["cox"] = {"file": "cox.pkl", "c_index_fold_mean": None}
    if scores:
        for key, value in scores.items():
            if key in models:
                models[key]["c_index_fold_mean"] = float(value)

    scored = {k: v["c_index_fold_mean"] for k, v in models.items() if v["c_index_fold_mean"]}
    recommended = max(scored, key=scored.get) if scored else "aft"

    sidecar = {
        "predictive_sigma": model.predictive_sigma,
        "params": {
            "max_depth": model.params.max_depth,
            "learning_rate": model.params.learning_rate,
            "n_rounds": model.params.n_rounds,
            "aft_sigma": model.params.aft_sigma,
        },
        "models": models,
        "recommended": recommended,
        "recipe": recipe.to_dict(),
        **meta,
    }
    (dir_path / "sidecar.json").write_text(json.dumps(sidecar, indent=2))


def load_model_bundle(dir_path: str | Path):
    """Returns (aft model, recipe, sidecar dict). Import here avoids a cycle:
    model.py must not depend on io.py."""
    import xgboost as xgb

    from .model import AFTParams, XGBoostAFT

    dir_path = Path(dir_path)
    booster_path = dir_path / "booster.json"
    sidecar_path = dir_path / "sidecar.json"
    for p in (booster_path, sidecar_path):
        if not p.exists():
            raise ValueError(f"not a model directory (missing {p.name}): {dir_path}")
    sidecar = json.loads(sidecar_path.read_text())
    p = sidecar["params"]
    model = XGBoostAFT(
        AFTParams(
            max_depth=int(p["max_depth"]),
            learning_rate=float(p["learning_rate"]),
            n_rounds=int(p["n_rounds"]),
            aft_sigma=float(p["aft_sigma"]),
        )
    )
    model.booster = xgb.Booster()
    model.booster.load_model(str(booster_path))
    model.predictive_sigma = sidecar["predictive_sigma"]
    recipe = EncodingRecipe.from_dict(sidecar["recipe"])
    return model, recipe, sidecar


def load_cox_from_bundle(dir_path: str | Path):
    """The Cox baseline saved alongside the AFT model, or None if the bundle
    predates two-model saving or the run had no Cox fit."""
    from .baseline import CoxBaseline

    path = Path(dir_path) / "cox.pkl"
    return CoxBaseline.load(path) if path.exists() else None


def check_minimum_data(n_rows: int, n_events: int, n_folds: int) -> str | None:
    """Refusal message when the dataset cannot support the requested folds,
    None when it can. Thresholds: 300 rows overall, and 40 observed events per
    fold on average. Below that, fold metrics are mostly noise and the refusal
    says so rather than printing unstable numbers.

    The event threshold is an aggregate proxy, not a per-fold guarantee: this
    function sees totals, so a dataset whose events cluster in one window can
    still pass with a thin fold. The per-fold sizes are printed in every
    report's fold table, which is where such a run shows itself.
    """
    min_rows = 300
    events_needed = 40 * n_folds
    problems = []
    if n_rows < min_rows:
        problems.append(f"{n_rows} rows is below the minimum of {min_rows}")
    if n_events < events_needed:
        problems.append(
            f"{n_events} observed events cannot support {n_folds} folds "
            f"(need {events_needed}, at 40 per fold); try fewer folds"
        )
    if not problems:
        return None
    return "refusing to fit: " + "; ".join(problems)
