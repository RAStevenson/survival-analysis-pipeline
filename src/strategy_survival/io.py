"""Loading, validating, and encoding real right-censored duration data.

The synthetic path builds its feature matrix from a known schema. This module
is the door for everything else: it maps user column names onto the canonical
(id, start date, duration, event) contract, collects every validation problem
before failing so one pass reports them all, and turns the remaining columns
into a model-ready feature table. Numeric columns pass through (NaN allowed;
XGBoost treats missing values natively, and the Cox baseline imputes train
medians downstream). Text columns are one-hot encoded. Constant and all-null
columns are dropped with a printed notice. The encoding recipe is kept so a
saved model can re-apply the identical encoding to new rows at prediction
time; a recipe mismatch is an error, never a silent zero-fill of a whole
column.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Canonical internal column names after mapping. Neutral wording on purpose:
# the contract covers churn and equipment failure as well as strategies.
ID = "row_id"
START = "start_date"
DURATION = "duration_days"
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
) -> LoadedData:
    """Read a duration CSV, validate the contract, encode the features.

    Raises ValueError listing every problem found, not just the first.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"no such file: {path}")
    raw = pd.read_csv(path)
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
    date_text = raw[date_col].astype(str).where(raw[date_col].notna())
    dates = pd.to_datetime(date_text, errors="coerce", format="mixed")
    bad_dates = raw[date_col][dates.isna()]
    if len(bad_dates) > 0:
        problems.append(
            f"date column {date_col!r} has {len(bad_dates)} unparseable values: "
            f"{_examples(bad_dates)}"
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

    if problems:
        raise ValueError(_format_problems(path, problems))
    features, recipe = _encode_features(raw[feature_cols])

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


def _encode_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, EncodingRecipe]:
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
        if series.isna().all() or series.nunique(dropna=True) <= 1:
            dropped.append(col)
        elif pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric.append(col)
        else:
            counts = series.dropna().astype(str).value_counts()
            kept = sorted(counts[counts >= min_level_count].index)
            n_collapsed = len(counts) - len(kept)
            if n_collapsed > 0:
                collapsed_total += n_collapsed
                kept = [*kept, OTHER]
            if len(kept) <= 1:
                dropped.append(col)
            else:
                categorical[col] = tuple(kept)

    if dropped:
        print(f"dropped {len(dropped)} constant or all-null feature columns: {', '.join(dropped)}")
    if collapsed_total:
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
        as_str = raw[col].astype(str).where(raw[col].notna())
        if OTHER in levels:
            known = set(levels) - {OTHER}
            as_str = as_str.where(as_str.isin(known) | as_str.isna(), OTHER)
        values = as_str.to_numpy()
        for level in levels:
            columns[f"{col}={level}"] = (values == level).astype(float)
    x = pd.DataFrame(columns, index=pd.RangeIndex(len(raw)))
    return x[list(recipe.feature_names)]


def encode_with_recipe(raw: pd.DataFrame, recipe: EncodingRecipe) -> pd.DataFrame:
    """Encode new rows exactly as the training data was encoded.

    Missing feature columns are an error (a model scoring rows that lack a
    training feature is silently wrong). Unseen categorical levels encode as
    all-zeros for that column's flags, with a printed notice. Extra columns
    are ignored with a printed notice.
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
        values = raw[col].dropna().astype(str)
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
    None when it can. Thresholds: 300 rows overall, 40 observed events per
    fold. Below that, fold metrics are mostly noise and the refusal says so
    rather than printing unstable numbers."""
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
