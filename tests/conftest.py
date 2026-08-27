from __future__ import annotations

import sys
from pathlib import Path

# Same bootstrap as scripts/run_*.py: put src/ on the import path so a fresh
# clone runs the tests with no install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from survival_analysis_pipeline.duration_csv import LoadedData, load_duration_csv
from survival_analysis_pipeline.synthetic_generator import GeneratorConfig, generate
from survival_analysis_pipeline.synthetic_schema import LATENT_COLUMNS


@pytest.fixture(scope="session")
def small_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate(GeneratorConfig(n_strategies=600, seed=123))


@pytest.fixture(scope="session")
def medium_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate(GeneratorConfig(n_strategies=2500, seed=99))


def _as_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write a generated frame out the way the pipeline always meets it: an
    ordinary duration CSV, with no latent column anywhere in it."""
    assert not set(LATENT_COLUMNS) & set(df.columns)
    df.to_csv(path, index=False)
    return path


def _load(path: Path) -> LoadedData:
    return load_duration_csv(
        path,
        id_col="strategy_id",
        date_col="discovery_date",
        duration_col="duration_days",
        event_col="event",
    )


@pytest.fixture(scope="session")
def small_csv(small_data, tmp_path_factory) -> Path:
    return _as_csv(small_data[0], tmp_path_factory.mktemp("export") / "strategies.csv")


@pytest.fixture(scope="session")
def small_loaded(small_csv: Path) -> LoadedData:
    return _load(small_csv)


@pytest.fixture(scope="session")
def small_features(small_loaded: LoadedData) -> pd.DataFrame:
    return small_loaded.features


@pytest.fixture(scope="session")
def medium_features(medium_data, tmp_path_factory) -> pd.DataFrame:
    path = _as_csv(medium_data[0], tmp_path_factory.mktemp("export-medium") / "strategies.csv")
    return _load(path).features
