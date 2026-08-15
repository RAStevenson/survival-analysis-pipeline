from __future__ import annotations

import sys
from pathlib import Path

# Same bootstrap as scripts/run_*.py: put src/ on the import path so a fresh
# clone runs the tests with no install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from survival_analysis_pipeline.features import build_features
from survival_analysis_pipeline.generate import GeneratorConfig, generate


@pytest.fixture(scope="session")
def small_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate(GeneratorConfig(n_strategies=600, seed=123))


@pytest.fixture(scope="session")
def medium_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate(GeneratorConfig(n_strategies=2500, seed=99))


@pytest.fixture(scope="session")
def small_features(small_data: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    return build_features(small_data[0])
