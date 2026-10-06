"""Ground-truth extras only the synthetic run can report.

The synthetic study goes through the same public door as any user file
(fit_evaluate.fit_evaluate), which by construction knows nothing about latent
truth. One measurement
therefore cannot come from that run and is computed here, afterwards, then
merged into the run's metrics.json: the oracle ceiling, the concordance of
the latent log survival time the generator actually used, which bounds
every model.

It is a label-only ranking: nothing is fitted, so it can be recomputed
from the file and the fold definitions without repeating the run. Fold
membership is rebuilt through the same `temporal_folds.temporal_folds` call the run
used, against the same frame the same loader produced, and checked against
the fold sizes the run already recorded before anything is joined.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .duration_csv import DURATION, EVENT, ROW_ID, START, load_duration_csv
from .evaluate_model import harrell_c
from .synthetic_generator import ASSET_CLASS_WEIGHTS, ASSET_LOG_TIME_EFFECT, GeneratorConfig
from .synthetic_schema import ASSET_CLASSES
from .temporal_folds import temporal_folds

# The synthetic file's column contract, in one place because two callers must
# agree on it exactly: the runner passes these to fit_evaluate, and the reload
# below has to reproduce that run's frame row for row.
ID_COL = "strategy_id"
DATE_COL = "discovery_date"
DURATION_COL = "duration_days"
EVENT_COL = "event"
KM_COL = "asset_class"


def _reconstruct_folds(dataset: pd.DataFrame, recorded_folds: list[dict], run_config: dict) -> list:
    """Rebuild the run's temporal folds from the reloaded frame and check their sizes against the
    metrics before anything is joined.
    """
    folds = temporal_folds(dataset[START], run_config["n_folds"], run_config["min_train_frac"])
    if len(folds) != len(recorded_folds):
        raise AssertionError(
            f"rebuilt {len(folds)} folds against {len(recorded_folds)} in the metrics; "
            "the run and this step disagree about the cross-validation scheme"
        )
    for i, (fold, fold_record) in enumerate(zip(folds, recorded_folds, strict=True), start=1):
        if (
            len(fold.train_rows) != fold_record["n_train"]
            or len(fold.test_rows) != fold_record["n_test"]
        ):
            raise AssertionError(
                f"fold {i} rebuilt as {len(fold.train_rows)} train / {len(fold.test_rows)} test "
                f"against {fold_record['n_train']} / {fold_record['n_test']} in the metrics. "
                "The reloaded frame "
                "does not match the one the run scored, so any latent joined onto it would be "
                "misaligned and the oracle figures would be silently wrong."
            )
    return folds


def add_synthetic_extras(
    run_dir: str | Path,
    data_path: str | Path,
    latents_path: str | Path,
    generator: GeneratorConfig,
) -> dict:
    """Merge the generator block and the oracle ceiling into the run's
    metrics.json. Returns the updated metrics."""
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())

    loaded = load_duration_csv(
        data_path,
        ID_COL,
        DATE_COL,
        DURATION_COL,
        EVENT_COL,
        time_unit=metrics["config"]["time_unit"],
    )
    dataset = loaded.frame
    folds = _reconstruct_folds(dataset, metrics["folds"], metrics["config"])

    latents = pd.read_csv(latents_path)
    log_time_eta = (
        latents.set_index(ID_COL)["log_time_eta"].reindex(dataset[ROW_ID]).to_numpy(dtype=float)
    )
    if not np.isfinite(log_time_eta).all():
        raise AssertionError(
            "some rows have no latent after the join on "
            f"{ID_COL!r}; the latents file does not cover the data file"
        )
    duration = dataset[DURATION].to_numpy(dtype=float)
    event = dataset[EVENT].to_numpy()

    for fold, fold_record in zip(folds, metrics["folds"], strict=True):
        fold_test_rows = fold.test_rows
        fold_record["c_oracle"] = harrell_c(
            duration[fold_test_rows], event[fold_test_rows], log_time_eta[fold_test_rows]
        )

    test_rows = np.concatenate([fold.test_rows for fold in folds])
    pooled_durations, pooled_events = duration[test_rows], event[test_rows]
    pooled_log_time_eta = log_time_eta[test_rows]
    n_test_rows = len(test_rows)
    pooled = metrics["pooled"]
    if n_test_rows != pooled["n_test"]:
        raise AssertionError(
            f"rebuilt {n_test_rows} out-of-fold rows against {pooled['n_test']} in the metrics"
        )
    pooled["c_oracle"] = harrell_c(pooled_durations, pooled_events, pooled_log_time_eta)

    # The installed class effects are module constants, not config, so they
    # ride along here for the notes to cite; a note quoting a constant by
    # hand went stale silently on a generator change.
    metrics["generator"] = asdict(generator) | {
        "installed": {
            "asset_log_time_effect": dict(ASSET_LOG_TIME_EFFECT),
            "asset_class_weights": dict(zip(ASSET_CLASSES, ASSET_CLASS_WEIGHTS, strict=True)),
        }
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics
