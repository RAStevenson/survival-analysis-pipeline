"""Ground-truth extras only the synthetic run can report.

The synthetic study goes through the same public door as any user file
(fit_evaluate.fit_evaluate), which by construction knows nothing about latent
truth or about which observable column drove selection. Two measurements
therefore cannot come from that run and are computed here, afterwards, then
merged into the run's metrics.json:

  the oracle ceiling   concordance of the latent log survival time the
                       generator actually used, which bounds every model
  the Sharpe baseline  concordance of ranking by validation Sharpe, the
                       metric the population was selected on

Both are label-only rankings: nothing is fitted, so they can be recomputed
from the file and the fold definitions without repeating the run. Fold
membership is rebuilt through the same `cv.temporal_folds` call the run
used, against the same frame the same loader produced, and checked against
the fold sizes the run already recorded before anything is joined.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .duration_csv import DURATION, EVENT, ID, START, load_duration_csv
from .evaluate_model import bootstrap_ci, harrell_c
from .synthetic_generator import GeneratorConfig
from .temporal_folds import temporal_folds

# The synthetic file's column contract, in one place because two callers must
# agree on it exactly: the runner passes these to fit_evaluate, and the reload
# below has to reproduce that run's frame row for row.
ID_COL = "strategy_id"
DATE_COL = "discovery_date"
DURATION_COL = "duration_days"
EVENT_COL = "event"
KM_COL = "asset_class"
SHARPE_COL = "val_sharpe"


def _reconstruct_folds(frame: pd.DataFrame, recorded: list[dict], cfg_block: dict) -> list:
    folds = temporal_folds(frame[START], cfg_block["n_folds"], cfg_block["min_train_frac"])
    if len(folds) != len(recorded):
        raise AssertionError(
            f"rebuilt {len(folds)} folds against {len(recorded)} in the metrics; "
            "the run and this step disagree about the cross-validation scheme"
        )
    for i, (fold, rec) in enumerate(zip(folds, recorded, strict=True), start=1):
        if len(fold.train_idx) != rec["n_train"] or len(fold.test_idx) != rec["n_test"]:
            raise AssertionError(
                f"fold {i} rebuilt as {len(fold.train_idx)} train / {len(fold.test_idx)} test "
                f"against {rec['n_train']} / {rec['n_test']} in the metrics. The reloaded frame "
                "does not match the one the run scored, so any latent joined onto it would be "
                "misaligned and the oracle and Sharpe figures would be silently wrong."
            )
    return folds


def add_synthetic_extras(
    run_dir: str | Path,
    data_path: str | Path,
    latents_path: str | Path,
    generator: GeneratorConfig,
) -> dict:
    """Merge the generator block, the oracle ceiling, and the validation-Sharpe
    baseline into the run's metrics.json. Returns the updated metrics."""
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())

    data = load_duration_csv(
        data_path,
        ID_COL,
        DATE_COL,
        DURATION_COL,
        EVENT_COL,
        time_unit=metrics["config"]["time_unit"],
    )
    frame = data.frame
    folds = _reconstruct_folds(frame, metrics["folds"], metrics["config"])

    latents = pd.read_csv(latents_path)
    eta = latents.set_index(ID_COL)["log_time_eta"].reindex(frame[ID]).to_numpy(dtype=float)
    if not np.isfinite(eta).all():
        raise AssertionError(
            "some rows have no latent after the join on "
            f"{ID_COL!r}; the latents file does not cover the data file"
        )
    sharpe = frame[SHARPE_COL].to_numpy(dtype=float)
    duration = frame[DURATION].to_numpy(dtype=float)
    event = frame[EVENT].to_numpy()

    for fold, rec in zip(folds, metrics["folds"], strict=True):
        idx = fold.test_idx
        rec["c_sharpe"] = harrell_c(duration[idx], event[idx], sharpe[idx])
        rec["c_oracle"] = harrell_c(duration[idx], event[idx], eta[idx])

    test_idx = np.concatenate([f.test_idx for f in folds])
    oof_dur, oof_ev = duration[test_idx], event[test_idx]
    oof_sharpe, oof_eta = sharpe[test_idx], eta[test_idx]
    n = len(test_idx)
    pooled = metrics["pooled"]
    if n != pooled["n_test"]:
        raise AssertionError(
            f"rebuilt {n} out-of-fold rows against {pooled['n_test']} in the metrics"
        )
    pooled["c_sharpe"] = harrell_c(oof_dur, oof_ev, oof_sharpe)
    pooled["c_sharpe_ci"] = bootstrap_ci(
        lambda i: harrell_c(oof_dur[i], oof_ev[i], oof_sharpe[i]),
        n,
        metrics["config"]["n_bootstrap"],
        seed=2,
    )
    # The natural control for an inverted metric: what the ranking scores
    # once its direction is known. The report cites it beside c_sharpe.
    pooled["c_sharpe_flipped"] = harrell_c(oof_dur, oof_ev, -oof_sharpe)
    pooled["c_oracle"] = harrell_c(oof_dur, oof_ev, oof_eta)

    metrics["generator"] = asdict(generator)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics
