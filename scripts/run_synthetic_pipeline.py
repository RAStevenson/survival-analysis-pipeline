#!/usr/bin/env python3
"""The full synthetic study: generate the seeded dataset, then run it through
the same pipeline any user file goes through, and rebuild the report.

    pip install -r requirements.txt
    python scripts/run_synthetic_pipeline.py

About two minutes. The data is regenerated from the seed on every run rather
than read from disk, so a run can never quietly describe data an older
version of the generator wrote. The generated CSV is an ordinary duration
file: it goes to `fit_evaluate` with column flags, exactly as a real dataset
does, which is what makes this run a test of the product rather than of a
sibling of it. Only afterwards does `synthetic_extras` add the two
measurements no user file could supply, the oracle ceiling and the
validation-Sharpe baseline.

    python scripts/run_synthetic_pipeline.py --no-report   # stop after metrics and figures
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import os
import subprocess

from survival_analysis_pipeline.generate import GeneratorConfig, generate
from survival_analysis_pipeline.pipeline import fit_evaluate
from survival_analysis_pipeline.synthetic_extras import (
    DATE_COL,
    DURATION_COL,
    EVENT_COL,
    ID_COL,
    KM_COL,
    add_synthetic_extras,
)

ROOT = Path(__file__).resolve().parents[1]
# Every path this run reads or writes is anchored to the repository, and the
# run records the ones it used. Working from the root keeps those records
# repo-relative, so the committed metrics name `data/strategies.csv` rather
# than whichever absolute path the machine that produced them happened to use.
DATA = Path("data")
RUN_DIR = Path("reports") / "synthetic"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_synthetic_pipeline.py",
        description="The full synthetic study with no arguments.",
    )
    parser.add_argument("--n", type=int, default=5000, help="strategies to generate")
    parser.add_argument("--seed", type=int, default=7, help="generator seed")
    parser.add_argument("--folds", type=int, default=5, help="temporal CV folds")
    parser.add_argument(
        "--no-report", action="store_true", help="skip rebuilding the HTML/PDF report"
    )
    args = parser.parse_args()
    os.chdir(ROOT)

    gen_cfg = GeneratorConfig(n_strategies=args.n, seed=args.seed)
    df, latents = generate(gen_cfg)
    DATA.mkdir(parents=True, exist_ok=True)
    data_path = DATA / "strategies.csv"
    latents_path = DATA / "latents.csv"
    df.to_csv(data_path, index=False)
    latents.to_csv(latents_path, index=False)
    print(f"generated {len(df):,} strategies at seed {args.seed} into {data_path}")

    fit_evaluate(
        # Posix spelling, because the run records the path it was given and
        # the committed record has to read the same on either platform.
        data_path.as_posix(),
        name="synthetic",
        id_col=ID_COL,
        date_col=DATE_COL,
        duration_col=DURATION_COL,
        event_col=EVENT_COL,
        km_col=KM_COL,
        n_folds=args.folds,
        out_dir=RUN_DIR,
    )
    metrics = add_synthetic_extras(RUN_DIR, data_path, latents_path, gen_cfg)

    pooled = metrics["pooled"]
    print(
        f"pooled C-index  xgb {pooled['c_xgb']:.3f} "
        f"[{pooled['c_xgb_ci'][0]:.3f}, {pooled['c_xgb_ci'][1]:.3f}]"
    )
    print(f"                cox {pooled['c_cox_by_fold_mean']:.3f} (fold mean)")
    print(f"             sharpe {pooled['c_sharpe']:.3f}")
    print(f"             oracle {pooled['c_oracle']:.3f}")
    print(f"{RUN_DIR.as_posix()}/ written")

    if not args.no_report:
        # Separate process on purpose: the report builder is its own entry
        # script, and a report-build failure should not read as a pipeline
        # failure.
        report_script = Path(__file__).resolve().parent / "run_build_report.py"
        subprocess.run([sys.executable, str(report_script)], check=True)


if __name__ == "__main__":
    main()
