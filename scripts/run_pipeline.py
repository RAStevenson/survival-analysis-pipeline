"""Full pipeline: generate data, run temporal CV, write metrics and figures,
then rebuild the report.

    pip install -r requirements.txt
    python scripts/run_pipeline.py

About two minutes. Flags:

    python scripts/run_pipeline.py --seed 8        # a different synthetic dataset
    python scripts/run_pipeline.py --no-report     # stop after metrics and figures
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import subprocess
from dataclasses import replace

from strategy_survival.generate import GeneratorConfig
from strategy_survival.pipeline import PipelineConfig, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Strategy survival meta-model: full run with no arguments.",
    )
    parser.add_argument("--n", type=int, default=5000, help="strategies to generate")
    parser.add_argument("--seed", type=int, default=7, help="generator seed")
    parser.add_argument("--folds", type=int, default=5, help="temporal CV folds")
    parser.add_argument(
        "--no-report", action="store_true", help="skip rebuilding the HTML/PDF report"
    )
    args = parser.parse_args()

    gen_cfg = GeneratorConfig(n_strategies=args.n, seed=args.seed)
    metrics = run_pipeline(replace(PipelineConfig(), generator=gen_cfg, n_folds=args.folds))

    pooled = metrics["pooled"]
    print(
        f"pooled C-index  xgb {pooled['c_xgb']:.3f} "
        f"[{pooled['c_xgb_ci'][0]:.3f}, {pooled['c_xgb_ci'][1]:.3f}]"
    )
    print(f"                cox {pooled['c_cox_by_fold_mean']:.3f} (fold mean)")
    print(f"             sharpe {pooled['c_sharpe']:.3f}")
    print(f"             oracle {pooled['c_oracle']:.3f}")
    print("reports/metrics.json and reports/figures/ written")

    if not args.no_report:
        # Separate process on purpose: the report builder is its own entry
        # script, and a report-build failure should not read as a pipeline
        # failure.
        report_script = Path(__file__).resolve().parent / "run_build_report.py"
        subprocess.run([sys.executable, str(report_script)], check=True)


if __name__ == "__main__":
    main()
