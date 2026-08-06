"""The full synthetic study: generate the seeded dataset, run temporal CV,
write metrics and figures, then rebuild the report.

    pip install -r requirements.txt
    python scripts/run_synthetic_pipeline.py

About two minutes. The data is regenerated from the seed on every run rather
than read from disk, so a run can never quietly describe data an older
version of the generator wrote. Flags:

    python scripts/run_synthetic_pipeline.py --no-report     # stop after metrics and figures

The committed seed-8 robustness check, which the report's limitations section
reads when it is present. It writes only its own metrics file, so the seed-7
figures and report it is compared against stay as they are:

    python scripts/run_synthetic_pipeline.py --seed 8 --no-report --no-figures \
        --metrics-name metrics_seed8.json
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
        prog="run_synthetic_pipeline.py",
        description="Strategy survival meta-model: the full synthetic study with no arguments.",
    )
    parser.add_argument("--n", type=int, default=5000, help="strategies to generate")
    parser.add_argument("--seed", type=int, default=7, help="generator seed")
    parser.add_argument("--folds", type=int, default=5, help="temporal CV folds")
    parser.add_argument(
        "--no-report", action="store_true", help="skip rebuilding the HTML/PDF report"
    )
    parser.add_argument(
        "--metrics-name",
        default="metrics.json",
        help="filename to write under reports/; a robustness run at another seed uses "
        "metrics_seed8.json so it does not overwrite what the report is built from",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="write only the metrics file, leaving data/ and reports/figures/ untouched",
    )
    args = parser.parse_args()

    gen_cfg = GeneratorConfig(n_strategies=args.n, seed=args.seed)
    metrics = run_pipeline(
        replace(
            PipelineConfig(),
            generator=gen_cfg,
            n_folds=args.folds,
            metrics_name=args.metrics_name,
            write_figures=not args.no_figures,
        )
    )

    pooled = metrics["pooled"]
    print(
        f"pooled C-index  xgb {pooled['c_xgb']:.3f} "
        f"[{pooled['c_xgb_ci'][0]:.3f}, {pooled['c_xgb_ci'][1]:.3f}]"
    )
    print(f"                cox {pooled['c_cox_by_fold_mean']:.3f} (fold mean)")
    print(f"             sharpe {pooled['c_sharpe']:.3f}")
    print(f"             oracle {pooled['c_oracle']:.3f}")
    written = f"reports/{args.metrics_name}"
    print(f"{written}{'' if args.no_figures else ' and reports/figures/'} written")

    if not args.no_report:
        # Separate process on purpose: the report builder is its own entry
        # script, and a report-build failure should not read as a pipeline
        # failure.
        report_script = Path(__file__).resolve().parent / "run_build_report.py"
        subprocess.run([sys.executable, str(report_script)], check=True)


if __name__ == "__main__":
    main()
