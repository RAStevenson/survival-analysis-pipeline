from __future__ import annotations

import argparse
import json
from dataclasses import replace

from .generate import GeneratorConfig, generate
from .pipeline import PipelineConfig, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="strategy_survival")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write synthetic data to data/")
    gen.add_argument("--n", type=int, default=5000)
    gen.add_argument("--seed", type=int, default=7)

    run = sub.add_parser("run", help="full pipeline: data, CV, metrics, SHAP, figures")
    run.add_argument("--n", type=int, default=5000)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--folds", type=int, default=5)

    args = parser.parse_args()
    gen_cfg = GeneratorConfig(n_strategies=args.n, seed=args.seed)

    if args.command == "generate":
        df, latents = generate(gen_cfg)
        cfg = PipelineConfig()
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.data_dir / "strategies.csv", index=False)
        latents.to_csv(cfg.data_dir / "latents.csv", index=False)
        print(f"wrote {len(df)} strategies to {cfg.data_dir / 'strategies.csv'}")
        print(
            f"event rate {df['event'].mean():.2f}, "
            f"median observed duration {df['duration_days'].median():.0f} days"
        )
        return

    metrics = run_pipeline(replace(PipelineConfig(), generator=gen_cfg, n_folds=args.folds))
    pooled = metrics["pooled"]
    print(json.dumps(metrics["params"], indent=2))
    print(
        f"pooled C-index  xgb {pooled['c_xgb']:.3f} "
        f"[{pooled['c_xgb_ci'][0]:.3f}, {pooled['c_xgb_ci'][1]:.3f}]"
    )
    print(f"                cox {pooled['c_cox_by_fold_mean']:.3f} (fold mean)")
    print(f"             sharpe {pooled['c_sharpe']:.3f}")
    print(f"             oracle {pooled['c_oracle']:.3f}")
    for h, scores in metrics["ipcw_brier"].items():
        print(
            f"IPCW Brier {h}: xgb {scores['xgb']:.4f}  cox {scores['cox']:.4f}  "
            f"marginal KM {scores['km_marginal']:.4f}"
        )
    print("reports/metrics.json and reports/figures/ written")


if __name__ == "__main__":
    main()
