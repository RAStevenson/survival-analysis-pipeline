"""Generate the synthetic dataset and stop. No model, no evaluation.

    python scripts/run_generate_data.py
    python scripts/run_generate_data.py --seed 8 --n 10000

Writes data/strategies.csv (the metadata the model sees) and data/latents.csv
(the hidden truth, used only for the oracle ceiling - never model input).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from strategy_survival.generate import GeneratorConfig, generate
from strategy_survival.pipeline import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(prog="run_generate_data.py")
    parser.add_argument("--n", type=int, default=5000, help="strategies to generate")
    parser.add_argument("--seed", type=int, default=7, help="generator seed")
    args = parser.parse_args()

    df, latents = generate(GeneratorConfig(n_strategies=args.n, seed=args.seed))
    data_dir = PipelineConfig().data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "strategies.csv", index=False)
    latents.to_csv(data_dir / "latents.csv", index=False)

    print(f"wrote {len(df)} strategies to {data_dir / 'strategies.csv'}")
    print(
        f"event rate {df['event'].mean():.2f}, "
        f"median observed duration {df['duration_days'].median():.0f} days"
    )


if __name__ == "__main__":
    main()
