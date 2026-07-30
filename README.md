# Strategy survival meta-model

A survival-analysis model that predicts how long an algorithmically discovered
trading strategy will stay profitable out-of-sample, using only the metadata
available on the day it is deployed. Everything in this repo runs on synthetic
data with known ground truth; nothing proprietary lives here.

**The full story is in the report:**
[PDF](reports/strategy_survival_report.pdf) (renders in GitHub's viewer) or
[self-contained HTML](reports/strategy_survival_report.html). This README is
the short version.

## Headline result

Pooled over 3,000 out-of-fold test strategies (seed 7, five expanding-window
temporal folds; full conditions in the report):

| Model | Harrell C-index |
|---|---|
| Oracle on latent log-time (noise ceiling) | 0.820 |
| XGBoost AFT | 0.782 [0.773, 0.790] |
| Cox PH, same features (fold mean) | 0.781 |
| Rank by validation Sharpe | 0.410 [0.397, 0.422] |

The headline is not the 0.78. It is the 0.41: ranking strategies by their
validation Sharpe is worse than random, because every strategy entered the
dataset by clearing a Sharpe threshold. Past that bar, a higher score is more
likely overfitting than edge, and the overfit strategies decay fastest. A model
reading walk-forward consistency instead recovers most of what the noise
ceiling permits, and it ties the much simpler Cox baseline, a tie I report
rather than tune away (report, section 6).

![C-index by temporal fold](reports/figures/fold_cindex.png)

## How it works

`generate.py` draws candidate strategies with two latent components, true edge
and overfit, and keeps only candidates whose validation Sharpe clears a
threshold, which is how strategies actually enter a deployment queue. That
selection step manufactures the winner's curse the headline table shows.
Survival time is log-normal in the latents plus observable effects, censored by
the observation cutoff and by administrative retirement.

The model is XGBoost with the `survival:aft` objective: a death is the interval
[t, t], a still-running strategy is [t, infinity). Evaluation is five
expanding-window temporal folds over discovery dates, and training labels are
re-censored at each fold's split date (`cv.recensor`), because a strategy that
died after the split was still alive as far as any model trained at that date
could know. Skipping that step looks fine and quietly leaks the future.

Hyperparameters are selected by held-out censored log-likelihood rather than
C-index. An earlier version selected by C-index, ranked exactly as well as the
final version, and still lost to a no-skill marginal forecast on 365-day Brier
score, because a ranking metric cannot see a broken probability scale. The
failure and the fix are documented in the report, section 5.3.

Because the generating process is known, the report also checks the model's
attributions against the mechanisms actually built in, and the test suite
asserts the model never outscores the oracle, since beating perfect information
would indicate a leak.

## Running it

Python 3.11+. From the repo root:

```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
.venv\Scripts\python -m strategy_survival run
.venv\Scripts\python reports/build_report.py
```

The `run` command regenerates the data (seed 7, 5,000 strategies), runs the
temporal CV, and writes `reports/metrics.json` plus all figures in about two
minutes. `build_report.py` regenerates the report from that metrics file, so
every number in it is traceable to a run rather than transcribed, and prints
the PDF copy when Chrome is available. The seed-dependence check reads
`reports/metrics_seed8.json`, produced by `run --seed 8`.

## Limitations

The generator encodes my prior about how strategy decay works, not evidence.
The model recovers structure I put there, so this validates a methodology, not
a market claim, and the C-index here is an upper bound on what production data
would give. Lifetimes are treated as independent although real strategies die
together in regime breaks, administrative retirement is modeled as independent
censoring although real retirement is informative, and results come from two
generator seeds, which is a consistency check rather than a variance estimate.
The full list, with the fixes each one implies, is in the report, sections 8
and 9.

## Layout

```
src/strategy_survival/
  generate.py       synthetic metadata + censored survival target
  features.py       feature matrix with a fixed column contract
  cv.py             expanding-window folds and label re-censoring
  model.py          XGBoost AFT wrapper + predictive-scale calibration
  baseline.py       Cox PH and the validation-Sharpe heuristic
  evaluate.py       Harrell C, IPCW Brier, decile calibration, bootstrap
  plots.py          report figures
  shap_analysis.py  attributions on the log-time margin
  pipeline.py       the whole run, end to end
tests/              41 tests covering generator invariants, leakage, metrics
reports/            metrics.json, figures, and the generated report
```
