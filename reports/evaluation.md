# Evaluation report

Numbers in this report come from the default seeded run (`python -m
strategy_survival run`, seed 7, 5,000 strategies). The pipeline writes the raw
values to `reports/metrics.json`; if you change the generator or the model,
regenerate and update this file.

## Setup

I evaluate with five expanding-window temporal folds. The earliest 40% of
strategies (by discovery date) is burn-in that is only ever trained on. Each
fold trains on every strategy discovered before its split date and tests on
the next 600 strategies. Training labels are re-censored at the split date:
a strategy that died after the split is treated as alive-and-censored, because
that is all a model trained at that moment could have known. Skipping this
step quietly imports future information into the training labels, and it is
the most common leak I see in duration models built on operational data.

Hyperparameters (tree depth and the AFT loss scale) are selected once, on the
first fold's training window, using an inner temporal split and held-out
censored log-likelihood. I select by likelihood rather than C-index because a
ranking metric cannot see a miscalibrated probability scale. After each fold's
model is refit, the predictive log-normal scale is calibrated on the inner
tail slice the early-stopping probe never trained on; the selected loss scale
was 0.6 and the calibrated predictive scale came out at 0.64.

Test labels use the full observation window. That is the standard asymmetry in
backtest-style evaluation: the model must not see the future, but the scorer
may.

## Discrimination

Pooled over the 3,000 out-of-fold test strategies:

| Model | Harrell C | 95% bootstrap CI |
|---|---|---|
| XGBoost AFT | 0.782 | [0.773, 0.790] |
| Cox PH (same features) | 0.781 (fold mean) | -- |
| Rank by validation Sharpe | 0.410 | [0.397, 0.422] |
| Oracle on latent log-time | 0.820 | -- |

Two things matter here. First, ranking by validation Sharpe is worse than
random, and reliably so across every fold (0.35 to 0.43). The generator builds
in a winner's curse: strategies only enter the dataset by clearing a Sharpe
threshold, so conditional on selection, an unusually high validation Sharpe is
more likely to be overfitting than edge. The anti-correlation is not a bug in
the metric. It is the phenomenon the meta-model exists to exploit.

Second, the boosted model does not beat the Cox baseline (0.782 vs 0.781).
The generative process is close to additive in the observable metadata, and
2,000 to 4,400 training rows is not enough for trees to find much beyond what
a penalized linear model in the log-hazard already captures. I report the tie
rather than tuning until the headline model wins. The oracle score of 0.820,
computed from the latent log-time predictor that generated the data, shows
both models sit about 0.04 below the ceiling that measurement noise imposes;
most of the remaining error is irreducible from this metadata.

Per-fold C-index is stable (0.768 to 0.817 for the AFT model). The last fold
scores highest, but 39% of its test strategies are censored versus about 2% in
the early folds, which changes the set of comparable pairs; C-index values are
not strictly comparable across folds with different censoring mixes, so I read
the per-fold plot as evidence of stability, not of improvement over time.

## Calibration

I turn each predicted median survival time into P(survive > h) under the
calibrated log-normal, then check those probabilities two ways.

IPCW Brier scores (lower is better), against a no-skill reference that assigns
every strategy the pooled Kaplan-Meier marginal:

| Horizon | AFT model | Cox | Marginal KM |
|---|---|---|---|
| 90 days | 0.148 | 0.150 | 0.249 |
| 180 days | 0.132 | 0.133 | 0.182 |
| 365 days | 0.051 | 0.051 | 0.056 |

The margin over the no-skill reference is wide at 90 days and narrow at 365,
which is what it should be: median survival is about 91 days, so by one year
nearly every strategy is dead and the marginal forecast is already close to
certain. There is little skill left to demonstrate at that horizon.

The decile calibration plot at 180 days (`figures/calibration_180d.png`)
tracks the diagonal from 0.001 to 0.68 predicted probability, with the largest
gap about 4.5 points in the sixth decile. Observed frequencies inside each bin
are Kaplan-Meier estimates, so censored strategies contribute correctly rather
than being dropped.

Before I calibrated the predictive scale this table looked different: with the
loss scale reused as the predictive scale, the model lost to the marginal KM
at 365 days. A one-parameter fit on held-out data fixed it. The general
lesson holds outside this project: AFT point predictions can rank well while
the implied probabilities are badly overdispersed, and nothing in the training
loop warns you.

## Uncertainty, honestly

The bootstrap intervals above resample test strategies i.i.d., which ignores
the fold structure and any dependence between strategies discovered in the
same period. They are best read as a lower bound on the real uncertainty.

Everything reported here is one dataset from one generator seed. I have not
run a multi-seed sweep, so I cannot separate model variance from data variance
(it is on the TODO list). More fundamentally, the oracle ceiling and the
SHAP-versus-truth comparison are luxuries synthetic data buys; on real
strategy metadata there is no oracle, the metadata distribution drifts as the
search system itself evolves, and survival times are correlated across
strategies through shared market regimes. I would expect a lower C-index and
wider intervals on production data, and I would trust nothing until the
temporal CV was repeated there with the same re-censoring discipline.
