**Reading the tie.** The boosted model and the Cox baseline land at
@val{pooled.c_xgb_by_fold_mean:.3f} and @val{pooled.c_cox_by_fold_mean:.3f}
on the fold mean, which is essentially a
tie. The synthetic data generator's
observable structure is close to additive, which leaves little for trees to find
beyond what a penalized linear model in the log-hazard captures. I chose to report
the tie rather than adding interactions to the generator until the
headline model wins, because a benchmark tuned until it loses is not a
benchmark.

**The calibration failure worth keeping.** An earlier version of this
pipeline selected hyperparameters by concordance and looked correct, then
lost to a no-skill forecast on 365-day Brier score. Concordance only compares ordering and is
invariant to the predictive scale. So the search had settled on a
distribution width far too wide and nothing in the training loop objected.
That failure is why selection now uses censored log-likelihood and why
every report this pipeline produces grades probability quality separately
from ranking.

**What the attribution shows, checked against the mechanism.**
The features the model leans on most are the walk-forward statistics, the record of how a strategy performed across successive validation windows. It is important to note that none of these statistics drive survival time directly in the generator. In the generator, survival time is driven by two hidden quantities. These are how much real edge a strategy has and how much of its validation score was overfitting. The walk-forward statistics are noisy proxies of those two quantities, and the model found them. The generator gives every feature family and every asset class a fixed shift on log survival time, and the model's attribution for each one carries the same sign the generator installed. The walk-forward statistics have no installed shift, but the generator does fix a direction. Walk-forward positive fraction rises with real edge in the generator, so it should push predicted survival up, and it does. Walk-forward Sharpe decay and fold-to-fold Sharpe dispersion both rise with overfitting, so they should push predicted survival down, and they do. Validation Sharpe itself barely registers in the attribution ranking. Past the selection bar it carries more overfitting than edge, so the model gains little from it. The model recovering the installed effects directly, and reaching the two hidden drivers through their proxies, is the evidence this run exists to produce. The pipeline fits real structure where the answer is known, which is the ground for trusting its evaluation on data where the answer is not.
