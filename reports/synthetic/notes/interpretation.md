**Reading the tie.** The boosted model and the Cox baseline land at
@val{pooled.c_xgb_by_fold_mean:.3f} and @val{pooled.c_cox_by_fold_mean:.3f}
on the fold mean, which is essentially a
tie. The synthetic data generator's
observable structure is close to additive, which leaves little for trees to find
beyond what a penalized linear model in the log-hazard captures. I  chose to report
the tie rather than adding interactions to the generator until the
headline model wins, because a benchmark tuned until it loses is not a
benchmark.

**The calibration failure worth keeping.** An earlier version of this
pipeline selected hyperparameters by concordance and looked correct, then
lost to a no-skill forecast on 365-day Brier score. Concordance is
invariant to the predictive scale, so the search had settled on a
distribution width far too wide and nothing in the training loop objected.
That failure is why selection now uses censored log-likelihood and why
every report this pipeline produces grades probability quality separately
from ranking.

**What the attribution shows, checked against the mechanism.** The
walk-forward statistics that dominate the ranking are not inputs to
survival time in the generator. They exist only as noisy proxies for the
latent edge-versus-overfit split that is the actual driver, and the model
found them and leaned on them. Feature-family and asset-class effects
return with the installed signs, among them the crypto shift of
@val{generator.installed.asset_log_time_effect.crypto:.2f} on log time. High walk-forward positive fraction pushes
predicted survival up, and a steeply negative walk-forward Sharpe slope
pushes it down hard. High fold-to-fold Sharpe dispersion is penalized.
All three directions match the generative design, where consistency rises
with true edge and falls with overfitting. Validation Sharpe itself is
nearly absent from the attribution ranking, which matches the design.
Past the selection bar it carries more overfit than edge, so the model
has no reason to trust it. The supportable reading of the magnitudes is
that the model uses positive fraction about twice as much as Sharpe
dispersion, not that positive fraction is twice as important, since the
three walk-forward statistics are correlated proxies of one latent
quantity.

**Intended use, refit on a real book.** Refit on production metadata, the
model would be a capital-allocation and review-scheduling prior, never a
trade signal. Predicted median
lifetime sets the first review date and is one input to initial sizing,
and the full curve gives a decay schedule to compare live performance
against. It prices day-one information only. Once a strategy is live,
realized performance carries information no discovery-time metadata can.
Before any such use, the same temporal cross-validation with the same
re-censoring has to be repeated on production metadata, and the resulting
concordance should be expected to come in lower, with wider intervals.
