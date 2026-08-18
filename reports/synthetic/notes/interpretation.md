**Why ranking by validation Sharpe inverts.** Every strategy entered the
population by clearing a validation-Sharpe threshold of
@val{generator.selection_sharpe:g}. An observed Sharpe is the sum of true
edge, an overfitting component, and noise. Conditioning on that sum
exceeding a bar means the survivors with the highest scores are
disproportionately the ones whose overfitting and noise broke upward. Past
the bar, additional observed Sharpe is more likely inflation than edge.
The inflated strategies are the overfit ones, which decay fastest, so
higher validation Sharpe actively predicts shorter working life. It scores
@val{pooled.c_sharpe:.3f} pooled and sits below a coin flip in every fold. This
is also what makes the modeling problem worth posing. If the backtest
metric ranked survival correctly there would be nothing to add. A
meta-model earns its place precisely because selection has already consumed
the obvious signal.

**Reading the tie.** The boosted model and the Cox baseline land at
@val{pooled.c_xgb_by_fold_mean:.3f} and @val{pooled.c_cox_by_fold_mean:.3f}
on the fold mean. The synthetic data generator's
observable structure is close to additive, which leaves little for trees to find
beyond what a penalized linear model in the log-hazard captures. I report
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
return with the installed signs, among them the crypto shift of -0.30
described in the data note. High walk-forward positive fraction pushes
predicted survival up, and a steeply negative walk-forward Sharpe slope
pushes it down hard. High fold-to-fold Sharpe dispersion is penalized.
All three directions match the generative design, where consistency rises
with true edge and falls with overfitting. Validation Sharpe itself is
nearly absent from the attribution ranking, which is the winner's curse
seen from inside the model. The supportable reading of the magnitudes is
that the model uses positive fraction about twice as much as Sharpe
dispersion, not that positive fraction is twice as important, since the
three walk-forward statistics are correlated proxies of one latent
quantity.

**Intended use, refit on a real book.** The model is a capital-allocation
and review-scheduling prior, never a trade signal. Predicted median
lifetime sets the first review date and is one input to initial sizing,
and the full curve gives a decay schedule to compare live performance
against. It prices day-one information only. Once a strategy is live,
realized performance carries information no discovery-time metadata can.
Before any such use, the same temporal cross-validation with the same
re-censoring has to be repeated on production metadata, and the resulting
concordance should be expected to come in lower, with wider intervals.
