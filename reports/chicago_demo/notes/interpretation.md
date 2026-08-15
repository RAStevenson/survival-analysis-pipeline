The comparison worth acting on is the model choice. The Cox baseline edges
the boosted model on the fold mean, @val{pooled.c_cox_by_fold_mean:.3f} to
@val{pooled.c_xgb_by_fold_mean:.3f}, and the honest reading is that on this
problem a penalized linear model is enough. Day-one paperwork mostly
identifies risky licence categories: ranking rows by their category's mean
prediction alone scores @val{within_group.c_group_mean:.3f} against the
model's pooled @val{pooled.c_xgb:.3f}, and comparisons within a category
score @val{within_group.c_within:.3f}, barely above chance. The metadata
says almost nothing about which handyman outlasts the others.

The one-year probabilities should not drive decisions. The model loses to
the no-skill reference on Brier score at that horizon,
@val{ipcw_brier.365d.xgb:.3f} against @val{ipcw_brier.365d.km_marginal:.3f},
so treat it as an ordering tool at short horizons rather than a probability
source there.

The printed bootstrap interval is narrower than the real uncertainty.
Licences in the same category and the same stretch of time tend to fail
together, and the interval resamples rows as if they were independent. The
per-fold spread in the results section is the better guide to how the
score moves across history.
