The comparison worth acting on is the model choice. The Cox baseline edges
the boosted model on the fold mean, @val{pooled.c_cox_by_fold_mean:.3f} to
@val{pooled.c_xgb_by_fold_mean:.3f}, and the honest reading is that on this
problem a penalized linear model is enough. Day-one paperwork mostly
identifies risky licence categories. Ranking rows by their category's mean
prediction alone scores @val{within_group.c_group_mean:.3f} against the
model's pooled @val{pooled.c_xgb:.3f}, and comparisons within a category
score @val{within_group.c_within:.3f}, barely above chance. So, for example,
a Home Repair business correlates with short life but the metadata says almost
nothing about which handyman outlasts the others.

The one-year probabilities should not drive decisions. The model loses to
the no-skill reference on Brier score at that horizon,
@val{ipcw_brier.365d.xgb:.3f} against @val{ipcw_brier.365d.km_marginal:.3f},
so treat it as an ordering tool at short horizons rather than a probability
source there.

Fold 3, the weakest window in the per-fold table, drops for a measurable
reason. Upon investigation it was discovered that its test block sits across
a shift in the city's category mix, and the shift is administrative rather
than economic. The city stopped issuing the Home Occupation and Home Repair
licence types in 2012, the same year Regulated Business License first
appears with a one-year spike of issues. The change is the city relabeling
businesses, not the businesses themselves changing, and this event was
discoverable in the data.

The two retired categories carry 11 percent of this fold's training rows and stop
appearing in the test block entirely. Regulated Business License nearly
triples its share. Another 2 percent of test rows fall in categories the
training window never saw. Re-selecting hyperparameters on that fold's own
window closes about a third of its gap to the Cox baseline on that fold. 
This demonstrates that a fold that drops as this one did is worth investigating
against the composition of its test block before being dismissed as model instability. 
The measurements in this paragraph are recomputed by `scripts/run_fold3_investigation.py`.

The printed bootstrap interval is narrower than the real uncertainty.
Licences in the same category and the same stretch of time tend to fail
together, and the interval resamples rows as if they were independent. The
per-fold spread in the results section is the better guide to how the
score moves across history.
