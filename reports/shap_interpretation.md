# SHAP interpretation

SHAP values for the AFT model live on the log-time margin: a value of +0.3
multiplies the predicted survival time by about 1.35, and negative values
shrink it. The figures referenced here are in `figures/` and come from the
final model fit on all 5,000 strategies (seed 7), explained on a 2,000-row
sample. Because the data is synthetic I can check every attribution against
the process that generated it, which is the point of this exercise: if SHAP
recovered the wrong story on data where I know the truth, I would not trust
it on data where I do not.

## What the model learned

Mean |SHAP| ranking (`figures/shap_bar.png`): the three walk-forward
consistency statistics dominate. `wf_positive_fraction` (0.29),
`wf_sharpe_decay` (0.22), and `wf_sharpe_std` (0.14) together carry more
attribution than everything else combined. The generator never feeds these
into survival time directly; they matter only because they are the least
noisy observable proxies for the latent edge-versus-overfit split. The model
found the proxies and leaned on them, which is exactly the right move.

The beeswarm (`figures/shap_beeswarm.png`) confirms the directions. High
positive-fraction pushes predictions up; a steeply negative walk-forward
Sharpe slope pushes them down hard, with the left tail reaching -1.5 log-days
(a factor of about 4.5 shorter); high fold-to-fold Sharpe dispersion is
penalized. All three match the generative design, where consistency rises
with true edge and falls with overfitting.

The dependence plots (`figures/shap_dependence.png`) add shape. The
`wf_positive_fraction` effect is monotone and close to linear in the fraction,
with clean vertical bands at the eight possible fold fractions and a swing of
about 1.3 log-days end to end. The `wf_sharpe_decay` effect saturates: once
the slope is positive the model stops rewarding it, sensible behavior given
that decay slopes above zero are mostly noise. The `wf_sharpe_std` effect is
a descending staircase that flattens past 1.5, where additional dispersion no
longer distinguishes anything.

## The winner's curse, seen from inside the model

`val_sharpe` does not appear in the top twelve features. On unconditional
data that would be strange; here it is the signature of selection. Every
strategy in the dataset cleared the same validation-Sharpe bar, so the
metric's remaining variation is mostly the overfit component plus noise, and
the model correctly routes its attention to walk-forward consistency instead.
This mirrors the C-index result in the evaluation report, where ranking by
validation Sharpe scores 0.41, worse than a coin flip.

`log_n_candidates_tested` (0.076) shows the multiple-testing penalty: more
candidates searched means more SHAP mass below zero. The generator applies
this twice, once through the overfit scale and once directly, and the model
picks up the aggregate. Regime concentration (0.079) is penalized above
roughly 0.45, matching the hinge the generator uses.

## Feature families and asset classes

The family flags sort in the order the generator's log-time multipliers imply:
`uses_value_carry` (+, 0.089) and `uses_volatility_premium` (+, 0.056) extend
predicted survival, while `uses_microstructure` and `uses_seasonality` shorten
it. Crypto strategies take a consistent penalty (`asset_crypto`, 0.075, almost
entirely negative for crypto rows), and rates futures a small bonus. These are
the generator's crowding-and-fragility assumptions read back verbatim.

One honest caveat: agreement in direction is easy to over-read. The family
flags enter the generator additively and independently, so any reasonable
model recovers them. The harder attribution problem, separating correlated
proxies of the same latent (the three walk-forward statistics), SHAP resolves
by spreading credit across all three, and the split among them reflects the
model's internal choices as much as the data. I would not conclude from these
numbers that positive-fraction is "twice as important" as Sharpe dispersion in
any causal sense; only that the model uses it about twice as much.
