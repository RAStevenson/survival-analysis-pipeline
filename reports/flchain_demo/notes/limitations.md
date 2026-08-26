**The folds are only as sharp as the dates.** Start dates here are years,
so every subject sampled in the same year shares one date, and the folds
cannot order them. A fold boundary that lands inside a year splits that
year's subjects arbitrarily, so read the per-fold numbers loosely on this
dataset.

**Use the model to rank, not to read off long-horizon probabilities.**
This limitation is measured by the Brier rows in the results section,
where lower is better. The boosted model's probabilities fail at every
horizon, and the report says so above. The Cox model's probabilities hold
up at five years and land just past the no-skill reference at ten years,
a Brier score of @val{ipcw_brier.3650d.cox:.3f} against the reference's
@val{ipcw_brier.3650d.km_marginal:.3f}. At long horizons the model still
says who is at higher risk, and that is the use to put it to.
