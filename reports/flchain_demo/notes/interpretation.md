**How this compares to the standard exercise.** This dataset is a teaching
classic, and the usual exercise evaluates it with a random split. A random
subset of people is hidden, the model trains on the rest, and its training
pool therefore spans every year of the study. A published benchmark reports a
concordance around 0.794 under that protocol. A random split suits the
original research goal. A researcher asking whether the blood assay
predicts mortality beyond age wants to know whether the relationship is
real, the whole completed study is fair evidence for that, and the
deliverable is the finding itself. Training only on the past suits the
deployment goal. A clinic scoring each new patient at intake stands at a
moment in time and knows only what has happened so far, and a test is only
honest if it rehearses that position. This pipeline runs the second
protocol, and its folds reconcile the two numbers. The late folds, whose
training windows contain most of the study, score @val{folds.3.c_cox:.3f}
and @val{folds.4.c_cox:.3f}, matching the published figure. The earliest
fold, trained before most deaths had happened, scores
@val{folds.0.c_cox:.3f}. The fold mean of @val{pooled.c_cox_by_fold_mean:.3f}
is the cost of training only on the past, and the fold column shows where
that cost concentrates.

**Why the boosted model's probabilities fail here.** Only
@val{dataset.event_rate:.0%} of subjects died while the study watched. For
everyone else, the data records only that they lived at least as long as
the observation, so most true lifetimes lie beyond anything the model ever
saw. The boosted model fits a fixed curve shape and places its guesses
near the edge of what it observed, so every probability comes out too
grim, and the Brier table above catches it losing badly to the no-skill
forecast. The Cox baseline reads its curve off the data by counting who is
still alive at each age, stays calibrated, and is the model this run's
bundle recommends. This is why the pipeline fits both models and lets the
data choose.

**What the attribution shows.** Age carries most of the ranking, and the
two light-chain measurements add signal on top, which is the original
study's own finding. The sex decomposition locates the model's
contribution. Sex alone ranks survival at a coin flip, comparisons within
a sex score @val{within_group.c_within:.3f}, and the difference is signal
the model found in the individual measurements.
