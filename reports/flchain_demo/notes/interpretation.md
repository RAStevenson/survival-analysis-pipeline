**How this compares to the standard exercise.** This dataset is a teaching
classic, and the usual exercise evaluates it with a random split. A random
subset of people is hidden, the model trains on the rest, and its training
pool therefore spans every year of the study. A benchmark paper (BigSurvSGD,
arXiv:2003.00116) reports a concordance around 0.794 under that protocol.

A random split suits the original research goal. A researcher asking
whether the blood assay predicts mortality beyond age wants to know
whether the relationship is real. The whole completed study is fair
evidence for that, and the deliverable is the finding itself. Training
only on the past suits the deployment goal. A clinic scoring each new
patient at intake stands at a moment in time and knows only what has
happened so far, and a test is only honest if it rehearses that position.
This pipeline runs the second protocol and reaches the same answer. The
Cox baseline's fold mean is @val{pooled.c_cox_by_fold_mean:.3f}, matching
the benchmark figure, with the three folds at @val{folds.0.c_cox:.3f},
@val{folds.1.c_cox:.3f}, and @val{folds.2.c_cox:.3f}.

**What the review process caught.** An earlier version of this run reported
a different result, and the review that retired it is worth recording. The
source file lists subjects within each sample year in age order. Start
dates here are year-only, so a fold boundary that lands inside a year
cannot cut on time and cuts between rows instead, and row order here was
age order. Two
folds trained on identical windows and scored 0.649 and 0.847 on the
unshuffled file, purely from who landed in their test slices. The
preparation script now shuffles within each year with a fixed seed, folds
that snap to the same split date merge into one, and the horizons now sit
within the longest training window's observed follow-up. A
random split can never meet this failure, because it never cuts along the
file at all. A temporal protocol on coarse dates has to, which is why the
run documents it.

**Why the boosted model still trails.** Only @val{dataset.event_rate:.0%}
of subjects died while the study watched. Under that much censoring the
boosted model trails the Cox baseline on ranking,
@val{pooled.c_xgb_by_fold_mean:.3f} against
@val{pooled.c_cox_by_fold_mean:.3f} on the fold mean, and on probability
quality, running slightly behind the no-skill forecast at every horizon
while Cox runs slightly ahead. The Brier table in the results section
grades this, and lower is better. The boosted model fits a fixed
curve shape and guesses lifetimes from mostly unfinished ones. Cox reads
its curve off the data by counting who is still alive at each age, which
holds up where the data can check it. This is why the pipeline fits both
models and lets the data choose, and this run's bundle recommends Cox.

**What the attribution shows.** Age carries most of the ranking, and the
two light-chain measurements add signal on top, which is the original
study's own finding. The decomposition by the assay's banding makes the
model's contribution precise. A low, middle, and top banding of the ten
assay groups alone ranks survival at
@val{within_group.c_group_mean:.3f}, and comparisons restricted to
subjects within the same band score @val{within_group.c_within:.3f}, so
the model adds real ranking beyond the assay's own grouping. The
Kaplan-Meier figure in the data section shows the banding's separation
directly.
