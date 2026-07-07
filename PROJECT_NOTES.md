# Project notes

A walkthrough of the decisions behind this repo, for a technical reader who
is not going to run the code. The measured results live in
[reports/evaluation.md](reports/evaluation.md) and
[reports/shap_interpretation.md](reports/shap_interpretation.md); this
document is about why the project looks the way it does.

## Why survival is a prediction problem at all

A strategy-search system produces a queue of candidates that all look good,
because looking good is the admission criterion. The operational questions
start after deployment: how much capital to give a new strategy, when to
schedule its first serious review, when to cut it. All three are functions of
how long the edge will last, and lifetimes vary by an order of magnitude even
among strategies with identical headline metrics.

The null hypothesis is that decay time is unpredictable noise and you should
treat every deployment the same. I do not believe that, and the reason is
structural rather than empirical: strategies die for causes that leave
fingerprints in discovery-time metadata. A strategy that was the best of a
hundred thousand candidates carries more selection bias than the best of a
hundred. A strategy whose walk-forward Sharpe was already sliding during
validation is telling you something. Crowded feature families decay on the
crowd's schedule, not yours. None of this is visible in a single validation
statistic, but all of it is sitting in metadata the search system already
records. If those fingerprints carry signal, a model can read them, and the
allocation and review-scheduling decisions stop being uniform.

Whether they carry *enough* signal is the empirical question, and on real
data I could not publish the answer. So the repo does the next honest thing:
it builds the generative structure I believe in, with the noise levels I
consider realistic, and shows the modeling approach extracts most of what
that structure permits. The oracle ceiling in the evaluation report exists
precisely to keep that claim bounded.

## The winner's curse, and why validation Sharpe fails

Every strategy in the dataset cleared a validation-Sharpe threshold, because
that is how strategies get deployed. This single fact poisons the most
intuitive predictor. Validation Sharpe is the sum of true edge, overfit, and
noise; conditioning on the sum exceeding a bar means the survivors with the
highest scores are disproportionately the ones whose overfit and noise
components broke upward. Past the bar, more observed Sharpe is more likely to
be inflation than edge.

The consequence shows up as the strangest number in the results: ranking by
validation Sharpe is not merely weak, it is directionally wrong, worse than a
coin flip in every temporal fold. I built the generator to reproduce this
because I have watched the pattern on production data, and I wanted the
synthetic benchmark to punish the same naive heuristic reality punishes. It
also gives the meta-model a fair reason to exist. If the backtest metric
ranked survival correctly there would be nothing to model; a meta-model earns
its keep exactly when the selection process has already consumed the obvious
signal.

## Design decisions that mattered

**Re-censoring at fold splits.** Temporal cross-validation for survival data
has a trap that ordinary temporal CV does not. Splitting by discovery date is
not enough: a strategy discovered two years before a fold's split date may
have died three months after it, and its recorded label contains that future.
A model trained on final labels is trained on information no deployment-time
model could have had. The fix is mechanical, rewrite every training label to
what was knowable at the split date, so a post-split death becomes a censoring
at the split. `cv.recensor` does this and the test suite pins it down. I made
it a first-class function rather than a loop inside the pipeline because it is
the piece of this project most worth stealing for other duration problems on
operational data.

**AFT over classification, and over plain regression.** Regression on
lifetime fails on a live book: the strategies still running have no label, and
they are systematically the long-lived ones, so dropping them biases the model
toward pessimism. Classification at a fixed horizon handles censoring awkwardly
and answers only one question. The accelerated-failure-time framing keeps every
row, expresses a censored strategy as the interval "at least this long", and
returns a full time distribution I can collapse to any horizon a capital
allocator asks about. XGBoost's `survival:aft` objective made the gradient
boosting side nearly free; the work was in the labels and the evaluation, which
is usually where survival projects actually live or die.

**Switching the selection metric after the calibration failure.** The first
working version selected hyperparameters, including the AFT loss scale, by
C-index on a held-out temporal slice, and it looked fine: the ranking metrics
were exactly where they ended up in the final version. Then I computed IPCW
Brier scores and found the model losing to a no-skill marginal forecast at the
one-year horizon. The cause took a while to see. C-index is invariant to the
predictive scale, so the grid search had happily chosen a loss scale whose
implied probability distribution was far too wide, and nothing in the training
loop objected. Two changes fixed it: hyperparameters are now selected by
held-out censored log-likelihood, which does penalize a broken scale, and the
predictive scale gets a one-parameter calibration on a temporal tail slice the
early-stopping probe never trained on. I left the failure in the written record
deliberately. A model can rank well and lie about probabilities at the same
time, and the only defense is evaluating both.

There is also a decision I made and want to flag rather than bury: the boosted
model ties the Cox baseline on discrimination, and I shipped the tie. The
generator's observable structure is close to additive, so a linear model in the
log-hazard captures most of it at this sample size. I could have added
interactions to the generator until the tree model pulled ahead, but then the
benchmark would exist to flatter the model, which inverts the point of building
a benchmark.

## What SHAP recovered, and why it is the expected answer

Synthetic data turns SHAP from decoration into a test. I know the generative
process, so I know what the attribution should say, and the interesting
question becomes whether it says it.

It does, in the ways that matter. The walk-forward consistency statistics
dominate the ranking, which is correct for a subtle reason: the generator never
feeds them into survival time directly. They matter only as the least-noisy
observable proxies for the latent edge-versus-overfit decomposition, and the
model routes its attention to them anyway. Validation Sharpe is nearly absent
from the attribution, which is the winner's curse seen from inside the model.
The feature-family and asset-class effects come back with the signs and rough
ordering I wrote into the generator, and the search-intensity penalty appears
where the multiple-testing story says it should.

I trust the directions more than the magnitudes, and the interpretation report
says so explicitly. The three walk-forward statistics are correlated proxies of
the same latent quantity, and how SHAP divides credit among correlated features
reflects the model's internal choices as much as the data. The honest reading
is "the model uses positive-fraction about twice as much as Sharpe dispersion",
not "positive-fraction is twice as important". On real data, where I cannot
check attributions against a known mechanism, that distinction is the
difference between interpretation and storytelling.

## Limitations, without the soft focus

The deepest one: this project validates a methodology, not a market claim. The
generator encodes my beliefs about how strategy decay works, selection bias,
proxy structure, family-level crowding, and a model that recovers structure I
put there says nothing about whether real metadata contains comparable signal.
The C-index here is an upper bound on what I would expect in production, where
the metadata distribution drifts as the search system evolves and where
lifetimes are correlated across strategies through shared regimes, a dependence
the generator omits entirely and the i.i.d. bootstrap intervals silently
ignore.

Administrative retirement is modeled as censoring independent of performance.
In a real book, strategies get retired *because* someone saw early decay, which
makes the censoring informative and biases survival estimates upward. Handling
that properly means competing risks, which is on the TODO list rather than in
the code.

Finally, the headline numbers come from one generator seed. I reran the whole
pipeline on a second seed and the story held, same hyperparameters selected,
same attribution ordering at the top, ranking metrics within the confidence
intervals, but two runs is a spot check, not a variance estimate. Until a
proper multi-seed sweep exists, the specific decimals in the reports deserve
less confidence than the relationships between them.
