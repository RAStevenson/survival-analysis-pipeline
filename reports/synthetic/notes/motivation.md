An automated strategy search produces a queue of candidates that all clear
validation, and they stop working at very different rates. How much capital
a new strategy should receive, when its first review should be scheduled, and when it
should be retired all depend on how long its edge will last. Treating every
deployment the same misallocates in both directions. It overfunds
strategies that will not survive the quarter and underfunds ones that
would have run for a year.

The question this run poses, on a synthetic population built to mimic that
setting, is whether discovery-time metadata carries enough signal to
separate them. A strategy selected from a hundred thousand
candidates carries more selection bias than one selected from a hundred, and
a strategy whose walk-forward Sharpe was already sliding during validation
had begun decaying before it was deployed. None of that is visible in a
single validation statistic, and all of it is already recorded. Death in this
simulated book is a bookkeeping event. It is the date a strategy stops
clearing the retention rule the generator installs. The lifetimes any model
learns are partly a property of that rule.
