**Folds merge where the dates cannot separate them.** Start dates are
years, so several requested folds can snap to the same split date. Those
merge into one fold with the combined test block rather than posing as
distinct models, which is why the report shows three folds against the
five the reproduce command requests.

**Year-floored dates leak a little future into training labels.** Every
start is recorded as January 1 of its sample year, backdating subjects by
up to a year. Re-censoring therefore keeps some deaths that truly
happened just after a split inside that split's training labels. The bias
runs toward optimism, is bounded by the deaths falling within a year of
each split, and finer dates would remove it. Its size is not measured
here.

**Horizons stay inside the longest window's observation.**
The training windows watch subjects for one to four years, so the report
grades probabilities at one, two, and four years only. An earlier version
of this run asked these windows for ten-year probabilities and got
extrapolations dressed as measurements. Support is still per fold. The
one-year row is inside every window's observation, while the four-year
row leans on the widest window, and the earlier folds contribute
extrapolated predictions there. The Brier rows in the results section
grade all of it, and lower is better there.
