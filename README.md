# Strategy survival meta-model

Predicting trading strategy lifespan is just like any other survival modeling
problem. Treating it as one is how this repo finds that validation Sharpe, the
number allocators rank on, predicts survival worse than a coin flip on data
built to test the question.

Trading strategies decay over time, as do many other things with a lifespan. An
edge that clears validation today is often unprofitable within a few months,
and some strategies last far longer than others.

For a live portfolio of strategies this raises three practical questions: how
much capital to give a new strategy, when to schedule its first review, and
when to cut it.

All three depend on how long the edge will last, so this project builds a
model that predicts a strategy's lifespan from the metadata available on the
day it is deployed.

Nothing proprietary lives here. The strategy data is synthetic and the
demonstration data is public. What the repo shows is the process.

This same pipeline, designed on synthetic data and demonstrated on publicly
available data, also fits and evaluates any right-censored duration CSV,
meaning a feature set where some rows may still be running when observation
stops. This includes churn, equipment failure, death, subscription lapse,
etc.

The results for the synthetic strategy survival validation and the
demonstration on 239,721 real Chicago business licences are both covered
below.

The full write-up is in the [report](reports/strategy_survival_report.pdf),
which covers the method, the results, the requirements, and the limitations.
Its headline finding is that ranking strategies by validation Sharpe predicts
survival worse than a coin flip (concordance 0.410 pooled across folds,
where 0.5 is chance, from `reports/metrics.json`). The report explains why.

## Quick start

Requires git and Python 3.11 or later.

In a terminal, run the following commands.

1. Clone the repository:

   ```
   git clone https://github.com/RAStevenson/strategy-survival-model.git
   ```

2. Navigate to the root directory:

   ```
   cd strategy-survival-model
   ```

3. Create a new environment and activate it (on Mac or Linux the second
   line is `source .venv/bin/activate`):

   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

4. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

5. Run the fully synthetic pipeline and generate the report (with default
   arguments):

   ```
   python scripts/run_synthetic_pipeline.py
   ```

The last command regenerates the dataset, runs the temporal cross-validation,
writes `reports/metrics.json` and the figures, and rebuilds the report. A full
run takes about two minutes on my machine. The versions in `requirements.txt` are the exact
ones the reported numbers were produced with.

Additional entry points and arguments:

```
python scripts/run_synthetic_pipeline.py --no-report    # stop after metrics and figures

python scripts/run_synthetic_pipeline.py --seed 8 --no-report --no-figures --metrics-name metrics_seed8.json

python scripts/run_build_report.py                      # rebuild the report only

pytest                                                  # the test suite
```

The second command is the seed-8 robustness run the report's addendum
reads. `--metrics-name` keeps it from overwriting `reports/metrics.json`.
A rerun without that flag replaces the file the committed report is
built from. The pipeline script also takes `--n`, the number of
strategies to generate, and `--folds`, the number of temporal folds.
One more script, `run_check_reproducibility.py`, compares a fresh
metrics file against the committed one and fails when a performance
number moves more than its tolerance. CI runs it on every push.

## Using it on your own data

The pipeline is not restricted to trading strategy survival studies.
It fits any right-censored duration CSV. The pipeline censors and
re-censors rows itself. A row that starts on a date, runs for some time,
and either ends while you are watching or is still going when observation
stops needs no special preparation.

The CSV requires a minimum of four columns, under any names: a unique id,
a start date, an observed duration (positive), and an event flag
(1 = the ending was observed, 0 = censored).

Every other column becomes a feature. Numeric columns pass through, missing
values included. Text columns are turned into one yes/no column per distinct
value (one-hot encoding). Values too rare to support an estimate are grouped
into a single `(other)` value. A column with gaps gets its own `(missing)`
value, so a gap stays visible to the model instead of looking identical to
the category a linear model measures the others against. Constant columns
are dropped with a notice.

An important note on data units and timescale:

Durations are read as days unless `--time-unit` says otherwise (seconds,
minutes, hours, days, weeks, months, or years). The duration column and
the `--horizons` checkpoints must share that same unit (but the
start-date column stays a calendar date). The unit specified must match the data.
Built-in leakage protection compares durations against calendar time, so
declaring a coarser unit than the data uses makes rows appear to end in
the future, and the system refuses. But it should be noted that there
is no easy way to detect a unit declared finer than the data uses, and
that mismatch silently corrupts the evaluation.

Separately, during training any survival duration shorter than one unit
is rounded up to one unit. So choose a unit that is small next to your
typical lifetimes. The tool refuses to fit when the median duration is
below one unit.

To run it on your own data, use the two commands below:

```
python scripts/run_fit_evaluate.py --data your.csv --name myrun --id-col id --date-col start_date --duration-col days --event-col ended

python scripts/run_predict.py --model runs/myrun --data new_rows.csv
```

The first command runs the same evaluation pipeline the synthetic data validates
(expanding temporal folds, training labels re-censored at each split
date, likelihood-based selection, held-out scale calibration), writes metrics
and figures to `runs/myrun/`, and renders the report. The second command
scores new rows: one output row per input row, with the predicted median
survival time in the dataset's time unit and the probability of surviving
past each horizon. The output column names carry the unit
(`predicted_median_days`, `p_survive_90d`). New rows with categorical
values the model never saw in training join the `(other)` bucket when
training created one. When it did not, they count as the category the
model measures the others against. Columns the model was not trained on
are ignored. Each case prints a notice.

The report can also carry your own prose about the dataset. Put a notes
file next to the data file, named after it (`your.notes.json` beside
`your.csv`). It takes four keys. `data` prints into the report's data
section, `limitations` into its limitations section, `km_caption`
under the Kaplan-Meier figure, and `worst_fold` replaces the report's
default "no cause is established" sentence about the weakest fold once
you have investigated it. Use it for context the pipeline cannot
know, such as how the file was collected or what its end dates really
mean. The report template itself asserts nothing about any particular
dataset, so this file is the only channel for dataset-specific claims.
The notes live beside the data rather than inside the output, which
means they survive every refit and rebuild. The Chicago report's
dataset-specific paragraphs come from
`datasets/chicago_licences.notes.json`, and that file is a working
example of the format.

### Arguments for `run_fit_evaluate.py`

Required:

- `--data`: path to the CSV, absolute or relative to where you run the
  command. Keep your data outside the repo, or in `data/`, which git
  ignores.
- `--name`: the run's name. It titles the report and, unless `--out` says
  otherwise, sets the output folder to `runs/<name>/`.
- `--id-col`: the column holding each row's unique id.
- `--date-col`: the column holding each row's start date.
- `--duration-col`: the column holding the observed duration, which must
  be positive and measured in the `--time-unit` unit.
- `--event-col`: the column saying how observation ended. 1 means the
  ending was observed; 0 means the row was censored, still running when
  observation stopped.

Optional:

- `--drop-cols`: comma-separated columns to exclude from the features. Use
  it for anything recorded after the outcome, which would leak the label
  into the model. By default every column is kept.
- `--categorical-cols`: comma-separated columns to treat as labels even
  though they read as numbers, such as a district, zip code, or product
  id. A code like this names a place or a product, and the size of the
  number means nothing. Left numeric, a linear model fits each code as a
  straight-line trend, as if risk rose steadily from district 1 to
  district 50.
- `--km-col`: a text column to draw a Kaplan-Meier figure from in the
  report, the survival curve of each group in that column, computed from
  the data before any model is fitted. Off by default. When set, the
  metrics and report also split the pooled concordance into what group
  membership alone explains and what ranking within a group adds.
- `--folds`: the number of expanding temporal folds in the evaluation.
  Default 5.
- `--horizons`: comma-separated checkpoints, in the `--time-unit` unit,
  where the report grades the model's survival probabilities; calibration
  uses the middle one. Default 90,180,365, which suits day-based data with
  lifetimes measured in months. Pick values that bracket your data's
  typical lifetime and name dates you would act on.
- `--time-unit`: the unit the duration column and `--horizons` are
  measured in. One of seconds, minutes, hours, days, weeks, months, or
  years; days is the default. As noted previously, the duration column and
  `--horizons` must share this one unit, and it must match the data.
- `--out`: the output directory override. Outputs go to the given path
  instead of the default `runs/<name>/`.
- `--no-report`: stop after metrics and figures, skipping the HTML and PDF
  render. `python scripts/run_build_report.py --run runs/<name>` renders
  it later without refitting.

### Arguments for `run_predict.py`

Required:

- `--model`: the run directory that `run_fit_evaluate.py` created, or its
  `model/` subfolder.
- `--data`: CSV of new rows to score. It must carry the id column and
  every feature column the model was trained on, under the same names.
  Outcome columns are not needed and are ignored if present.

Optional:

- `--horizons`: comma-separated checkpoints for the survival-probability
  columns in the output, in the time unit the model was trained with.
  Default 90,180,365.
- `--model-type`: `aft` or `cox`, choosing which saved model scores the
  rows. The default is whichever scored higher out of time during the run
  that saved them, and the command prints which one it used.
- `--out`: where the predictions CSV is written. Default is
  `<data>_predictions.csv` next to the input.

Both scripts print the same lists with `--help`.

Both fitted models are written to the run's `model/` folder. This includes
the boosted model (XGBoost with its accelerated-failure-time survival
objective, AFT) and the Cox baseline, the standard linear survival model.
Both are saved because keeping only the boosted model would misreport any
dataset the baseline wins. Models are build outputs and are not committed to
the repo.

One caveat is that on real data there is no ground truth, so there is no
oracle ceiling and no way to check feature attributions against a true
mechanism. The generated report states this rather than omitting it.

## A real-data demonstration - Chicago Business Licence Lifespans

The question here is how long a business keeps its licence. `datasets/chicago_licences.csv.gz`
holds every City of Chicago business licence whose first issue falls after
2002, excluding licence types that are temporary by construction (the special
event, pop-up, and itinerant permits). A temporary permit was always going to
expire, so its short life is intent rather than failure, and the dataset asks
instead about businesses that meant to stay open. This is 239,721 licences,
most of them closed by the 2026 cutoff and the rest still current. The full
statistics, source and cleaning are documented in
[datasets/README.md](datasets/README.md). The run lives in
`reports/chicago_demo/`.

The command below recreates it.

```
python scripts/run_fit_evaluate.py --data datasets/chicago_licences.csv.gz --name chicago_licences --id-col licence_id --date-col first_issued --duration-col licensed_days --event-col closed --categorical-cols ward,community_area,police_district,zip_code --km-col license_description --folds 5 --horizons 365,1095,1825 --out reports/chicago_demo
```

`--categorical-cols` matters here. Ward, community area, police district,
and zip code are administrative codes. Ward 43 is not more ward than ward
12; the number is just a name. Left unflagged, they would be read as
quantities.

`--horizons 365,1095,1825` scores the model at one, three, and five years.
This should be intelligently chosen per dataset. The median observed life
here is 904 days. `--out` routes the run into `reports/chicago_demo/`,
which is tracked by git (for demonstration purposes). A run on your own data
defaults into the gitignored `runs/` instead.

The dataset is committed, so that command reproduces the report as a snapshot.
But a user could use `scripts/run_prepare_chicago.py` and rebuild the dataset
from the city's API and updated records. Be aware that running it fetches
whatever the portal holds today, which will no longer match the numbers below.

Out-of-time fold-mean concordance, the share of pairs the model puts in the
right order where 0.5 is a coin flip (C-index), is 0.585 for the boosted
model and 0.592 for the Cox baseline. The simpler model wins outright, and
the saved run recommends it for scoring. That is a stronger form of the
synthetic lesson. On this problem a penalized linear model is enough. On
censoring-weighted Brier score both models beat a no-skill forecast at three
and five years and lose to it at one year, and the report says so rather
than rounding it up to a win.

Predicting which of these businesses fails is genuinely hard from day-one
paperwork. Most of the concordance comes from which licence category a row
is in. Ranking rows by their category's mean prediction alone scores 0.613
against the model's pooled 0.573, and comparisons within a category score
0.510, barely above chance. Day-one metadata mostly identifies risky
categories; it says almost nothing about which handyman outlasts the others.
The report gives the decomposition.

An earlier version of this demo kept the temporary permits and scored 0.697
pooled, with the three largest feature effects all one-off event permits.
That number was flattering and empty. Its model's biggest claim was that
licences issued for single events do not last, which is true by
construction, and the exclusion is what trades it for the honest number.

Full detail in
[reports/chicago_demo/report.pdf](reports/chicago_demo/report.pdf).
