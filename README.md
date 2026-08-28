# Survival analysis pipeline

This project's aim was to build a practical and useful survival analysis pipeline. It is
validated on synthetic data against an oracle, the best score any model could
reach given the generator's hidden truth, and demonstrated on real
data. It is a practical tool for survival analysis and report generation
on your own data.

The pipeline fits and evaluates any right-censored duration CSV, meaning a
feature set where some rows may still be running when observation stops.
This could include churn, equipment failure, death, subscription lapse, etc. On
every dataset it fits two models for comparison, XGBoost with its accelerated-failure-time
survival objective (AFT) and a Cox proportional hazards baseline fitted
with lifelines. It evaluates both on expanding temporal folds with training
labels re-censored at each split date, and it generates a report with
dynamic figures.

Nothing proprietary lives here. The validation data is synthetic and the
demonstration data is public. The repo shows the process and provides a tool.

This document covers setup, using the pipeline on your own data, and then
the three runs committed in this repo. This includes a synthetic validation simulating trading strategy lifespan, which checks
the pipeline against known ground truth, the demonstration on the
survival of 239,721 real Chicago business licences, and a benchmark run
on the widely taught flchain medical cohort.

## Setup

Requires git and Python 3.11 or later.

In a terminal, run the following commands.

1. Clone the repository:

   ```
   git clone https://github.com/RAStevenson/survival-analysis-pipeline.git
   ```

2. Navigate to the root directory:

   ```
   cd survival-analysis-pipeline
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

The versions in `requirements.txt` are the exact ones the committed
numbers were produced with.

## Using it on your own data

The pipeline fits any right-censored duration CSV. It censors and
re-censors rows itself. A row that starts on a date, runs for some time,
and either ends while you are watching or is still going when observation
stops needs no special preparation.

The CSV requires a minimum of four columns, under any names: a unique id,
a start date, an observed duration (positive), and an event flag
(1 = the ending was observed, 0 = censored). A True/False column works as
the event flag without conversion; anything else in that column is refused
by name.

A file structured like the below example is a complete input, with the four required columns
under the names the example commands below expect:

```
id,start_date,days,ended,monthly_fee,plan
c-101,2021-03-04,412,1,49.0,pro
c-102,2021-03-11,447,0,9.0,basic
c-103,2021-04-02,55,1,,basic
c-104,2021-05-19,378,0,199.0,enterprise
```

Every other column becomes a feature. Numeric columns pass through, missing
values included. Text columns are turned into one yes/no column per distinct
value (one-hot encoding). Values too rare to support an estimate are
grouped into a single `(other)` value. The bar is half a percent of the
rows, and it stops rising at 200 occurrences, so in a large file any value
seen 200 times stands alone. A column with gaps gets its own `(missing)`
value, so a gap stays visible to the model instead of looking identical to
the reference category, the one value a linear model measures the others
against. Constant columns are dropped with a notice.

Feature columns should hold their values as of each row's start date. A
column holding today's value instead, this month's fee rather than the
signup fee, quietly imports the future the same way a post-outcome column
does.

An important note on data units and timescale:

Durations are read as days unless `--time-unit` says otherwise (seconds,
minutes, hours, days, weeks, months, or years). The duration column and
the `--horizons` checkpoints must share that same unit, and it must match
the data. The start-date column of the data should stay a calendar date. Built-in leakage
protection compares durations against calendar time, so declaring a coarser
unit than the data uses makes rows appear to end in the future, and the
system refuses. Declaring 400 days of data as 400 weeks, for example,
pushes that row's apparent end years past the file's export date. But it should be noted that there is no easy way to detect
a unit declared finer than the data uses, and that mismatch silently
corrupts the evaluation.

Separately, during training any survival duration shorter than one unit
is rounded up to one unit. So choose a unit that is small next to your
typical lifetimes. The tool refuses to fit when the median duration is
below one unit.

To run it on your own data, use the two commands below:

```
python scripts/run_fit_evaluate.py --data your.csv --name myrun --id-col id --date-col start_date --duration-col days --event-col ended

python scripts/run_predict.py --model runs/myrun --data new_rows.csv
```

The first command runs the same evaluation pipeline the synthetic data
validates: expanding temporal folds, training labels re-censored at each
split date, and model settings chosen and probability widths measured on
held-out data. It writes metrics and figures to `runs/myrun/` and renders
the report. The second command
scores new rows: one output row per input row, with the predicted median
survival time in the dataset's time unit and the probability of surviving
past each horizon. The output column names carry the unit
(`predicted_median_days`, `p_survive_90d`). New rows with categorical
values the model never saw in training join the `(other)` bucket when
training created one. When it did not, that value's flags all read zero,
which the linear baseline reads as the reference category. Columns the model was not trained on
are ignored. Each case prints a notice.

## Arguments for `run_fit_evaluate.py`

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
  membership alone explains and what ranking within a group adds. The
  column may also be one named in --drop-cols, which plots and decomposes
  by a grouping the model itself never sees.
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

## Arguments for `run_predict.py`

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
  rows. The default is whichever scored higher on later-dated test data
  during the run that saved them, and the command prints which one it used.
- `--out`: where the predictions CSV is written. Default is
  `<data>_predictions.csv` next to the input.

Both scripts print the same lists with `--help`.

Both fitted models are written to the run's `model/` folder. This includes
the boosted model (XGBoost with its accelerated-failure-time survival
objective, AFT) and the Cox baseline, the standard linear survival model,
fitted with lifelines. Both are saved because keeping only the boosted
model would misreport any dataset the baseline wins. Models are build
outputs and are not committed to the repo.

## Adding your own prose to the report

The generated report states measurements in prose that would hold on any
dataset. Anything you wish to add to the report that the pipeline cannot
know goes in a notes folder, `runs/myrun/notes/`. Start with none,
because the report is complete without them. Expect a bare report to be
noticeably shorter than the three committed ones, since much of their
length is authored notes rather than pipeline output.

Notes can be added to four report sections, and only those four. Each
note is a markdown file named for the section you want to append to.
`motivation.md` and `interpretation.md` become their own report sections,
and `data.md` and `limitations.md` append paragraphs to the Data and
Limitations sections every report already has. A file under any other
name fails the report build, and the error lists the four valid names,
so a typo cannot silently drop your prose.

If you choose, notes can quote metric values through tokens such as
`@val{pooled.c_xgb:.3f}`, resolved from the run's `metrics.json` when the
report builds, so a quoted number cannot drift when a refit changes the
measurements. An unknown token fails the build instead of vanishing
silently. Refits rewrite metrics and figures but never touch the notes
folder, so notes survive every refit. Users can rebuild the report without
refitting via the below command:

```
python scripts/run_build_report.py --run runs/myrun
```

The committed runs are working examples of the format, in
`reports/synthetic/notes/`, `reports/chicago_demo/notes/`, and
`reports/flchain_demo/notes/`.

One caveat is that on real data there is no ground truth, so there is no
oracle ceiling and no way to check feature attributions against a true
mechanism. The generated report states this rather than omitting it.

## Validation on synthetic data - trading strategy survival

Trading strategies decay over time, as do many other things with a lifespan. An
edge that clears validation today is often unprofitable within a few months,
and some strategies last far longer than others. The pipeline's validation
run models a strategy's lifespan from the metadata available on the day it
is deployed. The data is synthetic and its generating process is known, so
the best achievable score is computable and the model's feature
attributions can be checked against the mechanisms actually installed.
That is what makes the run a validation rather than a demonstration. It
shows the system fits real structure and reports honestly against a known
answer.

The full write-up is in the autogenerated
[report](reports/synthetic/report.pdf), which covers the method, 
the results, the requirements, and the limitations. Its headline finding 
concerns validation Sharpe, the return-per-unit-of-risk score a strategy 
earned in testing and the usual way a search ranks its candidates. Ranking
by it predicts survival worse than a coin flip. Its concordance, the share
of pairs a ranking puts in the right order where 0.5 is chance, is 0.410
pooled across folds (from `reports/synthetic/metrics.json`). Readers should keep in mind that this
finding may not be reflective of real life data but is consistent with the
winner's curse expectation. The report explains why.

The validation run goes through the same door your own data would. It
generates the dataset, writes it to `data/strategies.csv`, and hands that
file to the same `fit_evaluate` the command above calls, with the same
column flags. Only afterwards does it add the two measurements no ordinary
CSV could supply: the oracle ceiling and the validation-Sharpe baseline,
both computable only because the generating process is known. So the run
tests the tool rather than a private path through it.

Reproduce the run, from the environment set up above, with:

```
python scripts/run_synthetic_pipeline.py
```

This regenerates the dataset, runs the temporal cross-validation, writes
`reports/synthetic/` and its figures, and rebuilds the report. A full run
takes about two minutes on my machine.

Additional entry points and arguments:

```
python scripts/run_synthetic_pipeline.py --no-report    # stop after metrics and figures

python scripts/run_build_report.py                      # rebuild the report only

pytest                                                  # the test suite
```

The pipeline script also takes `--n`, the number of strategies to
generate, `--seed`, the generator seed, and `--folds`, the number of
temporal folds. One more script, `run_check_reproducibility.py`, compares
a fresh metrics file against the committed one and fails when a
performance number moves more than its tolerance. CI runs it on every
push.

## A real-data demonstration - Chicago Business Licence Lifespans

The question examined here is how long a business keeps its licence. The dataset 
`datasets/chicago_licences.csv.gz` holds every City of Chicago business licence 
whose first issue falls after 2002. It excludes licence types that are temporary 
by construction (the special event, pop-up, and itinerant permits), because
they flood the data with lives that are short on purpose. A permit built to
expire tells you nothing about which businesses meant to stay open but did
not. For those types of businesses, short life is intentional and not failure. The analysis asks
instead about businesses that meant to stay open. This is 239,721 licences,
most of them closed by the 2026 cutoff and the rest still current. The full
statistics, source and cleaning are documented in
[datasets/README.md](datasets/README.md). The run lives in `reports/chicago_demo/`.

The command below recreates it.

```
python scripts/run_fit_evaluate.py --data datasets/chicago_licences.csv.gz --name chicago_licences --id-col licence_id --date-col first_issued --duration-col licensed_days --event-col closed --categorical-cols ward,community_area,police_district,zip_code --km-col license_description --folds 5 --horizons 365,1095,1825 --out reports/chicago_demo
```

`--categorical-cols` matters here. Ward, community area, police district,
and zip code are administrative codes. Ward 43 is not more ward than ward
12; the number is just a name. Left unflagged, they would be read as
quantities.

`--horizons 365,1095,1825` (units in days) scores the model at one, three, and five years.
This should be intelligently chosen per dataset. The median observed life
here is 904 days. `--out` routes the run into `reports/chicago_demo/`,
which is tracked by git (for demonstration purposes). A run on your own data
defaults into the gitignored `runs/` instead.

The dataset is committed, so that command reproduces the report as a snapshot.
You can also rebuild the dataset from the city's API with
`scripts/run_prepare_chicago.py`. Be aware that running it fetches
whatever the portal holds today, which will no longer match the numbers below.

Out-of-time fold-mean concordance (C-index) is 0.585 for the boosted
model and 0.592 for the Cox baseline. The simpler model wins outright, and
the saved run recommends it for scoring. That is a stronger form of the
synthetic lesson. On this problem a penalized linear model, its
coefficients shrunk toward zero to keep the fit stable, is enough. On
censoring-weighted Brier score both models beat a no-skill forecast at three
and five years and lose to it at one year, and the report says so rather
than rounding it up to a win.

Predicting which of these businesses fails is genuinely hard from day-one
paperwork. Most of the concordance comes from which licence category a row
is in. Ranking rows by their category's mean prediction alone scores 0.613
against the model's pooled 0.573, and comparisons within a category score
0.510, barely above chance. Day-one metadata mostly identifies risky
categories and not individual businesses. It says almost nothing about which handyman outlasts the others.
The report gives the decomposition.

An earlier version of this demo kept the temporary permits and scored
about 0.70 pooled, with the largest feature effects all one-off event
permits. That run predates the exclusion and is not reproducible from the
committed dataset. The number was flattering and empty. Its model's biggest claim was that
licences issued for single events do not last, which is true by
construction, and the exclusion is what trades it for the honest number.

Full detail in
[reports/chicago_demo/report.pdf](reports/chicago_demo/report.pdf).

## A benchmark demonstration - the flchain cohort

`datasets/flchain.csv.gz` is the serum free light chain cohort that ships
with R's survival package, one of the most widely taught survival
datasets. It is included because it is recognizable. A reader who knows
the dataset arrives with numbers they already trust and can check this
pipeline against them.

The pipeline matches the benchmark under a stricter protocol. A benchmark
paper reports a concordance near 0.794 from a random split, which
trains on people drawn from every year of the study. This pipeline trains
only on the past and reaches a Cox fold mean of 0.795 across three
expanding windows (0.797, 0.785, 0.803). The run also records a
failure the review caught. An earlier version scored 0.756, with fold
scores swinging from 0.649 to 0.847, because the source file hid an age
ordering within each sample year that the fold boundaries cut along. It
also graded probabilities at horizons years beyond what any training
window had observed. The prepare script now
neutralizes the ordering, folds that the year-only dates cannot separate
merge rather than repeat, and the horizons sit at one, two,
and four years, inside the longest window's observation. The boosted model still trails the
Cox baseline here on both ranking and probability quality, and the
pipeline recommends Cox for scoring. The full story is in
[reports/flchain_demo/report.pdf](reports/flchain_demo/report.pdf).

The commands below recreate it.

```
python scripts/run_prepare_flchain.py

python scripts/run_fit_evaluate.py --data datasets/flchain.csv.gz --name flchain --id-col subject_id --date-col sample_year --duration-col futime --event-col death --drop-cols flc_band --km-col flc_band --folds 5 --horizons 365,730,1460 --out reports/flchain_demo
```


