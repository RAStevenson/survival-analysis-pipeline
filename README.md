# Survival analysis pipeline

This is a survival analysis pipeline for right-censored duration data. It is
validated on synthetic data against an oracle, the best score any model could
reach given the generator's hidden truth, and demonstrated on real data with
real results.

In short, the purpose is not the results, it is the tool.

The pipeline fits and evaluates any right-censored duration CSV, meaning a
feature set where some rows may still be running when observation stops.
This includes churn, equipment failure, death, subscription lapse, etc. On
every dataset it fits two models, XGBoost with its accelerated-failure-time
survival objective (AFT) and a Cox proportional hazards baseline fitted
with lifelines. It evaluates both on expanding temporal folds with training
labels re-censored at each split date, and it generates a report with
dynamic figures.

Nothing proprietary lives here. The strategy data is synthetic and the
demonstration data is public. What the repo shows is the process.

This document covers setup, using the pipeline on your own data, and then
the two runs committed in this repo: the synthetic validation, which checks
the pipeline against known ground truth, and the demonstration on 239,721
real Chicago business licences.

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
the `--horizons` checkpoints must share that same unit, and it must match
the data (the start-date column stays a calendar date). Built-in leakage
protection compares durations against calendar time, so declaring a coarser
unit than the data uses makes rows appear to end in the future, and the
system refuses. But it should be noted that there is no easy way to detect
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

The full argument lists for both commands are documented at the end of
this file.

### Adding your own prose to the report

The generated report states measurements in prose that would hold on any
dataset; anything you know about your data that the pipeline cannot goes
in a notes folder, `runs/myrun/notes/`, one markdown file per section.
Start with none, because the report is complete without them. When you
have something to say, add a file named for the section you want to say it
in: `motivation.md` and `interpretation.md` become their own report
sections, and `data.md` and `limitations.md` append paragraphs to those.

Notes can quote metric values through tokens such as
`@val{pooled.c_xgb:.3f}`, resolved from the run's `metrics.json` when the
report builds, so a quoted number cannot drift from the measurement. An
unknown token or a misnamed file fails the build instead of vanishing
silently. Refits rewrite metrics and figures but never touch the notes
folder, so notes survive every refit; rebuild the report without
refitting via:

```
python scripts/run_build_report.py --run runs/myrun
```

Both committed runs are working examples of the format, in
`reports/chicago_demo/notes/` and `reports/synthetic/notes/`.

One caveat is that on real data there is no ground truth, so there is no
oracle ceiling and no way to check feature attributions against a true
mechanism. The generated report states this rather than omitting it.

## Validation on synthetic data - trading strategy survival

Predicting trading strategy lifespan is just like any other survival modeling 
problem. 

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
by it predicts survival worse than a coin flip (concordance 0.410 pooled 
across folds, where 0.5 is chance, from 
`reports/synthetic/metrics.json`). Readers should keep in mind that this
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
not; its short life is intent rather than failure. The analysis asks
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

`--horizons 365,1095,1825` scores the model at one, three, and five years.
This should be intelligently chosen per dataset. The median observed life
here is 904 days. `--out` routes the run into `reports/chicago_demo/`,
which is tracked by git (for demonstration purposes). A run on your own data
defaults into the gitignored `runs/` instead.

The dataset is committed, so that command reproduces the report as a snapshot.
You can also rebuild the dataset from the city's API with
`scripts/run_prepare_chicago.py`. Be aware that running it fetches
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
  rows. The default is whichever scored higher out of time during the run
  that saved them, and the command prints which one it used.
- `--out`: where the predictions CSV is written. Default is
  `<data>_predictions.csv` next to the input.

Both scripts print the same lists with `--help`.

Both fitted models are written to the run's `model/` folder. This includes
the boosted model (XGBoost with its accelerated-failure-time survival
objective, AFT) and the Cox baseline, the standard linear survival model,
fitted with lifelines. Both are saved because keeping only the boosted
model would misreport any dataset the baseline wins. Models are build
outputs and are not committed to the repo.
