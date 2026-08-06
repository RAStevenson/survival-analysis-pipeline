# Strategy survival meta-model

Trading strategies, like many other lifespan problems, decay over time. An
edge that clears validation today is often unprofitable within a few months,
and some strategies last far longer than others.

For a live portfolio of strategies this raises three practical questions: how
much capital to give a new strategy, when to schedule its first review, and
when to cut it.
All three depend on how long the edge will last, so this project builds a
model that predicts a strategy's lifespan from the metadata available on the
day it is deployed.

It should be noted that nothing proprietary lives here. The foundational data is
synthetic and the supporting data is publicly available. But what is being presented
here is not data but process.

This same pipeline, designed on synthetic data and demonstrated on publicly
available data, also fits and evaluates any right-censored duration CSV: a
feature set where some rows may still be running when observation stops. This
includes churn, equipment failure, death, subscription lapse, etc.

The results for the synthetic strategy survival validation and the
demonstration on 262,763 real Chicago business licences are both covered
below.

The full write-up is in the report, which covers the method, the results,
what the model relies on, and the limitations:
[PDF](reports/strategy_survival_report.pdf) (GitHub renders it inline) or
[HTML](reports/strategy_survival_report.html). Its headline finding is that
ranking strategies by validation Sharpe predicts survival worse than a coin
flip. The report explains why.

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
run takes about two minutes. The versions in `requirements.txt` are the exact
ones the reported numbers were produced with.

Additional entry points and arguments:

```
python scripts/run_synthetic_pipeline.py --seed 8       # rerun on a different synthetic dataset
python scripts/run_synthetic_pipeline.py --no-report    # stop after metrics and figures
python scripts/run_build_report.py                      # rebuild the report only
pytest                                                  # the test suite
```

## Using it on your own data

The pipeline is not restricted to trading strategy survival studies.
Any right-censored duration data can fit. The system handles censoring and
re-censoring appropriately (rows that start on a date, run for some days,
and either end while you are watching or are still going when observation
stops are handled automatically).

The CSV requires a minimum of four columns, under any names: a unique id,
a start date, an observed duration in days (positive), and an event flag
(1 = the ending was observed, 0 = censored).

Every other column becomes a feature. Numeric columns pass through, missing
values included. Text columns are turned into one yes/no column per distinct
value (one-hot encoding). Values too rare to support an estimate are grouped
into a single `(other)` value. A column with gaps gets its own `(missing)`
value, so a gap stays visible to the model instead of looking identical to
the category a linear model measures the others against. Constant columns
are dropped with a notice.

Two flags carry the judgement the system cannot handle automatically:
`--drop-cols` for anything recorded after the outcome, which would leak the label, and
`--categorical-cols` for codes that read as numbers but are labels, such as
a district, zip code, or product id. For these features and those like them,
meaning does not live in its sequential number, but in its unique id. But the
system cannot understand this fact unless told explicitly.

This is an essential step because, if these features were to be left as numeric, a linear
model would incorrectly fit each code like a straight-line trend. It would
behave as if, for example, risk rose steadily from district 1 to district 50.

```
python scripts/run_fit_evaluate.py --data your.csv --name myrun --id-col id --date-col start_date --duration-col days --event-col ended
python scripts/run_predict.py --model runs/myrun --data new_rows.csv
```

The first command runs the same evaluation the synthetic pipeline is verified
with (expanding temporal folds, training labels re-censored at each split
date, likelihood-based selection, held-out scale calibration), writes metrics
and figures to `runs/myrun/`, and renders the report. `--no-report` skips the
render, and `python scripts/run_build_report.py --run runs/myrun` rebuilds it
later without refitting. The second command scores new rows.

Both fitted models are written to `runs/myrun/model/`. This includes the boosted model
(XGBoost with its accelerated-failure-time survival objective, AFT) and the
Cox baseline, the standard linear survival model. The run records which
scored higher out of time.

`run_predict.py` uses that one by default and prints which model it used. Use the argument
`--model-type` to override the default. Saving only the boosted model would misreport any
dataset the baseline wins. Models are build outputs and are not committed to the repo.

One honest caveat: on real data there is no ground truth, so there is no
oracle ceiling and no way to check feature attributions against a true
mechanism. The generated report states this rather than omitting it.

## A real-data demonstration

How long does a business keep its licence? `datasets/chicago_licences.csv.gz`
holds every City of Chicago business licence whose first issue falls after
2002, across 149 licence types. This is 262,763 licences, 83.6 percent of them closed
by the 2026 cutoff and the rest still current. Source and cleaning are
documented in [datasets/README.md](datasets/README.md). The run lives in
`reports/chicago_demo/`.

A user can recreate it with the command below.

```
python scripts/run_fit_evaluate.py --data datasets/chicago_licences.csv.gz --name chicago_licences --id-col licence_id --date-col first_issued --duration-col licensed_days --event-col closed --categorical-cols ward,community_area,police_district,zip_code --folds 5 --horizons 365,1095,1825 --out reports/chicago_demo
```

`--categorical-cols` is imperative here. Ward, community area, police
district and zip code are administrative codes: ward 43 is not more ward
than ward 12, the number is just a name. Without the flag they are read as
quantities, and a linear model then fits each one a straight-line trend which
is incorrect. These feature columns must be listed as categorical.

The dataset is committed, so that command reproduces the report as it
stands. `scripts/run_prepare_chicago.py` rebuilds the dataset from the city's
API and records how it was assembled, but be aware that running it fetches
whatever the portal holds today, which will no longer match the numbers below.

Out-of-time fold-mean concordance, the share of pairs the model puts in the
right order where 0.5 is a coin flip (C-index), is 0.695 for both models. Cox is ahead
in the fourth decimal, 0.6951 against 0.6946, which is a tie in any sense
that matters; the run defaults to Cox for scoring because it has to pick one.
That is the same lesson as the synthetic tie. On this problem a
penalized linear model is enough. Both beat a no-skill forecast on
censoring-weighted Brier score at all three horizons.

The strongest single predictor is the licence type, and the largest effects
are the temporary permits (special event food, pop-up retail, special event
liquor), which is the sanity check you want. The model's biggest claim is that
licences issued for one-off events do not last, which is true by construction.

Full detail in
[reports/chicago_demo/report.pdf](reports/chicago_demo/report.pdf).
