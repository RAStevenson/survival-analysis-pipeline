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

The headline results run on synthetic data with known ground truth; nothing
proprietary is here. The same pipeline also fits and evaluates any
right-censored duration CSV, one where some rows are still running when
observation stops, as in churn, equipment failure, or subscription lapse.
The repo includes a demonstration on 262,763 real Chicago business licences.
Both are covered below.

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
   python scripts/run_pipeline.py
   ```

The last command regenerates the dataset, runs the temporal cross-validation,
writes `reports/metrics.json` and the figures, and rebuilds the report. A full
run takes about two minutes. The versions in `requirements.txt` are the exact
ones the reported numbers were produced with.

Additional entry points and arguments:

```

python scripts/run_pipeline.py --seed 8       # rerun on a different synthetic dataset
python scripts/run_pipeline.py --no-report    # stop after metrics and figures
python scripts/run_generate_data.py           # write synthetic data/ and stop
python scripts/run_build_report.py            # rebuild the report only
pytest                                        # the test suite

```

## Using it on your own data

The pipeline is not tied to trading strategies. Any right-censored duration
data fits: rows that start on a date, run for some days, and either end while
you are watching or are still going when observation stops. The CSV needs
four columns, under any names: a unique id, a start date, an observed
duration in days (positive), and an event flag (1 = the ending was observed,
0 = censored). Every other column becomes a feature. Numeric columns pass
through, missing values included. Text columns are turned into one yes/no
column per distinct value (one-hot encoding). Values too rare to support an
estimate are grouped into a single `(other)` value. A column with gaps gets
its own `(missing)` value, so a gap stays visible to the model instead of
looking identical to the category a linear model measures the others
against. Constant columns are dropped with a notice.

Two flags carry the judgement the file cannot: `--drop-cols` for anything
recorded after the outcome, which would leak the label, and
`--categorical-cols` for codes that read as numbers but are labels, such as
a ward, zip or product id. A code's meaning lives in which thing it names,
not in the size of the number, and nothing in a CSV marks the difference.
Left numeric, a linear model fits each code one straight-line trend, as if
risk rose steadily from ward 1 to ward 50.

```
python scripts/run_fit_evaluate.py --data your.csv --name myrun --id-col id --date-col start_date --duration-col days --event-col ended
python scripts/run_build_report.py --run runs/myrun
python scripts/run_predict.py --model runs/myrun --data new_rows.csv
```

The first command runs the same evaluation the synthetic pipeline is verified
with (expanding temporal folds, training labels re-censored at each split
date, likelihood-based selection, held-out scale calibration), and writes
metrics and figures to `runs/myrun/`. The second renders the report; the
third scores new rows.

Both fitted models are written to `runs/myrun/model/`: the boosted model
(XGBoost with its accelerated-failure-time survival objective, AFT) and the
Cox baseline, the standard linear survival model. The run records which
scored higher out of time.
`run_predict.py` uses that one by default and prints which it used;
`--model-type` overrides. Saving only the boosted model would misreport any
dataset the baseline wins, and on real data it can. Models are build outputs
and are not committed: they rebuild from the data and code, and binaries do
not delta-compress, so committing them would grow the history with every
refit.

One honest caveat: on your data there is no ground truth, so there is no
oracle ceiling and no way to check feature attributions against a true
mechanism. The generated report states this rather than omitting it.

## A real-data demonstration

How long does a business keep its licence? `datasets/chicago_licences.csv.gz`
holds every City of Chicago business licence whose first issue falls after
2002, across 149 licence types: 262,763 licences, 83.6 percent of them closed
by the 2026 cutoff and the rest still current. Source and cleaning are
documented in [datasets/README.md](datasets/README.md). The run lives in
`reports/chicago_demo/`:

```
python scripts/run_fit_evaluate.py --data datasets/chicago_licences.csv.gz --name chicago_licences --id-col licence_id --date-col first_issued --duration-col licensed_days --event-col closed --categorical-cols ward,community_area,police_district,zip_code --folds 5 --horizons 365,1095,1825 --out reports/chicago_demo
python scripts/run_build_report.py --run reports/chicago_demo
```

`--categorical-cols` is doing real work there. Ward, community area, police
district and zip code are administrative codes: ward 43 is not more ward
than ward 12, the number is just a name. Without the flag they are read as
quantities, and a linear model then fits each one a straight-line trend, as
if the risk of closing rose steadily from ward 1 to ward 50.

The dataset is committed, so those two commands reproduce the report as it
stands. `scripts/run_prepare_chicago.py` rebuilds the dataset from the city's
API and records how it was assembled, but running it fetches whatever the
portal holds today, which will no longer match the numbers below.

Out-of-time fold-mean concordance, the share of pairs the model puts in the
right order where 0.5 is a coin flip, is 0.695 for both models. Cox is ahead
in the fourth decimal, 0.6951 against 0.6946, which is a tie in any sense
that matters; the run defaults to Cox for scoring because it has to pick
one. That is the same lesson as the synthetic tie: on this problem a
penalized linear model is enough. Both beat a no-skill forecast on
censoring-weighted Brier score at all three horizons.

The strongest single predictor is the licence type, and the largest effects
are the temporary permits (special event food, pop-up retail, special event
liquor), which is the sanity check you want: the model's biggest claim is that
licences issued for one-off events do not last, which is true by
construction.

Full detail in
[reports/chicago_demo/report.pdf](reports/chicago_demo/report.pdf).