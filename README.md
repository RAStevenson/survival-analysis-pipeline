# Strategy survival meta-model

Trading strategies, like many other lifespan problems, decay over time. 
An edge that clears validation today is often unprofitable within a 
few months, and some strategies last far longer than others.

For a live portfolio of strategies this raises three practical questions: how much capital to
give a new strategy, when to schedule its first review, and when to cut it.
All three depend on how long the edge will last, so this project builds a
model that predicts a strategy's lifespan from the metadata available on the
day it is deployed.

Everything in this repo runs on synthetic data with known ground truth.
Nothing proprietary is here.

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
pytest                                        # run the 41 tests

```