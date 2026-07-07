# TODO

Polish items, roughly in the order I would do them.

- GitHub Actions workflow running ruff + pytest on push.
- Multi-seed generator sweep (10+ seeds) to separate data variance from model
  variance; report C-index spread instead of a single seeded number.
- Pin exact dependency versions (lockfile or constraints.txt) so the seeded
  results are reproducible across machines.
- Uno's C (IPCW concordance) alongside Harrell's; Harrell's is biased under
  heavy censoring and fold 5 is 39% censored.
- D-calibration test in addition to the single-horizon decile plot.
- Block bootstrap by discovery month to stop understating CI width.
- Competing-risks treatment of administrative retirement instead of
  independent censoring (cause-specific hazards or Fine-Gray).
- Per-fold hyperparameter re-selection to measure how sensitive fold metrics
  are to the shared-params shortcut.
- SHAP interaction values; check whether the model found the
  val_sharpe x n_trades interaction the generator implies.
- CLI flags for observation cutoff and discovery window.
- pytest filterwarnings for shap's matplotlib deprecation noise.
- Dark-mode variants of the report figures.
