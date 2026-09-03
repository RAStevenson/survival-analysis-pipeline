"""Survival analysis on right-censored duration data: fit and evaluate an
XGBoost AFT model against a Cox baseline on any duration CSV, on expanding
temporal folds with training labels re-censored at each split, and generate a
report from what was measured. The trading-strategy study drawn by
`synthetic_generator` is the validation run, not the subject."""

__version__ = "0.1.0"
