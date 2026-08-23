"""Tests for the reproducibility checker's decision logic: the two-tolerance
split, the composition-sensitive skips, structural key changes, and the SHAP
top-set check. These encode the fixture cases the tolerance split was verified
against by hand before the 2026-08-17 CI fix, so a future edit to the checker
cannot silently start passing real drift or failing good runs.

The script is loaded by file path because scripts/ is not a package.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_check_reproducibility.py"
_spec = importlib.util.spec_from_file_location("run_check_reproducibility", _SCRIPT)
assert _spec is not None and _spec.loader is not None
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)

BASE = {
    "pooled": {"c_xgb": 0.78, "c_cox": 0.77},
    "folds": [{"c_xgb": 0.75, "n_test": 200}, {"c_xgb": 0.76, "n_test": 300}],
    "calibration_180d": [{"n": 100, "predicted": 0.5, "observed_km": 0.52}],
    "shap_top": [{"feature": "a"}, {"feature": "b"}, {"feature": "c"}],
    "run": {"name": "x"},
}


def run_check(tmp_path, monkeypatch, capsys, ref: dict, cand: dict) -> tuple[int, str]:
    ref_path, cand_path = tmp_path / "ref.json", tmp_path / "cand.json"
    ref_path.write_text(json.dumps(ref))
    cand_path.write_text(json.dumps(cand))
    monkeypatch.setattr(
        sys, "argv", ["run_check_reproducibility.py", str(ref_path), str(cand_path)]
    )
    code = 0
    try:
        checker.main()
    except SystemExit as err:
        code = err.code if isinstance(err.code, int) else 1
    return code, capsys.readouterr().out


def test_fold_drift_within_fold_tolerance_passes(tmp_path, monkeypatch, capsys):
    # 3e-3 on a per-fold value: past the strict tolerance, inside the fold one.
    cand = copy.deepcopy(BASE)
    cand["folds"][0]["c_xgb"] = 0.753
    code, out = run_check(tmp_path, monkeypatch, capsys, BASE, cand)
    assert code == 0
    assert "OK" in out


def test_same_drift_on_a_pooled_value_fails(tmp_path, monkeypatch, capsys):
    cand = copy.deepcopy(BASE)
    cand["pooled"]["c_xgb"] = 0.783
    code, out = run_check(tmp_path, monkeypatch, capsys, BASE, cand)
    assert code == 1
    assert "pooled.c_xgb" in out


def test_fold_drift_past_fold_tolerance_fails(tmp_path, monkeypatch, capsys):
    cand = copy.deepcopy(BASE)
    cand["folds"][0]["c_xgb"] = 0.756
    code, out = run_check(tmp_path, monkeypatch, capsys, BASE, cand)
    assert code == 1
    assert "folds[0].c_xgb" in out


def test_missing_and_extra_keys_fail_regardless_of_tolerance(tmp_path, monkeypatch, capsys):
    cand = copy.deepcopy(BASE)
    del cand["pooled"]["c_cox"]
    cand["pooled"]["c_new"] = 0.5
    code, out = run_check(tmp_path, monkeypatch, capsys, BASE, cand)
    assert code == 1
    assert "missing from cand.json: .pooled.c_cox" in out
    assert "not present in ref.json: .pooled.c_new" in out


def test_calibration_bin_values_are_skipped(tmp_path, monkeypatch, capsys):
    # A bin-membership shift can move a bin's observed value arbitrarily far
    # without the model having changed; the checker must not fail on it.
    cand = copy.deepcopy(BASE)
    cand["calibration_180d"][0]["observed_km"] = 0.9
    code, out = run_check(tmp_path, monkeypatch, capsys, BASE, cand)
    assert code == 0
    assert "composition-sensitive" in out


def test_shap_reorder_within_top_set_passes(tmp_path, monkeypatch, capsys):
    cand = copy.deepcopy(BASE)
    cand["shap_top"] = [{"feature": "c"}, {"feature": "a"}, {"feature": "b"}]
    code, _ = run_check(tmp_path, monkeypatch, capsys, BASE, cand)
    assert code == 0


def test_shap_top_set_change_fails(tmp_path, monkeypatch, capsys):
    cand = copy.deepcopy(BASE)
    cand["shap_top"] = [{"feature": "a"}, {"feature": "b"}, {"feature": "d"}]
    code, out = run_check(tmp_path, monkeypatch, capsys, BASE, cand)
    assert code == 1
    assert "SHAP features changed" in out
