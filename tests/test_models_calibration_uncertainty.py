"""
Unit and Integration Tests for Base Models, Stacking, Calibration, and Uncertainty Gating
"""

import pytest
import numpy as np

from src.base_models import BaseEnsemble
from src.stacking import StackingClassifier
from src.calibration import ModelCalibrator, compute_ece
from src.uncertainty import UncertaintyGate, ThresholdOptimizer

def test_base_ensemble_4_models():
    np.random.seed(42)
    N, D = 80, 20
    X = np.random.randn(N, D)
    y = np.random.choice([0, 1], size=N)

    ensemble = BaseEnsemble()
    assert set(ensemble.models.keys()) == {"rf", "xgb", "lgbm", "lr"}

    ensemble.fit(X, y)
    probas = ensemble.predict_proba_dict(X)

    assert set(probas.keys()) == {"rf", "xgb", "lgbm", "lr"}
    for name, p in probas.items():
        assert p.shape == (N,)
        assert (p >= 0.0).all() and (p <= 1.0).all()

    stacked_features, probas_dict, disagreement = ensemble.predict_base_features(X)
    assert stacked_features.shape == (N, 5)  # 4 base models + disagreement
    assert disagreement.shape == (N,)
    assert (disagreement >= 0.0).all()

def test_stacking_classifier():
    np.random.seed(42)
    N, D = 60, 15
    X = np.random.randn(N, D)
    y = np.random.choice([0, 1], size=N)

    stacker = StackingClassifier(n_folds=3)
    stacker.fit(X, y)

    p_stack, disagreement, probas_dict = stacker.predict_proba(X)
    assert p_stack.shape == (N,)
    assert (p_stack >= 0.0).all() and (p_stack <= 1.0).all()
    assert disagreement.shape == (N,)

def test_calibration_and_ece():
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    p_uncal = np.array([0.9, 0.85, 0.7, 0.2, 0.1, 0.3, 0.6, 0.4, 0.75, 0.25])

    ece, bin_details = compute_ece(y_true, p_uncal)
    assert 0.0 <= ece <= 1.0
    assert "bin_accuracies" in bin_details
    assert "bin_confidences" in bin_details

    calibrator = ModelCalibrator(method="sigmoid")
    calibrator.fit(p_uncal, y_true)

    p_cal = calibrator.calibrate(p_uncal)
    assert p_cal.shape == p_uncal.shape
    assert (p_cal >= 0.0).all() and (p_cal <= 1.0).all()

def test_uncertainty_gate_and_threshold_sweep():
    gate = UncertaintyGate(tau=0.70, delta=0.15)

    # 1. High confidence REAL, low disagreement -> REAL
    res1 = gate.evaluate_single(p_calibrated=0.10, disagreement=0.04)
    assert res1["decision"] == "REAL"
    assert res1["trigger"] == "none"

    # 2. High confidence FAKE, low disagreement -> FAKE
    res2 = gate.evaluate_single(p_calibrated=0.90, disagreement=0.03)
    assert res2["decision"] == "FAKE"
    assert res2["trigger"] == "none"

    # 3. Low confidence only (0.55 confidence < 0.70 tau, 0.04 dis <= 0.15 delta) -> UNCERTAIN (low_confidence)
    res3 = gate.evaluate_single(p_calibrated=0.55, disagreement=0.04)
    assert res3["decision"] == "UNCERTAIN"
    assert res3["trigger"] == "low_confidence"
    assert res3["is_low_confidence"] and not res3["is_high_disagreement"]

    # 4. High disagreement only (0.88 conf >= 0.70 tau, 0.22 dis > 0.15 delta) -> UNCERTAIN (high_disagreement)
    res4 = gate.evaluate_single(p_calibrated=0.88, disagreement=0.22)
    assert res4["decision"] == "UNCERTAIN"
    assert res4["trigger"] == "high_disagreement"
    assert not res4["is_low_confidence"] and res4["is_high_disagreement"]

    # 5. Both triggers active (0.55 conf < 0.70 tau AND 0.22 dis > 0.15 delta) -> UNCERTAIN (both_triggers)
    res5 = gate.evaluate_single(p_calibrated=0.55, disagreement=0.22)
    assert res5["decision"] == "UNCERTAIN"
    assert res5["trigger"] == "both_triggers"
    assert res5["is_low_confidence"] and res5["is_high_disagreement"]

    # 6. Batch evaluation covering all states
    batch_res = gate.evaluate_batch(
        p_calibrated=np.array([0.10, 0.90, 0.55, 0.88, 0.55]),
        disagreements=np.array([0.04, 0.03, 0.04, 0.22, 0.22]),
    )
    assert list(batch_res["triggers"]) == ["none", "none", "low_confidence", "high_disagreement", "both_triggers"]
    assert list(batch_res["decisions"]) == ["REAL", "FAKE", "UNCERTAIN", "UNCERTAIN", "UNCERTAIN"]

    # Test threshold optimizer
    y_val = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    p_val = np.array([0.85, 0.15, 0.90, 0.10, 0.55, 0.45, 0.88, 0.12])
    dis_val = np.array([0.05, 0.04, 0.06, 0.03, 0.18, 0.08, 0.05, 0.04])

    optimizer = ThresholdOptimizer(tau_grid=[0.60, 0.70, 0.80], delta_grid=[0.10, 0.15, 0.20])
    sweep = optimizer.sweep(y_val, p_val, dis_val, target_coverage=0.70)
    assert "best_tau" in sweep
    assert "best_delta" in sweep
    assert "all_sweep_results" in sweep
