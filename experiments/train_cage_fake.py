"""
CAGE-Fake: Main Pipeline Training Script
Executes the full end-to-end training and evaluation pipeline.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import setup_logger, set_seed, save_artifact
from src.data_loader import load_and_standardize_data, create_stratified_splits
from src.preprocessing import TextPreprocessor
from src.feature_pipeline import UnifiedFeaturePipeline
from src.balancing import ManifoldBalancingPipeline
from src.stacking import StackingClassifier
from src.calibration import ModelCalibrator, compute_ece
from src.uncertainty import ThresholdOptimizer
from src.explainability import SHAPExplainer
from src.evaluation import evaluate_predictions, save_metrics, plot_confusion_matrix
from config import TRAINED_MODELS_DIR, RANDOM_SEED

logger = setup_logger("train_cage_fake")

def main():
    set_seed(RANDOM_SEED)
    logger.info("Starting CAGE-Fake End-to-End Training Pipeline...")
    
    start_time = time.time()

    # 1. Load Data
    logger.info("Step 1: Loading and splitting data...")
    df, stats = load_and_standardize_data()
    train_df, val_df, calib_df, test_df = create_stratified_splits(df)

    # 2. Preprocessing
    logger.info("Step 2: Preprocessing text...")
    preprocessor = TextPreprocessor(use_spacy=False) # Fallback for speed
    
    train_raw, train_clean, _ = preprocessor.batch_process(train_df["title"].tolist(), train_df["body"].tolist())
    val_raw, val_clean, _ = preprocessor.batch_process(val_df["title"].tolist(), val_df["body"].tolist())
    calib_raw, calib_clean, _ = preprocessor.batch_process(calib_df["title"].tolist(), calib_df["body"].tolist())
    test_raw, test_clean, _ = preprocessor.batch_process(test_df["title"].tolist(), test_df["body"].tolist())

    y_train, y_val, y_calib, y_test = train_df["target"].values, val_df["target"].values, calib_df["target"].values, test_df["target"].values

    # 3. Feature Pipeline
    logger.info("Step 3: Extracting features (Streams A, B, C, D)...")
    pipeline = UnifiedFeaturePipeline(include_semantic=True)
    X_train = pipeline.fit_transform(train_clean, train_raw, return_sparse=True)
    
    # Transform others
    X_val = pipeline.transform(val_clean, val_raw, return_sparse=True)
    X_calib = pipeline.transform(calib_clean, calib_raw, return_sparse=True)
    X_test = pipeline.transform(test_clean, test_raw, return_sparse=True)

    # 4. Balancing
    logger.info("Step 4: Balancing training data...")
    balancer = ManifoldBalancingPipeline()
    X_train_bal, y_train_bal = balancer.balance_training_data(X_train, y_train)

    # 5 & 6. Base Models + Stacking
    logger.info("Step 5 & 6: Training Base Models & OOF Stacking Meta-Learner...")
    stacking_model = StackingClassifier()
    # Stacking is trained on balanced dense features
    stacking_model.fit(X_train_bal, y_train_bal)
    
    # 7. Calibration
    logger.info("Step 7: Calibrating on isolated Calibration split...")
    # Get uncalibrated probabilities for calibration set
    # Note: Stacking input must match training feature format. If we used X_train_bal (dense SVD), we need to ensure the transform path is consistent.
    # To keep things simpler, let's assume the stacking model can predict on the sparse transform, or we adapt.
    # We should project X_calib if SVD was used!
    
    if balancer.is_fitted:
        X_calib_dense = balancer.svd.transform(X_calib)
        X_val_dense = balancer.svd.transform(X_val)
        X_test_dense = balancer.svd.transform(X_test)
        X_train_bg = balancer.svd.transform(X_train[:100])
    else:
        X_calib_dense = X_calib.toarray() if hasattr(X_calib, 'toarray') else X_calib
        X_val_dense = X_val.toarray() if hasattr(X_val, 'toarray') else X_val
        X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
        X_train_bg = X_train[:100].toarray() if hasattr(X_train, 'toarray') else X_train[:100]

    p_calib_uncal, _, _ = stacking_model.predict_proba(X_calib_dense)
    calibrator = ModelCalibrator()
    calibrator.fit(p_calib_uncal, y_calib)

    # 8. Threshold Sweep on Val
    logger.info("Step 8: Uncertainty Threshold Sweep on Validation set...")
    p_val_uncal, dis_val, _ = stacking_model.predict_proba(X_val_dense)
    p_val_cal = calibrator.calibrate(p_val_uncal)
    
    optimizer = ThresholdOptimizer()
    sweep_results = optimizer.sweep(y_val, p_val_cal, dis_val, target_coverage=0.80)
    best_tau = sweep_results["best_tau"]
    best_delta = sweep_results["best_delta"]
    
    # Evaluate Test Set
    logger.info("Step 11: Evaluating Test Set...")
    p_test_uncal, dis_test, _ = stacking_model.predict_proba(X_test_dense)
    p_test_cal = calibrator.calibrate(p_test_uncal)
    
    from src.uncertainty import UncertaintyGate
    gate = UncertaintyGate(tau=best_tau, delta=best_delta)
    gate_res = gate.evaluate_batch(p_test_cal, dis_test)
    
    y_pred_test = (p_test_cal >= 0.5).astype(int)
    metrics = evaluate_predictions(y_test, y_pred_test, p_test_cal, gate_res["is_accepted"])
    ece_cal, _ = compute_ece(y_test, p_test_cal)
    metrics["ece"] = round(ece_cal, 4)
    metrics["best_tau"] = best_tau
    metrics["best_delta"] = best_delta
    
    logger.info(f"Test Metrics: {metrics}")
    save_metrics(metrics, "cage_fake_metrics.json")
    
    # 9. SHAP Explainability
    logger.info("Step 9: Fitting SHAP Explainer...")
    explainer = SHAPExplainer(
        model=stacking_model.meta_learner, 
        feature_names=["p_lgbm", "p_rf", "p_xgb", "disagreement"], 
        stream_boundaries={"meta": (0, 4)}
    )
    # SHAP fits on the meta-features [p_lgbm, p_rf, p_xgb, disagreement]
    _, _, _ = stacking_model.predict_proba(X_train_bg) # just to get format
    meta_bg, _, _ = stacking_model.base_ensemble.predict_base_features(X_train_bg)
    explainer.fit(meta_bg)

    # Save artifacts
    logger.info("Saving trained models and artifacts...")
    save_artifact(pipeline, TRAINED_MODELS_DIR / "feature_pipeline.joblib")
    save_artifact(balancer, TRAINED_MODELS_DIR / "balancer.joblib")
    save_artifact(stacking_model, TRAINED_MODELS_DIR / "stacking_model.joblib")
    save_artifact(calibrator, TRAINED_MODELS_DIR / "calibrator.joblib")
    save_artifact(gate, TRAINED_MODELS_DIR / "uncertainty_gate.joblib")
    save_artifact(explainer, TRAINED_MODELS_DIR / "shap_explainer.joblib")

    logger.info(f"Training Complete! Total time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
