"""
CAGE-Fake: Global Configuration File
Calibration-Aware Gating and Evidence-Grounded Reasoning for Fake News Detection
"""

from pathlib import Path
import os

# Project Roots and Paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DATA_PATH = DATA_DIR / "news.csv"

MODELS_DIR = ROOT_DIR / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure all primary directories exist
for directory in [DATA_DIR, TRAINED_MODELS_DIR, CHECKPOINTS_DIR, FIGURES_DIR, METRICS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Reproducibility
RANDOM_SEED = 42

# Stratified Split Ratios (Sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
CALIBRATION_RATIO = 0.10
TEST_RATIO = 0.10

# Lexical Features (Stream A)
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_SUBLINEAR_TF = True
TFIDF_MIN_DF = 2

# Semantic Features (Stream B)
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
SBERT_DIM = 384
SBERT_BATCH_SIZE = 64
EMBEDDINGS_CACHE_FILE = DATA_DIR / "sbert_embeddings_cache.joblib"

# Stylometric Features (Stream C - 15 features)
STYLOMETRIC_NUM_FEATURES = 15

# Readability Features (Stream D - 4 features)
READABILITY_NUM_FEATURES = 4

# Total Expected Dimensions
EXPECTED_TOTAL_DIMS = 10403  # 10000 TF-IDF + ~4 lexical stats + 384 SBERT + 15 stylometric + 4 readability

# Balancing (SMOTE)
SVD_COMPONENTS = 100  # Project sparse TF-IDF prior to SMOTE to avoid distance metric collapse
APPLY_SMOTE = True

# Base Learners
LGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 16,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "eval_metric": "logloss",
}

# Stacking Meta-Learner
N_FOLDS_STACKING = 5

# Calibration
CALIBRATION_METHOD = "sigmoid"  # Platt scaling
ISOTONIC_METHOD = "isotonic"
CALIBRATION_BINS = 10

# Uncertainty Gating Threshold Sweeps
TAU_SWEEP = [0.60, 0.65, 0.70, 0.75, 0.80]
DELTA_SWEEP = [0.10, 0.15, 0.20, 0.25]
DEFAULT_TAU = 0.65  # Default calibrated confidence threshold (configurable, default 0.65)
DEFAULT_DELTA = 0.15

# Evidence-Grounded Triage Configuration
DEFAULT_CONFIDENCE_THRESHOLD = 0.65  # Configurable calibrated confidence threshold
EVIDENCE_SIMILARITY_THRESHOLD = 0.58  # SBERT cosine similarity for claim-evidence support
EVIDENCE_CONTRADICTION_THRESHOLD = 0.55  # SBERT cosine similarity for contradiction detection
HEADLINE_BODY_MISMATCH_THRESHOLD = 0.40  # SBERT similarity below which headline-body mismatch is flagged
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"

# Research Benchmark Target Metrics
TARGET_METRICS = {
    "macro_f1": 0.90,
    "ece": 0.08,
    "selective_acc_at_80_coverage": 0.95,
}

# Transformer Baseline Configuration
ENABLE_TRANSFORMER = True
TRANSFORMER_MODEL = "microsoft/deberta-v3-base"
FALLBACK_TRANSFORMER = "distilroberta-base"
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_MAX_LEN = 256
TRANSFORMER_EPOCHS = 3
TRANSFORMER_LR = 2e-5

# Explainability Caching
SHAP_BACKGROUND_SIZE = 80
SHAP_CACHE_FILE = OUTPUTS_DIR / "shap_values_cache.joblib"
