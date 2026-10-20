"""
CAGE-Fake: Base Learner Ensemble Module
Random Forest, XGBoost, LightGBM, and Logistic Regression models.
Extracts individual probabilities P(FAKE) and computes ensemble disagreement:
disagreement = std(p_rf, p_xgb, p_lgbm, p_lr).
"""

from typing import Dict, Tuple, List, Any
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from config import LGBM_PARAMS, RF_PARAMS, XGB_PARAMS, RANDOM_SEED
from src.utils import setup_logger

logger = setup_logger("base_models")

class BaseEnsemble:
    """
    Manages the diverse quartet of base learners:
    1. Random Forest (Bagging ensemble)
    2. XGBoost (Gradient Boosted Trees - exact/approx split)
    3. LightGBM (Gradient Boosted Trees - histogram based)
    4. Logistic Regression (Linear calibrated baseline)
    """

    def __init__(
        self,
        lgbm_params: Dict[str, Any] = None,
        rf_params: Dict[str, Any] = None,
        xgb_params: Dict[str, Any] = None,
        lr_params: Dict[str, Any] = None,
    ):
        self.lgbm_params = lgbm_params or LGBM_PARAMS
        self.rf_params = rf_params or RF_PARAMS
        self.xgb_params = xgb_params or XGB_PARAMS
        self.lr_params = lr_params or {"max_iter": 1000, "random_state": RANDOM_SEED, "C": 1.0}

        self.models = {
            "rf": RandomForestClassifier(**self.rf_params),
            "xgb": XGBClassifier(**self.xgb_params),
            "lgbm": LGBMClassifier(**self.lgbm_params),
            "lr": LogisticRegression(**self.lr_params),
        }
        self.is_fitted = False

    def fit(self, X_train, y_train) -> "BaseEnsemble":
        """Fits all base models on the training data."""
        logger.info(f"Training base learners on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        for name, model in self.models.items():
            logger.info(f"  Fitting {name.upper()}...")
            model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info("Base learners trained successfully.")
        return self

    def predict_proba_dict(self, X) -> Dict[str, np.ndarray]:
        """
        Returns dictionary of P(FAKE) probabilities:
        {'rf': p_rf, 'xgb': p_xgb, 'lgbm': p_lgbm, 'lr': p_lr}
        """
        if not self.is_fitted:
            raise RuntimeError("Base models must be fitted before predicting.")

        probas = {}
        for name, model in self.models.items():
            # Class 1 is FAKE
            p = model.predict_proba(X)[:, 1]
            probas[name] = p.astype(np.float32)
        return probas

    def predict_base_features(self, X) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes base probabilities and disagreement:
        Returns:
            stacked_features: np.ndarray of shape (N, 5) -> [p_rf, p_xgb, p_lgbm, p_lr, disagreement]
            probas_dict: dict of individual model probabilities
            disagreement: np.ndarray of shape (N,)
        """
        probas = self.predict_proba_dict(X)
        p_rf = probas["rf"]
        p_xgb = probas["xgb"]
        p_lgbm = probas["lgbm"]
        p_lr = probas["lr"]

        # Calculate disagreement = std(p_rf, p_xgb, p_lgbm, p_lr)
        probs_matrix = np.column_stack([p_rf, p_xgb, p_lgbm, p_lr])
        disagreement = np.std(probs_matrix, axis=1, ddof=0).astype(np.float32)

        stacked_features = np.column_stack([p_rf, p_xgb, p_lgbm, p_lr, disagreement])
        return stacked_features, probas, disagreement
