"""
CAGE-Fake: Base Learner Ensemble Module
LightGBM, Random Forest, and XGBoost models.
Extracts individual probabilities P(FAKE) and computes ensemble disagreement:
disagreement = std(p_lgbm, p_rf, p_xgb).
"""

from typing import Dict, Tuple, List, Any
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import LGBM_PARAMS, RF_PARAMS, XGB_PARAMS, RANDOM_SEED
from src.utils import setup_logger

logger = setup_logger("base_models")

class BaseEnsemble:
    """
    Manages the trio of diverse base learners:
    1. LightGBM (Gradient Boosted Trees - histogram based)
    2. Random Forest (Bagging ensemble)
    3. XGBoost (Gradient Boosted Trees - exact/approx split)
    """

    def __init__(
        self,
        lgbm_params: Dict[str, Any] = None,
        rf_params: Dict[str, Any] = None,
        xgb_params: Dict[str, Any] = None,
    ):
        self.lgbm_params = lgbm_params or LGBM_PARAMS
        self.rf_params = rf_params or RF_PARAMS
        self.xgb_params = xgb_params or XGB_PARAMS

        self.models = {
            "lgbm": LGBMClassifier(**self.lgbm_params),
            "rf": RandomForestClassifier(**self.rf_params),
            "xgb": XGBClassifier(**self.xgb_params),
        }
        self.is_fitted = False

    def fit(self, X_train, y_train) -> "BaseEnsemble":
        """Fits all three base models on the training data."""
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
        {'lgbm': p_lgbm, 'rf': p_rf, 'xgb': p_xgb}
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
            stacked_features: np.ndarray of shape (N, 4) -> [p_lgbm, p_rf, p_xgb, disagreement]
            probas_dict: dict of individual model probabilities
            disagreement: np.ndarray of shape (N,)
        """
        probas = self.predict_proba_dict(X)
        p_lgbm = probas["lgbm"]
        p_rf = probas["rf"]
        p_xgb = probas["xgb"]

        # Calculate disagreement = std(p_lgbm, p_rf, p_xgb)
        probs_matrix = np.column_stack([p_lgbm, p_rf, p_xgb])
        disagreement = np.std(probs_matrix, axis=1, ddof=0).astype(np.float32)

        stacked_features = np.column_stack([p_lgbm, p_rf, p_xgb, disagreement])
        return stacked_features, probas, disagreement
