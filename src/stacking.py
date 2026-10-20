"""
CAGE-Fake: Stacking Meta-Learner Module
Generates strictly leakage-free Out-Of-Fold (OOF) predictions via K-Fold CV on TRAIN split,
and trains a Logistic Regression meta-learner on [p_lgbm, p_rf, p_xgb, disagreement].
"""

from typing import Tuple, Dict, Any
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import N_FOLDS_STACKING, RANDOM_SEED, LGBM_PARAMS, RF_PARAMS, XGB_PARAMS
from src.base_models import BaseEnsemble
from src.utils import setup_logger

logger = setup_logger("stacking")

class StackingClassifier:
    """
    Leakage-free Stacking Classifier:
    1. Fits base learners across K-folds on TRAIN to produce OOF probability matrix.
    2. Fits LogisticRegression meta-learner on OOF [p_lgbm, p_rf, p_xgb, disagreement].
    3. Re-fits base learners on entire TRAIN split for downstream inference.
    """

    def __init__(self, n_folds: int = N_FOLDS_STACKING, random_state: int = RANDOM_SEED):
        self.n_folds = n_folds
        self.random_state = random_state
        self.meta_learner = LogisticRegression(random_state=self.random_state, max_iter=1000)
        self.base_ensemble = BaseEnsemble()
        self.is_fitted = False

    def fit(self, X_train, y_train, use_smote: bool = False) -> "StackingClassifier":
        """
        Executes strict leakage-free K-fold cross-validation on TRAIN split to generate OOF meta-features,
        fits the meta-learner on OOF predictions, and fits the final base ensemble on full TRAIN.

        Guarantees:
        1. Folds are partitioned from original real training examples.
        2. If SMOTE is enabled, it is applied strictly to the training portion of each fold.
        3. Held-out validation examples are never resampled, augmented, or seen during fold training.
        4. Exactly one valid OOF prediction is generated for every real training example.
        5. The meta-learner is trained on real labels without synthetic contamination.
        6. Calibration and test data are completely excluded from this procedure.
        """
        logger.info(f"Generating Out-of-Fold (OOF) predictions with {self.n_folds}-fold Stratified CV on TRAIN (use_smote={use_smote})...")
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        n_samples = X_train.shape[0]
        oof_p_rf = np.zeros(n_samples, dtype=np.float32)
        oof_p_xgb = np.zeros(n_samples, dtype=np.float32)
        oof_p_lgbm = np.zeros(n_samples, dtype=np.float32)
        oof_p_lr = np.zeros(n_samples, dtype=np.float32)

        visited_indices = set()

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            logger.info(f"  Training fold {fold + 1}/{self.n_folds} (train: {len(train_idx)}, val: {len(val_idx)})...")
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_va = X_train[val_idx]

            # Track visited validation indices to verify completeness
            visited_indices.update(val_idx)

            # Apply SMOTE strictly to fold training data if enabled
            if use_smote:
                from imblearn.over_sampling import SMOTE
                sm = SMOTE(random_state=self.random_state + fold)
                X_tr_fit, y_tr_fit = sm.fit_resample(X_tr, y_tr)
            else:
                X_tr_fit, y_tr_fit = X_tr, y_tr

            fold_rf = RandomForestClassifier(**RF_PARAMS)
            fold_xgb = XGBClassifier(**XGB_PARAMS)
            fold_lgbm = LGBMClassifier(**LGBM_PARAMS)
            fold_lr = LogisticRegression(max_iter=1000, random_state=self.random_state, C=1.0)

            fold_rf.fit(X_tr_fit, y_tr_fit)
            fold_xgb.fit(X_tr_fit, y_tr_fit)
            fold_lgbm.fit(X_tr_fit, y_tr_fit)
            fold_lr.fit(X_tr_fit, y_tr_fit)

            # Predict on strictly untouched, real held-out validation samples
            oof_p_rf[val_idx] = fold_rf.predict_proba(X_va)[:, 1]
            oof_p_xgb[val_idx] = fold_xgb.predict_proba(X_va)[:, 1]
            oof_p_lgbm[val_idx] = fold_lgbm.predict_proba(X_va)[:, 1]
            oof_p_lr[val_idx] = fold_lr.predict_proba(X_va)[:, 1]

        # Completeness verification
        assert len(visited_indices) == n_samples, f"OOF indices incomplete: visited {len(visited_indices)} of {n_samples}"

        # Calculate OOF disagreement = std(p_rf, p_xgb, p_lgbm, p_lr)
        oof_matrix = np.column_stack([oof_p_rf, oof_p_xgb, oof_p_lgbm, oof_p_lr])
        oof_disagreement = np.std(oof_matrix, axis=1, ddof=0).astype(np.float32)

        # Meta-features: [p_rf, p_xgb, p_lgbm, p_lr, disagreement]
        oof_meta_features = np.column_stack([oof_p_rf, oof_p_xgb, oof_p_lgbm, oof_p_lr, oof_disagreement])
        self.oof_meta_features = oof_meta_features

        logger.info(f"Fitting Logistic Regression meta-learner on {len(oof_meta_features)} real OOF features...")
        self.meta_learner.fit(oof_meta_features, y_train)

        # Train final base models on all of permitted TRAIN split for deployment
        logger.info("Fitting deployment base learners on full TRAIN split...")
        if use_smote:
            from imblearn.over_sampling import SMOTE
            sm_final = SMOTE(random_state=self.random_state)
            X_train_final, y_train_final = sm_final.fit_resample(X_train, y_train)
            self.base_ensemble.fit(X_train_final, y_train_final)
        else:
            self.base_ensemble.fit(X_train, y_train)

        self.is_fitted = True
        logger.info("Stacking Classifier fitted successfully with zero OOF leakage.")
        return self

    def predict_proba(self, X) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Inference:
        1. Predicts base probabilities and disagreement
        2. Predicts meta-learner probability p_stack = P(FAKE)
        Returns:
            p_stack: np.ndarray (N,) -> P(FAKE)
            disagreement: np.ndarray (N,) -> std across base learners
            probas_dict: dict with individual model probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("StackingClassifier must be fitted before predicting.")

        meta_features, probas_dict, disagreement = self.base_ensemble.predict_base_features(X)
        p_stack = self.meta_learner.predict_proba(meta_features)[:, 1]

        return p_stack.astype(np.float32), disagreement, probas_dict
