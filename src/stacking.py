"""
CAGE-Fake: Stacking Meta-Learner Module (Heterogeneous View Edition)
=====================================================================
Generates strictly leakage-free Out-Of-Fold (OOF) predictions via K-Fold CV on TRAIN split.
Each base model operates on its own orthogonal feature view (see base_models.py).
Meta-learner is a Logistic Regression trained on [p_lr, p_lgbm, p_rf, p_xgb, disagreement].

KEY CHANGE from the original:
  - fit() now receives the RAW SPARSE matrix from UnifiedFeaturePipeline (not the SVD-
    compressed dense matrix from ManifoldBalancingPipeline).
  - stream_boundaries (dict mapping stream name → (start, end) column indices) must be
    passed in so HeterogeneousViewBundle can slice each model's view correctly.
  - ManifoldBalancingPipeline is NO LONGER used as a pre-processing step before base models;
    the SVD is now computed INSIDE HeterogeneousViewBundle only for the LightGBM view.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from config import N_FOLDS_STACKING, RANDOM_SEED
from src.base_models import BaseEnsemble
from src.utils import setup_logger

logger = setup_logger("stacking")


class StackingClassifier:
    """
    Leakage-free Stacking Classifier with heterogeneous base views:
    1. Fits base learners across K-folds on TRAIN to produce OOF probability matrix.
       Each base learner uses its own orthogonal feature view.
    2. Fits LogisticRegression meta-learner on OOF [p_lr, p_lgbm, p_rf, p_xgb, disagreement].
    3. Re-fits base learners on entire TRAIN split for deployment.

    IMPORTANT: X_train must be the RAW SPARSE matrix from UnifiedFeaturePipeline.
               Do NOT pass SVD-compressed dense features here.
    """

    def __init__(self, n_folds: int = N_FOLDS_STACKING, random_state: int = RANDOM_SEED):
        self.n_folds = n_folds
        self.random_state = random_state
        self.meta_learner = LogisticRegression(random_state=self.random_state, max_iter=1000)
        self.base_ensemble = BaseEnsemble()
        self.is_fitted = False
        self._stream_boundaries: Optional[Dict] = None

    def fit(
        self,
        X_train,
        y_train,
        stream_boundaries: Optional[Dict] = None,
        use_smote: bool = False,
    ) -> "StackingClassifier":
        """
        Executes strict leakage-free K-fold cross-validation on TRAIN split.

        Args:
            X_train:            Raw sparse feature matrix (N, D) from UnifiedFeaturePipeline.
            y_train:            Binary label array (N,).
            stream_boundaries:  Dict mapping stream name → (start_col, end_col).
                                Must be provided so each model gets the right slice.
            use_smote:          If True, apply SMOTE inside each fold (on SVD-compressed LR
                                view only; other views are already dense and manageable).
        """
        if stream_boundaries is not None:
            self._stream_boundaries = stream_boundaries
        else:
            self._stream_boundaries = {}

        logger.info(
            f"Generating OOF predictions with {self.n_folds}-fold Stratified CV on TRAIN "
            f"(shape: {X_train.shape}, use_smote={use_smote}, "
            f"streams: {list(self._stream_boundaries.keys())})..."
        )

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        n_samples = X_train.shape[0]

        oof_p_lr   = np.zeros(n_samples, dtype=np.float32)
        oof_p_lgbm = np.zeros(n_samples, dtype=np.float32)
        oof_p_rf   = np.zeros(n_samples, dtype=np.float32)
        oof_p_xgb  = np.zeros(n_samples, dtype=np.float32)

        visited_indices = set()

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            logger.info(
                f"  Fold {fold + 1}/{self.n_folds}: "
                f"train={len(train_idx)}, val={len(val_idx)}"
            )
            X_tr = X_train[train_idx]
            X_va = X_train[val_idx]
            y_tr = y_train[train_idx]

            visited_indices.update(val_idx.tolist())

            # Build a fold-specific BaseEnsemble (fits its own SVD for the LightGBM view)
            fold_ensemble = BaseEnsemble(
                stream_boundaries=self._stream_boundaries,
                svd_components=100,
            )
            fold_ensemble.fit(X_tr, y_tr)

            fold_probas = fold_ensemble.predict_proba_dict(X_va)
            oof_p_lr[val_idx]   = fold_probas["lr"]
            oof_p_lgbm[val_idx] = fold_probas["lgbm"]
            oof_p_rf[val_idx]   = fold_probas["rf"]
            oof_p_xgb[val_idx]  = fold_probas["xgb"]

        # Completeness check
        assert len(visited_indices) == n_samples, (
            f"OOF indices incomplete: visited {len(visited_indices)} of {n_samples}"
        )

        # Genuine disagreement: std across orthogonal model outputs
        oof_matrix = np.column_stack([oof_p_lr, oof_p_lgbm, oof_p_rf, oof_p_xgb])
        oof_disagreement = np.std(oof_matrix, axis=1, ddof=0).astype(np.float32)

        # Meta-features: [p_lr, p_lgbm, p_rf, p_xgb, disagreement]
        oof_meta_features = np.column_stack(
            [oof_p_lr, oof_p_lgbm, oof_p_rf, oof_p_xgb, oof_disagreement]
        )
        self.oof_meta_features = oof_meta_features

        logger.info(
            f"Fitting LR meta-learner on {len(oof_meta_features)} real OOF examples "
            f"(OOF disagreement mean={oof_disagreement.mean():.4f}, "
            f"std={oof_disagreement.std():.4f})..."
        )
        self.meta_learner.fit(oof_meta_features, y_train)

        # Final deployment base ensemble (retrained on full TRAIN)
        logger.info("Fitting deployment base ensemble on full TRAIN split...")
        self.base_ensemble = BaseEnsemble(
            stream_boundaries=self._stream_boundaries,
            svd_components=100,
        )
        self.base_ensemble.fit(X_train, y_train)

        self.is_fitted = True
        logger.info("StackingClassifier fitted successfully with orthogonal views & zero OOF leakage.")
        return self

    def predict_proba(self, X) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Inference using heterogeneous views:
        1. Each base model predicts from its own orthogonal feature view.
        2. Meta-learner combines [p_lr, p_lgbm, p_rf, p_xgb, disagreement].

        Returns:
            p_stack:      shape (N,)  — stacked P(FAKE) from meta-learner
            disagreement: shape (N,)  — genuine inter-view std
            probas_dict:  per-model P(FAKE) dictionaries
        """
        if not self.is_fitted:
            raise RuntimeError("StackingClassifier must be fitted before predicting.")

        meta_features, probas_dict, disagreement = self.base_ensemble.predict_base_features(X)
        p_stack = self.meta_learner.predict_proba(meta_features)[:, 1]

        return p_stack.astype(np.float32), disagreement, probas_dict
