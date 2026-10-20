"""
CAGE-Fake: Data Balancing Module (SMOTE & TruncatedSVD)
Implements principled oversampling on lower-dimensional dense projections.

THEORETICAL JUSTIFICATION (Why SMOTE is NEVER applied on raw 10,000-d sparse TF-IDF):
1. Curse of Dimensionality & Distance Metric Collapse:
   In high-dimensional sparse spaces (e.g., 10,000-d text vectors), Euclidean distance between
   arbitrary pairs of vectors converges to nearly identical values (the distance concentration phenomenon).
   As a consequence, k-Nearest Neighbors (k-NN) fails to identify meaningful local semantic neighborhoods,
   resulting in arbitrary neighbors being selected for synthetic interpolation.
2. Sparsity Destruction & Semantic Incoherence:
   Convex combinations of two sparse one-hot or n-gram vectors produce dense synthetic vectors with non-zero
   fractional counts across disjoint vocabulary terms, generating unnatural linguistic artifacts that never
   occur in real-world corpora.
3. Principled Solution:
   We project the lexical subspace into a dense latent semantic space via TruncatedSVD (LSA) prior to SMOTE.
   This preserves the low-dimensional manifold geometry where Euclidean distances represent true distributional
   topic affinities, allowing SMOTE to generate coherent synthetic points.
"""

from typing import Tuple
import numpy as np
from scipy.sparse import issparse, csr_matrix
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE

from config import SVD_COMPONENTS, RANDOM_SEED
from src.utils import setup_logger

logger = setup_logger("balancing")

class ManifoldBalancingPipeline:
    """
    Applies TruncatedSVD dimensionality reduction on high-dimensional sparse text representations,
    and provides optional SMOTE oversampling exclusively on training data manifolds.
    """

    def __init__(self, svd_components: int = SVD_COMPONENTS, random_state: int = RANDOM_SEED):
        self.svd_components = svd_components
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=self.svd_components, random_state=self.random_state)
        self.smote = SMOTE(random_state=self.random_state)
        self.is_fitted = False

    def fit(self, X_train: csr_matrix | np.ndarray) -> "ManifoldBalancingPipeline":
        """Fits TruncatedSVD strictly on the training partition."""
        if issparse(X_train) and X_train.shape[1] > self.svd_components:
            logger.info(f"Fitting TruncatedSVD ({self.svd_components} components) on train split...")
            self.svd.fit(X_train)
            self.is_fitted = True
        else:
            self.is_fitted = False
        return self

    def transform(self, X: csr_matrix | np.ndarray) -> np.ndarray:
        """Projects unseen data to dense SVD space if fitted, else converts to dense array."""
        if self.is_fitted:
            return self.svd.transform(X).astype(np.float32)
        return X.toarray().astype(np.float32) if issparse(X) else np.asarray(X, dtype=np.float32)

    def fit_transform(self, X_train: csr_matrix | np.ndarray) -> np.ndarray:
        """Fits TruncatedSVD on X_train and returns dense projection."""
        self.fit(X_train)
        return self.transform(X_train)

    def balance_training_data(
        self,
        X_train: csr_matrix | np.ndarray,
        y_train: np.ndarray,
        apply_smote: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balances training split:
        1. Projects sparse matrix X_train to dense SVD manifold (strictly on train split).
        2. If apply_smote=True, applies SMOTE on the dense representation.
           If classes are already naturally balanced (e.g. 50.9% vs 49.1%), notes that SMOTE
           synthesizes very few points.
        Returns: (X_train_dense, y_train_dense)
        """
        counts = np.bincount(y_train)
        logger.info(f"Training split class distribution: {counts} ({counts[1]/len(y_train)*100:.1f}% minority/majority)")

        # Step 1: Dense Projection
        if not self.is_fitted:
            X_dense = self.fit_transform(X_train)
        else:
            X_dense = self.transform(X_train)

        # Step 2: SMOTE (if requested)
        if apply_smote:
            diff = abs(counts[0] - counts[1])
            if diff < len(y_train) * 0.05:
                logger.info(f"Dataset is naturally balanced (difference = {diff} samples out of {len(y_train)}).")
            logger.info("Applying SMOTE on dense SVD projection...")
            X_bal, y_bal = self.smote.fit_resample(X_dense, y_train)
            logger.info(f"Balanced training set shape: {X_bal.shape}, class distribution {np.bincount(y_bal)}")
            return X_bal.astype(np.float32), y_bal.astype(int)

        return X_dense.astype(np.float32), y_train.astype(int)
