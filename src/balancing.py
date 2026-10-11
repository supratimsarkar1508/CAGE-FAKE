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
    Applies SMOTE exclusively on training data after projecting high-dimensional
    sparse features via TruncatedSVD to prevent distance metric degradation.
    """

    def __init__(self, svd_components: int = SVD_COMPONENTS, random_state: int = RANDOM_SEED):
        self.svd_components = svd_components
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=self.svd_components, random_state=self.random_state)
        self.smote = SMOTE(random_state=self.random_state)
        self.is_fitted = False

    def balance_training_data(
        self,
        X_train: csr_matrix | np.ndarray,
        y_train: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balances training split:
        1. Projects sparse matrix X_train to dense SVD manifold.
        2. Applies SMOTE to synthesize minority class instances.
        Returns: (X_train_balanced_dense, y_train_balanced)
        """
        logger.info(f"Balancing training set: original shape {X_train.shape}, class distribution {np.bincount(y_train)}")

        # Check if already balanced
        counts = np.bincount(y_train)
        if len(counts) > 1 and abs(counts[0] - counts[1]) / len(y_train) < 0.02:
            logger.info("Classes are already well-balanced (<2% difference). Retaining natural training distribution.")
            if issparse(X_train):
                return X_train, y_train
            return X_train, y_train

        # If sparse and dimensional > 500, apply SVD first
        if issparse(X_train) and X_train.shape[1] > self.svd_components:
            logger.info(f"Fitting TruncatedSVD ({self.svd_components} components) on train split...")
            X_dense = self.svd.fit_transform(X_train)
            self.is_fitted = True
            logger.info("Applying SMOTE on dense SVD projection...")
            X_bal, y_bal = self.smote.fit_resample(X_dense, y_train)
            logger.info(f"Balanced training set shape: {X_bal.shape}, class distribution {np.bincount(y_bal)}")
            return X_bal, y_bal
        else:
            X_dense = X_train.toarray() if issparse(X_train) else X_train
            X_bal, y_bal = self.smote.fit_resample(X_dense, y_train)
            logger.info(f"Balanced training set shape: {X_bal.shape}, class distribution {np.bincount(y_bal)}")
            return X_bal, y_bal
