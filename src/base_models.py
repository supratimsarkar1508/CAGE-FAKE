"""
CAGE-Fake: Heterogeneous Base Learner Ensemble with Orthogonal Feature Views
===========================================================================
DESIGN RATIONALE (why the old design was broken):
-------------------------------------------------
The original pipeline fed all four base models (RF, XGB, LightGBM, LR) the SAME
100-dimensional TruncatedSVD projection of the combined feature matrix. This is the
root cause of correlated failures:

  - All models see identical information → make identical errors together.
  - The inter-model std (disagreement) stays low even when ALL models are wrong.
  - "5-model consensus" was really one opinion repeated five times.

FIX: Each model now receives a DIFFERENT orthogonal feature view, ensuring that
the disagreement signal reflects genuine epistemic disagreement:

  View A → LR      : Raw sparse TF-IDF (full vocabulary). Linear boundary
                     in high-dim lexical space. In-distribution lexical expert.
  View B → LightGBM: 100 SVD LSA components (dense). Fast, memory-efficient.
                     Captures latent topic structure. Best at distribution shift.
  View C → RF      : 19 dense stylometric + readability features. Captures
                     writing-style fingerprint. Domain-agnostic (no vocabulary).
  View D → XGB     : 384-d SBERT embedding. Semantic expert. Likely to fire
                     differently on OOD content (Indian news vs. US political).

When models now disagree, it means genuinely different feature spaces are
sending contradictory signals — which is the correct interpretation of uncertainty.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from scipy.sparse import issparse, csr_matrix
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from config import LGBM_PARAMS, RF_PARAMS, XGB_PARAMS, RANDOM_SEED
from src.utils import setup_logger

logger = setup_logger("base_models")


class HeterogeneousViewBundle:
    """
    Slices a combined feature matrix into orthogonal per-model views.
    Each model receives a structurally different portion of the features,
    preventing correlated failures and making disagreement meaningful.

    Expected layout of the combined feature matrix (produced by UnifiedFeaturePipeline):
        [0 : lexical_end)           → Stream A: TF-IDF lexical sparse (10,005 dims)
        [lexical_end : sem_end)     → Stream B: SBERT 384-d semantic embeddings
        [sem_end : stylo_end)       → Stream C: 15 stylometric features
        [stylo_end : total)         → Stream D: 4 readability features

    Derived views (used at inference time with .transform()):
        LR    view → Streams A (sparse TF-IDF only)
        LGBM  view → SVD-compressed full matrix (100 components, as before)
        RF    view → Streams C + D (dense 19-d stylometric + readability)
        XGB   view → Stream B (SBERT 384-d dense)
    """

    def __init__(self, stream_boundaries: Dict[str, Tuple[int, int]], svd_components: int = 100):
        self.boundaries = stream_boundaries
        self.svd_components = svd_components
        self._svd = None

    def fit_svd(self, X_sparse_full) -> "HeterogeneousViewBundle":
        """Fit SVD on full training matrix (for LightGBM view only)."""
        from sklearn.decomposition import TruncatedSVD
        n_components = min(self.svd_components, X_sparse_full.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        self._svd.fit(X_sparse_full)
        logger.info(f"HeterogeneousViewBundle: SVD fitted ({n_components} components) for LightGBM view.")
        return self

    def _dense(self, X, cols):
        """Extracts a column slice and returns as dense ndarray."""
        start, end = cols
        if issparse(X):
            return X[:, start:end].toarray().astype(np.float32)
        return np.asarray(X[:, start:end], dtype=np.float32)

    def view_lr(self, X_sparse_full) -> csr_matrix:
        """View A: Raw sparse TF-IDF for Logistic Regression."""
        s, e = self.boundaries.get("lexical", (0, X_sparse_full.shape[1]))
        if issparse(X_sparse_full):
            return X_sparse_full[:, s:e]
        return csr_matrix(X_sparse_full[:, s:e].astype(np.float32))

    def view_lgbm(self, X_sparse_full) -> np.ndarray:
        """View B: SVD-compressed full matrix for LightGBM."""
        if self._svd is None:
            raise RuntimeError("Call fit_svd() before view_lgbm().")
        return self._svd.transform(X_sparse_full).astype(np.float32)

    def view_rf(self, X_sparse_full) -> np.ndarray:
        """View C: Stylometric + Readability features for Random Forest."""
        parts = []
        if "stylometric" in self.boundaries:
            parts.append(self._dense(X_sparse_full, self.boundaries["stylometric"]))
        if "readability" in self.boundaries:
            parts.append(self._dense(X_sparse_full, self.boundaries["readability"]))
        if not parts:
            # Fallback: use all dense features if boundaries are unknown
            logger.warning("Stream boundaries for RF not found; using full dense fallback.")
            return (X_sparse_full.toarray() if issparse(X_sparse_full)
                    else np.asarray(X_sparse_full, dtype=np.float32))
        return np.hstack(parts)

    def view_xgb(self, X_sparse_full) -> np.ndarray:
        """View D: SBERT 384-d embeddings for XGBoost."""
        if "semantic" in self.boundaries:
            return self._dense(X_sparse_full, self.boundaries["semantic"])
        # Fallback: SVD if semantic stream not available
        logger.warning("Semantic stream not found for XGB view; falling back to SVD view.")
        return self.view_lgbm(X_sparse_full)


class BaseEnsemble:
    """
    Heterogeneous base learner ensemble using four orthogonal feature views.

    Each model is an expert on a different representation of the article:
      - LR    : vocabulary specialist (sparse TF-IDF)
      - LightGBM: topic specialist (latent semantic space via SVD)
      - RF    : style specialist (writing fingerprint features)
      - XGB   : semantic specialist (sentence-level SBERT embeddings)

    This design guarantees that inter-model disagreement reflects real uncertainty
    rather than correlated failures from a shared compressed representation.
    """

    def __init__(self, stream_boundaries: Optional[Dict[str, Tuple[int, int]]] = None,
                 svd_components: int = 100):
        self.stream_boundaries = stream_boundaries or {}
        self.svd_components = svd_components

        # Model definitions (each gets different hyper-params & feature view)
        lgbm_p = dict(LGBM_PARAMS)
        self.models = {
            "lr":   LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED,
                                       solver="saga", n_jobs=-1),
            "lgbm": LGBMClassifier(**lgbm_p),
            "rf":   RandomForestClassifier(**RF_PARAMS),
            "xgb":  XGBClassifier(**XGB_PARAMS),
        }

        self.view_bundle: Optional[HeterogeneousViewBundle] = None
        self.is_fitted = False

    def _make_bundle(self, X_sparse_full) -> HeterogeneousViewBundle:
        bundle = HeterogeneousViewBundle(self.stream_boundaries, self.svd_components)
        bundle.fit_svd(X_sparse_full)
        return bundle

    def __getattr__(self, name: str):
        """
        Backward-compatibility shim: old saved BaseEnsemble objects (trained before the
        heterogeneous-view redesign) are missing 'view_bundle' and '_stream_boundaries'.
        Return None for those attributes instead of raising AttributeError, so the
        predict.py compatibility check can detect the old architecture and fall back.
        """
        if name in ("view_bundle", "stream_boundaries", "svd_components"):
            return None
        raise AttributeError(f"'BaseEnsemble' object has no attribute '{name}'")

    def _get_views(self, X) -> Dict[str, np.ndarray]:
        """Returns per-model feature views from the combined sparse matrix."""
        assert self.view_bundle is not None, "view_bundle not initialised"
        return {
            "lr":   self.view_bundle.view_lr(X),
            "lgbm": self.view_bundle.view_lgbm(X),
            "rf":   self.view_bundle.view_rf(X),
            "xgb":  self.view_bundle.view_xgb(X),
        }

    def fit(self, X_train_sparse, y_train,
            stream_boundaries: Optional[Dict[str, Tuple[int, int]]] = None) -> "BaseEnsemble":
        """
        Fits all four base models on their respective orthogonal feature views.
        X_train_sparse: combined sparse feature matrix from UnifiedFeaturePipeline.
        """
        if stream_boundaries is not None:
            self.stream_boundaries = stream_boundaries

        logger.info(
            f"Training Heterogeneous BaseEnsemble on {X_train_sparse.shape[0]} samples "
            f"(full matrix shape: {X_train_sparse.shape}) with 4 orthogonal views..."
        )

        self.view_bundle = self._make_bundle(X_train_sparse)
        views = self._get_views(X_train_sparse)

        for name, model in self.models.items():
            view = views[name]
            logger.info(f"  Fitting {name.upper()} on view shape {view.shape if hasattr(view, 'shape') else '?'}...")
            model.fit(view, y_train)

        self.is_fitted = True
        logger.info("Heterogeneous BaseEnsemble fitted successfully.")
        return self

    def predict_proba_dict(self, X) -> Dict[str, np.ndarray]:
        """
        Returns dictionary of P(FAKE) probabilities from each model's view:
        {'lr': p_lr, 'lgbm': p_lgbm, 'rf': p_rf, 'xgb': p_xgb}

        For OLD pre-redesign models (view_bundle is None), falls back to the
        original behaviour: all models predict on the same X (SVD-dense matrix).
        """
        if not self.is_fitted:
            raise RuntimeError("BaseEnsemble must be fitted before predicting.")

        # --- LEGACY FALLBACK for old joblib artifacts ---
        if self.view_bundle is None:
            probas = {}
            for name, model in self.models.items():
                p = model.predict_proba(X)[:, 1]
                probas[name] = p.astype(np.float32)
            return probas

        # --- NEW: orthogonal view routing ---
        views = self._get_views(X)
        probas = {}
        for name, model in self.models.items():
            p = model.predict_proba(views[name])[:, 1]
            probas[name] = p.astype(np.float32)
        return probas

    def predict_base_features(self, X) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Computes per-model probabilities and genuine inter-view disagreement.

        For OLD models (view_bundle is None), all models share the same X input
        (legacy SVD-dense path). Disagreement is computed the same way; it will
        just be lower because all models see identical features.

        Returns:
            stacked_features: shape (N, 5)
            probas_dict:      dict of individual P(FAKE) per model
            disagreement:     shape (N,) = std across model outputs
        """
        probas = self.predict_proba_dict(X)

        # Collect in consistent order regardless of dict ordering
        keys = ["lr", "lgbm", "rf", "xgb"]
        # Fall back gracefully if old model only has rf/xgb/lgbm/lr
        available = [k for k in keys if k in probas]
        # Also pick up any keys the old model might have used
        for k in probas:
            if k not in available:
                available.append(k)

        p_arrays = [probas[k] for k in available]
        probs_matrix = np.column_stack(p_arrays)
        disagreement = np.std(probs_matrix, axis=1, ddof=0).astype(np.float32)

        # Build stacked features in canonical order [p_lr, p_lgbm, p_rf, p_xgb, dis]
        p_lr   = probas.get("lr",   probs_matrix[:, 0])
        p_lgbm = probas.get("lgbm", probs_matrix[:, 0])
        p_rf   = probas.get("rf",   probs_matrix[:, 0])
        p_xgb  = probas.get("xgb",  probs_matrix[:, 0])

        stacked_features = np.column_stack([p_lr, p_lgbm, p_rf, p_xgb, disagreement])
        return stacked_features, probas, disagreement
