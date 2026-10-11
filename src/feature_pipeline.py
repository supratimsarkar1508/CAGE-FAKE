"""
CAGE-Fake: Unified Feature Pipeline Module
Integrates Stream A (Lexical), Stream B (Semantic), Stream C (Stylometric), and Stream D (Readability).
Strictly preserves stream boundaries for modular explainability and SHAP attribution.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from scipy.sparse import issparse, csr_matrix, hstack

from src.lexical_features import LexicalFeatureExtractor
from src.semantic_features import SemanticFeatureExtractor
from src.stylometric_features import StylometricFeatureExtractor
from src.readability_features import ReadabilityFeatureExtractor
from src.utils import setup_logger

logger = setup_logger("feature_pipeline")

class UnifiedFeaturePipeline:
    """
    Orchestrates all 4 feature streams:
    - Stream A: Lexical (10,000 TF-IDF + 5 auxiliary)
    - Stream B: Semantic (384 SBERT embeddings)
    - Stream C: Stylometric (15 markers)
    - Stream D: Readability (4 metrics)
    Combined total ≈ 10,403 dimensions.
    """

    def __init__(self, include_semantic: bool = True):
        self.include_semantic = include_semantic
        self.lexical_extractor = LexicalFeatureExtractor()
        self.semantic_extractor = SemanticFeatureExtractor() if include_semantic else None
        self.stylometric_extractor = StylometricFeatureExtractor()
        self.readability_extractor = ReadabilityFeatureExtractor()

        self.stream_boundaries: Dict[str, Tuple[int, int]] = {}
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, train_clean_texts: List[str], train_raw_texts: List[str]) -> "UnifiedFeaturePipeline":
        """Fits all stream extractors strictly on TRAIN data."""
        logger.info("Fitting Unified Feature Pipeline...")

        # 1. Fit Lexical
        self.lexical_extractor.fit(train_clean_texts, train_raw_texts)
        lexical_names = self.lexical_extractor.get_feature_names()

        # 2. Fit Stylometric
        self.stylometric_extractor.fit(train_raw_texts, train_clean_texts)
        stylo_names = self.stylometric_extractor.get_feature_names()

        # 3. Fit Readability
        self.readability_extractor.fit(train_raw_texts)
        read_names = self.readability_extractor.get_feature_names()

        # Track Stream Names & Boundaries
        self.feature_names = list(lexical_names)
        start_idx = 0
        self.stream_boundaries["lexical"] = (start_idx, start_idx + len(lexical_names))
        start_idx += len(lexical_names)

        if self.include_semantic and self.semantic_extractor is not None:
            sem_names = self.semantic_extractor.get_feature_names()
            self.feature_names.extend(sem_names)
            self.stream_boundaries["semantic"] = (start_idx, start_idx + len(sem_names))
            start_idx += len(sem_names)

        self.feature_names.extend(stylo_names)
        self.stream_boundaries["stylometric"] = (start_idx, start_idx + len(stylo_names))
        start_idx += len(stylo_names)

        self.feature_names.extend(read_names)
        self.stream_boundaries["readability"] = (start_idx, start_idx + len(read_names))
        start_idx += len(read_names)

        self.is_fitted = True
        logger.info(f"Unified Feature Pipeline fitted. Total Dimensions: {len(self.feature_names)}")
        for stream, (s, e) in self.stream_boundaries.items():
            logger.info(f"  Stream '{stream}': dims [{s} : {e}] (total {e - s})")

        return self

    def transform(
        self,
        clean_texts: List[str],
        raw_texts: List[str],
        return_sparse: bool = True,
    ) -> csr_matrix | np.ndarray:
        """Transforms clean and raw texts into combined feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("UnifiedFeaturePipeline must be fitted before calling transform.")

        # Stream A
        feat_a = self.lexical_extractor.transform(clean_texts, raw_texts)  # csr_matrix

        # Stream B
        if self.include_semantic and self.semantic_extractor is not None:
            feat_b_dense = self.semantic_extractor.encode(clean_texts)
            feat_b = csr_matrix(feat_b_dense)
        else:
            feat_b = None

        # Stream C
        feat_c_dense = self.stylometric_extractor.transform(raw_texts, clean_texts)
        feat_c = csr_matrix(feat_c_dense)

        # Stream D
        feat_d_dense = self.readability_extractor.transform(raw_texts)
        feat_d = csr_matrix(feat_d_dense)

        # Combine
        parts = [feat_a]
        if feat_b is not None:
            parts.append(feat_b)
        parts.extend([feat_c, feat_d])

        combined = hstack(parts, format="csr")

        if not return_sparse:
            return combined.toarray()
        return combined

    def fit_transform(
        self,
        train_clean_texts: List[str],
        train_raw_texts: List[str],
        return_sparse: bool = True,
    ) -> csr_matrix | np.ndarray:
        return self.fit(train_clean_texts, train_raw_texts).transform(
            train_clean_texts, train_raw_texts, return_sparse=return_sparse
        )

    def get_stream_slices(self) -> Dict[str, slice]:
        """Returns slice objects for slicing dense or sparse matrix by feature stream."""
        return {
            stream: slice(start, end)
            for stream, (start, end) in self.stream_boundaries.items()
        }

    def get_feature_names(self) -> List[str]:
        return list(self.feature_names)
