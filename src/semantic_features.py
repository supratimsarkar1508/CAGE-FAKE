"""
CAGE-Fake: Stream B - Semantic Feature Extraction Module
Dense 384-dimensional semantic embeddings using Sentence-Transformers (all-MiniLM-L6-v2).
Includes robust local disk caching to guarantee reproducible, instant reloads.
"""

from typing import List, Optional
from pathlib import Path
import hashlib
import numpy as np
import joblib

from config import (
    SBERT_MODEL_NAME,
    SBERT_DIM,
    SBERT_BATCH_SIZE,
    EMBEDDINGS_CACHE_FILE,
)
from src.utils import setup_logger

logger = setup_logger("semantic_features")

class SemanticFeatureExtractor:
    """
    Extracts Stream B Semantic Features:
    - Sentence-Transformers 384-d embeddings
    - Persistent caching mechanism keyed by text hash / split ID
    """

    def __init__(
        self,
        model_name: str = SBERT_MODEL_NAME,
        dim: int = SBERT_DIM,
        batch_size: int = SBERT_BATCH_SIZE,
        cache_path: Path = EMBEDDINGS_CACHE_FILE,
    ):
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.model = None
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Loads disk cache if it exists (.npy or .joblib)."""
        npy_path = self.cache_path.with_suffix(".npy")
        if npy_path.exists():
            try:
                data = np.load(npy_path, allow_pickle=True).item()
                if isinstance(data, dict):
                    return data
            except Exception as e:
                logger.warning(f"Failed to load npy cache from {npy_path}: {e}")

        if self.cache_path.exists():
            try:
                return joblib.load(self.cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache from {self.cache_path}: {e}. Initializing fresh cache.")
        return {}

    def _save_cache(self) -> None:
        """Persists cache to disk in both .joblib and .npy formats."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._cache, self.cache_path)
            npy_path = self.cache_path.with_suffix(".npy")
            np.save(npy_path, self._cache, allow_pickle=True)
        except Exception as e:
            logger.warning(f"Could not persist embeddings cache: {e}")

    def _init_model(self) -> None:
        """Initializes SentenceTransformer model on CPU/GPU."""
        if self.model is None:
            logger.info(f"Loading Sentence-Transformer model: '{self.model_name}'...")
            try:
                from sentence_transformers import SentenceTransformer
                try:
                    self.model = SentenceTransformer(self.model_name, local_files_only=True)
                except Exception:
                    self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error(f"Error loading SentenceTransformer: {e}")
                raise

    @staticmethod
    def _hash_text(text: str) -> str:
        """Computes SHA-256 hash of text for caching."""
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    def encode(self, texts: List[str], show_progress_bar: bool = True) -> np.ndarray:
        """
        Encodes list of texts into dense (N, 384) matrix.
        Retrieves already encoded samples from cache to minimize computation.
        """
        n_samples = len(texts)
        embeddings = np.zeros((n_samples, self.dim), dtype=np.float32)
        uncached_indices = []
        uncached_texts = []

        for i, t in enumerate(texts):
            h = self._hash_text(t)
            if h in self._cache:
                embeddings[i] = self._cache[h]
            else:
                uncached_indices.append(i)
                uncached_texts.append(t)

        if uncached_texts:
            logger.info(f"Encoding {len(uncached_texts)} uncached samples (found {n_samples - len(uncached_texts)} in cache)...")
            self._init_model()
            new_embeddings = self.model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            for idx, emb, text in zip(uncached_indices, new_embeddings, uncached_texts):
                h = self._hash_text(text)
                self._cache[h] = emb
                embeddings[idx] = emb

            self._save_cache()
            logger.info(f"Successfully cached {len(uncached_texts)} new embeddings.")
        else:
            logger.info(f"Retrieved all {n_samples} embeddings directly from cache.")

        return embeddings

    def get_feature_names(self) -> List[str]:
        return [f"sbert_dim_{i}" for i in range(self.dim)]
