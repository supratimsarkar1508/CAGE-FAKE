"""
CAGE-Fake: Stream A - Lexical Feature Extraction Module
TF-IDF uni+bigram (10,000 features, sublinear_tf=True) + auxiliary lexical diversity & density statistics.
Fitted strictly on TRAIN split to prevent data leakage.
"""

from typing import List, Tuple, Dict, Any
import string
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_SUBLINEAR_TF,
    TFIDF_MIN_DF,
)
from src.utils import setup_logger

logger = setup_logger("lexical_features")

PUNCTUATION_CHARS = set(string.punctuation)
WORD_TOKEN_REGEX = re.compile(r"\b[a-zA-Z0-9_\'-]+\b")

class LexicalFeatureExtractor:
    """
    Extracts Stream A Lexical Features:
    - 10,000 TF-IDF uni+bigrams (sublinear_tf=True)
    - 5 auxiliary lexical statistics:
        1. lexical_diversity (TTR)
        2. rare_word_ratio
        3. repeated_word_ratio
        4. uppercase_token_density
        5. punctuation_density
    """

    def __init__(
        self,
        max_features: int = TFIDF_MAX_FEATURES,
        ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
        sublinear_tf: bool = TFIDF_SUBLINEAR_TF,
        min_df: int = TFIDF_MIN_DF,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            min_df=self.min_df,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.auxiliary_feature_names = [
            "lex_diversity",
            "lex_rare_word_ratio",
            "lex_repeated_word_ratio",
            "lex_uppercase_density",
            "lex_punctuation_density",
        ]

    def _compute_auxiliary_stats(self, raw_texts: List[str], clean_texts: List[str]) -> np.ndarray:
        """Computes 5 dense auxiliary lexical metrics for each document."""
        stats = []
        for raw, clean in zip(raw_texts, clean_texts):
            raw = str(raw or "")
            clean = str(clean or "")

            raw_len = max(1, len(raw))
            punct_count = sum(1 for ch in raw if ch in PUNCTUATION_CHARS)
            punct_density = punct_count / raw_len

            raw_tokens = raw.split()
            total_raw_tokens = max(1, len(raw_tokens))
            upper_tokens = sum(1 for tok in raw_tokens if tok.isupper() and len(tok) > 1)
            upper_density = upper_tokens / total_raw_tokens

            tokens = [t.lower() for t in WORD_TOKEN_REGEX.findall(clean)]
            total_tokens = max(1, len(tokens))
            unique_tokens = len(set(tokens))
            lex_diversity = unique_tokens / total_tokens

            # Word frequencies for repeated & rare words
            token_counts: Dict[str, int] = {}
            for t in tokens:
                token_counts[t] = token_counts.get(t, 0) + 1

            # Repeated words: sum of top 3 most frequent tokens over total tokens
            sorted_counts = sorted(token_counts.values(), reverse=True)
            repeated_ratio = sum(sorted_counts[:3]) / total_tokens if sorted_counts else 0.0

            # Rare words: tokens occurring exactly once (hapax in doc) over total tokens
            rare_count = sum(1 for c in token_counts.values() if c == 1)
            rare_ratio = rare_count / total_tokens

            stats.append([
                lex_diversity,
                rare_ratio,
                repeated_ratio,
                upper_density,
                punct_density,
            ])

        return np.array(stats, dtype=np.float32)

    def fit(self, train_clean_texts: List[str], train_raw_texts: List[str]) -> "LexicalFeatureExtractor":
        """Fits TF-IDF and auxiliary scaler strictly on TRAIN data."""
        logger.info(f"Fitting TF-IDF Vectorizer with max_features={self.max_features} on {len(train_clean_texts)} training samples...")
        self.vectorizer.fit(train_clean_texts)

        aux_train = self._compute_auxiliary_stats(train_raw_texts, train_clean_texts)
        self.scaler.fit(aux_train)

        self.is_fitted = True
        logger.info(f"Lexical Stream A fitted: {len(self.vectorizer.get_feature_names_out())} TF-IDF features + {len(self.auxiliary_feature_names)} auxiliary.")
        return self

    def transform(self, clean_texts: List[str], raw_texts: List[str]) -> csr_matrix:
        """Transforms text into concatenated sparse feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("LexicalFeatureExtractor must be fitted before calling transform.")

        tfidf_sparse = self.vectorizer.transform(clean_texts)
        aux_dense = self._compute_auxiliary_stats(raw_texts, clean_texts)
        aux_scaled = self.scaler.transform(aux_dense)
        aux_sparse = csr_matrix(aux_scaled)

        combined = hstack([tfidf_sparse, aux_sparse], format="csr")
        return combined

    def fit_transform(self, train_clean_texts: List[str], train_raw_texts: List[str]) -> csr_matrix:
        """Fits on train and returns transformed matrix."""
        return self.fit(train_clean_texts, train_raw_texts).transform(train_clean_texts, train_raw_texts)

    def get_feature_names(self) -> List[str]:
        """Returns full list of feature names for Stream A."""
        tfidf_names = [f"tfidf_{w}" for w in self.vectorizer.get_feature_names_out()]
        return tfidf_names + self.auxiliary_feature_names
