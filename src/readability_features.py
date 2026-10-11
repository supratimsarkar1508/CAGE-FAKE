"""
CAGE-Fake: Stream D - Readability Feature Extraction Module
Extracts 4 standard cognitive readability and complexity metrics:
1. Flesch Reading Ease
2. Flesch-Kincaid Grade Level
3. Gunning Fog Index
4. SMOG Index
StandardScaler fitted strictly on TRAIN split.
"""

from typing import List
import math
import re
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils import setup_logger

logger = setup_logger("readability_features")

WORD_REGEX = re.compile(r"\b[a-zA-Z]+\b")
SENTENCE_REGEX = re.compile(r"(?<=[.!?])\s+")

def count_syllables_word(word: str) -> int:
    """Estimates syllable count using vowel group heuristics."""
    w = word.lower().strip()
    if not w:
        return 0
    if len(w) <= 3:
        return 1

    # Remove silent trailing 'e'
    if w.endswith("e") and not w.endswith("le"):
        w = w[:-1]

    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    return max(1, count)

class ReadabilityFeatureExtractor:
    """
    Extracts Stream D Readability Features:
    1. flesch_reading_ease
    2. flesch_kincaid_grade
    3. gunning_fog
    4. smog_index
    """

    FEATURE_NAMES = [
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "gunning_fog",
        "smog_index",
    ]

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def compute_single(self, text: str) -> np.ndarray:
        """Computes unscaled 4-d readability vector for single article."""
        raw_text = str(text or "").strip()
        if not raw_text:
            return np.zeros(4, dtype=np.float32)

        # Sentences & Words
        sentences = [s.strip() for s in SENTENCE_REGEX.split(raw_text) if s.strip()]
        num_sentences = max(1, len(sentences))

        words = WORD_REGEX.findall(raw_text)
        num_words = max(1, len(words))

        syllable_counts = [count_syllables_word(w) for w in words]
        total_syllables = sum(syllable_counts)
        complex_words = sum(1 for s in syllable_counts if s >= 3)

        # 1. Flesch Reading Ease
        words_per_sentence = num_words / num_sentences
        syllables_per_word = total_syllables / num_words
        flesch_ease = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
        # Clip reasonable bounds to prevent outlier explosion
        flesch_ease = float(np.clip(flesch_ease, -100.0, 150.0))

        # 2. Flesch-Kincaid Grade Level
        fk_grade = (0.39 * words_per_sentence) + (11.8 * syllables_per_word) - 15.59
        fk_grade = float(np.clip(fk_grade, 0.0, 30.0))

        # 3. Gunning Fog Index
        percent_complex = (complex_words / num_words) * 100.0
        gunning_fog = 0.4 * (words_per_sentence + percent_complex)
        gunning_fog = float(np.clip(gunning_fog, 0.0, 35.0))

        # 4. SMOG Index
        smog = 1.0430 * math.sqrt(30.0 * (complex_words / num_sentences)) + 3.1291
        smog = float(np.clip(smog, 0.0, 35.0))

        return np.array([flesch_ease, fk_grade, gunning_fog, smog], dtype=np.float32)

    def compute_batch(self, texts: List[str]) -> np.ndarray:
        """Batch extraction of unscaled 4-d readability vectors."""
        matrix = [self.compute_single(t) for t in texts]
        return np.array(matrix, dtype=np.float32)

    def fit(self, train_texts: List[str]) -> "ReadabilityFeatureExtractor":
        """Fits StandardScaler strictly on TRAIN data."""
        logger.info(f"Computing and fitting Readability Stream D (4 features) on {len(train_texts)} training samples...")
        raw_feats = self.compute_batch(train_texts)
        self.scaler.fit(raw_feats)
        self.is_fitted = True
        logger.info("Readability Stream D fitted successfully.")
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transforms and scales readability features."""
        if not self.is_fitted:
            raise RuntimeError("ReadabilityFeatureExtractor must be fitted before calling transform.")
        raw_feats = self.compute_batch(texts)
        return self.scaler.transform(raw_feats)

    def fit_transform(self, train_texts: List[str]) -> np.ndarray:
        return self.fit(train_texts).transform(train_texts)

    def get_feature_names(self) -> List[str]:
        return list(self.FEATURE_NAMES)
