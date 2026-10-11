"""
CAGE-Fake: Stream C - Stylometric Feature Extraction Module
15 Stylometric markers capturing linguistic fingerprinting, structural variance,
and three novel epistemic markers: hedge-word ratio, tense-consistency ratio, and attribution-verb density.
StandardScaler fitted strictly on TRAIN split.
"""

from typing import List, Dict, Set
import re
import string
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils import setup_logger

logger = setup_logger("stylometric_features")

# Common English stopwords (self-contained list for zero external download dependency)
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
    "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

# Novel Marker 1: Epistemic Hedge Words (speculation, unverified claims, rumour)
HEDGE_WORDS: Set[str] = {
    "alleged", "allegedly", "apparently", "reportedly", "purportedly", "supposedly",
    "rumor", "rumored", "rumours", "claims", "claimed", "claiming", "suggests", "suggested",
    "suggesting", "could", "might", "maybe", "perhaps", "possibly", "possible", "potential",
    "seem", "seems", "seemed", "seeming", "appear", "appears", "appeared", "appearing",
    "speculate", "speculated", "speculation", "assumed", "presumably", "unconfirmed",
    "unverified", "likely", "unlikely", "suspect", "suspected", "believed"
}

# Novel Marker 2: Attribution Verbs (rigorous source citation vs unanchored assertions)
ATTRIBUTION_VERBS: Set[str] = {
    "said", "stated", "confirmed", "reported", "announced", "declared", "explained",
    "noted", "remarked", "testified", "admitted", "acknowledged", "cited", "asserted",
    "concluded", "emphasized", "insisted", "indicated", "revealed", "told", "warned"
}

# Verb markers for Tense Consistency (past vs present indicators)
PAST_VERB_MARKERS: Set[str] = {
    "was", "were", "had", "did", "went", "saw", "said", "took", "made", "came",
    "told", "found", "gave", "thought", "felt", "became", "left", "brought", "began"
}
PRESENT_VERB_MARKERS: Set[str] = {
    "is", "are", "am", "has", "have", "does", "do", "goes", "sees", "says",
    "takes", "makes", "comes", "tells", "finds", "gives", "thinks", "feels", "becomes"
}

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")
WORD_TOKEN_REGEX = re.compile(r"\b[a-zA-Z0-9_\'-]+\b")
PUNCTUATION_CHARS = set(string.punctuation)

class StylometricFeatureExtractor:
    """
    Extracts 15 stylometric features:
    1. avg_sentence_len
    2. sentence_len_variance
    3. avg_word_len
    4. ttr (Type-Token Ratio)
    5. hapax_ratio
    6. punctuation_density
    7. comma_density
    8. exclamation_density
    9. question_mark_density
    10. uppercase_ratio
    11. digit_ratio
    12. stopword_ratio
    13. hedge_word_ratio (novel)
    14. tense_consistency_ratio (novel)
    15. attribution_verb_density (novel)
    """

    FEATURE_NAMES = [
        "avg_sentence_len",
        "sentence_len_variance",
        "avg_word_len",
        "ttr",
        "hapax_ratio",
        "punctuation_density",
        "comma_density",
        "exclamation_density",
        "question_mark_density",
        "uppercase_ratio",
        "digit_ratio",
        "stopword_ratio",
        "hedge_word_ratio",
        "tense_consistency_ratio",
        "attribution_verb_density",
    ]

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_features_single(self, raw_text: str, clean_text: str) -> np.ndarray:
        """Computes raw unscaled 15-d vector for a single article."""
        raw = str(raw_text or "")
        clean = str(clean_text or "")

        total_chars = max(1, len(raw))
        alpha_chars = max(1, sum(1 for ch in raw if ch.isalpha()))

        # 1-2. Sentence statistics
        sentences = [s.strip() for s in SENTENCE_SPLIT_REGEX.split(raw) if s.strip()]
        sentence_count = max(1, len(sentences))
        sent_word_counts = [len(WORD_TOKEN_REGEX.findall(s)) for s in sentences]
        avg_sentence_len = float(np.mean(sent_word_counts)) if sent_word_counts else 0.0
        sentence_len_variance = float(np.var(sent_word_counts)) if len(sent_word_counts) > 1 else 0.0

        # Word tokens
        tokens = [w.lower() for w in WORD_TOKEN_REGEX.findall(clean)]
        total_words = max(1, len(tokens))

        # 3. Avg word length
        avg_word_len = float(sum(len(w) for w in tokens) / total_words)

        # 4. TTR (Type-Token Ratio)
        unique_words = len(set(tokens))
        ttr = float(unique_words / total_words)

        # 5. Hapax legomena ratio (words occurring exactly once)
        word_freqs: Dict[str, int] = {}
        for w in tokens:
            word_freqs[w] = word_freqs.get(w, 0) + 1
        hapax_count = sum(1 for c in word_freqs.values() if c == 1)
        hapax_ratio = float(hapax_count / total_words)

        # 6-9. Punctuation densities
        punct_count = sum(1 for ch in raw if ch in PUNCTUATION_CHARS)
        punctuation_density = float(punct_count / total_chars)
        comma_density = float(raw.count(",") / total_chars)
        exclamation_density = float(raw.count("!") / total_chars)
        question_mark_density = float(raw.count("?") / total_chars)

        # 10. Uppercase ratio
        upper_chars = sum(1 for ch in raw if ch.isupper())
        uppercase_ratio = float(upper_chars / alpha_chars)

        # 11. Digit ratio
        digit_count = sum(1 for ch in raw if ch.isdigit())
        digit_ratio = float(digit_count / total_chars)

        # 12. Stopword ratio
        stopword_count = sum(1 for w in tokens if w in STOPWORDS)
        stopword_ratio = float(stopword_count / total_words)

        # 13. Hedge-word ratio (novel)
        hedge_count = sum(1 for w in tokens if w in HEDGE_WORDS)
        hedge_word_ratio = float(hedge_count / total_words)

        # 14. Tense consistency ratio (novel)
        past_count = sum(1 for w in tokens if w in PAST_VERB_MARKERS or (w.endswith("ed") and len(w) > 4))
        present_count = sum(1 for w in tokens if w in PRESENT_VERB_MARKERS or (w.endswith("ing") and len(w) > 4))
        total_tense_verbs = past_count + present_count
        if total_tense_verbs > 0:
            tense_consistency_ratio = float(max(past_count, present_count) / total_tense_verbs)
        else:
            tense_consistency_ratio = 1.0  # neutral default

        # 15. Attribution-verb density (novel)
        attribution_count = sum(1 for w in tokens if w in ATTRIBUTION_VERBS)
        attribution_verb_density = float(attribution_count / total_words)

        features = [
            avg_sentence_len,
            sentence_len_variance,
            avg_word_len,
            ttr,
            hapax_ratio,
            punctuation_density,
            comma_density,
            exclamation_density,
            question_mark_density,
            uppercase_ratio,
            digit_ratio,
            stopword_ratio,
            hedge_word_ratio,
            tense_consistency_ratio,
            attribution_verb_density,
        ]

        return np.array(features, dtype=np.float32)

    def extract_features_batch(self, raw_texts: List[str], clean_texts: List[str]) -> np.ndarray:
        """Batch extraction of unscaled 15-d vectors."""
        matrix = [
            self.extract_features_single(r, c)
            for r, c in zip(raw_texts, clean_texts)
        ]
        return np.array(matrix, dtype=np.float32)

    def fit(self, train_raw_texts: List[str], train_clean_texts: List[str]) -> "StylometricFeatureExtractor":
        """Fits StandardScaler strictly on TRAIN data."""
        logger.info(f"Computing and fitting Stylometric Stream C (15 features) on {len(train_raw_texts)} training samples...")
        raw_feats = self.extract_features_batch(train_raw_texts, train_clean_texts)
        self.scaler.fit(raw_feats)
        self.is_fitted = True
        logger.info("Stylometric Stream C fitted successfully.")
        return self

    def transform(self, raw_texts: List[str], clean_texts: List[str]) -> np.ndarray:
        """Transforms and scales features."""
        if not self.is_fitted:
            raise RuntimeError("StylometricFeatureExtractor must be fitted before calling transform.")
        raw_feats = self.extract_features_batch(raw_texts, clean_texts)
        return self.scaler.transform(raw_feats)

    def fit_transform(self, train_raw_texts: List[str], train_clean_texts: List[str]) -> np.ndarray:
        return self.fit(train_raw_texts, train_clean_texts).transform(train_raw_texts, train_clean_texts)

    def get_feature_names(self) -> List[str]:
        return list(self.FEATURE_NAMES)
