"""
CAGE-Fake: Text Preprocessing and Dual-Representation Pipeline
Extracts raw_text (for stylometry) and clean_text (for lexical & semantic modeling),
computes structural linguistic properties (token count, sentence count, avg sentence len, TTR).
"""

from typing import Dict, List, Any, Tuple
import re
import html
import unicodedata

# Pre-compiled regex patterns for clean_text
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_REGEX = re.compile(r"<[^>]+>")
WHITESPACE_REGEX = re.compile(r"\s+")
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")
WORD_TOKEN_REGEX = re.compile(r"\b[a-zA-Z0-9_\'-]+\b")

class TextPreprocessor:
    """
    Dual-pipeline preprocessor producing:
    1. raw_text: Untouched text preserving uppercase, punctuation patterns for stylometry.
    2. clean_text: Normalized, entity-decoded, URL/HTML-stripped text for TF-IDF & Embeddings.
    3. linguistic_stats: Token/Sentence counts, avg sentence length, TTR.
    """

    def __init__(self, use_spacy: bool = True):
        self.nlp = None
        self._full_nlp = None
        if use_spacy:
            try:
                import spacy
                try:
                    # Lightweight sentence + token pipeline (sentencizer only) for high throughput
                    self.nlp = spacy.blank("en")
                    self.nlp.add_pipe("sentencizer")
                except Exception:
                    self.nlp = None
            except ImportError:
                self.nlp = None

    def _get_full_nlp(self):
        """Lazy loader for full tagger, ner, and lemmatizer pipeline when requested."""
        if self._full_nlp is None:
            try:
                import spacy
                self._full_nlp = spacy.load("en_core_web_sm", disable=["parser"])
                if "senter" not in self._full_nlp.pipe_names and "sentencizer" not in self._full_nlp.pipe_names:
                    self._full_nlp.add_pipe("sentencizer")
            except Exception:
                self._full_nlp = self.nlp
        return self._full_nlp

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans text: strips HTML, URLs, unescapes entities, normalizes unicode,
        and collapses excessive whitespace.
        """
        if not text or not isinstance(text, str):
            return ""

        # Unescape HTML entities (&amp; -> &, &quot; -> ", etc.)
        t = html.unescape(text)

        # Remove HTML tags
        t = HTML_TAG_REGEX.sub(" ", t)

        # Remove URLs
        t = URL_REGEX.sub(" ", t)

        # Normalize Unicode
        t = unicodedata.normalize("NFKD", t)

        # Strip control characters
        t = "".join(ch for ch in t if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t", " "))

        # Normalize whitespace
        t = WHITESPACE_REGEX.sub(" ", t).strip()

        return t

    def extract_linguistic_stats(self, raw_text: str, clean_text: str) -> Dict[str, float]:
        """
        Computes core structural linguistic metrics:
        - sentence_count
        - token_count
        - avg_sentence_length
        - type_token_ratio (TTR)
        """
        if not clean_text:
            return {
                "sentence_count": 0.0,
                "token_count": 0.0,
                "avg_sentence_length": 0.0,
                "ttr": 0.0,
            }

        # Token extraction
        tokens = [w.lower() for w in WORD_TOKEN_REGEX.findall(clean_text)]
        token_count = len(tokens)

        # Sentence extraction
        if self.nlp is not None and token_count < 10000:
            doc = self.nlp(clean_text[:50000])
            sentences = list(doc.sents)
            sentence_count = len(sentences) if sentences else 1
        else:
            sentences = [s for s in SENTENCE_SPLIT_REGEX.split(raw_text) if s.strip()]
            sentence_count = max(1, len(sentences))

        avg_sentence_len = float(token_count / max(1, sentence_count))
        unique_tokens = len(set(tokens))
        ttr = float(unique_tokens / max(1, token_count))

        return {
            "sentence_count": float(sentence_count),
            "token_count": float(token_count),
            "avg_sentence_length": round(avg_sentence_len, 4),
            "ttr": round(ttr, 4),
        }

    def analyze_spacy(self, text: str) -> Dict[str, Any]:
        """
        Extracts detailed spaCy linguistic annotations:
        sentences, tokens, POS tags, named entities, and lemmas.
        """
        full_nlp = self._get_full_nlp()
        if full_nlp is None:
            # Fallback
            words = WORD_TOKEN_REGEX.findall(text)
            return {
                "sentences": [s.strip() for s in SENTENCE_SPLIT_REGEX.split(text) if s.strip()],
                "tokens": words,
                "pos_tags": [],
                "entities": [],
                "lemmas": [w.lower() for w in words],
            }

        doc = full_nlp(text[:50000])
        return {
            "sentences": [s.text.strip() for s in doc.sents if s.text.strip()],
            "tokens": [t.text for t in doc],
            "pos_tags": [(t.text, t.pos_) for t in doc],
            "entities": [(ent.text, ent.label_) for ent in doc.ents],
            "lemmas": [t.lemma_ for t in doc if not t.is_space],
        }

    def process_article(self, headline: str, body: str) -> Dict[str, Any]:
        """
        Processes a single article into dual representations and linguistic statistics.
        """
        headline = str(headline or "").strip()
        body = str(body or "").strip()

        if headline and body:
            raw_text = f"{headline}\n\n{body}"
        elif headline:
            raw_text = headline
        else:
            raw_text = body

        clean = self.clean_text(raw_text)
        stats = self.extract_linguistic_stats(raw_text, clean)

        return {
            "raw_text": raw_text,
            "clean_text": clean,
            "headline": headline,
            "body": body,
            "stats": stats,
        }

    def batch_process(self, headlines: List[str], bodies: List[str]) -> Tuple[List[str], List[str], List[Dict[str, float]]]:
        """
        Batch processes headline + body pairs.
        Returns: (raw_texts, clean_texts, stats_list)
        """
        raw_texts = []
        clean_texts = []
        stats_list = []

        for h, b in zip(headlines, bodies):
            res = self.process_article(h, b)
            raw_texts.append(res["raw_text"])
            clean_texts.append(res["clean_text"])
            stats_list.append(res["stats"])

        return raw_texts, clean_texts, stats_list
