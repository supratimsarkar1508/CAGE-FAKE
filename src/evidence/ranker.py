"""
CAGE-Fake: Hybrid BM25 + Dense Semantic Evidence Ranker
Splits retrieved documents into sentences, ranks by hybrid BM25 and SBERT cosine similarity,
and returns top-k most relevant evidence sentences for each claim.
"""

from typing import List, Dict, Any, Optional
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.utils import setup_logger

logger = setup_logger("ranker")


class HybridEvidenceRanker:
    """
    Ranks retrieved sentences using a linear fusion of:
    - Sparse BM25 lexical matching (token-level precision)
    - Dense SBERT embeddings (semantic conceptual relevance)
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._sbert: Optional[SentenceTransformer] = None

    @property
    def sbert(self) -> SentenceTransformer:
        if self._sbert is None:
            logger.info(f"Loading dense ranker embedding model: {self.model_name}")
            self._sbert = SentenceTransformer(self.model_name)
        return self._sbert

    def _split_into_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        # Robust sentence splitting on punctuation + whitespace
        raw_sents = re.split(r"(?<=[.!?])\s+", text)
        clean_sents = []
        for s in raw_sents:
            s_clean = s.strip()
            # Retain informative sentences between 6 and 60 words
            words = s_clean.split()
            if 6 <= len(words) <= 60:
                clean_sents.append(s_clean)
        return clean_sents

    def rank_sentences(
        self,
        claim: str,
        retrieved_docs: List[Dict[str, Any]],
        top_k: int = 3,
        bm25_weight: float = 0.40,
        dense_weight: float = 0.60,
    ) -> List[Dict[str, Any]]:
        """
        Extracts sentences across all retrieved documents, scores each against the claim,
        and returns the top-k highest scoring candidate evidence sentences.
        """
        all_candidates = []
        for doc in retrieved_docs:
            content = doc.get("content", "")
            sents = self._split_into_sentences(content)
            for s in sents:
                all_candidates.append({
                    "sentence": s,
                    "url": doc.get("url", ""),
                    "domain": doc.get("domain", ""),
                    "source_type": doc.get("source_type", ""),
                    "is_trusted": doc.get("is_trusted", False),
                    "factcheck_rating": doc.get("factcheck_rating"),
                })

        if not all_candidates:
            return []

        # Deduplicate candidates by text
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            key = c["sentence"].lower().strip()
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        candidate_texts = [c["sentence"] for c in unique_candidates]

        # 1. BM25 Scoring
        tokenized_corpus = [s.lower().split() for s in candidate_texts]
        tokenized_claim = claim.lower().split()

        try:
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = np.array(bm25.get_scores(tokenized_claim), dtype=np.float32)
            max_b = bm25_scores.max()
            if max_b > 0:
                bm25_norm = bm25_scores / max_b
            else:
                bm25_norm = np.zeros_like(bm25_scores)
        except Exception as e:
            logger.debug(f"BM25 ranking error: {e}")
            bm25_norm = np.zeros(len(unique_candidates), dtype=np.float32)

        # 2. Dense Semantic Cosine Similarity
        try:
            claim_emb = self.sbert.encode([claim], convert_to_numpy=True)
            sent_embs = self.sbert.encode(candidate_texts, convert_to_numpy=True)

            # Cosine similarity
            dot = np.dot(sent_embs, claim_emb.T).flatten()
            norm_c = np.linalg.norm(claim_emb)
            norm_s = np.linalg.norm(sent_embs, axis=1)
            dense_scores = dot / (np.maximum(norm_s * norm_c, 1e-8))
            dense_norm = np.clip(dense_scores, 0.0, 1.0)
        except Exception as e:
            logger.debug(f"Dense ranking error: {e}")
            dense_norm = np.zeros(len(unique_candidates), dtype=np.float32)

        # 3. Hybrid Fusion Score
        hybrid_scores = bm25_weight * bm25_norm + dense_weight * dense_norm

        # Boost trusted sources slightly
        for idx, c in enumerate(unique_candidates):
            if c["is_trusted"]:
                hybrid_scores[idx] += 0.05

        ranked_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            item = dict(unique_candidates[idx])
            item["hybrid_score"] = round(float(hybrid_scores[idx]), 3)
            item["bm25_score"] = round(float(bm25_norm[idx]), 3)
            item["dense_score"] = round(float(dense_norm[idx]), 3)
            results.append(item)

        return results
