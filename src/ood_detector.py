"""
CAGE-Fake: Out-Of-Distribution (OOD) Detector
Computes domain shift via TF-IDF vocabulary coverage and SBERT Mahalanobis distance.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

from src.utils import setup_logger, save_artifact, load_artifact

logger = setup_logger("ood_detector")


class OODDetector:
    """
    Detects domain shift / out-of-distribution text using two complementary signals:
    1. Lexical Vocabulary Coverage: Fraction of tokens present in the training TF-IDF vocabulary.
    2. Semantic Mahalanobis Distance: Distance of SBERT embedding from the training distribution.
    """

    def __init__(self, ridge_alpha: float = 1e-4):
        self.ridge_alpha = ridge_alpha
        self.vocab: set = set()
        self.mean_vec: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None
        self.threshold_rho: float = 0.50
        self.vocab_coverage_threshold: float = 0.60
        self.mean_dist: float = 1.0
        self.std_dist: float = 1.0

    def fit(self, texts: List[str], embeddings: np.ndarray, vocab: Optional[set] = None) -> "OODDetector":
        """
        Fits training distribution statistics:
        - TF-IDF vocabulary set
        - SBERT embedding centroid and regularized inverse covariance matrix
        """
        logger.info("Fitting OOD detector on training distribution...")

        # Vocabulary
        if vocab is not None:
            self.vocab = set(vocab)
        else:
            # Simple whitespace/word tokenizer fallback if vocab not directly provided
            all_words = set()
            for t in texts:
                for w in str(t).lower().split():
                    clean_w = "".join(c for c in w if c.isalnum())
                    if len(clean_w) > 2:
                        all_words.add(clean_w)
            self.vocab = all_words

        # Semantic embedding distribution
        emb_arr = np.array(embeddings, dtype=np.float64)
        if emb_arr.ndim == 1:
            emb_arr = emb_arr.reshape(1, -1)

        self.mean_vec = np.mean(emb_arr, axis=0)

        # Empirical covariance with Ledoit-Wolf or ridge regularization
        cov = np.cov(emb_arr, rowvar=False)
        d = cov.shape[0]
        # Add ridge along diagonal for numerical stability
        cov_reg = cov + np.eye(d) * self.ridge_alpha
        try:
            self.inv_cov = np.linalg.pinv(cov_reg)
        except Exception as e:
            logger.warning(f"Covariance inversion failed, using diagonal inverse: {e}")
            var = np.var(emb_arr, axis=0) + self.ridge_alpha
            self.inv_cov = np.diag(1.0 / var)

        # Compute training distances to establish normalization scale
        train_dists = [self._compute_mahalanobis(emb) for emb in emb_arr]
        self.mean_dist = float(np.mean(train_dists)) if len(train_dists) > 0 else 1.0
        self.std_dist = float(np.std(train_dists)) if len(train_dists) > 0 else 1.0
        if self.std_dist == 0:
            self.std_dist = 1.0

        logger.info(f"OOD detector fit complete. Vocab size: {len(self.vocab)}, Mean dist: {self.mean_dist:.2f}")
        return self

    def _compute_mahalanobis(self, emb: np.ndarray) -> float:
        """Computes regularized Mahalanobis distance from the training centroid."""
        diff = np.array(emb, dtype=np.float64).flatten() - self.mean_vec
        dist_sq = float(np.dot(np.dot(diff, self.inv_cov), diff.T))
        return float(np.sqrt(max(0.0, dist_sq)))

    def compute_vocab_coverage(self, text: str) -> float:
        """
        Computes the ratio of unique tokens in text that exist in training vocabulary.
        Returns value in [0.0, 1.0]. Lower values indicate domain shift.
        """
        if not self.vocab or not text:
            return 1.0
        tokens = [w for w in "".join(c if c.isalnum() else " " for c in text.lower()).split() if len(w) > 2]
        if not tokens:
            return 1.0
        in_vocab = sum(1 for w in tokens if w in self.vocab)
        return float(in_vocab / len(tokens))

    def evaluate_single(self, text: str, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Computes OOD signals for a single article instance:
        - vocab_coverage: fraction of words in training vocabulary
        - mahalanobis_dist: semantic distance from training cluster
        - ood_score: normalized composite metric in [0.0, 1.0]
        - is_ood: bool (True if ood_score >= threshold_rho)
        """
        coverage = self.compute_vocab_coverage(text)
        dist = self._compute_mahalanobis(embedding)

        # Normalized distance (z-score transformed to [0, 1] sigmoid)
        z_dist = (dist - self.mean_dist) / max(0.1, self.std_dist)
        dist_norm = float(1.0 / (1.0 + np.exp(-z_dist)))

        # Composite OOD score: high distance OR low vocabulary coverage elevates OOD
        vocab_ood_penalty = max(0.0, 1.0 - coverage)
        ood_score = float(np.clip(0.5 * dist_norm + 0.5 * vocab_ood_penalty, 0.0, 1.0))
        is_ood = bool(ood_score >= self.threshold_rho or coverage < self.vocab_coverage_threshold)

        reason = []
        if coverage < self.vocab_coverage_threshold:
            reason.append(f"Low vocabulary coverage ({coverage:.1%} < {self.vocab_coverage_threshold:.1%})")
        if dist_norm > 0.70:
            reason.append(f"High semantic embedding distance ({dist:.1f} vs mean {self.mean_dist:.1f})")

        return {
            "ood_score": round(ood_score, 4),
            "is_ood": is_ood,
            "vocab_coverage": round(coverage, 4),
            "mahalanobis_dist": round(dist, 2),
            "threshold_rho": round(self.threshold_rho, 4),
            "explanation": " | ".join(reason) if reason else "In-distribution relative to training corpus",
        }

    def tune_threshold(
        self,
        val_texts: List[str],
        val_embeddings: np.ndarray,
        target_in_distribution_quantile: float = 0.95,
    ) -> float:
        """
        Tunes the threshold rho on validation set to capture at least
        target_in_distribution_quantile (e.g. 95%) of in-distribution samples.
        """
        scores = []
        coverages = []
        for t, emb in zip(val_texts, val_embeddings):
            res = self.evaluate_single(t, emb)
            scores.append(res["ood_score"])
            coverages.append(res["vocab_coverage"])

        # Set rho at chosen quantile of in-distribution validation scores
        self.threshold_rho = float(np.quantile(scores, target_in_distribution_quantile))
        # Set coverage floor at 5th percentile
        self.vocab_coverage_threshold = max(0.40, float(np.quantile(coverages, 0.05)))

        logger.info(f"Tuned OOD threshold rho = {self.threshold_rho:.4f}, min coverage = {self.vocab_coverage_threshold:.2%}")
        return self.threshold_rho

    def plot_ood_histogram(
        self,
        val_scores: List[float],
        out_score_pdf: Path,
        out_score_png: Optional[Path] = None,
        test_scores: Optional[List[float]] = None,
    ) -> None:
        """Saves OOD score distribution plot as standalone PDF and PNG."""
        fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
        ax.hist(val_scores, bins=25, color="#3b82f6", alpha=0.7, edgecolor="black", label="Validation (In-Dist)")
        if test_scores:
            ax.hist(test_scores, bins=25, color="#ef4444", alpha=0.5, edgecolor="black", label="Test Evaluation")

        ax.axvline(self.threshold_rho, color="#dc2626", linestyle="--", linewidth=2.0, label=f"OOD Threshold rho = {self.threshold_rho:.3f}")
        ax.set_title("Out-Of-Distribution (OOD) Domain Shift Score Distribution", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Composite OOD Score (0 = In-Distribution, 1 = Out-Of-Distribution)", fontsize=9)
        ax.set_ylabel("Article Frequency", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)

        out_score_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_score_pdf), format="pdf", bbox_inches="tight")
        if out_score_png:
            fig.savefig(str(out_score_png), format="png", bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved OOD score distribution plot to {out_score_pdf}")
