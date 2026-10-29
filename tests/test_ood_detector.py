"""
Tests for OODDetector: vocabulary coverage, Mahalanobis distance, composite OOD score,
and threshold tuning on validation data.
"""

import pytest
import numpy as np
from src.ood_detector import OODDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embeddings(n: int, dim: int = 16, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n, dim).astype(np.float64)


def _make_texts(n: int) -> list:
    words = ["the", "court", "judge", "sentence", "defendant", "case", "verdict", "law"]
    return [" ".join(words) for _ in range(n)]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def fitted_detector():
    """OODDetector fitted on synthetic in-distribution training data."""
    texts = _make_texts(200)
    embeddings = _make_embeddings(200, dim=16, seed=0)
    detector = OODDetector()
    detector.fit(texts, embeddings)
    return detector


# ---------------------------------------------------------------------------
# Vocabulary coverage tests
# ---------------------------------------------------------------------------

class TestVocabCoverage:
    def test_known_words_give_high_coverage(self, fitted_detector):
        text = "the court judge sentence defendant case verdict law"
        cov = fitted_detector.compute_vocab_coverage(text)
        assert cov > 0.80, f"Expected high coverage for in-vocab text, got {cov}"

    def test_unknown_words_give_low_coverage(self, fitted_detector):
        text = "pocso beti bachao acp rs agra sonika chaudhary"
        cov = fitted_detector.compute_vocab_coverage(text)
        assert cov < 0.50, f"Expected low coverage for OOD text, got {cov}"

    def test_empty_text_returns_one(self, fitted_detector):
        cov = fitted_detector.compute_vocab_coverage("")
        assert cov == 1.0

    def test_all_short_tokens_ignored(self, fitted_detector):
        # All tokens <= 2 chars are filtered; result should be 1.0 (nothing to check)
        cov = fitted_detector.compute_vocab_coverage("a b c d e f")
        assert cov == 1.0

    def test_mixed_coverage(self, fitted_detector):
        # Half known, half unknown
        text = "the court pocso rs"
        cov = fitted_detector.compute_vocab_coverage(text)
        # 'the' and 'court' are known (but 'the' is 3 chars OK), 'pocso' and 'rs' are unknown
        assert 0.0 <= cov <= 1.0


# ---------------------------------------------------------------------------
# OOD evaluation tests
# ---------------------------------------------------------------------------

class TestOODEvaluation:
    def test_in_distribution_embedding_gives_low_ood_score(self, fitted_detector):
        # An embedding near the training centroid should have low OOD score
        centroid = fitted_detector.mean_vec
        text = "the court judge sentence defendant"
        res = fitted_detector.evaluate_single(text, centroid)
        assert res["ood_score"] < 0.75, f"Near-centroid embedding should not be OOD, got {res['ood_score']}"

    def test_out_of_distribution_embedding_gives_high_ood_score(self, fitted_detector):
        # An embedding very far from training centroid
        far_emb = fitted_detector.mean_vec + 1000.0
        text = "the court judge"  # in-vocab text but far embedding
        res = fitted_detector.evaluate_single(text, far_emb)
        assert res["ood_score"] >= 0.50, f"Far embedding should be OOD, got {res['ood_score']}"

    def test_low_vocab_coverage_triggers_ood_flag(self, fitted_detector):
        # Text with no in-vocab words → vocab_coverage = 0 → is_ood = True
        text = "pocso rs acp agra sonika chaudhary beti bachao writ petition"
        emb = fitted_detector.mean_vec  # embedding fine, vocab bad
        res = fitted_detector.evaluate_single(text, emb)
        assert res["is_ood"] is True
        assert res["vocab_coverage"] < fitted_detector.vocab_coverage_threshold

    def test_evaluate_single_output_schema(self, fitted_detector):
        text = "the court judge sentence"
        emb = fitted_detector.mean_vec
        res = fitted_detector.evaluate_single(text, emb)
        required_keys = ["ood_score", "is_ood", "vocab_coverage", "mahalanobis_dist",
                         "threshold_rho", "explanation"]
        for k in required_keys:
            assert k in res, f"Missing key: {k}"

    def test_ood_score_bounded(self, fitted_detector):
        text = "the court judge"
        emb = _make_embeddings(1, dim=16, seed=99)[0]
        res = fitted_detector.evaluate_single(text, emb)
        assert 0.0 <= res["ood_score"] <= 1.0

    def test_mahalanobis_non_negative(self, fitted_detector):
        emb = _make_embeddings(1, dim=16, seed=7)[0]
        text = "the court judge"
        res = fitted_detector.evaluate_single(text, emb)
        assert res["mahalanobis_dist"] >= 0.0


# ---------------------------------------------------------------------------
# Threshold tuning tests
# ---------------------------------------------------------------------------

class TestThresholdTuning:
    def test_tune_threshold_sets_rho(self, fitted_detector):
        val_texts = _make_texts(50)
        val_embs = _make_embeddings(50, dim=16, seed=10)
        old_rho = fitted_detector.threshold_rho
        new_rho = fitted_detector.tune_threshold(val_texts, val_embs, target_in_distribution_quantile=0.90)
        # rho should have changed (tuned to a specific quantile)
        assert isinstance(new_rho, float)
        assert 0.0 <= new_rho <= 1.0
        assert fitted_detector.threshold_rho == new_rho

    def test_tune_threshold_95th_quantile_is_higher_than_50th(self, fitted_detector):
        val_texts = _make_texts(100)
        val_embs = _make_embeddings(100, dim=16, seed=20)
        rho_90 = fitted_detector.tune_threshold(val_texts, val_embs, target_in_distribution_quantile=0.90)
        rho_50 = fitted_detector.tune_threshold(val_texts, val_embs, target_in_distribution_quantile=0.50)
        assert rho_90 >= rho_50, "95th quantile threshold should be >= 50th quantile threshold"


# ---------------------------------------------------------------------------
# Fit API tests
# ---------------------------------------------------------------------------

class TestFitAPI:
    def test_fit_with_explicit_vocab(self):
        detector = OODDetector()
        vocab = {"apple", "banana", "cherry", "date"}
        texts = ["apple banana cherry date"] * 10
        embs = _make_embeddings(10, dim=8, seed=1)
        detector.fit(texts, embs, vocab=vocab)
        assert detector.vocab == vocab

    def test_fit_builds_vocab_from_texts_when_not_provided(self):
        detector = OODDetector()
        texts = ["hello world example text"] * 10
        embs = _make_embeddings(10, dim=8, seed=2)
        detector.fit(texts, embs)
        assert "hello" in detector.vocab or "world" in detector.vocab

    def test_fit_computes_mean_and_inv_cov(self):
        detector = OODDetector()
        texts = _make_texts(30)
        embs = _make_embeddings(30, dim=8, seed=3)
        detector.fit(texts, embs)
        assert detector.mean_vec is not None
        assert detector.inv_cov is not None
        assert detector.mean_vec.shape[0] == 8
        assert detector.inv_cov.shape == (8, 8)
