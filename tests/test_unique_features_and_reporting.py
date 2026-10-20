"""
Unit and Integration Tests for Unique Trustworthiness Features:
1. Model Disagreement Meter (separating 4-base gate disagreement and 5-model consensus spread)
2. News Credibility Fingerprint (with Bayesian length-smoothing)
3. Counterfactual Explanation (signed probability shift, exact changed fragments, and unmodified text handling)
4. 'Why this prediction?' Diagnostic Synthesis (strict trigger correspondence)
5. URL Extractor & Error Handling
6. PDF Audit Report Generation
"""

import pytest
from pathlib import Path
import numpy as np

from src.unique_features import (
    ModelDisagreementMeter,
    NewsCredibilityFingerprint,
    CounterfactualExplainer,
    synthesize_why_prediction,
)
from src.url_extractor import validate_url, extract_article_from_url, ArticleExtractionError
from src.pdf_generator import generate_pdf_report
from config import FIGURES_DIR, REPORTS_DIR


def test_model_disagreement_meter():
    probas = {"rf": 0.85, "xgb": 0.88, "lgbm": 0.90, "lr": 0.82}
    meter_info = ModelDisagreementMeter.compute(
        probas_dict=probas,
        p_calibrated_stack=0.87,
        base_disagreement_std=0.03,
        threshold_delta=0.15,
    )

    assert len(meter_info["models"]) == 5
    assert meter_info["fake_votes"] == 5
    assert meter_info["real_votes"] == 0
    assert meter_info["consensus_level"] == "Unanimous Consensus"
    assert not meter_info["is_vote_split"]
    assert not meter_info["is_gate_disagreement_triggered"]
    assert meter_info["gate_base_disagreement"] == 0.03
    assert "[" in meter_info["meter_bar"] and "]" in meter_info["meter_bar"]

    # Test conflicting votes case
    conflict_probas = {"rf": 0.85, "xgb": 0.35, "lgbm": 0.90, "lr": 0.20}
    conflict_info = ModelDisagreementMeter.compute(
        probas_dict=conflict_probas,
        p_calibrated_stack=0.60,
        base_disagreement_std=0.31,
        threshold_delta=0.20,
    )
    assert conflict_info["is_vote_split"]
    assert conflict_info["is_gate_disagreement_triggered"]
    assert conflict_info["consensus_level"] == "High Dissent / Conflicting Predictions"

    # Test plot generation
    fig_path = FIGURES_DIR / "test_meter.png"
    ModelDisagreementMeter.plot_meter(meter_info, fig_path)
    assert fig_path.exists()
    assert fig_path.stat().st_size > 1000


def test_news_credibility_fingerprint():
    mock_signals = {
        "sensationalism": {"score": 0.2, "level": "Low"},
        "attribution": {"score": 1.5, "level": "High"},
        "hedging": {"score": 0.3, "level": "Low"},
        "punctuation_intensity": {"score": 0.1},
        "readability": {"words_per_sentence": 19.0},
        "lexical_diversity": {"score": 0.55},
    }

    # Standard article (N=400 words)
    fp = NewsCredibilityFingerprint.compute(mock_signals, word_count=400)
    assert "scores" in fp
    assert len(fp["scores"]) == 6
    assert 15.0 <= fp["overall_credibility_index"] <= 95.0
    assert "credibility_grade" in fp
    for k, v in fp["scores"].items():
        assert 15.0 <= v <= 95.0, f"Score for {k} out of bounds: {v}"

    # Short article (N=25 words) - should shrink toward neutral baseline ~65
    fp_short = NewsCredibilityFingerprint.compute(mock_signals, word_count=25)
    assert fp_short["length_weight"] < 0.5
    for k, v in fp_short["scores"].items():
        assert 45.0 <= v <= 85.0, f"Short article score {k} failed shrinkage: {v}"

    radar_path = FIGURES_DIR / "test_radar.png"
    NewsCredibilityFingerprint.plot_radar(fp, radar_path)
    assert radar_path.exists()
    assert radar_path.stat().st_size > 1000


def test_counterfactual_explanation():
    clean_text = "shocking bombshell report alleges secret conspiracy and panic"
    raw_text = "SHOCKING BOMBSHELL: Report alleges secret conspiracy and panic!!!"

    def mock_predict(clean, raw):
        sens_count = sum(1 for w in ["shocking", "bombshell", "conspiracy", "panic"] if w in clean.lower())
        return min(0.95, 0.2 + sens_count * 0.18)

    orig_p = mock_predict(clean_text, raw_text)
    cf = CounterfactualExplainer.generate(
        clean_text=clean_text,
        raw_text=raw_text,
        original_p_fake=orig_p,
        predict_fn=mock_predict,
    )

    assert cf["original_prob"] == round(orig_p, 4)
    assert cf["simulated_prob"] < cf["original_prob"]
    assert cf["signed_prob_shift"] < 0
    assert cf["signed_shift_pct"] == round(cf["signed_prob_shift"] * 100.0, 2)
    assert len(cf["changed_fragments"]) > 0
    assert cf["is_modified"]
    assert "disclaimer" in cf

    # Test clean text with NO sensational words
    neutral_text = "The committee held a regular public session on infrastructure planning."
    cf_neutral = CounterfactualExplainer.generate(
        clean_text=neutral_text,
        raw_text=neutral_text,
        original_p_fake=0.20,
        predict_fn=mock_predict,
    )
    assert not cf_neutral["is_modified"]
    assert cf_neutral["signed_prob_shift"] == 0.0
    assert "no counterfactual perturbations were applied" in cf_neutral["summary"]


def test_synthesize_why_prediction_triggers():
    disagreement_info = {
        "consensus_level": "Strong Consensus",
        "fake_votes": 4,
        "real_votes": 1,
        "mean_prob": 0.72,
    }
    credibility_info = {
        "overall_credibility_index": 74.0,
        "credibility_grade": "B (Standard News Distribution)",
    }
    local_features = {
        "fake_signals": [{"name": "Random Forest Prediction [P(FAKE)]", "impact": 0.15}],
        "real_signals": [{"name": "Journalistic Source Attribution Density", "impact": -0.20}],
    }

    # Case 1: Low Confidence Only
    gate_low_conf = {
        "decision": "UNCERTAIN",
        "trigger": "low_confidence",
        "confidence": 0.62,
        "disagreement": 0.08,
        "tau": 0.80,
        "delta": 0.20,
    }
    why_lc = synthesize_why_prediction(gate_low_conf, disagreement_info, credibility_info, local_features)
    assert "Gating Trigger (Low Confidence)" in why_lc
    assert "acceptable consensus" in why_lc
    assert "Dual Uncertainty Active" not in why_lc

    # Case 2: High Disagreement Only
    gate_high_dis = {
        "decision": "UNCERTAIN",
        "trigger": "high_disagreement",
        "confidence": 0.88,
        "disagreement": 0.28,
        "tau": 0.80,
        "delta": 0.20,
    }
    why_hd = synthesize_why_prediction(gate_high_dis, disagreement_info, credibility_info, local_features)
    assert "Gating Trigger (High Disagreement)" in why_hd
    assert "base classifiers sharply diverged" in why_hd
    assert "Low Confidence" not in why_hd

    # Case 3: Both Triggers Active
    gate_both = {
        "decision": "UNCERTAIN",
        "trigger": "both_triggers",
        "confidence": 0.58,
        "disagreement": 0.26,
        "tau": 0.80,
        "delta": 0.20,
    }
    why_both = synthesize_why_prediction(gate_both, disagreement_info, credibility_info, local_features)
    assert "Dual Uncertainty Active" in why_both
    assert "Both safety triggers were activated" in why_both

    # Case 4: Neither Trigger Active (REAL)
    gate_real = {
        "decision": "REAL",
        "trigger": "none",
        "confidence": 0.94,
        "disagreement": 0.04,
        "tau": 0.80,
        "delta": 0.20,
    }
    why_real = synthesize_why_prediction(gate_real, disagreement_info, credibility_info, local_features)
    assert "Grounded Linguistic Anchors" in why_real or "Model Consensus" in why_real
    assert "UNCERTAIN" not in why_real


def test_url_extractor_validation_and_errors():
    # Valid syntax
    assert validate_url("https://www.bbc.com/news")
    assert validate_url("http://example.org/article")

    # Invalid syntax
    assert not validate_url("ftp://example.com")
    assert not validate_url("not_a_url")
    assert not validate_url("")
    assert not validate_url(None)

    # Inaccessible URL raises ArticleExtractionError
    with pytest.raises(ArticleExtractionError) as exc_info:
        extract_article_from_url("https://nonexistent-domain-test-12345.org/fake")
    assert "Network error" in str(exc_info.value) or "Could not retrieve" in str(exc_info.value) or "error" in str(exc_info.value).lower()


def test_pdf_report_generation():
    pdf_path = REPORTS_DIR / "test_audit_report.pdf"

    article_meta = {
        "title": "Senate Approves Major Clean Energy Infrastructure Bill",
        "url": "https://example.com/senate-clean-energy",
        "domain": "example.com",
        "word_count": 450,
    }
    gate_res = {
        "decision": "REAL",
        "confidence": 0.942,
        "disagreement": 0.035,
        "tau": 0.80,
        "delta": 0.20,
        "trigger": "none",
        "explanation": "High confidence and low ensemble disagreement.",
    }
    disagreement_info = ModelDisagreementMeter.compute(
        probas_dict={"rf": 0.08, "xgb": 0.05, "lgbm": 0.06, "lr": 0.09},
        p_calibrated_stack=0.058,
        base_disagreement_std=0.035,
        threshold_delta=0.20,
    )
    credibility_info = {
        "scores": {
            "Sensationalism Discipline": 92.0,
            "Source Attribution": 85.0,
            "Epistemic Grounding": 88.0,
            "Punctuation Discipline": 95.0,
            "Readability Structure": 80.0,
            "Lexical Diversity": 78.0,
        },
        "overall_credibility_index": 86.8,
        "credibility_grade": "A (Restrained / Professional Copy)",
        "methodology_note": "Evaluates predefined stylistic patterns.",
    }
    counterfactual_info = {
        "original_prob": 0.058,
        "simulated_prob": 0.058,
        "signed_prob_shift": 0.0,
        "signed_shift_pct": 0.0,
        "summary": "Article employs factual, restrained syntax.",
    }
    why_explanation = "The article presents well-attributed statements with high inter-model consensus."
    local_features = {
        "fake_signals": [{"name": "Lexical Diversity", "impact": 0.01}],
        "real_signals": [{"name": "Journalistic Source Attribution Density", "impact": -0.25}],
    }
    comparison_table = [
        {"model_name": "Random Forest", "accuracy": 0.89, "macro_f1": 0.89, "ece": 0.07, "coverage": 1.0, "selective_accuracy": 0.89},
        {"model_name": "CAGE-Fake (Calibrated + Gated)", "accuracy": 0.93, "macro_f1": 0.93, "ece": 0.03, "coverage": 0.82, "selective_accuracy": 0.97},
    ]

    out_file = generate_pdf_report(
        output_path=pdf_path,
        article_meta=article_meta,
        gate_result=gate_res,
        disagreement_info=disagreement_info,
        credibility_info=credibility_info,
        counterfactual_info=counterfactual_info,
        why_explanation=why_explanation,
        local_features=local_features,
        comparison_table=comparison_table,
    )

    assert out_file.exists()
    assert out_file.stat().st_size > 2000, "PDF file was generated but is too small or corrupted"
