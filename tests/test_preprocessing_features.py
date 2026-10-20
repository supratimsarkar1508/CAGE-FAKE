"""
Unit and Integration Tests for Preprocessing and Multi-Stream Feature Engineering
"""

import pytest
import numpy as np

from src.preprocessing import TextPreprocessor
from src.stylometric_features import StylometricFeatureExtractor
from src.readability_features import ReadabilityFeatureExtractor
from src.feature_pipeline import UnifiedFeaturePipeline

def test_text_preprocessor():
    raw_sample = "<b>Breaking News:</b> Check out https://example.com/breaking! President stated &quot;we are united&quot;."
    clean = TextPreprocessor.clean_text(raw_sample)

    assert "<b>" not in clean
    assert "https://" not in clean
    assert "&quot;" not in clean
    assert '"we are united"' in clean

def test_15_stylometric_features():
    extractor = StylometricFeatureExtractor()
    assert len(extractor.FEATURE_NAMES) == 15

    raw_texts = [
        "The president allegedly claimed that the election had several irregularities! Was it confirmed?",
        "According to official reports, the committee stated and confirmed that the findings were accurate.",
        "Breaking: Shocking revelations emerged today regarding the budget crisis.",
    ]
    clean_texts = [TextPreprocessor.clean_text(t) for t in raw_texts]

    # Test single extraction
    feats_single = extractor.extract_features_single(raw_texts[0], clean_texts[0])
    assert feats_single.shape == (15,)
    assert not np.isnan(feats_single).any()

    # Test fit and transform
    matrix = extractor.fit_transform(raw_texts, clean_texts)
    assert matrix.shape == (3, 15)

    # Check novel markers present
    names = extractor.get_feature_names()
    assert "hedge_word_ratio" in names
    assert "tense_consistency_ratio" in names
    assert "attribution_verb_density" in names

def test_4_readability_features():
    extractor = ReadabilityFeatureExtractor()
    assert len(extractor.FEATURE_NAMES) == 4

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Epistemological formulations substantiate empirical corroboration across socioeconomic strata.",
    ]
    matrix = extractor.fit_transform(texts)
    assert matrix.shape == (2, 4)
    assert not np.isnan(matrix).any()

    names = extractor.get_feature_names()
    assert names == [
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "gunning_fog",
        "smog_index",
    ]

def test_unified_feature_pipeline():
    train_raw = [
        "Senator Smith stated that the bill will pass with bipartisan consensus.",
        "Conspiracy exposed! Sources allegedly report massive secret meetings taking place.",
        "Scientists confirmed that global surface temperatures rose by 1.2 degrees Celsius.",
        "Unbelievable bombshell: The secret documents they do not want you to see!",
    ]
    train_clean = [TextPreprocessor.clean_text(t) for t in train_raw]

    pipeline = UnifiedFeaturePipeline(include_semantic=False)
    X = pipeline.fit_transform(train_clean, train_raw)

    assert X.shape[0] == 4
    # Streams A, C, D present
    assert "lexical" in pipeline.stream_boundaries
    assert "stylometric" in pipeline.stream_boundaries
    assert "readability" in pipeline.stream_boundaries
    assert pipeline.stream_boundaries["stylometric"][1] - pipeline.stream_boundaries["stylometric"][0] == 15
    assert pipeline.stream_boundaries["readability"][1] - pipeline.stream_boundaries["readability"][0] == 4
