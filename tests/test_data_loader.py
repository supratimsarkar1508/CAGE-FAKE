"""
Unit and Integration Tests for CAGE-Fake Data Loader and Partitioning
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_PATH, RANDOM_SEED
from src.data_loader import auto_detect_columns, load_and_standardize_data, create_stratified_splits

def test_auto_detect_columns():
    df = pd.DataFrame({
        "Unnamed: 0": [0, 1, 2],
        "headline": ["Breaking News", "Election Update", "Local Weather"],
        "content": ["Article text here...", "Another story text...", "Sunny skies today..."],
        "veracity": ["REAL", "FAKE", "REAL"],
    })
    title_col, body_col, label_col = auto_detect_columns(df)
    assert title_col == "headline"
    assert body_col == "content"
    assert label_col == "veracity"

def test_load_real_dataset():
    assert DATA_PATH.exists(), f"Dataset file not found at {DATA_PATH}"
    df, stats = load_and_standardize_data(DATA_PATH)

    assert len(df) > 5000, f"Expected > 5000 rows after deduplication, got {len(df)}"
    assert set(df.columns) >= {"title", "body", "target", "target_name"}
    assert set(df["target"].unique()) == {0, 1}
    assert (df["target"] == 1).sum() > 0
    assert (df["target"] == 0).sum() > 0

    # Ensure no empty bodies
    assert (df["body"].str.len() == 0).sum() == 0

    # Ensure no duplicate bodies
    assert df.duplicated(subset=["body"]).sum() == 0

def test_stratified_splits_integrity():
    df, _ = load_and_standardize_data(DATA_PATH)
    train_df, val_df, calib_df, test_df = create_stratified_splits(df)

    total_split_rows = len(train_df) + len(val_df) + len(calib_df) + len(test_df)
    assert total_split_rows == len(df), "Split rows do not sum to total dataset length"

    # Strict partition ratios
    assert np.isclose(len(train_df) / len(df), 0.70, atol=0.01)
    assert np.isclose(len(val_df) / len(df), 0.10, atol=0.01)
    assert np.isclose(len(calib_df) / len(df), 0.10, atol=0.01)
    assert np.isclose(len(test_df) / len(df), 0.10, atol=0.01)

    # Class balance stratification preserved across all 4 splits
    full_fake_ratio = df["target"].mean()
    assert np.isclose(train_df["target"].mean(), full_fake_ratio, atol=0.02)
    assert np.isclose(val_df["target"].mean(), full_fake_ratio, atol=0.03)
    assert np.isclose(calib_df["target"].mean(), full_fake_ratio, atol=0.03)
    assert np.isclose(test_df["target"].mean(), full_fake_ratio, atol=0.03)


def test_unrecognized_label_rejection(tmp_path):
    """Verifies that unmapped labels raise ValueError rather than silently defaulting to REAL."""
    csv_file = tmp_path / "corrupt_labels.csv"
    df = pd.DataFrame({
        "title": ["Article A", "Article B"],
        "text": ["This is text for article A.", "This is text for article B."],
        "label": ["REAL", "CORRUPT_TYPO_LABEL"]
    })
    df.to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="Unrecognized class label"):
        load_and_standardize_data(csv_file)


def test_conflicting_duplicate_bodies_policy(tmp_path):
    """Verifies that duplicate bodies with conflicting labels are discarded to prevent contradictory supervision."""
    csv_file = tmp_path / "conflicting_dupes.csv"
    df = pd.DataFrame({
        "title": ["Title A", "Title B", "Title C"],
        "text": ["Identical duplicate body text", "Identical duplicate body text", "Unique distinct body text"],
        "label": ["REAL", "FAKE", "REAL"]
    })
    df.to_csv(csv_file, index=False)

    clean_df, stats = load_and_standardize_data(csv_file)
    assert stats["conflicting_duplicate_bodies_count"] == 1
    # Conflicting body should be discarded, keeping only the non-conflicting unique record
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["title"] == "Title C"

