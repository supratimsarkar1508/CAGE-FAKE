"""
CAGE-Fake: Data Loader and Stratified Partitioning Module
Automatic schema detection, dataset inspection, and strict 4-way stratified splitting.
"""

from pathlib import Path
from typing import Dict, Tuple, Any
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    CALIBRATION_RATIO,
    TEST_RATIO,
)
from src.utils import setup_logger

logger = setup_logger("data_loader")

def auto_detect_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Auto-detects title, body, and label columns without hard-coding names or positions.
    Returns: (title_col, body_col, label_col)
    """
    columns = list(df.columns)
    
    # 1. Detect Label column
    label_col = None
    # Check for known categorical target values
    for col in columns:
        unique_vals = set(str(v).strip().upper() for v in df[col].dropna().unique())
        if unique_vals.issubset({"FAKE", "REAL", "TRUE", "FALSE", "0", "1", "0.0", "1.0"}) and 1 < len(unique_vals) <= 3:
            label_col = col
            break
            
    if label_col is None:
        for col in columns:
            if re.search(r"(label|target|class|veracity)", str(col), re.IGNORECASE):
                label_col = col
                break
                
    if label_col is None:
        raise ValueError("Could not auto-detect the label column in the dataset.")

    candidate_text_cols = [c for c in columns if c != label_col]

    # 2. Detect Title column
    title_col = None
    for col in candidate_text_cols:
        if re.search(r"(title|headline|header|subject)", str(col), re.IGNORECASE):
            title_col = col
            break

    # 3. Detect Body/Text column
    body_col = None
    for col in candidate_text_cols:
        if col != title_col and re.search(r"(text|body|content|article|story)", str(col), re.IGNORECASE):
            body_col = col
            break

    # Fallback by text length if column names are ambiguous
    if title_col is None or body_col is None:
        remaining = [c for c in candidate_text_cols if c not in (title_col, body_col)]
        # Calculate mean character length
        avg_lens = {
            col: df[col].astype(str).str.len().mean()
            for col in remaining
        }
        sorted_cols = sorted(avg_lens.items(), key=lambda x: x[1])
        if title_col is None and sorted_cols:
            title_col = sorted_cols[0][0]  # shorter is title
            sorted_cols = [x for x in sorted_cols if x[0] != title_col]
        if body_col is None and sorted_cols:
            body_col = sorted_cols[-1][0]  # longer is body

    logger.info(f"Auto-detected columns -> Title: '{title_col}', Body: '{body_col}', Label: '{label_col}'")
    return title_col, body_col, label_col

def inspect_dataset(df: pd.DataFrame, title_col: str, body_col: str, label_col: str) -> Dict[str, Any]:
    """
    Computes and logs comprehensive dataset statistics.
    """
    total_rows, total_cols = df.shape
    missing_counts = df.isnull().sum().to_dict()
    duplicates_count = int(df.duplicated().sum())

    # Standardize label representation for metrics
    label_series = df[label_col].astype(str).str.strip().str.upper()
    class_counts = label_series.value_counts().to_dict()
    class_percentages = {k: round(v / total_rows * 100, 2) for k, v in class_counts.items()}

    stats = {
        "shape": (total_rows, total_cols),
        "columns": list(df.columns),
        "missing_counts": missing_counts,
        "duplicates_count": duplicates_count,
        "class_counts": class_counts,
        "class_percentages": class_percentages,
    }

    logger.info(f"Dataset Shape: {total_rows} rows x {total_cols} columns")
    logger.info(f"Missing Values: {missing_counts}")
    logger.info(f"Duplicate Rows: {duplicates_count}")
    logger.info(f"Class Balance: {class_percentages}% ({class_counts})")

    return stats

def load_and_standardize_data(csv_path: Path | str = DATA_PATH) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw CSV, auto-detects columns, standardizes labels (1=FAKE, 0=REAL),
    fills missing text with headline fallback, and returns standardized DataFrame.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)
    title_col, body_col, label_col = auto_detect_columns(df)
    stats = inspect_dataset(df, title_col, body_col, label_col)

    # Standardize DataFrame
    standardized_df = pd.DataFrame()
    standardized_df["title"] = df[title_col].fillna("").astype(str).str.strip()
    standardized_df["body"] = df[body_col].fillna("").astype(str).str.strip()

    # Fallback: if body is empty or whitespace, use title
    empty_body_mask = standardized_df["body"].str.len() == 0
    if empty_body_mask.sum() > 0:
        logger.warning(f"Filling {empty_body_mask.sum()} empty body texts with corresponding title.")
        standardized_df.loc[empty_body_mask, "body"] = standardized_df.loc[empty_body_mask, "title"]

    # Map labels: 1 = FAKE, 0 = REAL
    raw_labels = df[label_col].astype(str).str.strip().str.upper()
    label_map = {
        "FAKE": 1,
        "1": 1,
        "1.0": 1,
        "TRUE": 1,   # if target refers to fake as positive class
        "REAL": 0,
        "0": 0,
        "0.0": 0,
        "FALSE": 0,
    }
    # Check if target is explicitly FAKE/REAL
    if set(raw_labels.unique()).issubset({"FAKE", "REAL"}):
        standardized_df["target"] = raw_labels.map({"FAKE": 1, "REAL": 0})
        standardized_df["target_name"] = raw_labels
    else:
        standardized_df["target"] = raw_labels.map(lambda x: label_map.get(x, 0))
        standardized_df["target_name"] = standardized_df["target"].map({1: "FAKE", 0: "REAL"})

    return standardized_df, stats

def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    calib_ratio: float = CALIBRATION_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates strict 4-way stratified splits: 70 train / 10 val / 10 calibration / 10 test.
    Ensures calibration is never touched during training and test remains untouched until final eval.
    """
    assert np.isclose(train_ratio + val_ratio + calib_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    # Step 1: Split into Train (70%) and Temp (30%)
    temp_ratio = val_ratio + calib_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["target"],
        random_state=seed,
    )

    # Step 2: Split Temp (30%) into Val (10%), Calib (10%), Test (10%)
    # Out of temp: val is 1/3, calib is 1/3, test is 1/3
    val_size_in_temp = val_ratio / temp_ratio
    val_df, remaining_temp_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_size_in_temp),
        stratify=temp_df["target"],
        random_state=seed,
    )

    calib_size_in_remaining = calib_ratio / (calib_ratio + test_ratio)
    calib_df, test_df = train_test_split(
        remaining_temp_df,
        test_size=(1.0 - calib_size_in_remaining),
        stratify=remaining_temp_df["target"],
        random_state=seed,
    )

    logger.info(
        f"Stratified Splits Created -> Train: {len(train_df)} ({train_df['target'].mean()*100:.1f}% FAKE), "
        f"Val: {len(val_df)} ({val_df['target'].mean()*100:.1f}% FAKE), "
        f"Calibration: {len(calib_df)} ({calib_df['target'].mean()*100:.1f}% FAKE), "
        f"Test: {len(test_df)} ({test_df['target'].mean()*100:.1f}% FAKE)"
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), calib_df.reset_index(drop=True), test_df.reset_index(drop=True)
