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

    candidate_text_cols = [c for c in columns if c != label_col and not str(c).lower().startswith("unnamed")]

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
    Computes and logs comprehensive dataset statistics, clearly distinguishing
    different categories of duplicate content and missing values.
    """
    total_rows, total_cols = df.shape
    missing_counts = df.isnull().sum().to_dict()

    # Exact duplicate analysis
    exact_duplicates_count = int(df.duplicated().sum())
    duplicate_title_body_pairs = int(df.duplicated(subset=[title_col, body_col]).sum())
    duplicate_body_only = int(df.duplicated(subset=[body_col]).sum())
    duplicate_title_only = int(df.duplicated(subset=[title_col]).sum())

    # Standardize label representation for metrics
    label_series = df[label_col].astype(str).str.strip().str.upper()
    class_counts = label_series.value_counts().to_dict()
    class_percentages = {k: round(v / total_rows * 100, 2) for k, v in class_counts.items()}

    stats = {
        "shape": (total_rows, total_cols),
        "columns": list(df.columns),
        "missing_counts": missing_counts,
        "exact_duplicates_count": exact_duplicates_count,
        "duplicate_title_body_pairs": duplicate_title_body_pairs,
        "duplicate_body_only": duplicate_body_only,
        "duplicate_title_only": duplicate_title_only,
        "class_counts": class_counts,
        "class_percentages": class_percentages,
    }

    logger.info(f"Dataset Raw Shape: {total_rows} rows x {total_cols} columns")
    logger.info(f"Missing Values: {missing_counts}")
    logger.info(
        f"Duplicate Breakdown -> Exact Rows: {exact_duplicates_count}, "
        f"Title+Body Pairs: {duplicate_title_body_pairs}, "
        f"Body-Only: {duplicate_body_only}, Title-Only: {duplicate_title_only}"
    )
    logger.info(f"Class Balance: {class_percentages}% ({class_counts})")

    return stats

def load_and_standardize_data(csv_path: Path | str = DATA_PATH) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw CSV, auto-detects columns, standardizes labels (1=FAKE, 0=REAL),
    handles missing values, distinguishes duplicate categories, and deduplicates
    to guarantee zero train-test leakage.
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
    empty_body_count = int(empty_body_mask.sum())
    stats["empty_body_fallback_count"] = empty_body_count
    if empty_body_count > 0:
        logger.warning(f"Filling {empty_body_count} empty body texts with corresponding title.")
        standardized_df.loc[empty_body_mask, "body"] = standardized_df.loc[empty_body_mask, "title"]

    # Map labels: 1 = FAKE, 0 = REAL
    raw_labels = df[label_col].astype(str).str.strip().str.upper()
    label_map = {
        "FAKE": 1,
        "1": 1,
        "1.0": 1,
        "TRUE": 1,
        "REAL": 0,
        "0": 0,
        "0.0": 0,
        "FALSE": 0,
    }

    # Strict label validation: reject unrecognized labels to prevent silent coercion to REAL
    unrecognized = [lbl for lbl in raw_labels.unique() if lbl not in label_map]
    if unrecognized:
        raise ValueError(
            f"Unrecognized class label(s) detected in column '{label_col}': {unrecognized}. "
            f"Expected one of: {sorted(list(label_map.keys()))}. "
            f"Unmapped labels are rejected to prevent silent misclassification as REAL."
        )

    standardized_df["target"] = raw_labels.map(label_map)
    standardized_df["target_name"] = standardized_df["target"].map({1: "FAKE", 0: "REAL"})
    standardized_df["source_dataset"] = csv_path.name

    # Check for additional datasets in data/extra/*.csv
    extra_dir = csv_path.parent / "extra"
    extra_dir.mkdir(parents=True, exist_ok=True)
    extra_dfs = [standardized_df]

    for extra_path in sorted(extra_dir.glob("*.csv")):
        try:
            e_df = pd.read_csv(extra_path)
            if len(e_df) == 0:
                continue
            e_title_col, e_body_col, e_label_col = auto_detect_columns(e_df)
            e_std = pd.DataFrame()
            e_std["title"] = e_df[e_title_col].fillna("").astype(str).str.strip()
            e_std["body"] = e_df[e_body_col].fillna("").astype(str).str.strip()

            e_empty_body = e_std["body"].str.len() == 0
            if e_empty_body.any():
                e_std.loc[e_empty_body, "body"] = e_std.loc[e_empty_body, "title"]

            e_raw_labels = e_df[e_label_col].astype(str).str.strip().str.upper()
            e_std["target"] = e_raw_labels.map(label_map)
            # Filter unmapped labels if any
            valid_mask = e_std["target"].notnull()
            e_std = e_std[valid_mask].copy()
            e_std["target"] = e_std["target"].astype(int)
            e_std["target_name"] = e_std["target"].map({1: "FAKE", 0: "REAL"})
            e_std["source_dataset"] = extra_path.name
            extra_dfs.append(e_std)
            logger.info(f"Loaded extra dataset '{extra_path.name}' with {len(e_std)} records.")
        except Exception as exc:
            logger.warning(f"Could not load extra dataset '{extra_path}': {exc}")

    if len(extra_dfs) > 1:
        standardized_df = pd.concat(extra_dfs, ignore_index=True)
        logger.info(f"Merged {len(extra_dfs)} datasets. Total combined records before deduplication: {len(standardized_df)}")

    # Drop non-informative empty records (both title and body empty)
    non_empty_mask = (standardized_df["title"].str.len() > 0) | (standardized_df["body"].str.len() > 0)
    standardized_df = standardized_df[non_empty_mask].reset_index(drop=True)

    # Detect duplicate bodies with conflicting labels (Policy: Discard conflicting clusters)
    body_label_groups = standardized_df.groupby("body")["target"].nunique()
    conflicting_bodies = set(body_label_groups[body_label_groups > 1].index)
    conflict_count = len(conflicting_bodies)
    stats["conflicting_duplicate_bodies_count"] = conflict_count

    if conflict_count > 0:
        logger.warning(
            f"Detected {conflict_count} duplicate body cluster(s) with conflicting veracity labels. "
            f"Policy: Discarding all conflicting occurrences to prevent contradictory supervision."
        )
        standardized_df = standardized_df[~standardized_df["body"].isin(conflicting_bodies)].reset_index(drop=True)
    else:
        logger.info("Conflicting duplicate bodies check: 0 conflicting label clusters detected.")

    # Handle duplicates across datasets to prevent leakage across splits (retain first clean occurrence)
    initial_len = len(standardized_df)
    standardized_df = standardized_df.drop_duplicates(subset=["body"]).reset_index(drop=True)
    dropped_dupes = initial_len - len(standardized_df)
    stats["deduplicated_dropped_count"] = dropped_dupes
    stats["final_clean_rows"] = len(standardized_df)

    # Recompute clean class counts and percentages on final clean dataset
    clean_counts = standardized_df["target_name"].value_counts().to_dict()
    clean_percentages = {k: round(v / len(standardized_df) * 100, 2) for k, v in clean_counts.items()}
    stats["raw_class_counts"] = dict(stats["class_counts"])
    stats["raw_class_percentages"] = dict(stats["class_percentages"])
    stats["clean_class_counts"] = clean_counts
    stats["clean_class_percentages"] = clean_percentages
    stats["class_counts"] = clean_counts
    stats["class_percentages"] = clean_percentages
    stats["source_distribution"] = standardized_df["source_dataset"].value_counts().to_dict()

    if dropped_dupes > 0:
        logger.info(
            f"Deduplication: removed {dropped_dupes} duplicate body texts to prevent train-test data leakage. "
            f"Final clean unique articles: {len(standardized_df)} | Clean Class Counts: {clean_counts} ({clean_percentages}%)"
        )
    logger.info(f"Dataset source distribution: {stats['source_distribution']}")

    return standardized_df, stats

def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    calib_ratio: float = CALIBRATION_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
    save_manifest: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates strict 4-way stratified splits: 70 train / 10 val / 10 calibration / 10 test.
    Stratifies by label AND source_dataset (with graceful fallback if rare strata exist).
    Ensures calibration is never touched during training and test remains untouched until final eval.
    Generates a cryptographic split manifest verifying zero leakage.
    """
    import hashlib
    import json
    from config import METRICS_DIR

    assert np.isclose(train_ratio + val_ratio + calib_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    # Joint stratification by label AND source_dataset
    if "source_dataset" in df.columns and df["source_dataset"].nunique() > 1:
        joint_strata = df["target"].astype(str) + "_" + df["source_dataset"].astype(str)
        if (joint_strata.value_counts() < 4).any():
            logger.warning("Some (label, source_dataset) strata have < 4 samples; stratifying by target only.")
            strat_key = df["target"]
        else:
            logger.info(f"Stratifying splits jointly by label AND source_dataset ({joint_strata.nunique()} strata).")
            strat_key = joint_strata
    else:
        strat_key = df["target"]

    # Step 1: Split into Train (70%) and Temp (30%)
    temp_ratio = val_ratio + calib_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=strat_key,
        random_state=seed,
    )

    # Step 2: Split Temp (30%) into Val (10%), Calib (10%), Test (10%)
    if "source_dataset" in temp_df.columns and temp_df["source_dataset"].nunique() > 1:
        temp_joint = temp_df["target"].astype(str) + "_" + temp_df["source_dataset"].astype(str)
        temp_strat = temp_df["target"] if (temp_joint.value_counts() < 3).any() else temp_joint
    else:
        temp_strat = temp_df["target"]

    val_size_in_temp = val_ratio / temp_ratio
    val_df, remaining_temp_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_size_in_temp),
        stratify=temp_strat,
        random_state=seed,
    )

    if "source_dataset" in remaining_temp_df.columns and remaining_temp_df["source_dataset"].nunique() > 1:
        rem_joint = remaining_temp_df["target"].astype(str) + "_" + remaining_temp_df["source_dataset"].astype(str)
        rem_strat = remaining_temp_df["target"] if (rem_joint.value_counts() < 2).any() else rem_joint
    else:
        rem_strat = remaining_temp_df["target"]

    calib_size_in_remaining = calib_ratio / (calib_ratio + test_ratio)
    calib_df, test_df = train_test_split(
        remaining_temp_df,
        test_size=(1.0 - calib_size_in_remaining),
        stratify=rem_strat,
        random_state=seed,
    )

    # Cryptographic verification of zero partition overlap
    def hash_text(t: str) -> str:
        return hashlib.sha256(t.strip().encode("utf-8", errors="replace")).hexdigest()

    train_hashes = set(train_df["body"].map(hash_text))
    val_hashes = set(val_df["body"].map(hash_text))
    calib_hashes = set(calib_df["body"].map(hash_text))
    test_hashes = set(test_df["body"].map(hash_text))

    assert len(train_hashes.intersection(val_hashes)) == 0, "Leakage detected: train intersects val"
    assert len(train_hashes.intersection(calib_hashes)) == 0, "Leakage detected: train intersects calib"
    assert len(train_hashes.intersection(test_hashes)) == 0, "Leakage detected: train intersects test"
    assert len(val_hashes.intersection(calib_hashes)) == 0, "Leakage detected: val intersects calib"
    assert len(val_hashes.intersection(test_hashes)) == 0, "Leakage detected: val intersects test"
    assert len(calib_hashes.intersection(test_hashes)) == 0, "Leakage detected: calib intersects test"

    def partition_root_digest(h_set: set) -> str:
        return hashlib.sha256("".join(sorted(list(h_set))).encode("utf-8")).hexdigest()

    manifest = {
        "description": "Partition audit manifest recording sample counts, SHA-256 partition root digests, class balances, and pairwise zero-overlap verification.",
        "random_seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "calib": calib_ratio,
            "test": test_ratio,
        },
        "counts": {
            "train": len(train_df),
            "val": len(val_df),
            "calib": len(calib_df),
            "test": len(test_df),
            "total": len(df),
        },
        "sha256_unique_hash_counts": {
            "train": len(train_hashes),
            "val": len(val_hashes),
            "calib": len(calib_hashes),
            "test": len(test_hashes),
        },
        "partition_sha256_root_digests": {
            "train": partition_root_digest(train_hashes),
            "val": partition_root_digest(val_hashes),
            "calib": partition_root_digest(calib_hashes),
            "test": partition_root_digest(test_hashes),
        },
        "pairwise_overlap_counts": {
            "train_val": len(train_hashes.intersection(val_hashes)),
            "train_calib": len(train_hashes.intersection(calib_hashes)),
            "train_test": len(train_hashes.intersection(test_hashes)),
            "val_calib": len(val_hashes.intersection(calib_hashes)),
            "val_test": len(val_hashes.intersection(test_hashes)),
            "calib_test": len(calib_hashes.intersection(test_hashes)),
        },
        "class_balance_fake_pct": {
            "train": round(float(train_df["target"].mean() * 100), 2),
            "val": round(float(val_df["target"].mean() * 100), 2),
            "calib": round(float(calib_df["target"].mean() * 100), 2),
            "test": round(float(test_df["target"].mean() * 100), 2),
            "full_dataset": round(float(df["target"].mean() * 100), 2),
        },
        "conflicting_duplicate_bodies_count": 0,
        "zero_overlap_verified": True,
    }

    if save_manifest:
        try:
            METRICS_DIR.mkdir(parents=True, exist_ok=True)
            with open(METRICS_DIR / "split_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=4)
            logger.info(f"Split manifest saved to {METRICS_DIR / 'split_manifest.json'}")
        except Exception as e:
            logger.warning(f"Could not save split manifest: {e}")

    logger.info(
        f"Stratified Splits Created -> Train: {len(train_df)} ({train_df['target'].mean()*100:.1f}% FAKE), "
        f"Val: {len(val_df)} ({val_df['target'].mean()*100:.1f}% FAKE), "
        f"Calibration: {len(calib_df)} ({calib_df['target'].mean()*100:.1f}% FAKE), "
        f"Test: {len(test_df)} ({test_df['target'].mean()*100:.1f}% FAKE)"
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), calib_df.reset_index(drop=True), test_df.reset_index(drop=True)
