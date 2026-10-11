"""
CAGE-Fake: Evaluation Module
Calculates benchmark metrics: Accuracy, Macro-F1, Precision, Recall, Selective Accuracy, and Coverage.
Includes utilities for generating confusion matrices and research reports.
"""

from typing import Dict, Any, List
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from config import METRICS_DIR, FIGURES_DIR, TARGET_METRICS
from src.utils import setup_logger

logger = setup_logger("evaluation")

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_pred: np.ndarray = None,
    accepted_mask: np.ndarray = None,
    training_time: float = 0.0,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Computes all classification metrics.
    If accepted_mask is provided, computes Selective Accuracy and Coverage.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    n_total = len(y_true)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    metrics = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "training_time_s": round(training_time, 2),
    }
    
    if accepted_mask is not None:
        n_accepted = int(np.sum(accepted_mask))
        coverage = n_accepted / n_total if n_total > 0 else 0.0
        
        if n_accepted > 0:
            sel_acc = accuracy_score(y_true[accepted_mask], y_pred[accepted_mask])
        else:
            sel_acc = 1.0
            
        metrics["coverage"] = round(coverage, 4)
        metrics["selective_accuracy"] = round(sel_acc, 4)
        
    return metrics

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path = None,
) -> Path:
    """Plots and saves confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["REAL (0)", "FAKE (1)"], yticklabels=["REAL (0)", "FAKE (1)"])
    plt.title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    
    out_path = save_path or (FIGURES_DIR / f"cm_{model_name.replace(' ', '_').lower()}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    logger.info(f"Confusion matrix saved to {out_path}")
    return out_path

def save_metrics(metrics: Dict[str, Any], filename: str = "metrics.json") -> Path:
    """Saves metrics dictionary to JSON."""
    out_path = METRICS_DIR / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Metrics saved to {out_path}")
    return out_path
