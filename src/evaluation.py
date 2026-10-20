"""
CAGE-Fake: Evaluation Module
Calculates benchmark metrics: Accuracy, Macro-F1, Precision, Recall, Selective Accuracy, and Coverage.
Includes utilities for generating confusion matrices and research reports.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from config import METRICS_DIR, FIGURES_DIR, TARGET_METRICS
from src.utils import setup_logger

logger = setup_logger("evaluation")

def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes Wilson score interval for a binomial proportion k/n.
    Guarantees robust confidence bounds without normal approximation breakdown near 0 or 1.
    """
    if n == 0:
        return (0.0, 1.0)
    import scipy.stats as stats
    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z * np.sqrt((p * (1.0 - p) / n) + ((z**2) / (4.0 * (n**2))))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return float(round(float(lower), 4)), float(round(float(upper), 4))

def compute_brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Mean Squared Error between predicted probabilities and binary ground truth."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    return float(np.mean((p - y) ** 2))

def compute_log_loss_score(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-15) -> float:
    """Binary cross-entropy / log-loss."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_pred: np.ndarray = None,
    accepted_mask: np.ndarray = None,
    training_time: float = 0.0,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Computes all classification metrics, Brier score, log-loss, and Wilson confidence intervals.
    If accepted_mask is provided, computes Selective Accuracy and Coverage.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    n_total = len(y_true)
    n_correct = int(np.sum(y_true == y_pred))
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    acc_ci_low, acc_ci_high = compute_wilson_ci(n_correct, n_total)
    acc_ci_formatted = f"[{float(acc_ci_low):.4f}, {float(acc_ci_high):.4f}]"

    from src.calibration import compute_ece
    ece_val = None
    brier_val = None
    log_loss_val = None
    if p_pred is not None:
        ece_val, _ = compute_ece(y_true, p_pred)
        ece_val = round(float(ece_val), 4)
        brier_val = round(compute_brier_score(y_true, p_pred), 4)
        log_loss_val = round(compute_log_loss_score(y_true, p_pred), 4)

    metrics = {
        "model_name": model_name,
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": round(acc, 4),
        "accuracy_ci_95": acc_ci_formatted,
        "macro_f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "ece": ece_val,
        "brier_score": brier_val,
        "log_loss": log_loss_val,
        "training_time_s": round(training_time, 2),
    }

    if accepted_mask is not None:
        accepted_mask = np.asarray(accepted_mask, dtype=bool)
        n_accepted = int(np.sum(accepted_mask))
        coverage = n_accepted / n_total if n_total > 0 else 0.0

        if n_accepted > 0:
            y_acc_true = y_true[accepted_mask]
            y_acc_pred = y_pred[accepted_mask]
            n_acc_correct = int(np.sum(y_acc_true == y_acc_pred))
            sel_acc = accuracy_score(y_acc_true, y_acc_pred)
            sel_ci_low, sel_ci_high = compute_wilson_ci(n_acc_correct, n_accepted)
            sel_ci_formatted = f"[{float(sel_ci_low):.4f}, {float(sel_ci_high):.4f}]"
        else:
            n_acc_correct = 0
            sel_acc = 1.0
            sel_ci_formatted = "[1.0000, 1.0000]"

        metrics["n_accepted"] = n_accepted
        metrics["n_accepted_correct"] = n_acc_correct
        metrics["n_accepted_errors"] = n_accepted - n_acc_correct
        metrics["coverage"] = round(coverage, 4)
        metrics["selective_accuracy"] = round(sel_acc, 4)
        metrics["selective_risk"] = round(1.0 - sel_acc, 4)
        metrics["selective_accuracy_ci_95"] = sel_ci_formatted

        # Abstained metrics
        abstained_mask = ~accepted_mask
        n_abstained = int(np.sum(abstained_mask))
        metrics["n_abstained"] = n_abstained
        if n_abstained > 0:
            y_abs_true = y_true[abstained_mask]
            y_abs_pred = y_pred[abstained_mask]
            abs_correct = int(np.sum(y_abs_true == y_abs_pred))
            abs_acc = abs_correct / n_abstained
            metrics["n_abstained_correct"] = abs_correct
            metrics["n_abstained_errors"] = n_abstained - abs_correct
            metrics["abstained_accuracy"] = round(abs_acc, 4)
            metrics["abstained_error_rate"] = round(1.0 - abs_acc, 4)
        else:
            metrics["n_abstained_correct"] = 0
            metrics["n_abstained_errors"] = 0
            metrics["abstained_accuracy"] = 0.0
            metrics["abstained_error_rate"] = 0.0
    else:
        metrics["coverage"] = 1.0
        metrics["selective_accuracy"] = round(acc, 4)
        metrics["selective_risk"] = round(1.0 - acc, 4)
        metrics["selective_accuracy_ci_95"] = acc_ci_formatted

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
