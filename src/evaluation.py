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

    # Compute per-class precision, recall, and F1
    from sklearn.metrics import precision_recall_fscore_support
    prec_k, rec_k, f1_k, supp_k = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    class_metrics = {
        "REAL": {
            "precision": round(float(prec_k[0]), 4),
            "recall": round(float(rec_k[0]), 4),
            "f1": round(float(f1_k[0]), 4),
            "support": int(supp_k[0]),
        },
        "FAKE": {
            "precision": round(float(prec_k[1]), 4),
            "recall": round(float(rec_k[1]), 4),
            "f1": round(float(f1_k[1]), 4),
            "support": int(supp_k[1]),
        },
    }

    metrics = {
        "model_name": model_name,
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": round(acc, 4),
        "accuracy_ci_95": acc_ci_formatted,
        "macro_f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "class_metrics": class_metrics,
        "precision_real": round(float(prec_k[0]), 4),
        "recall_real": round(float(rec_k[0]), 4),
        "f1_real": round(float(f1_k[0]), 4),
        "precision_fake": round(float(prec_k[1]), 4),
        "recall_fake": round(float(rec_k[1]), 4),
        "f1_fake": round(float(f1_k[1]), 4),
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

            # Per-class metrics on accepted subset
            acc_prec_k, acc_rec_k, acc_f1_k, acc_supp_k = precision_recall_fscore_support(
                y_acc_true, y_acc_pred, labels=[0, 1], zero_division=0
            )
            class_metrics["REAL_accepted"] = {
                "precision": round(float(acc_prec_k[0]), 4),
                "recall": round(float(acc_rec_k[0]), 4),
                "f1": round(float(acc_f1_k[0]), 4),
                "support": int(acc_supp_k[0]),
            }
            class_metrics["FAKE_accepted"] = {
                "precision": round(float(acc_prec_k[1]), 4),
                "recall": round(float(acc_rec_k[1]), 4),
                "f1": round(float(acc_f1_k[1]), 4),
                "support": int(acc_supp_k[1]),
            }
        else:
            n_acc_correct = 0
            sel_acc = 1.0
            sel_ci_formatted = "[1.0000, 1.0000]"

        metrics["n_accepted"] = n_accepted
        metrics["n_accepted_correct"] = n_acc_correct
        metrics["n_accepted_errors"] = n_accepted - n_acc_correct
        metrics["coverage"] = round(coverage, 4)
        metrics["uncertain_rate"] = round(1.0 - coverage, 4)
        metrics["selective_accuracy"] = round(sel_acc, 4)
        metrics["selective_risk"] = round(1.0 - sel_acc, 4)
        metrics["selective_accuracy_ci_95"] = sel_ci_formatted

        # Abstained / Uncertain metrics
        abstained_mask = ~accepted_mask
        n_abstained = int(np.sum(abstained_mask))
        metrics["n_abstained"] = n_abstained
        metrics["n_uncertain"] = n_abstained
        class_metrics["UNCERTAIN"] = {
            "precision": 1.0 if n_abstained > 0 else 0.0,
            "recall": round(n_abstained / n_total, 4) if n_total > 0 else 0.0,
            "f1": round(2.0 * (n_abstained / n_total) / (1.0 + (n_abstained / n_total)), 4) if n_total > 0 else 0.0,
            "support": n_abstained,
        }
        metrics["precision_uncertain"] = class_metrics["UNCERTAIN"]["precision"]
        metrics["recall_uncertain"] = class_metrics["UNCERTAIN"]["recall"]
        metrics["f1_uncertain"] = class_metrics["UNCERTAIN"]["f1"]

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
        metrics["uncertain_rate"] = 0.0
        metrics["n_uncertain"] = 0
        metrics["selective_accuracy"] = round(acc, 4)
        metrics["selective_risk"] = round(1.0 - acc, 4)
        metrics["selective_accuracy_ci_95"] = acc_ci_formatted
        class_metrics["UNCERTAIN"] = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "support": 0,
        }
        metrics["precision_uncertain"] = 0.0
        metrics["recall_uncertain"] = 0.0
        metrics["f1_uncertain"] = 0.0

    return metrics


def evaluate_3way_predictions(
    y_true: List[str],
    y_pred: List[str],
    p_pred: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Evaluates explicit 3-way triage verdicts (REAL, FAKE, UNCERTAIN).
    Computes per-class precision, recall, F1, overall accuracy, UNCERTAIN rate, and ECE.
    """
    classes = ["REAL", "FAKE", "UNCERTAIN"]
    y_t = [str(y).upper() for y in y_true]
    y_p = [str(y).upper() for y in y_pred]
    n_total = len(y_t)

    from sklearn.metrics import precision_recall_fscore_support
    prec_k, rec_k, f1_k, supp_k = precision_recall_fscore_support(
        y_t, y_p, labels=classes, zero_division=0
    )

    class_metrics = {}
    for i, cls_name in enumerate(classes):
        class_metrics[cls_name] = {
            "precision": round(float(prec_k[i]), 4),
            "recall": round(float(rec_k[i]), 4),
            "f1": round(float(f1_k[i]), 4),
            "support": int(supp_k[i]),
        }

    n_correct = sum(1 for yt, yp in zip(y_t, y_p) if yt == yp)
    n_uncertain = sum(1 for yp in y_p if yp == "UNCERTAIN")
    uncertain_rate = round(n_uncertain / n_total, 4) if n_total > 0 else 0.0

    from src.calibration import compute_ece
    ece_val = None
    if p_pred is not None:
        # Convert true labels to binary (1 for FAKE, 0 for REAL/other) for ECE
        binary_true = np.array([1 if y == "FAKE" else 0 for y in y_t])
        ece_val, _ = compute_ece(binary_true, np.asarray(p_pred))
        ece_val = round(float(ece_val), 4)

    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": round(n_correct / n_total, 4) if n_total > 0 else 0.0,
        "macro_f1": round(float(np.mean(f1_k)), 4),
        "uncertain_rate": uncertain_rate,
        "n_uncertain": n_uncertain,
        "ece": ece_val,
        "class_metrics": class_metrics,
        "precision_real": class_metrics["REAL"]["precision"],
        "recall_real": class_metrics["REAL"]["recall"],
        "f1_real": class_metrics["REAL"]["f1"],
        "precision_fake": class_metrics["FAKE"]["precision"],
        "recall_fake": class_metrics["FAKE"]["recall"],
        "f1_fake": class_metrics["FAKE"]["f1"],
        "precision_uncertain": class_metrics["UNCERTAIN"]["precision"],
        "recall_uncertain": class_metrics["UNCERTAIN"]["recall"],
        "f1_uncertain": class_metrics["UNCERTAIN"]["f1"],
    }

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path = None,
    pdf_number_name: Optional[str] = None,
) -> Path:
    """Plots and saves confusion matrix as high-res PNG and publication-quality vector PDF."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["REAL (0)", "FAKE (1)"], yticklabels=["REAL (0)", "FAKE (1)"])
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    
    out_path = save_path or (FIGURES_DIR / f"cm_{model_name.replace(' ', '_').lower()}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)

    # Also save vector PDF in outputs/figures/pdf/
    pdf_dir = FIGURES_DIR / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    clean_model_tag = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    pdf_filename = f"{pdf_number_name}.pdf" if pdf_number_name else f"04_confusion_{clean_model_tag}.pdf"
    pdf_path = pdf_dir / pdf_filename
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Confusion matrix saved to {out_path} and {pdf_path}")
    return out_path

def save_metrics(metrics: Dict[str, Any], filename: str = "metrics.json") -> Path:
    """Saves metrics dictionary to JSON."""
    out_path = METRICS_DIR / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Metrics saved to {out_path}")
    return out_path
