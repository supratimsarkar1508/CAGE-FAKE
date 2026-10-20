"""
CAGE-Fake: Uncertainty Gate and Threshold Sweep Module
Dual-Signal Uncertainty Gating Mechanism (with optional Evidence Hook):
1. Low Confidence Trigger: max(p, 1-p) < tau -> UNCERTAIN
2. High Disagreement Trigger: std(p_rf, p_xgb, p_lgbm, p_lr) > delta -> UNCERTAIN
3. Evidence Conflict Hook: Optional interface for external fact-verification conflict score

Includes grid sweep across (tau, delta) Pareto frontier on validation/calibration splits,
and generates Risk-Coverage curves.
"""

from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from config import (
    TAU_SWEEP,
    DELTA_SWEEP,
    DEFAULT_TAU,
    DEFAULT_DELTA,
    FIGURES_DIR,
)
from src.utils import setup_logger

logger = setup_logger("uncertainty")

class UncertaintyGate:
    """
    Triple-trigger uncertainty gating logic for trustworthy triage.
    """

    def __init__(
        self,
        tau: float = DEFAULT_TAU,
        delta: float = DEFAULT_DELTA,
    ):
        self.tau = tau
        self.delta = delta

    def evaluate_single(
        self,
        p_calibrated: float,
        disagreement: float,
        evidence_conflict: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates gate on a single article using:
        1. Calibrated confidence: max(p, 1-p)
        2. Base-model disagreement: std across 4 base learners (RF, XGB, LGBM, LR)
        3. Optional external evidence conflict hook

        Identifies:
        - 'both_triggers' when confidence < tau AND disagreement > delta
        - 'low_confidence' when confidence < tau and disagreement <= delta
        - 'high_disagreement' when confidence >= tau and disagreement > delta
        - 'evidence_conflict' when external conflict > 0.5
        - 'none' when neither condition triggers (verdict is accepted as REAL or FAKE)
        """
        confidence = float(max(p_calibrated, 1.0 - p_calibrated))
        disagreement = float(disagreement)

        is_low_conf = confidence < self.tau
        is_high_dis = disagreement > self.delta
        is_ev_conflict = evidence_conflict is not None and evidence_conflict > 0.5

        if is_low_conf and is_high_dis:
            decision = "UNCERTAIN"
            trigger = "both_triggers"
            explanation = (
                f"Both safety triggers active: Calibrated confidence ({confidence:.1%}) is below threshold tau "
                f"({self.tau:.1%}), AND base-model disagreement ({disagreement:.3f}) exceeds threshold delta ({self.delta:.3f})."
            )
        elif is_low_conf:
            decision = "UNCERTAIN"
            trigger = "low_confidence"
            explanation = (
                f"Confidence ({confidence:.1%}) is below threshold tau ({self.tau:.1%}); "
                f"base-model disagreement ({disagreement:.3f}) is within threshold delta ({self.delta:.3f})."
            )
        elif is_high_dis:
            decision = "UNCERTAIN"
            trigger = "high_disagreement"
            explanation = (
                f"Base-model disagreement ({disagreement:.3f}) exceeds threshold delta ({self.delta:.3f}); "
                f"calibrated confidence is {confidence:.1%}."
            )
        elif is_ev_conflict:
            decision = "UNCERTAIN"
            trigger = "evidence_conflict"
            explanation = f"Conflict with verified external evidence detected ({evidence_conflict:.2f})."
        else:
            decision = "FAKE" if p_calibrated >= 0.5 else "REAL"
            trigger = "none"
            explanation = f"High confidence ({confidence:.1%}) and acceptable base-model disagreement ({disagreement:.3f})."

        return {
            "decision": decision,
            "trigger": trigger,
            "is_accepted": decision != "UNCERTAIN",
            "is_low_confidence": is_low_conf,
            "is_high_disagreement": is_high_dis,
            "is_evidence_conflict": is_ev_conflict,
            "confidence": confidence,
            "disagreement": disagreement,
            "p_calibrated": p_calibrated,
            "tau": self.tau,
            "delta": self.delta,
            "explanation": explanation,
            "evidence_conflict_available": evidence_conflict is not None,
            "evidence_note": (
                "External evidence active."
                if evidence_conflict is not None
                else "Evidence conflict unavailable because no gold-standard external evidence is provided at prediction time."
            ),
        }

    def evaluate_batch(
        self,
        p_calibrated: np.ndarray,
        disagreements: np.ndarray,
        evidence_conflicts: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Evaluates gate on batch of samples."""
        p_cal = np.asarray(p_calibrated)
        dis = np.asarray(disagreements)
        confs = np.maximum(p_cal, 1.0 - p_cal)

        n = len(p_cal)
        decisions = []
        triggers = []
        is_accepted = np.ones(n, dtype=bool)

        for i in range(n):
            conf = confs[i]
            d = dis[i]
            ev = evidence_conflicts[i] if evidence_conflicts is not None else None

            is_lc = conf < self.tau
            is_hd = d > self.delta
            is_ev = ev is not None and ev > 0.5

            if is_lc and is_hd:
                decisions.append("UNCERTAIN")
                triggers.append("both_triggers")
                is_accepted[i] = False
            elif is_lc:
                decisions.append("UNCERTAIN")
                triggers.append("low_confidence")
                is_accepted[i] = False
            elif is_hd:
                decisions.append("UNCERTAIN")
                triggers.append("high_disagreement")
                is_accepted[i] = False
            elif is_ev:
                decisions.append("UNCERTAIN")
                triggers.append("evidence_conflict")
                is_accepted[i] = False
            else:
                decisions.append("FAKE" if p_cal[i] >= 0.5 else "REAL")
                triggers.append("none")

        return {
            "decisions": np.array(decisions),
            "triggers": np.array(triggers),
            "is_accepted": is_accepted,
            "coverage": float(np.mean(is_accepted)),
        }

class ThresholdOptimizer:
    """
    Performs grid search over tau and delta on validation/calibration partitions
    to identify the Pareto-optimal operating point.
    """

    def __init__(
        self,
        tau_grid: List[float] = None,
        delta_grid: List[float] = None,
    ):
        self.tau_grid = tau_grid or TAU_SWEEP
        self.delta_grid = delta_grid or DELTA_SWEEP

    def sweep(
        self,
        y_true: np.ndarray,
        p_calibrated: np.ndarray,
        disagreements: np.ndarray,
        target_coverage: float = 0.80,
    ) -> Dict[str, Any]:
        """
        Executes threshold sweep across grid.
        Calculates Coverage, Selective Accuracy, Selective Risk.
        Selects optimal (tau*, delta*) satisfying coverage >= target_coverage with max selective accuracy.
        """
        y_true = np.asarray(y_true).astype(int)
        p_cal = np.asarray(p_calibrated)
        dis = np.asarray(disagreements)

        results = []
        best_acc = -1.0
        best_tau = self.tau_grid[0]
        best_delta = self.delta_grid[0]

        for tau in self.tau_grid:
            for delta in self.delta_grid:
                gate = UncertaintyGate(tau=tau, delta=delta)
                res = gate.evaluate_batch(p_cal, dis)
                accepted_mask = res["is_accepted"]
                coverage = res["coverage"]

                if np.sum(accepted_mask) > 0:
                    y_acc = y_true[accepted_mask]
                    preds_acc = (p_cal[accepted_mask] >= 0.5).astype(int)
                    sel_acc = float(np.mean(preds_acc == y_acc))
                    sel_risk = 1.0 - sel_acc
                else:
                    sel_acc = 1.0
                    sel_risk = 0.0

                results.append({
                    "tau": tau,
                    "delta": delta,
                    "coverage": round(coverage, 4),
                    "selective_accuracy": round(sel_acc, 4),
                    "selective_risk": round(sel_risk, 4),
                    "accepted_count": int(np.sum(accepted_mask)),
                })

                if coverage >= target_coverage and sel_acc > best_acc:
                    best_acc = sel_acc
                    best_tau = tau
                    best_delta = delta

        # Fallback if no pair met target_coverage
        if best_acc < 0:
            best_pair = max(results, key=lambda x: (x["selective_accuracy"], x["coverage"]))
            best_tau = best_pair["tau"]
            best_delta = best_pair["delta"]
            best_acc = best_pair["selective_accuracy"]

        logger.info(
            f"Optimal Thresholds Selected on Validation: tau*={best_tau}, delta*={best_delta} "
            f"(Selective Acc={best_acc:.4f})"
        )

        return {
            "best_tau": best_tau,
            "best_delta": best_delta,
            "best_selective_acc": best_acc,
            "all_sweep_results": results,
        }

    @staticmethod
    def plot_risk_coverage(
        sweep_results: List[Dict[str, Any]],
        optimal_tau: float,
        optimal_delta: float,
        save_path: Path = None,
    ) -> Path:
        """Plots the Risk-Coverage trade-off curve across evaluated threshold pairs."""
        coverages = [r["coverage"] for r in sweep_results]
        risks = [r["selective_risk"] for r in sweep_results]
        accs = [r["selective_accuracy"] for r in sweep_results]

        plt.figure(figsize=(8, 5))
        scatter = plt.scatter(coverages, risks, c=accs, cmap="viridis_r", s=70, edgecolors="k", alpha=0.85)
        plt.colorbar(scatter, label="Selective Accuracy")

        # Highlight optimal point
        opt = next((r for r in sweep_results if r["tau"] == optimal_tau and r["delta"] == optimal_delta), None)
        if opt:
            plt.scatter(
                [opt["coverage"]],
                [opt["selective_risk"]],
                color="red",
                s=160,
                marker="*",
                label=f"Selected (tau={optimal_tau}, delta={optimal_delta})",
                zorder=5,
            )

        plt.title("Risk-Coverage Curve (Selective Classification)", fontsize=12, fontweight="bold")
        plt.xlabel("Coverage (Fraction of Articles Triaged)", fontsize=11)
        plt.ylabel("Selective Risk (Error Rate on Accepted Articles)", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left")
        plt.tight_layout()

        out_path = save_path or (FIGURES_DIR / "risk_coverage_curve.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()

        logger.info(f"Risk-coverage curve saved to {out_path}")
        return out_path
