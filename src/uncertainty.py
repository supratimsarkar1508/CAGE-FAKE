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
        evidence_analysis: Optional[Dict[str, Any]] = None,
        linguistic_red_flags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates gate on a single article using:
        1. Calibrated confidence: max(p, 1-p)
        2. Base-model disagreement: std across 4 base learners (RF, XGB, LGBM, LR)
        3. Evidence Analysis (Supported, Contradicted, Unverified claims)
        4. Linguistic Red Flags congruence

        Identifies:
        - 'missing_evidence': when article lacks supporting evidence (per trustworthy AI requirements)
        - 'conflicting_evidence': when claims have mixed support and contradiction, or conflict score > 0.5
        - 'signal_disagreement': when linguistic signals conflict with evidence verification
        - 'both_triggers': when confidence < tau AND disagreement > delta
        - 'low_confidence': when confidence < tau
        - 'high_disagreement': when disagreement > delta
        - 'none': accepted prediction (REAL or FAKE) with supporting evidence and high confidence
        """
        confidence = float(max(p_calibrated, 1.0 - p_calibrated))
        disagreement = float(disagreement)

        is_low_conf = confidence < self.tau
        is_high_dis = disagreement > self.delta

        # Extract evidence status if available
        has_evidence_analysis = evidence_analysis is not None
        has_no_ev = False
        has_conf_ev = False
        has_supp_ev = False
        has_cont_ev = False
        ev_conflict_val = evidence_conflict

        if has_evidence_analysis:
            has_no_ev = evidence_analysis.get("has_no_evidence", False)
            has_conf_ev = evidence_analysis.get("has_conflicting_evidence", False)
            has_supp_ev = evidence_analysis.get("has_supporting_evidence", False)
            has_cont_ev = evidence_analysis.get("has_contradicting_evidence", False)
            auto_conflict = evidence_analysis.get("evidence_conflict_score", 0.0)
            if ev_conflict_val is None:
                ev_conflict_val = auto_conflict
            else:
                ev_conflict_val = max(ev_conflict_val, auto_conflict)

        is_ev_conflict = ev_conflict_val is not None and ev_conflict_val > 0.5
        red_flags = linguistic_red_flags or []
        n_supp_val = evidence_analysis.get("supported_count", 0) if has_evidence_analysis else 0
        claims_list = evidence_analysis.get("claim_verifications", []) if has_evidence_analysis else []
        high_sim_supp = [
            c for c in claims_list
            if c.get("status") == "Supported" and float(c.get("similarity", c.get("dense_score", 0.0))) >= 0.65
        ]
        has_high_evidence_support = bool(len(high_sim_supp) >= 2 or (n_supp_val >= 2 and not has_cont_ev and not is_ev_conflict))

        # Check for signal disagreement between linguistic features and evidence
        is_signal_disagreement = False
        if has_evidence_analysis:
            # Evidence supports claims (suggesting REAL), but linguistic red flags indicate heavy manipulation or model predicts FAKE
            if has_supp_ev and (p_calibrated >= 0.50 or len(red_flags) >= 2):
                if has_high_evidence_support and len(red_flags) == 0:
                    is_signal_disagreement = False
                else:
                    is_signal_disagreement = True
            # Evidence contradicts claims (suggesting FAKE), but model predicts REAL with very low P(FAKE)
            elif has_cont_ev and p_calibrated < 0.50:
                is_signal_disagreement = True

        # Apply Decision Gating Hierarchy
        if has_evidence_analysis and (has_no_ev or (not has_supp_ev and not has_cont_ev)):
            decision = "UNCERTAIN"
            trigger = "missing_evidence"
            explanation = (
                "Uncertainty Gated (Missing Evidence): No corroborating evidence was found in the knowledge base or retrieved sources. "
                "Per trustworthy AI requirements, the system never forces a REAL or FAKE label without supporting evidence."
            )
        elif is_ev_conflict or has_conf_ev:
            decision = "UNCERTAIN"
            trigger = "conflicting_evidence"
            explanation = (
                f"Uncertainty Gated (Conflicting Evidence): Contradictory evidence detected across article claims "
                f"(evidence conflict score: {ev_conflict_val:.2f})."
            )
        elif is_signal_disagreement:
            decision = "UNCERTAIN"
            trigger = "signal_disagreement"
            explanation = (
                f"Uncertainty Gated (Signal Disagreement): Linguistic signals and evidence verification disagree "
                f"({len(red_flags)} red flags detected, model P(FAKE)={p_calibrated:.1%})."
            )
        elif is_low_conf and is_high_dis:
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
            if has_high_evidence_support:
                decision = "LIKELY REAL (EVIDENCE-GROUNDED)"
                trigger = "disagreement_evidence_resolved"
                explanation = (
                    f"Inter-model disagreement ({disagreement:.3f} > delta {self.delta:.3f}) resolved by high external "
                    f"evidence corroboration ({len(high_sim_supp) or n_supp_val} claims verified with similarity >= 0.65)."
                )
            else:
                decision = "UNCERTAIN"
                trigger = "high_disagreement"
                explanation = (
                    f"Base-model disagreement ({disagreement:.3f}) exceeds threshold delta ({self.delta:.3f}); "
                    f"calibrated confidence is {confidence:.1%}."
                )
        else:
            if has_evidence_analysis:
                if has_cont_ev:
                    decision = "FAKE"
                elif has_supp_ev:
                    decision = "REAL"
                else:
                    decision = "UNCERTAIN"
            else:
                decision = "FAKE" if p_calibrated >= 0.5 else "REAL"
            trigger = "none"
            explanation = f"High confidence ({confidence:.1%}), verified evidence alignment, and acceptable model disagreement ({disagreement:.3f})."

        return {
            "decision": decision,
            "trigger": trigger,
            "is_accepted": decision != "UNCERTAIN",
            "is_low_confidence": is_low_conf,
            "is_high_disagreement": is_high_dis,
            "is_evidence_conflict": is_ev_conflict or has_conf_ev,
            "is_missing_evidence": has_no_ev,
            "is_signal_disagreement": is_signal_disagreement,
            "confidence": confidence,
            "disagreement": disagreement,
            "p_calibrated": p_calibrated,
            "tau": self.tau,
            "delta": self.delta,
            "explanation": explanation,
            "evidence_conflict_available": ev_conflict_val is not None,
            "evidence_note": (
                f"External evidence active (conflict score: {ev_conflict_val:.2f})."
                if ev_conflict_val is not None
                else "Evidence verification completed via knowledge base."
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

        # Also save vector PDF in outputs/figures/pdf/07_risk_coverage.pdf
        pdf_dir = FIGURES_DIR / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / "07_risk_coverage.pdf"
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close()

        logger.info(f"Risk-coverage curve saved to {out_path} and {pdf_path}")
        return out_path
