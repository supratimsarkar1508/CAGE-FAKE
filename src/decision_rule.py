"""
CAGE-Fake: Evidence-First Decision Rule & Tri-Trigger Gating (v2 — Evidence Absence Fix)
=========================================================================================
Implements evidence priority arbitration with a CRITICAL FIX for the "Unverified = Silent"
bug that caused a real POCSO court article to be classified FAKE at 95.9% confidence.

ROOT CAUSE OF BUG:
------------------
  When article claims cannot be found online (domain-shifted content, e.g. Indian court
  reports that never appeared in US political training corpora), the evidence pipeline
  returns status="Unverified" for all claims. The original decision engine treated this
  identically to "no evidence layer" — it fell through to Rule 4 (stylometric fallback)
  and confidently output FAKE based on SVD-compressed US-news features alone.

  This is wrong. "Cannot find corroboration" ≠ "Article is fake".
  It means "We don't know" → system must abstain.

FIX — Rule 3b (Evidence Absence + High P(FAKE)):
  If ALL claims are Unverified (retrieved no sources at all) AND the style model
  predicts FAKE with high confidence → UNCERTAIN (abstain).
  The style signal alone is insufficient when there is ZERO factual corroboration
  and the content is clearly domain-shifted from the training distribution.

Decision precedence order:
  1. ≥60% claims supported by ≥2 trusted domains, no contradictions → REAL
  2. Any core claim contradicted or numeric clash → FAKE
  3. No evidence + OOD domain shift → UNCERTAIN (abstain)
  3b. All claims Unverified (no sources found) + high style P(FAKE) → UNCERTAIN (abstain)
  4. In-distribution style fallback via tri-trigger gate (tau, delta, rho) → Style verdict
"""

from typing import Dict, Any, List, Optional
from src.utils import setup_logger

logger = setup_logger("decision_rule")

# Fraction of claims that must be Unverified (no live sources found)
# before triggering the evidence-absence abstention rule.
UNVERIFIED_ABSTAIN_THRESHOLD = 0.75   # 75%+ claims unverified → cannot rely on style alone

# Style confidence must exceed this to trigger abstention under rule 3b.
# (If style model is also unsure, it would abstain via low_confidence trigger anyway.)
STYLE_FAKE_CONFIDENCE_FOR_ABSTAIN = 0.55  # Any meaningful FAKE prediction with no evidence


class EvidenceFirstDecisionEngine:
    """
    Arbitrates multi-view expert predictions, live evidence verifications,
    and domain-shift (OOD) indicators into a definitive triage decision.
    """

    def __init__(self, tau: float = 0.80, delta: float = 0.20, rho: float = 0.50):
        self.tau = tau      # Calibrated confidence threshold
        self.delta = delta  # Multi-view disagreement threshold
        self.rho = rho      # Out-of-distribution domain shift threshold

    def evaluate(
        self,
        evidence_summary: Dict[str, Any],
        multi_view_summary: Dict[str, Any],
        ood_summary: Dict[str, Any],
        custom_tau: Optional[float] = None,
        custom_delta: Optional[float] = None,
        custom_rho: Optional[float] = None,
    ) -> Dict[str, Any]:
        tau_val   = custom_tau   if custom_tau   is not None else self.tau
        delta_val = custom_delta if custom_delta is not None else self.delta
        rho_val   = custom_rho   if custom_rho   is not None else self.rho

        # --- Evidence signals ---
        pct_supp       = evidence_summary.get("percent_supported", 0.0)
        n_supp_domains = evidence_summary.get("trusted_domains_count", 0)
        has_contra     = evidence_summary.get("has_any_contradiction", False)
        has_num_conflict = evidence_summary.get("has_numeric_conflict", False)
        total_claims   = evidence_summary.get("total_claims", 0)
        n_unverified   = evidence_summary.get("unverified_count", 0)

        # Fraction of claims that had ZERO live sources found
        pct_unverified = (n_unverified / total_claims) if total_claims > 0 else 1.0

        # --- Style signals ---
        mean_p_fake = multi_view_summary.get("mean_p_fake", 0.50)
        view_dis    = multi_view_summary.get("view_disagreement", 0.0)
        p_style     = multi_view_summary.get("p_style", mean_p_fake)

        # --- OOD signals ---
        is_ood     = ood_summary.get("is_ood", False)
        ood_score  = ood_summary.get("ood_score", 0.0)
        vocab_cov  = ood_summary.get("vocab_coverage", 1.0)

        style_evidence_conflict = False
        conflict_explanation = None

        # =====================================================================
        # RULE 1: STRONG CORROBORATING EVIDENCE FROM TRUSTED SOURCES
        # =====================================================================
        if (pct_supp >= 0.60) and (n_supp_domains >= 2) and (not has_contra) and (not has_num_conflict):
            decision   = "REAL"
            confidence = max(0.92, float(pct_supp))
            trigger    = "evidence_corroborated"
            qualifier  = "Verified External Evidence"

            if p_style >= 0.55 or mean_p_fake >= 0.55:
                style_evidence_conflict = True
                conflict_explanation = (
                    f"STYLE-EVIDENCE CONFLICT DETECTED: Style experts predicted FAKE "
                    f"(P(FAKE)={p_style:.1%}, ensemble mean {mean_p_fake:.1%}), but {pct_supp:.0%} of "
                    f"factual claims were independently corroborated by {n_supp_domains} trusted domains. "
                    f"External factual evidence strictly supersedes stylometric heuristics."
                )
            explanation = conflict_explanation or (
                f"Factual claims corroborated by {n_supp_domains} independent authoritative domains."
            )

            return _make_result(
                decision, confidence, trigger, qualifier, explanation,
                mean_p_fake, tau_val, delta_val, rho_val,
                style_evidence_conflict, conflict_explanation,
            )

        # =====================================================================
        # RULE 2: CORE CLAIM CONTRADICTION OR NUMERIC/ENTITY CLASH
        # =====================================================================
        if has_contra or has_num_conflict:
            decision   = "FAKE"
            confidence = 0.95
            trigger    = "evidence_contradicted"
            qualifier  = "Authoritative Refutation / Numeric Discrepancy"

            contra_details = []
            if has_contra:
                contra_details.append(
                    "Core claims refuted by authoritative external fact-checkers or primary sources"
                )
            if has_num_conflict:
                contra_details.append(
                    "Numerical figures or dates clash directly with corroborated reporting"
                )
            explanation = " | ".join(contra_details)

            return _make_result(
                decision, confidence, trigger, qualifier, explanation,
                mean_p_fake, tau_val, delta_val, rho_val,
            )

        # =====================================================================
        # RULE 3: NO EVIDENCE + DOMAIN SHIFT (OUT-OF-DISTRIBUTION)
        # =====================================================================
        if (n_supp_domains == 0 or pct_supp == 0.0) and is_ood:
            decision   = "UNCERTAIN"
            confidence = 0.50
            trigger    = "domain_shift_no_evidence"
            qualifier  = "Abstain (Domain Shift & Zero Corroboration)"
            explanation = (
                f"Article exhibits severe domain shift (OOD score {ood_score:.3f} >= rho {rho_val:.3f}, "
                f"vocab coverage {vocab_cov:.1%}) with zero corroborating evidence found. "
                f"System abstains to avoid confident out-of-domain classification errors."
            )
            return _make_result(
                decision, confidence, trigger, qualifier, explanation,
                mean_p_fake, tau_val, delta_val, rho_val,
            )

        # =====================================================================
        # RULE 3b: EVIDENCE ABSENCE — All Claims Unverified (no live sources found)
        #          + Style model predicts FAKE → ABSTAIN (critical fix for domain-shift bug)
        # =====================================================================
        # WHY: When the evidence pipeline finds NO live sources for most claims, it means
        # the article topic is absent from the web indices or retriever pool. This is NOT
        # evidence of fakeness; it is absence of evidence. If the style model also happens
        # to fire FAKE (e.g., because legal court vocabulary statistically resembles sensational
        # political fake-news in the training corpus), we would produce a confident but
        # groundless FAKE verdict. This rule catches that case and forces abstention.
        if pct_unverified >= UNVERIFIED_ABSTAIN_THRESHOLD and mean_p_fake >= STYLE_FAKE_CONFIDENCE_FOR_ABSTAIN:
            decision   = "UNCERTAIN"
            confidence = 0.50
            trigger    = "evidence_absence_abstain"
            qualifier  = "Abstain (Evidence Absent — Cannot Validate Style Prediction)"
            explanation = (
                f"{n_unverified}/{total_claims} claims ({pct_unverified:.0%}) returned no live "
                f"corroborating sources. Style model predicts FAKE ({mean_p_fake:.1%}) but this "
                f"signal cannot be trusted without factual grounding — the vocabulary may simply "
                f"be absent from the training distribution (domain shift). "
                f"System abstains; manual editorial review required."
            )
            logger.warning(
                f"[RULE 3b] Evidence-absence abstention triggered: "
                f"{pct_unverified:.0%} unverified, P(FAKE)={mean_p_fake:.1%}"
            )
            return _make_result(
                decision, confidence, trigger, qualifier, explanation,
                mean_p_fake, tau_val, delta_val, rho_val,
            )

        # =====================================================================
        # RULE 4: IN-DISTRIBUTION STYLE-EXPERT FALLBACK (TRI-TRIGGER GATE)
        # =====================================================================
        # Only reached when:
        #   - No strong corroboration (Rule 1 missed)
        #   - No contradiction (Rule 2 missed)
        #   - Not OOD (Rule 3 missed)
        #   - Some evidence was found but inconclusive (evidence_absence rule didn't fire)

        # Trigger A: Residual OOD check
        if is_ood:
            decision   = "UNCERTAIN"
            confidence = 0.50
            trigger    = "domain_shift_unverified"
            qualifier  = "Abstain (OOD Trigger)"
            explanation = f"OOD score ({ood_score:.3f}) exceeds safety threshold rho ({rho_val:.3f})."

        # Trigger B: Inter-view disagreement check (now genuine with orthogonal views)
        elif view_dis > delta_val:
            decision   = "UNCERTAIN"
            confidence = 0.50
            trigger    = "high_view_disagreement"
            qualifier  = "Abstain (Multi-View Dissent)"
            explanation = (
                f"Orthogonal multi-view expert disagreement (std={view_dis:.3f}) "
                f"exceeds threshold delta ({delta_val:.2f}). "
                f"Conflicting signals from lexical, semantic, style, and topic experts."
            )

        # Trigger C: Confidence threshold check
        else:
            conf = max(mean_p_fake, 1.0 - mean_p_fake)
            if conf < tau_val:
                decision   = "UNCERTAIN"
                confidence = round(conf, 4)
                trigger    = "low_confidence"
                qualifier  = "Abstain (Low Confidence)"
                explanation = (
                    f"Model confidence ({conf:.1%}) falls below required threshold tau ({tau_val:.1%})."
                )
            else:
                decision   = "FAKE" if mean_p_fake >= 0.50 else "REAL"
                confidence = round(conf, 4)
                trigger    = "tri_gate_accepted"
                qualifier  = "Stylometric Pattern (Partial Evidence)"
                explanation = (
                    f"In-distribution multi-view consensus passes tri-trigger safety gate "
                    f"(conf={conf:.1%} ≥ tau={tau_val:.1%}, dis={view_dis:.3f} ≤ delta={delta_val:.2f})."
                )

        return _make_result(
            decision, confidence, trigger, qualifier, explanation,
            mean_p_fake, tau_val, delta_val, rho_val,
            disagreement=view_dis,
        )


def _make_result(
    decision: str,
    confidence: float,
    trigger: str,
    qualifier: str,
    explanation: str,
    mean_p_fake: float,
    tau: float,
    delta: float,
    rho: float,
    style_evidence_conflict: bool = False,
    conflict_explanation: Optional[str] = None,
    disagreement: float = 0.0,
) -> Dict[str, Any]:
    """Constructs the standardized gate result dict."""
    p_fake = mean_p_fake if decision == "FAKE" else (1.0 - confidence if decision == "REAL" else mean_p_fake)
    p_real = 1.0 - p_fake
    return {
        "decision": decision,
        "confidence": round(float(confidence), 4),
        "p_fake": round(float(p_fake), 4),
        "p_real": round(float(p_real), 4),
        "trigger": trigger,
        "qualifier": qualifier,
        "explanation": explanation,
        "style_evidence_conflict": style_evidence_conflict,
        "conflict_explanation": conflict_explanation,
        "disagreement": round(float(disagreement), 4),
        "tau": tau,
        "delta": delta,
        "rho": rho,
    }
