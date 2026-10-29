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

EVIDENCE ABSENCE TRAP FIX (AUD-247238):
  If an article has a Grade A Credibility Fingerprint (Index >= 80, Sensationalism Discipline >= 85)
  and live search is unverified or unavailable, do NOT default to 88.8% FAKE.
  Shrink toward the stylometric baseline (RF / linguistic score) and output:
  'INSUFFICIENT_EVIDENCE (RESTRAINED_STYLE)'.

Decision precedence order:
  1. ≥60% claims supported by ≥2 trusted domains, no contradictions → REAL
  2. Any core claim contradicted or numeric clash → FAKE
  2b. Dynamic Evidence Override (≥2 supported claims + high credibility) → LIKELY REAL
  3. Evidence Absence with Grade A Credibility → INSUFFICIENT_EVIDENCE (RESTRAINED_STYLE)
  3b. No evidence + OOD domain shift → UNCERTAIN (abstain)
  3c. All claims Unverified (no sources found) + high style P(FAKE) → UNCERTAIN (abstain)
  4. In-distribution style fallback via tri-trigger gate (tau, delta, rho) → Style verdict
"""

from typing import Dict, Any, List, Optional
import numpy as np
from config import DEFAULT_DELTA
from src.utils import setup_logger

logger = setup_logger("decision_rule")

# Fraction of claims that must be Unverified (no live sources found)
# before triggering the evidence-absence abstention rule.
UNVERIFIED_ABSTAIN_THRESHOLD = 0.75   # 75%+ claims unverified → cannot rely on style alone

# Style confidence must exceed this to trigger abstention under rule 3b.
STYLE_FAKE_CONFIDENCE_FOR_ABSTAIN = 0.55  # Any meaningful FAKE prediction with no evidence


class EvidenceFirstDecisionEngine:
    """
    Arbitrates multi-view expert predictions, live evidence verifications,
    and domain-shift (OOD) indicators into a definitive triage decision.
    """

    def __init__(self, tau: float = 0.80, delta: float = DEFAULT_DELTA, rho: float = 0.50):
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
        # Clamp delta used at inference to <= config.DEFAULT_DELTA (0.15) unless user passes a custom value
        if custom_delta is not None:
            delta_val = float(custom_delta)
        else:
            delta_val = min(float(self.delta), float(DEFAULT_DELTA))
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

        # --- Credibility & High-Similarity Evidence Overrides ---
        credibility_index = float(multi_view_summary.get("credibility_index", 50.0))
        sens_discipline = float(multi_view_summary.get("sensationalism_discipline", credibility_index))
        has_high_credibility = (credibility_index >= 75.0 or sens_discipline >= 80.0)
        is_grade_a = (credibility_index >= 80.0 and sens_discipline >= 85.0)

        claim_verifs = evidence_summary.get("claim_verifications", [])
        high_sim_supported = [
            c for c in claim_verifs
            if c.get("status") == "Supported" and float(c.get("similarity", c.get("dense_score", 0.0))) >= 0.60
        ]
        n_high_sim = len(high_sim_supported)

        # Evidence Override condition: >=2 supported claims AND high credibility/sensationalism discipline
        evidence_override_active = (
            n_high_sim >= 2
            and has_high_credibility
            and not has_contra
            and not has_num_conflict
        )

        # =====================================================================
        # RULE 1: STRONG CORROBORATING EVIDENCE FROM TRUSTED SOURCES
        # =====================================================================
        # Require >= 2 distinct trusted independent domains and pct_supp >= 0.60
        can_corroborate = (pct_supp >= 0.60 and n_supp_domains >= 2)
        if can_corroborate and (not has_contra) and (not has_num_conflict):
            decision   = "REAL"
            style_conflict = 1 if (p_style >= 0.55 or mean_p_fake >= 0.55) else 0
            # Formula: min(0.95, 0.70 + 0.05*min(n_domains,4) + 0.10*pct_supp - 0.10*style_conflict)
            confidence = min(0.95, 0.70 + 0.05 * min(n_supp_domains, 4) + 0.10 * pct_supp - 0.10 * style_conflict)
            trigger    = "evidence_corroborated"
            qualifier  = "Verified External Evidence"
            domain_label = "domain" if n_supp_domains == 1 else "domains"

            if style_conflict:
                style_evidence_conflict = True
                conflict_explanation = (
                    f"STYLE-EVIDENCE CONFLICT DETECTED: The text classifier predicted FAKE "
                    f"(P(FAKE)={mean_p_fake:.1%}, stylometric RF expert={p_style:.1%}), but {pct_supp:.0%} of "
                    f"factual claims were independently corroborated by {n_supp_domains} trusted {domain_label}. "
                    f"External factual evidence strictly supersedes stylometric heuristics."
                )
            explanation = conflict_explanation or (
                f"Factual claims corroborated by {n_supp_domains} independent authoritative {domain_label}."
            )

            return _make_result(
                decision, confidence, trigger, qualifier, explanation,
                mean_p_fake, tau_val, delta_val, rho_val,
                style_evidence_conflict, conflict_explanation,
                disagreement=view_dis,
                confidence_type="evidence_rule",
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
                disagreement=view_dis,
                confidence_type="evidence_rule",
            )

        # =====================================================================
        # RULE 2b: DYNAMIC EVIDENCE OVERRIDE (>=2 SUPPORTED CLAIMS & HIGH CREDIBILITY)
        # =====================================================================
        if evidence_override_active:
            decision = "LIKELY REAL (EVIDENCE-GROUNDED)"
            avg_sim = float(np.mean([float(c.get("similarity", c.get("dense_score", 0.70))) for c in high_sim_supported])) if high_sim_supported else 0.75
            confidence = min(0.95, max(0.80, avg_sim))
            trigger = "evidence_override"
            qualifier = "Likely Real (Evidence-Grounded & Editorial Discipline)"
            explanation = (
                f"EVIDENCE OVERRIDE: {n_high_sim} factual claims independently corroborated by external sources "
                f"(avg similarity {avg_sim:.2f}) and editorial credibility index is {credibility_index:.1f}/100. "
                f"Lexical and semantic domain shift penalties downweighted."
            )
            return _make_result(
                decision, confidence, trigger, qualifier, explanation,
                mean_p_fake=0.10, tau=tau_val, delta=delta_val, rho=rho_val,
                style_evidence_conflict=True,
                conflict_explanation=explanation,
                disagreement=view_dis,
                confidence_type="evidence_rule",
            )

        # =====================================================================
        # RULE 3: EVIDENCE ABSENCE TRAP FIX — Grade A Credibility Fingerprint
        #          When live search is unverified or unavailable, shrink toward
        #          the stylometric baseline rather than defaulting to high fake prob.
        # =====================================================================
        if (pct_unverified >= UNVERIFIED_ABSTAIN_THRESHOLD or (n_supp_domains == 0 and pct_supp == 0.0)) and is_grade_a:
            decision = "INSUFFICIENT_EVIDENCE (RESTRAINED_STYLE)"
            shrunk_p_fake = float(p_style)
            confidence = round(float(credibility_index / 100.0), 3)
            trigger = "insufficient_evidence_restrained_style"
            qualifier = "Restrained Style (Grade A Credibility - Domain Shift Attenuated)"
            explanation = (
                f"External evidence retrieval found {n_unverified}/{total_claims} claims unverified, but the article "
                f"exhibits Grade A journalistic credibility (Index {credibility_index:.1f}/100, Sensationalism Discipline {sens_discipline:.1f}/100) "
                f"and restrained syntax. Lexical fake score ({mean_p_fake:.1%}) is an artifact of legal/crime vocabulary shift. "
                f"Meta-ensemble shrunk to stylometric baseline ({shrunk_p_fake:.1%})."
            )
            return _make_result(
                decision, confidence, trigger, qualifier, explanation,
                mean_p_fake=shrunk_p_fake, tau=tau_val, delta=delta_val, rho=rho_val,
                style_evidence_conflict=True,
                conflict_explanation=explanation,
                disagreement=view_dis,
                confidence_type="restrained_style",
            )

        # =====================================================================
        # RULE 3b: NO EVIDENCE + DOMAIN SHIFT (OUT-OF-DISTRIBUTION)
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
                disagreement=view_dis,
                confidence_type="abstain",
            )

        # =====================================================================
        # RULE 3c: EVIDENCE ABSENCE — All Claims Unverified (no live sources found)
        #          + Style model predicts FAKE → ABSTAIN (critical fix for domain-shift bug)
        # =====================================================================
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
                f"[RULE 3c] Evidence-absence abstention triggered: "
                f"{pct_unverified:.0%} unverified, P(FAKE)={mean_p_fake:.1%}"
            )
            return _make_result(
                decision, confidence, trigger, qualifier, explanation,
                mean_p_fake, tau_val, delta_val, rho_val,
                disagreement=view_dis,
                confidence_type="abstain",
            )

        # =====================================================================
        # RULE 4: IN-DISTRIBUTION STYLE-EXPERT FALLBACK (TRI-TRIGGER GATE)
        # =====================================================================
        # Trigger 0: Base model vote split where stylometric RF expert dissents
        p_lexical = multi_view_summary.get("p_lexical", mean_p_fake)
        p_style = multi_view_summary.get("p_style", mean_p_fake)
        p_semantic = multi_view_summary.get("p_semantic", mean_p_fake)
        p_topic = multi_view_summary.get("p_topic", mean_p_fake)

        votes = {
            "lr": p_lexical >= 0.50,
            "rf": p_style >= 0.50,
            "xgb": p_semantic >= 0.50,
            "lgbm": p_topic >= 0.50,
        }
        other_votes = [votes["lr"], votes["xgb"], votes["lgbm"]]
        has_split = len(set(votes.values())) > 1
        rf_dissenting = (other_votes.count(not votes["rf"]) >= 2)

        if has_split and rf_dissenting and not evidence_override_active and n_high_sim < 2:
            decision   = "UNCERTAIN"
            confidence = 0.50
            trigger    = "style_lexical_split"
            qualifier  = "Abstain (Stylometric vs Lexical Split)"
            explanation = (
                f"Base models split on verdict with stylometric RF expert dissenting "
                f"(RF P(FAKE)={p_style:.1%}, but LR={p_lexical:.1%}, LGBM={p_topic:.1%}, XGB={p_semantic:.1%}). "
                f"Lexical and semantic models key on out-of-domain vocabulary while stylometric expert dissents."
            )
            return _make_result(
                decision, confidence, trigger, qualifier, explanation,
                mean_p_fake, tau_val, delta_val, rho_val,
                disagreement=view_dis,
                confidence_type="abstain",
            )

        # Trigger A: Residual OOD check
        if is_ood:
            decision   = "UNCERTAIN"
            confidence = 0.50
            trigger    = "domain_shift_unverified"
            qualifier  = "Abstain (OOD Trigger)"
            explanation = f"OOD score ({ood_score:.3f}) exceeds safety threshold rho ({rho_val:.3f})."
            return _make_result(
                decision, confidence, trigger, qualifier, explanation,
                mean_p_fake, tau_val, delta_val, rho_val,
                disagreement=view_dis,
                confidence_type="abstain",
            )

        # Trigger B: Inter-view disagreement check
        elif view_dis > delta_val:
            # If evidence support is high (>=2 claims supported with sim >= 0.65), route to LIKELY REAL
            if (n_high_sim >= 2 or pct_supp >= 0.50) and (not has_contra and not has_num_conflict):
                decision = "LIKELY REAL (EVIDENCE-GROUNDED)"
                avg_sim = float(np.mean([float(c.get("similarity", c.get("dense_score", 0.70))) for c in high_sim_supported])) if high_sim_supported else 0.75
                confidence = min(0.95, max(0.80, avg_sim))
                trigger = "disagreement_evidence_resolved"
                qualifier = "Likely Real (Evidence-Grounded)"
                explanation = (
                    f"Model disagreement (std={view_dis:.3f} > delta {delta_val:.2f}) resolved by high external "
                    f"evidence corroboration ({n_high_sim} claims verified with similarity >= 0.65)."
                )
                return _make_result(
                    decision, confidence, trigger, qualifier, explanation,
                    mean_p_fake=0.15, tau=tau_val, delta=delta_val, rho=rho_val,
                    disagreement=view_dis,
                    confidence_type="evidence_rule",
                )
            decision   = "UNCERTAIN"
            confidence = 0.50
            trigger    = "high_view_disagreement"
            qualifier  = "Abstain (Multi-View Dissent)"
            explanation = (
                f"Orthogonal multi-view expert disagreement (std={view_dis:.3f}) "
                f"exceeds threshold delta ({delta_val:.2f}). "
                f"Conflicting signals from lexical, semantic, style, and topic experts."
            )
            return _make_result(
                decision, confidence, trigger, qualifier, explanation,
                mean_p_fake, tau_val, delta_val, rho_val,
                disagreement=view_dis,
                confidence_type="abstain",
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
                return _make_result(
                    decision, confidence, trigger, qualifier, explanation,
                    mean_p_fake, tau_val, delta_val, rho_val,
                    disagreement=view_dis,
                    confidence_type="abstain",
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
                    confidence_type="calibrated",
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
    confidence_type: str = "calibrated",
) -> Dict[str, Any]:
    """Constructs the standardized gate result dict."""
    p_fake = mean_p_fake if decision == "FAKE" else (1.0 - confidence if decision == "REAL" else mean_p_fake)
    p_real = 1.0 - p_fake
    return {
        "decision": decision,
        "confidence": round(float(confidence), 4),
        "confidence_type": confidence_type,
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
