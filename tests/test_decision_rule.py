"""
Tests for EvidenceFirstDecisionEngine (all decision rules including the POCSO fix).
Tests the 4 decision rules plus Rule 3b (evidence-absence abstention).
"""

import pytest
from src.decision_rule import EvidenceFirstDecisionEngine, UNVERIFIED_ABSTAIN_THRESHOLD, STYLE_FAKE_CONFIDENCE_FOR_ABSTAIN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Default decision engine with typical threshold values."""
    return EvidenceFirstDecisionEngine(tau=0.80, delta=0.20, rho=0.50)


def _evidence(
    pct_supp=0.0, n_supp_domains=0, has_contra=False, has_num_conflict=False,
    total_claims=4, n_unverified=0
):
    """Helper to build a minimal evidence_summary dict."""
    return {
        "percent_supported": pct_supp,
        "trusted_domains_count": n_supp_domains,
        "has_any_contradiction": has_contra,
        "has_numeric_conflict": has_num_conflict,
        "total_claims": total_claims,
        "unverified_count": n_unverified,
    }


def _multiview(mean_p_fake=0.50, view_dis=0.05, p_style=None):
    return {
        "mean_p_fake": mean_p_fake,
        "view_disagreement": view_dis,
        "p_style": p_style if p_style is not None else mean_p_fake,
    }


def _ood(is_ood=False, ood_score=0.0, vocab_cov=0.95):
    return {
        "is_ood": is_ood,
        "ood_score": ood_score,
        "vocab_coverage": vocab_cov,
    }


# ---------------------------------------------------------------------------
# Rule 1: Strong corroboration → REAL
# ---------------------------------------------------------------------------

class TestRule1Corroboration:
    def test_corroborated_gives_real(self, engine):
        ev = _evidence(pct_supp=0.75, n_supp_domains=3)
        mv = _multiview(mean_p_fake=0.30)
        ood = _ood()
        res = engine.evaluate(ev, mv, ood)
        assert res["decision"] == "REAL"
        assert res["trigger"] == "evidence_corroborated"
        assert res["confidence"] >= 0.75

    def test_corroborated_but_style_says_fake_flags_conflict(self, engine):
        """Rule 1 should still return REAL but report style-evidence conflict."""
        ev = _evidence(pct_supp=0.80, n_supp_domains=4)
        mv = _multiview(mean_p_fake=0.85, p_style=0.85)
        ood = _ood()
        res = engine.evaluate(ev, mv, ood)
        assert res["decision"] == "REAL"
        assert res["style_evidence_conflict"] is True
        assert res["conflict_explanation"] is not None

    def test_corroboration_threshold_just_below_fails(self, engine):
        """55% support and only 1 domain does NOT trigger Rule 1."""
        ev = _evidence(pct_supp=0.55, n_supp_domains=1)
        mv = _multiview(mean_p_fake=0.30, view_dis=0.05)
        ood = _ood()
        res = engine.evaluate(ev, mv, ood)
        # Rule 1 not triggered; falls through to Rule 4
        assert res["decision"] != "REAL" or res["trigger"] != "evidence_corroborated"

    def test_corroboration_blocked_by_contradiction(self, engine):
        """Even if pct_supp >= 60%, a contradiction must block Rule 1."""
        ev = _evidence(pct_supp=0.70, n_supp_domains=3, has_contra=True)
        mv = _multiview(mean_p_fake=0.30)
        ood = _ood()
        res = engine.evaluate(ev, mv, ood)
        assert res["decision"] == "FAKE"
        assert res["trigger"] == "evidence_contradicted"


# ---------------------------------------------------------------------------
# Rule 2: Contradiction → FAKE
# ---------------------------------------------------------------------------

class TestRule2Contradiction:
    def test_direct_contradiction_gives_fake(self, engine):
        ev = _evidence(has_contra=True)
        mv = _multiview(mean_p_fake=0.40)
        ood = _ood()
        res = engine.evaluate(ev, mv, ood)
        assert res["decision"] == "FAKE"
        assert res["trigger"] == "evidence_contradicted"
        assert res["confidence"] == 0.95

    def test_numeric_conflict_gives_fake(self, engine):
        ev = _evidence(has_num_conflict=True)
        mv = _multiview(mean_p_fake=0.40)
        ood = _ood()
        res = engine.evaluate(ev, mv, ood)
        assert res["decision"] == "FAKE"
        assert res["trigger"] == "evidence_contradicted"

    def test_numeric_conflict_overrides_style_real(self, engine):
        """Even if style says REAL (p_fake=0.1), a numeric clash must still yield FAKE."""
        ev = _evidence(has_num_conflict=True)
        mv = _multiview(mean_p_fake=0.10)
        ood = _ood()
        res = engine.evaluate(ev, mv, ood)
        assert res["decision"] == "FAKE"


# ---------------------------------------------------------------------------
# Rule 3: No evidence + OOD → UNCERTAIN
# ---------------------------------------------------------------------------

class TestRule3OOD:
    def test_no_evidence_and_ood_gives_uncertain(self, engine):
        ev = _evidence(pct_supp=0.0, n_supp_domains=0, total_claims=4, n_unverified=0)
        mv = _multiview(mean_p_fake=0.80)
        ood = _ood(is_ood=True, ood_score=0.72, vocab_cov=0.35)
        res = engine.evaluate(ev, mv, ood)
        assert res["decision"] == "UNCERTAIN"
        assert res["trigger"] == "domain_shift_no_evidence"

    def test_ood_without_no_evidence_does_not_trigger_rule3(self, engine):
        """If some domains support claims, Rule 3 shouldn't fire even if OOD."""
        ev = _evidence(pct_supp=0.50, n_supp_domains=2)
        mv = _multiview(mean_p_fake=0.85, view_dis=0.05)
        ood = _ood(is_ood=True, ood_score=0.72)
        res = engine.evaluate(ev, mv, ood)
        # Should NOT be domain_shift_no_evidence since we have some support
        assert res["trigger"] != "domain_shift_no_evidence"


# ---------------------------------------------------------------------------
# Rule 3b: Evidence absence + high p_fake → UNCERTAIN (POCSO fix)
# ---------------------------------------------------------------------------

class TestRule3bEvidenceAbstention:
    def test_all_claims_unverified_and_high_pfake_gives_uncertain(self, engine):
        """
        This is the exact scenario that produced P(FAKE)=95.9% on the POCSO court article.
        With 100% unverified claims and high style confidence, Rule 3b must abstain.
        """
        ev = _evidence(total_claims=5, n_unverified=5, pct_supp=0.0, n_supp_domains=0)
        mv = _multiview(mean_p_fake=0.959)
        ood = _ood(is_ood=False)  # Even without OOD flag, should still abstain
        res = engine.evaluate(ev, mv, ood)
        assert res["decision"] == "UNCERTAIN"
        assert res["trigger"] == "evidence_absence_abstain"

    def test_partial_unverified_below_threshold_does_not_trigger_3b(self, engine):
        """If only 50% of claims are unverified, Rule 3b should NOT fire."""
        ev = _evidence(total_claims=4, n_unverified=2, pct_supp=0.25, n_supp_domains=1)
        mv = _multiview(mean_p_fake=0.85, view_dis=0.05)
        ood = _ood(is_ood=False)
        res = engine.evaluate(ev, mv, ood)
        # 50% unverified < UNVERIFIED_ABSTAIN_THRESHOLD (0.75) → should NOT fire 3b
        assert res["trigger"] != "evidence_absence_abstain"

    def test_all_unverified_but_low_pfake_does_not_trigger_3b(self, engine):
        """
        If all claims are unverified but style predicts REAL with low P(FAKE),
        Rule 3b should not fire (the style signal is consistent with REAL, not suspicious).
        """
        ev = _evidence(total_claims=4, n_unverified=4, pct_supp=0.0, n_supp_domains=0)
        mv = _multiview(mean_p_fake=0.30)  # below STYLE_FAKE_CONFIDENCE_FOR_ABSTAIN
        ood = _ood(is_ood=False)
        res = engine.evaluate(ev, mv, ood)
        assert res["trigger"] != "evidence_absence_abstain"

    def test_rule_3b_threshold_values_are_correct(self):
        """Verify the sentinel constants match documented design."""
        assert UNVERIFIED_ABSTAIN_THRESHOLD == 0.75
        assert STYLE_FAKE_CONFIDENCE_FOR_ABSTAIN == 0.55


# ---------------------------------------------------------------------------
# Rule 4: In-distribution style fallback
# ---------------------------------------------------------------------------

class TestRule4StyleFallback:
    def test_high_confidence_real_verdict(self, engine):
        ev = _evidence(total_claims=4, n_unverified=1, pct_supp=0.25, n_supp_domains=1)
        mv = _multiview(mean_p_fake=0.15, view_dis=0.05)
        ood = _ood(is_ood=False)
        res = engine.evaluate(ev, mv, ood)
        assert res["decision"] == "REAL"
        assert res["trigger"] == "tri_gate_accepted"

    def test_high_confidence_fake_verdict(self, engine):
        ev = _evidence(total_claims=4, n_unverified=1, pct_supp=0.25, n_supp_domains=1)
        mv = _multiview(mean_p_fake=0.90, view_dis=0.05)
        ood = _ood(is_ood=False)
        res = engine.evaluate(ev, mv, ood)
        assert res["decision"] == "FAKE"
        assert res["trigger"] == "tri_gate_accepted"

    def test_low_confidence_abstains(self, engine):
        """When model confidence is below tau, should abstain (low_confidence trigger)."""
        ev = _evidence(total_claims=4, n_unverified=1, pct_supp=0.25, n_supp_domains=1)
        # p_fake = 0.55 → confidence = max(0.55, 0.45) = 0.55, which is < tau=0.80
        mv = _multiview(mean_p_fake=0.55, view_dis=0.05)
        ood = _ood(is_ood=False)
        res = engine.evaluate(ev, mv, ood)
        assert res["decision"] == "UNCERTAIN"
        assert res["trigger"] == "low_confidence"

    def test_high_disagreement_abstains(self, engine):
        """When view disagreement exceeds delta, should abstain."""
        ev = _evidence(total_claims=4, n_unverified=1, pct_supp=0.25, n_supp_domains=1)
        mv = _multiview(mean_p_fake=0.90, view_dis=0.35)  # > delta=0.20
        ood = _ood(is_ood=False)
        res = engine.evaluate(ev, mv, ood)
        assert res["decision"] == "UNCERTAIN"
        assert res["trigger"] == "high_view_disagreement"


# ---------------------------------------------------------------------------
# Custom threshold overrides
# ---------------------------------------------------------------------------

class TestCustomThresholds:
    def test_custom_tau_override(self, engine):
        """Passing custom_tau=0.50 should lower the bar for acceptance."""
        ev = _evidence(total_claims=4, n_unverified=1, pct_supp=0.25, n_supp_domains=1)
        mv = _multiview(mean_p_fake=0.70, view_dis=0.05)  # conf=0.70, tau default=0.80
        ood = _ood(is_ood=False)
        # Default tau=0.80 → abstain
        res_default = engine.evaluate(ev, mv, ood)
        assert res_default["decision"] == "UNCERTAIN"
        # Custom tau=0.60 → accept
        res_custom = engine.evaluate(ev, mv, ood, custom_tau=0.60)
        assert res_custom["decision"] == "FAKE"
        assert res_custom["trigger"] == "tri_gate_accepted"

    def test_output_schema_keys_always_present(self, engine):
        """All result dicts must contain the standard schema keys."""
        ev = _evidence()
        mv = _multiview()
        ood = _ood()
        res = engine.evaluate(ev, mv, ood)
        required_keys = [
            "decision", "confidence", "p_fake", "p_real",
            "trigger", "qualifier", "explanation",
            "style_evidence_conflict", "disagreement", "tau", "delta", "rho",
        ]
        for k in required_keys:
            assert k in res, f"Missing key: {k}"
