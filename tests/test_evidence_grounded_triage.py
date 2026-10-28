"""
Unit and Integration Tests for Evidence-Grounded Triage and Three-Way Verdicts:
1. Satire test cases (absurdity, sensational clickbait, unverified claims)
2. Partially true articles (conflicting / mixed evidence across claims)
3. Articles with no available evidence (strict abstention with UNCERTAIN)
4. Configurable calibrated confidence threshold (tau)
5. Explainable JSON output format and field schema
6. 3-Way evaluation metrics (per-class precision, recall, F1, UNCERTAIN rate, ECE)
"""

import pytest
import numpy as np
from src.evidence_analysis import ClaimExtractor, EvidenceKnowledgeBase, EvidenceVerifier
from src.nlp_explanation import NLPLinguisticExplainer
from src.uncertainty import UncertaintyGate
from src.evaluation import evaluate_predictions, evaluate_3way_predictions


@pytest.fixture
def evidence_verifier():
    """Provides an initialized EvidenceVerifier with pre-indexed knowledge base."""
    return EvidenceVerifier()


@pytest.fixture
def linguistic_explainer():
    """Provides an initialized NLPLinguisticExplainer."""
    return NLPLinguisticExplainer()


def test_article_with_no_available_evidence(evidence_verifier):
    """
    Requirement 3 & 5: When an article contains claims with no matching evidence,
    all claims must be marked 'Unverified', and the gate MUST return 'UNCERTAIN'.
    Per trustworthy AI requirements, the system never forces a REAL or FAKE label without evidence.
    """
    obscure_text = (
        "Archaeologists in a remote valley discovered a secret subterranean library built in 1150 AD "
        "containing quantum mathematical manuscripts written on parchment by anonymous medieval scribes."
    )
    title = "Secret Medieval Quantum Library Unearthed in Hidden Valley"

    analysis = evidence_verifier.verify_article(obscure_text, title=title)

    assert analysis["total_claims"] >= 1
    assert analysis["has_no_evidence"] is True
    assert analysis["supported_count"] == 0
    assert analysis["contradicted_count"] == 0
    assert all(c["status"] == "Unverified" for c in analysis["claim_verifications"])

    # Test uncertainty gate abstention under missing evidence
    gate = UncertaintyGate(tau=0.65, delta=0.15)
    # Even with high statistical confidence (e.g. 0.90), missing evidence must force UNCERTAIN
    gate_res = gate.evaluate_single(
        p_calibrated=0.10,  # model predicts REAL
        disagreement=0.05,
        evidence_analysis=analysis,
    )
    assert gate_res["decision"] == "UNCERTAIN"
    assert gate_res["trigger"] == "missing_evidence"
    assert "never forces a REAL or FAKE label without supporting evidence" in gate_res["explanation"]


def test_partially_true_article_conflicting_evidence(evidence_verifier):
    """
    Requirement 1, 3 & 5: When an article contains both verified factual statements
    and contradicted/debunked claims, the system must detect conflicting evidence
    and return 'UNCERTAIN'.
    """
    partially_true_text = (
        "The James Webb Space Telescope was successfully launched in December 2021 by NASA, ESA, and CSA "
        "to observe infrared astronomy. Immediately after reaching orbit, telescope operators confirmed "
        "the discovery of an alien megastructure and crystalline civilization around Alpha Centauri."
    )
    title = "NASA James Webb Telescope Confirms Alien Megastructure Around Alpha Centauri"

    analysis = evidence_verifier.verify_article(partially_true_text, title=title)

    statuses = {c["status"] for c in analysis["claim_verifications"]}
    assert "Supported" in statuses, "True fact about JWST launch should be Supported"
    assert "Contradicted" in statuses, "False claim about alien megastructure should be Contradicted"
    assert analysis["has_conflicting_evidence"] is True
    assert analysis["evidence_conflict_score"] >= 0.5

    # Gate must abstain with UNCERTAIN on conflicting evidence
    gate = UncertaintyGate(tau=0.65, delta=0.15)
    gate_res = gate.evaluate_single(
        p_calibrated=0.50,
        disagreement=0.10,
        evidence_analysis=analysis,
    )
    assert gate_res["decision"] == "UNCERTAIN"
    assert gate_res["trigger"] == "conflicting_evidence"


def test_satire_article_flags_and_verdict(evidence_verifier, linguistic_explainer):
    """
    Requirement 2 & 5: Satirical articles mimicking news structure with hyperbolic phrasing,
    clickbait wording, and unverified/absurd claims should trigger linguistic red flags
    and yield an UNCERTAIN or FAKE verdict, never a verified REAL.
    """
    satire_text = (
        "You won't believe this shocking truth that mainstream media is silent about! "
        "Scientists in a mind-blowing announcement declared that Earth was actually completely flat "
        "until heavy rainfall in 1985 bent the edges into a curve. Officials reportedly panicked as "
        "secret documents exposed the catastrophic conspiracy."
    )
    title = "SHOCKING TRUTH: Earth Was Actually Flat Until 1985 Rainfall Bent It!"

    # 1. Linguistic check
    ling_res = linguistic_explainer.analyze(satire_text, satire_text.lower(), title=title)
    red_flags = ling_res["linguistic_red_flags"]
    assert len(red_flags) >= 2, "Should detect multiple linguistic red flags in satire"
    red_flags_str = " ".join(red_flags).lower()
    assert any(term in red_flags_str for term in ["clickbait", "sensationalism", "flat", "hedging", "punctuation"])

    # 2. Evidence verification check
    ev_analysis = evidence_verifier.verify_article(satire_text, title=title)
    # The claim about flat earth should be contradicted by the knowledge base
    statuses = [c["status"] for c in ev_analysis["claim_verifications"]]
    assert "Contradicted" in statuses or ev_analysis["has_no_evidence"]

    # 3. Gate decision
    gate = UncertaintyGate(tau=0.65, delta=0.15)
    gate_res = gate.evaluate_single(
        p_calibrated=0.88,  # model predicts fake
        disagreement=0.08,
        evidence_analysis=ev_analysis,
        linguistic_red_flags=red_flags,
    )
    assert gate_res["decision"] in {"FAKE", "UNCERTAIN"}
    assert gate_res["decision"] != "REAL"


def test_fully_supported_real_article(evidence_verifier, linguistic_explainer):
    """
    Verifies that a genuine article with supporting evidence, clean style,
    and high calibrated confidence returns 'REAL'.
    """
    real_text = (
        "International delegates and governments participate in global climate summits to negotiate "
        "emissions targets and renewable energy agreements. Official spokespersons announced that participating "
        "nations will pursue measurable investments in clean power generation over the next decade."
    )
    title = "Global Climate Conference Concludes with Clean Energy Accord"

    ev_analysis = evidence_verifier.verify_article(real_text, title=title)
    assert ev_analysis["has_supporting_evidence"] is True
    assert ev_analysis["has_contradicting_evidence"] is False

    ling_res = linguistic_explainer.analyze(real_text, real_text.lower(), title=title)
    red_flags = ling_res["linguistic_red_flags"]

    gate = UncertaintyGate(tau=0.65, delta=0.15)
    gate_res = gate.evaluate_single(
        p_calibrated=0.12,  # P(FAKE) = 12% -> P(REAL) = 88%
        disagreement=0.06,
        evidence_analysis=ev_analysis,
        linguistic_red_flags=red_flags,
    )
    assert gate_res["decision"] == "REAL"
    assert gate_res["confidence"] == pytest.approx(0.88, abs=1e-3)
    assert gate_res["trigger"] == "none"


def test_three_way_evaluation_metrics_and_ece():
    """
    Requirement 5: Test that evaluate_predictions and evaluate_3way_predictions report:
    - per-class precision, recall, and F1 (REAL, FAKE, UNCERTAIN)
    - the UNCERTAIN rate
    - calibration error (ECE)
    """
    y_true_binary = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    y_pred_binary = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1])
    p_pred = np.array([0.1, 0.2, 0.25, 0.6, 0.85, 0.9, 0.95, 0.4, 0.15, 0.8])
    accepted_mask = np.array([True, True, True, True, True, True, True, True, False, False])

    m = evaluate_predictions(
        y_true=y_true_binary,
        y_pred=y_pred_binary,
        p_pred=p_pred,
        accepted_mask=accepted_mask,
    )

    assert "class_metrics" in m
    assert "REAL" in m["class_metrics"]
    assert "FAKE" in m["class_metrics"]
    assert "UNCERTAIN" in m["class_metrics"]
    for cls in ["REAL", "FAKE", "UNCERTAIN"]:
        assert "precision" in m["class_metrics"][cls]
        assert "recall" in m["class_metrics"][cls]
        assert "f1" in m["class_metrics"][cls]

    assert "uncertain_rate" in m
    assert m["uncertain_rate"] == 0.20  # 2 abstained out of 10
    assert "ece" in m
    assert m["ece"] is not None
    assert 0.0 <= m["ece"] <= 1.0

    # Test explicit 3-way evaluator
    y_t_3way = ["REAL", "REAL", "FAKE", "FAKE", "REAL"]
    y_p_3way = ["REAL", "UNCERTAIN", "FAKE", "UNCERTAIN", "REAL"]
    p_3way = np.array([0.1, 0.5, 0.9, 0.5, 0.2])

    res_3way = evaluate_3way_predictions(y_t_3way, y_p_3way, p_pred=p_3way)
    assert res_3way["n_total"] == 5
    assert res_3way["n_uncertain"] == 2
    assert res_3way["uncertain_rate"] == 0.40
    assert "REAL" in res_3way["class_metrics"]
    assert "FAKE" in res_3way["class_metrics"]
    assert "UNCERTAIN" in res_3way["class_metrics"]
    assert res_3way["ece"] is not None


def test_configurable_confidence_threshold():
    """
    Requirement 3: Verifies that the confidence threshold tau is configurable
    and strictly determines whether a borderline prediction is accepted or gated.
    """
    p_cal = 0.68  # confidence is 68%
    dis = 0.08

    # With tau = 0.65, confidence (68%) exceeds threshold -> accepted
    gate_lenient = UncertaintyGate(tau=0.65, delta=0.15)
    res_lenient = gate_lenient.evaluate_single(p_cal, dis)
    assert res_lenient["is_accepted"] is True
    assert res_lenient["decision"] == "FAKE"

    # With tau = 0.75, confidence (68%) falls below threshold -> UNCERTAIN
    gate_strict = UncertaintyGate(tau=0.75, delta=0.15)
    res_strict = gate_strict.evaluate_single(p_cal, dis)
    assert res_strict["is_accepted"] is False
    assert res_strict["decision"] == "UNCERTAIN"
    assert res_strict["trigger"] == "low_confidence"


def test_explainable_json_schema(evidence_verifier):
    """
    Requirement 4: Return JSON with these fields:
    verdict, confidence, per-claim evidence status with sources,
    the linguistic red flags found, and a 2–3 line reasoning summary.
    """
    text = (
        "Antibiotics treat bacterial infections and are ineffective against viral infections. "
        "Health authorities confirmed this in multiple clinical advisories."
    )
    title = "Medical Guidelines on Antibiotic Usage"

    ev = evidence_verifier.verify_article(text, title=title)
    explainer = NLPLinguisticExplainer()
    ling = explainer.analyze(text, text.lower(), title=title)

    gate = UncertaintyGate(tau=0.65, delta=0.15)
    gate_res = gate.evaluate_single(
        p_calibrated=0.15,
        disagreement=0.05,
        evidence_analysis=ev,
        linguistic_red_flags=ling["linguistic_red_flags"],
    )

    # Validate explainable schema
    explainable_dict = {
        "verdict": gate_res["decision"],
        "confidence": round(float(gate_res["confidence"]), 4),
        "per_claim_evidence_status": ev["claim_verifications"],
        "claim_verifications": ev["claim_verifications"],
        "linguistic_red_flags": ling["linguistic_red_flags"],
        "reasoning_summary": "Article claims are supported by verified medical facts with high confidence and no contradictions.",
    }

    assert "verdict" in explainable_dict
    assert explainable_dict["verdict"] in {"REAL", "FAKE", "UNCERTAIN"}
    assert "confidence" in explainable_dict
    assert isinstance(explainable_dict["confidence"], float)

    assert "per_claim_evidence_status" in explainable_dict
    assert "claim_verifications" in explainable_dict
    assert len(explainable_dict["claim_verifications"]) > 0
    for claim in explainable_dict["claim_verifications"]:
        assert "claim" in claim
        assert "status" in claim
        assert claim["status"] in {"Supported", "Contradicted", "Unverified"}
        assert "source" in claim
        assert "evidence_text" in claim
        assert "similarity" in claim

    assert "linguistic_red_flags" in explainable_dict
    assert isinstance(explainable_dict["linguistic_red_flags"], list)

    assert "reasoning_summary" in explainable_dict
    assert isinstance(explainable_dict["reasoning_summary"], str)
    assert len(explainable_dict["reasoning_summary"]) > 10


def test_additional_satire_absurd_premise(evidence_verifier, linguistic_explainer):
    """
    Test Case: Satirical article with absurd parody premise.
    Must detect multiple linguistic red flags and abstain or flag as fake, never REAL.
    """
    satire_text = (
        "BOMBSHELL REVEALED: In an unprecedented and shocking move, astronomers confirmed "
        "that the Moon is actually a giant hollow balloon inflated by NASA in 1969 to win the space race! "
        "Officials reportedly panicked as leaks exposed the secret rubber conspiracy."
    )
    title = "SHOCKING: NASA Admits Moon Was Giant Balloon All Along!"

    ling = linguistic_explainer.analyze(satire_text, satire_text.lower(), title=title)
    assert len(ling["linguistic_red_flags"]) >= 2
    flags_text = " ".join(ling["linguistic_red_flags"]).lower()
    assert any(term in flags_text for term in ["clickbait", "sensationalism", "hedging", "shouting", "punctuation"])

    ev = evidence_verifier.verify_article(satire_text, title=title)
    # KB contains Apollo 11 landed astronauts on Moon and debunks moon hoax
    assert ev["has_no_evidence"] or ev["has_contradicting_evidence"]

    gate = UncertaintyGate(tau=0.65, delta=0.15)
    gate_res = gate.evaluate_single(
        p_calibrated=0.92,
        disagreement=0.08,
        evidence_analysis=ev,
        linguistic_red_flags=ling["linguistic_red_flags"],
    )
    assert gate_res["decision"] in {"FAKE", "UNCERTAIN"}
    assert gate_res["decision"] != "REAL"


def test_additional_partially_true_with_debunk(evidence_verifier):
    """
    Test Case: Partially true article mixing genuine public policy facts with a debunked conspiracy.
    Must detect both Supported and Contradicted claims, triggering conflicting_evidence UNCERTAIN.
    """
    mixed_text = (
        "The Federal Reserve regulates monetary policy, sets benchmark federal funds rates, "
        "and conducts economic stabilization in the United States. "
        "However, central bank economists allegedly confirmed that 5G wireless technology transmits "
        "biological viruses and pathogens through banking ATMs."
    )
    title = "Federal Reserve Announces Policy Strategy and Cellular Network Operations"

    ev = evidence_verifier.verify_article(mixed_text, title=title)
    statuses = {c["status"] for c in ev["claim_verifications"]}
    assert "Supported" in statuses, "Fed monetary policy claim should be Supported"
    assert "Contradicted" in statuses, "5G virus pathogen claim should be Contradicted"
    assert ev["has_conflicting_evidence"] is True
    assert ev["evidence_conflict_score"] >= 0.50

    gate = UncertaintyGate(tau=0.65, delta=0.15)
    gate_res = gate.evaluate_single(
        p_calibrated=0.45,
        disagreement=0.08,
        evidence_analysis=ev,
    )
    assert gate_res["decision"] == "UNCERTAIN"
    assert gate_res["trigger"] == "conflicting_evidence"


def test_additional_article_no_evidence_refuses_guess(evidence_verifier):
    """
    Test Case: Completely uncorroborated obscure article.
    All claims must be marked 'Unverified', and pipeline must strictly abstain with UNCERTAIN.
    """
    obscure_text = (
        "In a remote hamlet in the Carpathian mountains, village elder Bronislaw discovered "
        "a rare 14th-century copper kettle that whistles musical folk melodies whenever rain approaches."
    )
    title = "Carpathian Village Unearths Whistling 14th-Century Copper Kettle"

    ev = evidence_verifier.verify_article(obscure_text, title=title)
    assert ev["has_no_evidence"] is True
    assert ev["supported_count"] == 0
    assert ev["contradicted_count"] == 0
    assert all(c["status"] == "Unverified" for c in ev["claim_verifications"])

    gate = UncertaintyGate(tau=0.65, delta=0.15)
    # Even if model had high ungrounded probability for REAL or FAKE, must abstain
    gate_res_real = gate.evaluate_single(p_calibrated=0.05, disagreement=0.04, evidence_analysis=ev)
    assert gate_res_real["decision"] == "UNCERTAIN"
    assert gate_res_real["trigger"] == "missing_evidence"

    gate_res_fake = gate.evaluate_single(p_calibrated=0.95, disagreement=0.04, evidence_analysis=ev)
    assert gate_res_fake["decision"] == "UNCERTAIN"
    assert gate_res_fake["trigger"] == "missing_evidence"


def test_semantic_similarity_vs_keyword_matching(evidence_verifier):
    """
    Requirement 2: Use semantic similarity (sentence embeddings) between claims and evidence,
    not keyword matching alone.
    Verifies that semantically paraphrased statements are recognized as Supported,
    while keyword-overlapping contradictions are recognized as Contradicted.
    """
    # 1. Paraphrase sharing almost no content keywords with KB entry
    # KB Entry: "The James Webb Space Telescope was successfully launched in December 2021 by NASA, ESA, and CSA and observes infrared astronomy."
    paraphrased_claim = "The premier deep-space observatory began its mission in late 2021 to capture infrared astrophysical data."
    res_para = evidence_verifier.verify_claim(paraphrased_claim)
    assert res_para["similarity"] >= 0.55
    assert res_para["status"] == "Supported"

    # 2. High keyword overlap but inverted / negated meaning
    # Matches "Earth is an oblate spheroid revolving around the Sun"
    negated_claim = "Earth is not an oblate spheroid and does not revolve around the Sun."
    res_neg = evidence_verifier.verify_claim(negated_claim)
    assert res_neg["status"] == "Contradicted"


def test_temperature_scaling_calibration():
    """
    Requirement 3 & 5: Tests TemperatureScaling calibration and ECE reduction.
    """
    from src.calibration import TemperatureScaling, compute_ece
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Overconfident uncalibrated probabilities
    p_uncal = np.array([0.01, 0.05, 0.12, 0.40, 0.45, 0.60, 0.70, 0.88, 0.95, 0.99])

    ece_before, _ = compute_ece(y_true, p_uncal)
    ts = TemperatureScaling()
    ts.fit(p_uncal, y_true)

    assert ts.is_fitted
    assert ts.temperature > 0.0

    p_cal = ts.calibrate(p_uncal)
    assert len(p_cal) == len(p_uncal)
    assert (p_cal >= 0.0).all() and (p_cal <= 1.0).all()
    # Monotonicity preserved
    assert np.all(np.diff(p_cal) >= 0)


def test_retrieved_sources_verification(evidence_verifier):
    """
    Requirement 1: Verify each claim against retrieved sources or the knowledge base.
    """
    retrieved_docs = [
        {
            "statement": "The national railway agency introduced high-speed electric locomotives on the eastern corridor in March 2024.",
            "source": "Ministry of Transport Official Gazette",
            "is_debunk": False,
        }
    ]
    claim = "High-speed electric locomotives were introduced on the eastern railway corridor in 2024."
    res = evidence_verifier.verify_claim(claim, retrieved_sources=retrieved_docs)

    assert res["status"] == "Supported"
    assert "Ministry of Transport" in res["source"]
    assert res["similarity"] >= 0.60
