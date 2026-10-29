"""
CAGE-Fake: End-to-End Prediction Pipeline
Executes prediction without retraining, loads saved model artifacts from disk,
computes multi-stream features, extracts and verifies claims against live evidence,
evaluates inter-model and multi-view disagreement, detects out-of-distribution domain shift,
applies the evidence-first decision engine with tri-trigger gating, synthesizes explainability,
and generates publication-quality PDF audit reports alongside full modular dossiers.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd

from config import (
    TRAINED_MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
)
from src.utils import setup_logger, load_artifact
from src.url_extractor import extract_article_from_url
from src.preprocessing import TextPreprocessor
from src.nlp_explanation import NLPLinguisticExplainer
from src.evidence.evidence_pipeline import EvidencePipeline
from src.evidence.domain_whitelist import is_trusted_domain
from src.ood_detector import OODDetector
from src.decision_rule import EvidenceFirstDecisionEngine
from src.report_pdf_dossier import ArticleReportDossier
from src.unique_features import (
    ModelDisagreementMeter,
    CounterfactualExplainer,
    NewsCredibilityFingerprint,
    synthesize_why_prediction,
)
from src.pdf_generator import generate_pdf_report

logger = setup_logger("predict")


class PredictionPipeline:
    """
    Manages loading of saved artifacts and executing step-by-step evidence-grounded inference.
    """

    def __init__(self):
        self.feature_pipeline = None
        self.balancer = None
        self.stacking_model = None
        self.calibrator = None
        self.gate = None
        self.explainer = None
        self.multi_view_ensemble = None
        self.ood_detector = None
        self.comparison_table = None
        self.preprocessor = TextPreprocessor(use_spacy=False)
        self.nlp_explainer = NLPLinguisticExplainer()
        self.evidence_pipeline = EvidencePipeline(use_live_retrieval=True)
        self.decision_engine = EvidenceFirstDecisionEngine()
        self.report_dossier = ArticleReportDossier(REPORTS_DIR)
        self.is_loaded = False
        self._uses_new_arch = False  # updated after load_artifacts()

    def load_artifacts(self) -> "PredictionPipeline":
        """Loads all pre-trained models, calibration objects, and optional multi-view experts from disk."""
        logger.info("Loading pre-trained CAGE-Fake artifacts from disk...")

        artifacts = {
            "feature_pipeline": TRAINED_MODELS_DIR / "feature_pipeline.joblib",
            "balancer": TRAINED_MODELS_DIR / "balancer.joblib",
            "stacking_model": TRAINED_MODELS_DIR / "stacking_model.joblib",
            "calibrator": TRAINED_MODELS_DIR / "calibrator.joblib",
            "uncertainty_gate": TRAINED_MODELS_DIR / "uncertainty_gate.joblib",
            "shap_explainer": TRAINED_MODELS_DIR / "shap_explainer.joblib",
        }

        missing = [k for k, p in artifacts.items() if not p.exists()]
        if missing:
            raise RuntimeError(
                f"Missing required trained model artifacts: {missing}.\n"
                f"Please train the pipeline first by running: python run_pipeline.py --train"
            )

        self.feature_pipeline = load_artifact(artifacts["feature_pipeline"])
        if hasattr(self.feature_pipeline, "semantic_extractor") and self.feature_pipeline.semantic_extractor is not None:
            from config import EMBEDDINGS_CACHE_FILE
            self.feature_pipeline.semantic_extractor.cache_path = EMBEDDINGS_CACHE_FILE
        self.balancer = load_artifact(artifacts["balancer"])
        self.stacking_model = load_artifact(artifacts["stacking_model"])
        self.calibrator = load_artifact(artifacts["calibrator"])
        self.gate = load_artifact(artifacts["uncertainty_gate"])
        self.explainer = load_artifact(artifacts["shap_explainer"])

        # Update decision engine defaults from loaded gate
        if hasattr(self.gate, "tau"):
            self.decision_engine.tau = self.gate.tau
        if hasattr(self.gate, "delta"):
            self.decision_engine.delta = self.gate.delta

        # Optional multi-view ensemble
        mve_path = TRAINED_MODELS_DIR / "multi_view_ensemble.joblib"
        if mve_path.exists():
            try:
                self.multi_view_ensemble = load_artifact(mve_path)
                logger.info("Loaded Multi-View Ensemble (Lexical, Style, Semantic experts).")
            except Exception as e:
                logger.warning(f"Could not load multi_view_ensemble: {e}")
                self.multi_view_ensemble = None

        # Optional OOD detector
        ood_path = TRAINED_MODELS_DIR / "ood_detector.joblib"
        if ood_path.exists():
            try:
                self.ood_detector = load_artifact(ood_path)
                self.decision_engine.rho = self.ood_detector.threshold_rho
                logger.info("Loaded OOD Detector (Vocab coverage & SBERT Mahalanobis distance).")
            except Exception as e:
                logger.warning(f"Could not load ood_detector: {e}")
                self.ood_detector = None

        # Load comparison metrics if available for the PDF report
        comp_csv = METRICS_DIR / "comparison_table.csv"
        if comp_csv.exists():
            try:
                self.comparison_table = pd.read_csv(comp_csv).to_dict(orient="records")
            except Exception as e:
                logger.warning(f"Could not load comparison table: {e}")
                self.comparison_table = None

        self.is_loaded = True
        logger.info("All CAGE-Fake model artifacts successfully loaded.")

        # Detect architecture: new heterogeneous-view model has non-empty _stream_boundaries.
        # Old pre-redesign models have only 'models' dict and no view_bundle.
        _sb = getattr(self.stacking_model, '_stream_boundaries', None)
        self._uses_new_arch = bool(_sb)
        if self._uses_new_arch:
            logger.info("Detected NEW heterogeneous-view architecture (4 orthogonal model views).")
        else:
            logger.warning(
                "Detected OLD shared-SVD architecture. Predictions still work but disagreement "
                "is NOT independent across models. Retrain with --train to enable the new design."
            )
        return self

    def _predict_fn_for_counterfactual(self, clean_text: str, raw_text: str) -> float:
        """Helper callable for counterfactual perturbation simulation."""
        feat_sparse = self.feature_pipeline.transform([clean_text], [raw_text], return_sparse=True)
        # Detect architecture: new (sparse) vs old (SVD-dense)
        if self._uses_new_arch:
            p_uncal, _, _ = self.stacking_model.predict_proba(feat_sparse)
        else:
            feat_dense = self.balancer.transform(feat_sparse)
            p_uncal, _, _ = self.stacking_model.predict_proba(feat_dense)
        return float(self.calibrator.calibrate(np.array([p_uncal[0]]))[0])

    def predict(
        self,
        title: Optional[str] = None,
        body: Optional[str] = None,
        url: Optional[str] = None,
        retrieved_sources: Optional[List[Any]] = None,
        confidence_threshold: Optional[float] = None,
        save_pdf: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end evidence-grounded inference with multi-view experts and OOD domain shift detection.
        """
        t0 = time.time()
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")

        if not self.is_loaded:
            self.load_artifacts()

        # Step 1: Content Extraction / Validation
        if url:
            if verbose:
                print(f"\n[1/7] Extracting article content from URL: {url}...")
            extracted = extract_article_from_url(url)
            title = extracted["title"]
            body = extracted["text"]
            domain = extracted["domain"]
            word_count = extracted["word_count"]
            extraction_status = "Successfully Extracted via Trafilatura"
        else:
            if not body or len(body.strip()) < 20:
                raise ValueError("Article text is too short or empty (minimum 20 characters required).")
            domain = "Manual Text Submission"
            word_count = len(body.split())
            extraction_status = "Direct Manual Input"

        short_text_warning = None
        if word_count < 50:
            short_text_warning = (
                f"Input text contains only {word_count} words (< 50 words recommended). "
                "Stylometric distributions (type-token ratio, hedge frequency, attribution density) and readability formulas "
                "exhibit higher variance on short passages; article-level diagnostics should be interpreted with caution."
            )

        article_meta = {
            "title": title or "Untitled Article",
            "body": body,
            "url": url,
            "domain": domain,
            "word_count": word_count,
            "extraction_status": extraction_status,
            "short_text_warning": short_text_warning,
            "timestamp": timestamp_str,
        }

        # Step 2: Evidence Grounding & Live Multi-Source Verification
        if verbose:
            print("[2/7] Extracting key claims and verifying against live web, fact-checks, and Wikipedia...")
        evidence_res = self.evidence_pipeline.verify_article(
            body, title=title, custom_claims=retrieved_sources, source_url=url
        )
        evidence_res["is_trusted_publisher"] = is_trusted_domain(domain)

        # Step 3: Preprocessing & Linguistic Red Flags
        if verbose:
            print("[3/7] Preprocessing text and computing linguistic signals & red flags...")
        clean_text = self.preprocessor.clean_text(f"{title}. {body}")
        raw_text = f"{title}. {body}"

        linguistic_analysis = self.nlp_explainer.analyze(raw_text, clean_text, title=title)
        signals = linguistic_analysis["signals"]
        red_flags = linguistic_analysis["linguistic_red_flags"]
        credibility_info = NewsCredibilityFingerprint.compute(signals, word_count=word_count)
        cred_index = float(credibility_info["overall_credibility_index"])

        # Step 4: Multi-Stream Feature Extraction
        if verbose:
            print("[4/7] Extracting multi-stream features (Lexical, SBERT, 15 Stylometric, 4 Readability)...")
        X_sparse = self.feature_pipeline.transform([clean_text], [raw_text], return_sparse=True)

        # Step 5: Model Inference & Multi-View Disagreement
        if verbose:
            arch_label = "heterogeneous-view" if self._uses_new_arch else "legacy shared-SVD"
            print(f"[5/7] Running inference across base models ({arch_label})...")

        if self._uses_new_arch:
            # NEW: each model slices its own orthogonal view internally
            p_uncal, dis_raw, base_probas = self.stacking_model.predict_proba(X_sparse)
        else:
            # LEGACY: compress to SVD-dense first (old trained models expect this)
            X_dense = self.balancer.transform(X_sparse)
            p_uncal, dis_raw, base_probas = self.stacking_model.predict_proba(X_dense)

        p_uncal_val = float(p_uncal[0])
        dis_val = float(dis_raw[0])
        p_cal_val = float(self.calibrator.calibrate(np.array([p_uncal_val]))[0])

        # Dynamic Evidence Override Policy:
        # If external evidence verification has >=2 supported claims (cosine similarity >= 0.65)
        # AND stylometric credibility index is >= 80:
        # Penalize semantic/lexical fake probabilities by dynamically downweighting them before triage gating.
        sens_discipline = float(credibility_info["scores"].get("Sensationalism Discipline", cred_index))
        has_high_credibility = (cred_index >= 75.0 or sens_discipline >= 80.0)
        claim_verifs = evidence_res.get("claim_verifications", [])
        high_sim_supported = [
            c for c in claim_verifs
            if c.get("status") == "Supported" and float(c.get("similarity", c.get("dense_score", 0.0))) >= 0.60
        ]
        n_high_sim = len(high_sim_supported)
        has_contra = evidence_res.get("has_any_contradiction", False) or evidence_res.get("has_numeric_conflict", False)
        evidence_override_active = (n_high_sim >= 2 and has_high_credibility and not has_contra)

        if evidence_override_active:
            avg_sim = float(np.mean([float(c.get("similarity", c.get("dense_score", 0.70))) for c in high_sim_supported])) if high_sim_supported else 0.75
            attenuation = max(0.10, 1.0 - avg_sim)
            base_probas = dict(base_probas)
            for k in ["lr", "xgb", "lgbm"]:
                if k in base_probas:
                    base_probas[k] = float(np.atleast_1d(base_probas[k])[0]) * attenuation
            avail = [float(np.atleast_1d(base_probas[k])[0]) for k in ["lr", "lgbm", "rf", "xgb"] if k in base_probas]
            dis_val = float(np.std(avail))
            p_uncal_val = float(np.mean(avail))
            p_cal_val = float(self.calibrator.calibrate(np.array([p_uncal_val]))[0])

        # Dynamic Temperature Scaling across ensemble conditioned on model disagreement delta
        t_dyn = 1.0 + 1.5 * max(0.0, dis_val)
        if 0.0 < p_cal_val < 1.0:
            import math
            logit = math.log(p_cal_val / (1.0 - p_cal_val))
            p_cal_val = float(1.0 / (1.0 + math.exp(-logit / t_dyn)))

        # Step 5.5: Out-Of-Distribution (OOD) Domain Shift Check
        sbert_emb = None
        if hasattr(self.feature_pipeline, "semantic_extractor") and self.feature_pipeline.semantic_extractor is not None:
            try:
                sbert_emb = self.feature_pipeline.semantic_extractor.encode([clean_text])[0]
            except Exception as e:
                logger.warning(f"Could not compute SBERT embedding for OOD detector: {e}")

        ood_detector_loaded = bool(self.ood_detector is not None)
        if self.ood_detector is not None and sbert_emb is not None:
            ood_res = self.ood_detector.evaluate_single(raw_text, sbert_emb)
            ood_res["ood_detector_loaded"] = True
        else:
            logger.warning(
                "OOD detector artifact missing from models/trained/ood_detector.joblib. "
                "Computing fallback vocabulary coverage against TF-IDF vocabulary..."
            )
            vocab = None
            if hasattr(self, "feature_pipeline") and hasattr(self.feature_pipeline, "lexical_extractor"):
                vec = getattr(self.feature_pipeline.lexical_extractor, "vectorizer", None)
                if vec is not None and hasattr(vec, "vocabulary_"):
                    vocab = vec.vocabulary_

            if vocab:
                tokens = [w for w in "".join(c if c.isalnum() else " " for c in raw_text.lower()).split() if len(w) > 2]
                if tokens:
                    in_vocab = sum(1 for w in tokens if w in vocab)
                    coverage = float(in_vocab / len(tokens))
                else:
                    coverage = 1.0
            else:
                coverage = 1.0

            is_ood = bool(coverage < 0.60)
            ood_score = round(max(0.0, 1.0 - coverage), 4)
            ood_res = {
                "is_ood": is_ood,
                "ood_score": ood_score,
                "vocab_coverage": round(coverage, 4),
                "mahalanobis_dist": 0.0,
                "threshold_rho": self.decision_engine.rho,
                "explanation": (
                    f"Fallback vocabulary coverage ({coverage:.1%} {'< 60% (OOD Domain Shift)' if is_ood else '>= 60% (In-Distribution)'}); "
                    "ood_detector artifact missing"
                ),
                "ood_detector_loaded": False,
            }

        def _p(key):
            """Safely extract a float P(FAKE) from base_probas (handles 0-d and 1-d arrays)."""
            v = base_probas.get(key, p_cal_val)
            return float(np.atleast_1d(v)[0])

        # Map orthogonal view outputs to semantic labels:
        #   LR   -> lexical vocabulary expert
        #   LGBM -> topic / SVD expert
        #   RF   -> stylometric writing expert  (this is the "style" signal)
        #   XGB  -> semantic SBERT expert
        multi_view_res = {
            "mean_p_fake": p_cal_val,
            "view_disagreement": dis_val,
            "credibility_index": cred_index,
            "sensationalism_discipline": sens_discipline,
            "p_lexical":  _p("lr"),
            "p_style":    _p("rf"),    # RF = stylometric expert
            "p_semantic": _p("xgb"),  # XGB = SBERT semantic
            "p_topic":    _p("lgbm"), # LGBM = SVD topic
            "lexical_explanation": {"p_fake": _p("lr"),  "why": []},
            "style_explanation":   {"p_fake": _p("rf"),  "why": []},
            "semantic_explanation":{"p_fake": _p("xgb"), "why": []},
        }

        # Step 6: Evidence-First Decision Engine with Tri-Trigger Gating
        if verbose:
            print("[6/7] Evaluating Evidence-First Decision Engine (tri-trigger gate: tau, delta, rho)...")

        gate_res = self.decision_engine.evaluate(
            evidence_summary=evidence_res,
            multi_view_summary=multi_view_res,
            ood_summary=ood_res,
            custom_tau=confidence_threshold,
        )

        verdict = gate_res["decision"]
        confidence = gate_res["confidence"]

        # Step 7: Disagreement Meter, Credibility Fingerprint & Explanations
        if verbose:
            print("[7/7] Computing credibility fingerprint, attributions, counterfactuals, and report...")
        disagreement_info = ModelDisagreementMeter.compute(
            probas_dict=base_probas,
            p_calibrated_stack=p_cal_val,
            base_disagreement_std=dis_val,
            threshold_delta=gate_res["delta"],
        )
        meter_fig = FIGURES_DIR / f"disagreement_meter_{timestamp_str}.png"
        ModelDisagreementMeter.plot_meter(disagreement_info, meter_fig)

        credibility_info = NewsCredibilityFingerprint.compute(signals, word_count=word_count)
        radar_fig = FIGURES_DIR / f"credibility_radar_{timestamp_str}.png"
        NewsCredibilityFingerprint.plot_radar(credibility_info, radar_fig)

        # Local Feature Attributions
        # Order matches StackingClassifier meta-learner training: [p_lr, p_lgbm, p_rf, p_xgb, disagreement]
        meta_features_vec = np.array([
            float(np.atleast_1d(base_probas["lr"])[0]),    # View A: LR on TF-IDF lexical
            float(np.atleast_1d(base_probas["lgbm"])[0]),  # View B: LightGBM on SVD topics
            float(np.atleast_1d(base_probas["rf"])[0]),    # View C: RF on stylometrics
            float(np.atleast_1d(base_probas["xgb"])[0]),  # View D: XGB on SBERT semantic
            dis_val,                                        # Inter-view disagreement (std)
        ])
        local_features = self.explainer.explain_local_instance(meta_features_vec, top_k=5)

        # Counterfactual Analysis
        counterfactual_info = CounterfactualExplainer.generate(
            clean_text=clean_text,
            raw_text=raw_text,
            original_p_fake=p_cal_val,
            predict_fn=self._predict_fn_for_counterfactual,
        )

        # 'Why this prediction?' Synthesis
        why_explanation = synthesize_why_prediction(
            gate_result=gate_res,
            disagreement_info=disagreement_info,
            credibility_info=credibility_info,
            local_features=local_features,
        )

        # Synthesize Reasoning Summary
        reasoning_summary = self._generate_reasoning_summary(
            verdict=verdict,
            confidence=confidence,
            evidence_res=evidence_res,
            red_flags=red_flags,
            disagreement=dis_val,
            gate_res=gate_res,
        )

        gate_res["ood_detector_loaded"] = ood_detector_loaded
        article_meta["ood_detector_loaded"] = ood_detector_loaded

        # Construct Explainable Output JSON (DUPLICATION REMOVED: claim_verifications omitted)
        explainable_json = {
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "confidence_type": gate_res.get("confidence_type", "calibrated"),
            "per_claim_evidence_status": evidence_res["claim_verifications"],
            "linguistic_red_flags": red_flags,
            "reasoning_summary": reasoning_summary,
            "ood_status": ood_res,
            "ood_detector_loaded": ood_detector_loaded,
            "policy_category": gate_res["qualifier"],
            "style_evidence_conflict": gate_res.get("style_evidence_conflict", False),
        }

        # Step 8: PDF Report & Modular Dossier Generation
        pdf_path = None
        dossier_pdf_path = None
        if save_pdf:
            pdf_path = REPORTS_DIR / f"audit_report_{timestamp_str}.pdf"
            fig_map = {
                "meter": meter_fig,
                "radar": radar_fig,
            }
            rel_fig = FIGURES_DIR / "reliability_diagram.png"
            if rel_fig.exists():
                fig_map["reliability"] = rel_fig
            rc_fig = FIGURES_DIR / "risk_coverage_curve.png"
            if rc_fig.exists():
                fig_map["risk_coverage"] = rc_fig

            generate_pdf_report(
                output_path=pdf_path,
                article_meta=article_meta,
                gate_result=gate_res,
                disagreement_info=disagreement_info,
                credibility_info=credibility_info,
                counterfactual_info=counterfactual_info,
                why_explanation=why_explanation,
                local_features=local_features,
                comparison_table=self.comparison_table,
                figures=fig_map,
                claim_verifications=evidence_res.get("claim_verifications"),
                linguistic_red_flags=red_flags,
                reasoning_summary=reasoning_summary,
            )

            # Generate the 11-part modular dossier and merged full_report.pdf
            try:
                dossier_pdf_path = self.report_dossier.generate_dossier(
                    timestamp_str=timestamp_str,
                    article_meta=article_meta,
                    decision_res=gate_res,
                    multi_view_res=multi_view_res,
                    evidence_res=evidence_res,
                    ood_res=ood_res,
                    credibility_info=credibility_info,
                    figures=fig_map,
                )
            except Exception as e:
                logger.warning(f"Dossier compilation failed: {e}")

        elapsed = time.time() - t0

        result = {
            "verdict": verdict,
            "decision": verdict,
            "confidence": confidence,
            "confidence_type": gate_res.get("confidence_type", "calibrated"),
            "ood_detector_loaded": ood_detector_loaded,
            "p_fake": gate_res.get("p_fake", p_cal_val),
            "p_real": gate_res.get("p_real", 1.0 - p_cal_val),
            "per_claim_evidence_status": evidence_res["claim_verifications"],
            "evidence_analysis": evidence_res,
            "linguistic_red_flags": red_flags,
            "reasoning_summary": reasoning_summary,
            "explainable_json": explainable_json,
            "article_meta": article_meta,
            "gate_disagreement": dis_val,
            "gate_threshold_delta": gate_res["delta"],
            "gate_threshold_tau": gate_res["tau"],
            "gate_threshold_rho": gate_res["rho"],
            "trigger": gate_res["trigger"],
            "qualifier": gate_res["qualifier"],
            "explanation": gate_res["explanation"],
            "style_evidence_conflict": gate_res.get("style_evidence_conflict", False),
            "conflict_explanation": gate_res.get("conflict_explanation"),
            "ood_status": ood_res,
            "disagreement_info": disagreement_info,
            "credibility_info": credibility_info,
            "counterfactual_info": counterfactual_info,
            "why_explanation": why_explanation,
            "local_features": local_features,
            "pdf_report": str(pdf_path) if pdf_path else None,
            "full_dossier_pdf": str(dossier_pdf_path) if dossier_pdf_path else None,
            "elapsed_seconds": round(elapsed, 2),
        }

        if verbose:
            self.display_terminal_report(result)

        return result

    def _generate_reasoning_summary(
        self,
        verdict: str,
        confidence: float,
        evidence_res: Dict[str, Any],
        red_flags: List[str],
        disagreement: float,
        gate_res: Dict[str, Any],
    ) -> str:
        """Synthesizes an executive 2-3 sentence grounded justification for the decision."""
        if gate_res.get("style_evidence_conflict"):
            return gate_res["conflict_explanation"]

        trig = gate_res["trigger"]
        if trig == "evidence_corroborated":
            return (
                f"Article claims were independently corroborated by {evidence_res.get('trusted_domains_count', 0)} "
                f"authoritative whitelisted newsrooms ({evidence_res.get('percent_supported', 0.0):.0%} of claims verified). "
                f"Decision is grounded in external factual corroboration with high confidence ({confidence:.1%})."
            )
        elif trig == "evidence_contradicted":
            return (
                f"Article contains critical factual refutations or direct numerical discrepancies verified against primary records. "
                f"Triaged as FAKE ({confidence:.1%}) based on authoritative factual contradiction."
            )
        elif trig == "domain_shift_no_evidence":
            return (
                f"Article exhibits severe lexical and semantic domain shift relative to training data, with zero external corroboration. "
                f"The pipeline abstains (UNCERTAIN) to eliminate confident out-of-distribution classification errors."
            )
        elif verdict == "UNCERTAIN":
            return (
                f"System abstains with UNCERTAIN triage ({gate_res.get('explanation')}). "
                f"Multi-view divergence (std={disagreement:.3f}) mandates human editorial review."
            )
        elif verdict == "REAL":
            return (
                f"Restrained journalistic presentation with consistent stylometrics ({confidence:.1%} confidence). "
                f"Multi-view ensemble reaches safe consensus without sensational red flags."
            )
        else:
            flags_str = f" ({len(red_flags)} linguistic red flags)" if red_flags else ""
            return (
                f"Predictive models identify deceptive stylometric patterns and emotional sensationalism{flags_str}. "
                f"Triaged as FAKE ({confidence:.1%} confidence)."
            )

    @staticmethod
    def display_terminal_report(res: Dict[str, Any]) -> None:
        """Renders rich ASCII terminal report including claim-level evidence and red flags."""
        meta = res["article_meta"]
        dec = res["verdict"]
        conf = res["confidence"]
        dis = res["gate_disagreement"]
        p_fake = res["p_fake"]
        p_real = res["p_real"]
        tau = res["gate_threshold_tau"]
        delta = res["gate_threshold_delta"]
        rho = res["gate_threshold_rho"]
        trig = res["trigger"]

        d_info = res["disagreement_info"]
        c_info = res["credibility_info"]
        claims = res.get("per_claim_evidence_status", [])
        red_flags = res.get("linguistic_red_flags", [])

        # ANSI colors
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        CYAN = "\033[96m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        if dec == "REAL":
            v_color = GREEN
        elif dec == "FAKE":
            v_color = RED
        else:
            v_color = YELLOW

        print("\n" + "=" * 78)
        print(f"{BOLD}{CYAN}CAGE-Fake: EVIDENCE-GROUNDED ARTICLE TRIAGE REPORT{RESET}")
        print("=" * 78)

        # Style-Evidence Conflict Alert (if triggered)
        if res.get("style_evidence_conflict"):
            print(f"\n{BOLD}{RED}[!] STYLE-EVIDENCE CONFLICT DETECTED:{RESET}")
            print(f"    {res['conflict_explanation']}")

        # Mandatory Disclaimer
        print(f"\n{YELLOW}[!] TRUSTWORTHY FACT-CHECKING POLICY:{RESET}")
        print("    Decisions prioritize semantic claim verification against verified whitelisted sources.")
        print("    The pipeline strictly abstains (UNCERTAIN) under domain shift or when evidence is missing.")

        print(f"\n{BOLD}ARTICLE METADATA:{RESET}")
        print(f"  Title             : {meta['title'][:70]}")
        print(f"  Source/URL        : {meta.get('url') or 'Manual Text Input'}")
        print(f"  Extraction Status : {meta.get('extraction_status', 'OK')}")
        print(f"  Word Count        : {meta['word_count']} words")

        if meta.get("short_text_warning"):
            print(f"\n{YELLOW}[!] SHORT-TEXT STATISTICAL CAVEAT:{RESET}")
            print(f"    {meta['short_text_warning']}")

        # Verdict Card
        print("\n" + "-" * 78)
        print(f"  {BOLD}TRIAGE VERDICT         :{RESET} {v_color}{BOLD}{dec}{RESET} ({res.get('qualifier', 'Evaluated')})")
        print(f"  Calibrated Confidence    : {conf:.1%} (Locked Threshold tau = {tau:.1%})")
        print(f"  Calibrated Probabilities : P(FAKE) = {p_fake:.1%}  |  P(REAL) = {p_real:.1%}")
        print(f"  Model Disagreement       : {dis:.3f} (Threshold delta = {delta:.2f})")
        ood = res.get("ood_status", {})
        print(f"  Domain Shift (OOD Score) : {ood.get('ood_score', 0.0):.3f} (Threshold rho = {rho:.2f}, Vocab Cov = {ood.get('vocab_coverage', 1.0):.1%})")
        print(f"  Active Gating Trigger    : {trig}")
        print("-" * 78)

        # Section 1: Claim-by-Claim Evidence Verification
        print(f"\n{BOLD}1. CLAIM-BY-CLAIM EVIDENCE VERIFICATION ({len(claims)} claims checked):{RESET}")
        if claims:
            for idx, c in enumerate(claims, 1):
                st = c["status"]
                st_color = GREEN if st == "Supported" else (RED if st == "Contradicted" else YELLOW)
                print(f"   [{idx}] Status : {st_color}{BOLD}{st.upper()}{RESET} (Semantic Similarity: {c.get('similarity', 0.0):.2f})")
                print(f"       Claim  : \"{c['claim']}\"")
                print(f"       Source : {c['source']}")
                if c.get("evidence_text"):
                    print(f"       Matched: \"{c['evidence_text'][:90]}...\"")
                if c.get("conflicts"):
                    for conf_msg in c["conflicts"]:
                        print(f"       {RED}[!] Conflict: {conf_msg}{RESET}")
        else:
            print("   No distinct factual claims extracted.")

        # Section 2: Linguistic Red Flags
        print(f"\n{BOLD}2. LINGUISTIC & SEMANTIC RED FLAGS ({len(red_flags)} detected):{RESET}")
        if red_flags:
            for rf in red_flags:
                print(f"   {RED}[x]{RESET} {rf}")
        else:
            print(f"   {GREEN}[v]{RESET} Clean journalistic presentation; no clickbait, emotional distortion, or contradictions.")

        # Section 3: Reasoning Summary
        print(f"\n{BOLD}3. REASONING SUMMARY:{RESET}")
        print(f"   {res['reasoning_summary']}")

        # Section 4: Multi-Model Consensus Breakdown
        print(f"\n{BOLD}4. FIVE-MODEL CONSENSUS BREAKDOWN:{RESET}")
        print(f"   Consensus Status : {d_info['consensus_level']} (Votes: {d_info['fake_votes']} FAKE / {d_info['real_votes']} REAL)")
        print(f"   Gate Disagreement: {d_info['meter_bar']} (4-base std = {d_info['gate_base_disagreement']:.3f} vs delta = {delta:.2f})")
        print("   " + "-" * 62)
        print(f"   {'Model Name':<28} {'P(FAKE)':<10} {'Vote':<10} {'Confidence':<10}")
        print("   " + "-" * 62)
        for m in d_info["models"]:
            vote_c = RED if m["vote"] == "FAKE" else GREEN
            print(f"   {m['name']:<28} {m['prob_fake'] * 100:>5.1f}%     {vote_c}{m['vote']:<10}{RESET} {m['confidence'] * 100:>5.1f}%")
        print("   " + "-" * 62)

        # Section 5: News Credibility Fingerprint
        print(f"\n{BOLD}5. NEWS CREDIBILITY FINGERPRINT:{RESET}")
        print(f"   Overall Credibility Index : {c_info['overall_credibility_index']}/100 ({c_info['credibility_grade']})")
        for dim, score in c_info["scores"].items():
            s_color = GREEN if score >= 75 else (YELLOW if score >= 50 else RED)
            bar = "#" * int(score // 5)
            print(f"   - {dim:<26} : {s_color}{score:>5.1f}/100  [{bar:<20}]{RESET}")

        # PDF Report Links
        if res.get("pdf_report"):
            print("\n" + "=" * 78)
            print(f"{BOLD}{GREEN}[v] Full 2-Page Executive Audit Report Saved:{RESET} {res['pdf_report']}")
            if res.get("full_dossier_pdf"):
                print(f"{BOLD}{GREEN}[v] Complete 11-Part Multi-Page Dossier Saved:{RESET} {res['full_dossier_pdf']}")
            print("=" * 78)
