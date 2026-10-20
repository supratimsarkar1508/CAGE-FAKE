"""
CAGE-Fake: End-to-End Prediction Pipeline
Executes prediction without retraining, loads saved model artifacts from disk,
computes multi-stream features, evaluates inter-model disagreement and Platt calibration,
applies the dual-signal uncertainty gate, synthesizes explainability,
and generates publication-quality PDF audit reports.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import time
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
    Manages loading of saved artifacts and executing step-by-step inference.
    """

    def __init__(self):
        self.feature_pipeline = None
        self.balancer = None
        self.stacking_model = None
        self.calibrator = None
        self.gate = None
        self.explainer = None
        self.comparison_table = None
        self.preprocessor = TextPreprocessor(use_spacy=False)
        self.nlp_explainer = NLPLinguisticExplainer()
        self.is_loaded = False

    def load_artifacts(self) -> "PredictionPipeline":
        """Loads all pre-trained models and calibration objects from disk."""
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
        self.balancer = load_artifact(artifacts["balancer"])
        self.stacking_model = load_artifact(artifacts["stacking_model"])
        self.calibrator = load_artifact(artifacts["calibrator"])
        self.gate = load_artifact(artifacts["uncertainty_gate"])
        self.explainer = load_artifact(artifacts["shap_explainer"])

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
        return self

    def _predict_fn_for_counterfactual(self, clean_text: str, raw_text: str) -> float:
        """Helper callable for counterfactual perturbation simulation."""
        feat_sparse = self.feature_pipeline.transform([clean_text], [raw_text], return_sparse=True)
        feat_dense = self.balancer.transform(feat_sparse)
        p_uncal, _, _ = self.stacking_model.predict_proba(feat_dense)
        p_cal = self.calibrator.calibrate(p_uncal)[0]
        return float(p_cal)

    def predict(
        self,
        title: str = "",
        body: str = "",
        url: Optional[str] = None,
        save_pdf: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference step by step:
        1. Preprocessing (clean & raw text)
        2. Feature Extraction (TF-IDF, SBERT, 15 Stylometric, 4 Readability)
        3. Dense SVD Projection
        4. 5-Model Probabilities & Inter-Model Disagreement
        5. Post-Hoc Platt Calibration & Uncertainty Gating
        6. News Credibility Fingerprint & Linguistic Diagnostics
        7. SHAP / Local Feature Attributions
        8. Counterfactual Sensitivity Simulation
        9. PDF Audit Report Synthesis
        """
        t0 = time.time()
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")

        # Step 1: Content Extraction / Validation (validate before loading heavy artifacts)
        if url:
            if verbose:
                print(f"\n[1/6] Extracting article content from URL: {url}...")
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

        if not self.is_loaded:
            self.load_artifacts()

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

        # Step 2: Preprocessing
        if verbose:
            print("[2/6] Preprocessing text and computing structural linguistic properties...")
        clean_text = self.preprocessor.clean_text(f"{title}. {body}")
        raw_text = f"{title}. {body}"

        # Step 3: Feature Extraction
        if verbose:
            print("[3/6] Extracting multi-stream features (Lexical, SBERT, 15 Stylometric, 4 Readability)...")
        X_sparse = self.feature_pipeline.transform([clean_text], [raw_text], return_sparse=True)
        X_dense = self.balancer.transform(X_sparse)

        # Step 4: Model Inference across all 5 Models
        if verbose:
            print("[4/6] Running 5-model inference and calculating ensemble disagreement...")
        p_uncal, dis_raw, base_probas = self.stacking_model.predict_proba(X_dense)
        p_uncal_val = float(p_uncal[0])
        dis_val = float(dis_raw[0])

        # Step 5: Post-Hoc Platt Calibration & Uncertainty Gating
        if verbose:
            print("[5/6] Applying Platt probability calibration and uncertainty gate (confidence + disagreement)...")
        p_cal_val = float(self.calibrator.calibrate(np.array([p_uncal_val]))[0])
        gate_res = self.gate.evaluate_single(p_cal_val, dis_val)

        # Step 6: 5-Model Disagreement Meter
        disagreement_info = ModelDisagreementMeter.compute(
            probas_dict=base_probas,
            p_calibrated_stack=p_cal_val,
            base_disagreement_std=dis_val,
            threshold_delta=self.gate.delta,
        )
        meter_fig = FIGURES_DIR / f"disagreement_meter_{timestamp_str}.png"
        ModelDisagreementMeter.plot_meter(disagreement_info, meter_fig)

        # Step 7: Linguistic Signals & News Credibility Fingerprint
        if verbose:
            print("[6/6] Computing credibility fingerprint, SHAP attributions, and counterfactuals...")
        signals = self.nlp_explainer.analyze(raw_text, clean_text)
        credibility_info = NewsCredibilityFingerprint.compute(signals, word_count=word_count)
        radar_fig = FIGURES_DIR / f"credibility_radar_{timestamp_str}.png"
        NewsCredibilityFingerprint.plot_radar(credibility_info, radar_fig)

        # Step 8: Local Feature Attributions (on meta-features: RF, XGB, LGBM, LR, Disagreement)
        meta_features_vec = np.array([
            float(np.atleast_1d(base_probas["rf"])[0]),
            float(np.atleast_1d(base_probas["xgb"])[0]),
            float(np.atleast_1d(base_probas["lgbm"])[0]),
            float(np.atleast_1d(base_probas["lr"])[0]),
            dis_val,
        ])
        local_features = self.explainer.explain_local_instance(meta_features_vec, top_k=5)

        # Step 9: Counterfactual Analysis
        counterfactual_info = CounterfactualExplainer.generate(
            clean_text=clean_text,
            raw_text=raw_text,
            original_p_fake=p_cal_val,
            predict_fn=self._predict_fn_for_counterfactual,
        )

        # Step 10: 'Why this prediction?' Synthesis
        why_explanation = synthesize_why_prediction(
            gate_result=gate_res,
            disagreement_info=disagreement_info,
            credibility_info=credibility_info,
            local_features=local_features,
        )

        # Step 11: PDF Audit Report Generation
        pdf_path = None
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
            )

        elapsed = time.time() - t0

        result = {
            "article_meta": article_meta,
            "decision": gate_res["decision"],
            "confidence": gate_res["confidence"],
            "p_fake": p_cal_val,
            "p_real": 1.0 - p_cal_val,
            "gate_disagreement": dis_val,
            "gate_threshold_delta": self.gate.delta,
            "gate_threshold_tau": self.gate.tau,
            "trigger": gate_res["trigger"],
            "explanation": gate_res["explanation"],
            "disagreement_info": disagreement_info,
            "credibility_info": credibility_info,
            "counterfactual_info": counterfactual_info,
            "why_explanation": why_explanation,
            "local_features": local_features,
            "pdf_report": str(pdf_path) if pdf_path else None,
            "elapsed_seconds": round(elapsed, 2),
        }

        if verbose:
            self.display_terminal_report(result)

        return result

    @staticmethod
    def display_terminal_report(res: Dict[str, Any]) -> None:
        """Renders rich ASCII terminal report."""
        meta = res["article_meta"]
        dec = res["decision"]
        conf = res["confidence"]
        dis = res["gate_disagreement"]
        p_fake = res["p_fake"]
        p_real = res["p_real"]
        tau = res["gate_threshold_tau"]
        delta = res["gate_threshold_delta"]
        trig = res["trigger"]

        d_info = res["disagreement_info"]
        c_info = res["credibility_info"]
        cf = res["counterfactual_info"]

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

        trigger_labels = {
            "both_triggers": "Dual Safety Triggers (Low Confidence AND High Disagreement)",
            "low_confidence": f"Low Confidence (Confidence {conf:.1%} < Threshold {tau:.1%})",
            "high_disagreement": f"High Disagreement (Base std {dis:.3f} > Threshold {delta:.2f})",
            "evidence_conflict": "External Evidence Conflict",
            "none": "None (High Confidence & Multi-Model Consensus)",
        }
        trigger_display = trigger_labels.get(trig, trig.replace("_", " ").title())

        print("\n" + "=" * 78)
        print(f"{BOLD}{CYAN}CAGE-Fake: ARTICLE TRUSTWORTHINESS & UNCERTAINTY TRIAGE REPORT{RESET}")
        print("=" * 78)

        # Mandatory Disclaimer
        print(f"\n{YELLOW}[!] METHODOLOGICAL DISCLAIMER:{RESET}")
        print("    A trained fake-news classifier predicts linguistic and stylistic patterns learned")
        print("    from dataset benchmarks; it does NOT independently verify factual truth.")
        print("    URL extraction, model prediction, and external fact verification remain separate.")

        print(f"\n{BOLD}ARTICLE METADATA:{RESET}")
        print(f"  Title             : {meta['title'][:70]}")
        print(f"  Source/URL        : {meta.get('url') or 'Manual Text Input'}")
        print(f"  Extraction Status : {meta.get('extraction_status', 'OK')}")
        print(f"  Word Count        : {meta['word_count']} words")

        if meta.get("short_text_warning"):
            print(f"\n{YELLOW}[!] SHORT-TEXT STATISTICAL CAVEAT:{RESET}")
            print(f"    {meta['short_text_warning']}")

        # Verdict
        print("\n" + "-" * 78)
        print(f"  {BOLD}TRIAGE VERDICT         :{RESET} {v_color}{BOLD}{dec}{RESET}")
        print(f"  Calibrated Probabilities : P(FAKE) = {p_fake:.1%}  |  P(REAL) = {p_real:.1%}")
        print(f"  Calibrated Confidence    : {conf:.1%} (Safety Threshold tau = {tau:.1%})")
        print(f"  Base Model Disagreement  : {dis:.3f} (Safety Threshold delta = {delta:.2f})")
        print(f"  Active Gate Trigger      : {trigger_display}")
        print("-" * 78)

        # 5-Model Disagreement Meter
        print(f"\n{BOLD}1. FIVE-MODEL CONSENSUS BREAKDOWN:{RESET}")
        print(f"   Consensus Status : {d_info['consensus_level']} (Votes: {d_info['fake_votes']} FAKE / {d_info['real_votes']} REAL)")
        print(f"   Gate Disagreement: {d_info['meter_bar']} (4-base std = {d_info['gate_base_disagreement']:.3f} vs delta = {delta:.2f})")
        print(f"   Five-Model Spread: std across all 5 models = {d_info['five_model_std']:.3f}")
        print("   " + "-" * 62)
        print(f"   {'Model Name':<28} {'P(FAKE)':<10} {'Vote':<10} {'Confidence':<10}")
        print("   " + "-" * 62)
        for m in d_info["models"]:
            vote_c = RED if m["vote"] == "FAKE" else GREEN
            print(f"   {m['name']:<28} {m['prob_fake'] * 100:>5.1f}%     {vote_c}{m['vote']:<10}{RESET} {m['confidence'] * 100:>5.1f}%")
        print("   " + "-" * 62)

        # News Credibility Fingerprint
        print(f"\n{BOLD}2. NEWS CREDIBILITY FINGERPRINT (Length-Smoothed):{RESET}")
        print(f"   Overall Credibility Index : {c_info['overall_credibility_index']}/100 ({c_info['credibility_grade']})")
        for dim, score in c_info["scores"].items():
            s_color = GREEN if score >= 75 else (YELLOW if score >= 50 else RED)
            bar = "#" * int(score // 5)
            print(f"   - {dim:<26} : {s_color}{score:>5.1f}/100  [{bar:<20}]{RESET}")

        # 'Why this prediction?'
        print(f"\n{BOLD}3. WHY THIS PREDICTION? (DIAGNOSTIC REASONING):{RESET}")
        for line in res["why_explanation"].split("\n"):
            print(f"   {line}")

        # Counterfactual Analysis
        print(f"\n{BOLD}4. COUNTERFACTUAL EXPLANATION (Signed Probability Shift):{RESET}")
        print(f"   {cf['summary']}")
        if cf.get("changed_fragments"):
            print("   Targeted Neutralizations:")
            for item in cf["changed_fragments"][:4]:
                print(f"     * [{item['category']}] {item['description']}")
        print(f"   Original P(FAKE)   : {cf['original_prob']:.1%}")
        print(f"   Transformed P(FAKE): {cf['simulated_prob']:.1%}")
        print(f"   Signed Shift Delta : {cf['signed_shift_pct']:+.1f} percentage points (shift = {cf['signed_prob_shift']:+.4f})")

        # Local Features
        print(f"\n{BOLD}5. TOP LOCAL ATTRIBUTION SIGNALS:{RESET}")
        top_fake = [f['name'] for f in res['local_features'].get('fake_signals', [])[:3]]
        top_real = [f['name'] for f in res['local_features'].get('real_signals', [])[:3]]
        print(f"   Indicators pushing towards FAKE : {', '.join(top_fake) if top_fake else 'None prominent'}")
        print(f"   Anchors pushing towards REAL    : {', '.join(top_real) if top_real else 'None prominent'}")

        # PDF Report Link
        if res.get("pdf_report"):
            print("\n" + "=" * 78)
            print(f"{BOLD}{GREEN}[v] Full Publication-Quality PDF Report Saved:{RESET} {res['pdf_report']}")
            print("=" * 78)
        print(f"Inference completed in {res['elapsed_seconds']}s\n")
