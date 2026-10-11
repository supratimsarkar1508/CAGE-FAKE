"""
CAGE-Fake: Streamlit Interactive Dashboard
Tabs: Home, Detector, NLP Analysis, Explanation, Uncertainty, Model Comparison, Evaluation, About.
"""

import sys
from pathlib import Path
import streamlit as st
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from src.preprocessing import TextPreprocessor
from src.nlp_explanation import NLPLinguisticExplainer
from src.utils import load_artifact
from config import TRAINED_MODELS_DIR

st.set_page_config(page_title="CAGE-Fake", layout="wide")

@st.cache_resource
def load_models():
    """Loads all trained artifacts from disk."""
    try:
        pipeline = load_artifact(TRAINED_MODELS_DIR / "feature_pipeline.joblib")
        balancer = load_artifact(TRAINED_MODELS_DIR / "balancer.joblib")
        stacking = load_artifact(TRAINED_MODELS_DIR / "stacking_model.joblib")
        calibrator = load_artifact(TRAINED_MODELS_DIR / "calibrator.joblib")
        gate = load_artifact(TRAINED_MODELS_DIR / "uncertainty_gate.joblib")
        explainer = load_artifact(TRAINED_MODELS_DIR / "shap_explainer.joblib")
        return pipeline, balancer, stacking, calibrator, gate, explainer
    except FileNotFoundError:
        return None, None, None, None, None, None

def main():
    st.title("CAGE-Fake: Calibration-Aware Fake News Triage")
    st.markdown("Trustworthy prediction that knows when not to trust itself.")
    
    tabs = st.tabs([
        "Home", 
        "Detector", 
        "NLP Analysis", 
        "Explanation", 
        "Uncertainty", 
        "Model Comparison", 
        "Evaluation", 
        "About"
    ])
    
    pipeline, balancer, stacking, calibrator, gate, explainer = load_models()
    preprocessor = TextPreprocessor(use_spacy=False)
    nlp_explainer = NLPLinguisticExplainer()

    with tabs[0]:
        st.header("Welcome to CAGE-Fake")
        st.write("Enter an article in the Detector tab to evaluate its veracity.")
        if not pipeline:
            st.error("Models not found! Please run `python experiments/train_cage_fake.py` first.")

    with tabs[1]:
        st.header("Fake News Detector")
        headline = st.text_input("Headline")
        body = st.text_area("Article Body", height=200)
        
        if st.button("Analyze Article") and pipeline:
            with st.spinner("Analyzing..."):
                raw, clean, _ = preprocessor.batch_process([headline], [body])
                
                # NLP Analysis
                nlp_res = nlp_explainer.analyze(raw[0], clean[0])
                st.session_state["nlp_res"] = nlp_res
                
                # Features
                X_sparse = pipeline.transform(clean, raw, return_sparse=True)
                X_dense = balancer.svd.transform(X_sparse) if balancer.is_fitted else X_sparse.toarray()
                
                # Models
                p_uncal, dis, probas_dict = stacking.predict_proba(X_dense)
                p_cal = calibrator.calibrate(p_uncal)[0]
                d = dis[0]
                
                # Gate
                gate_res = gate.evaluate_single(p_cal, d)
                st.session_state["gate_res"] = gate_res
                st.session_state["probas_dict"] = probas_dict
                
                # UI Display
                decision = gate_res["decision"]
                if decision == "UNCERTAIN":
                    st.warning(f"**Automated decision withheld** because the system detected insufficient confidence or model disagreement. ({gate_res['explanation']})")
                elif decision == "FAKE":
                    st.error(f"**Prediction: FAKE** (Confidence: {gate_res['confidence']:.1%})")
                else:
                    st.success(f"**Prediction: REAL** (Confidence: {gate_res['confidence']:.1%})")
                    
                st.metric("Model Disagreement", f"{d:.3f}")
                st.metric("Calibrated Probability of Fake", f"{p_cal:.1%}")

    with tabs[2]:
        st.header("NLP Linguistic Signal Summary")
        if "nlp_res" in st.session_state:
            nlp_res = st.session_state["nlp_res"]
            st.write(nlp_res["summary"])
            st.json(nlp_res["signals"])
        else:
            st.info("Run the detector first.")

    with tabs[3]:
        st.header("Explanation")
        st.info("SHAP explanations and Stream contributions will be rendered here.")

    with tabs[4]:
        st.header("Uncertainty & Gating")
        if "gate_res" in st.session_state:
            res = st.session_state["gate_res"]
            st.write(f"**Trigger:** {res['trigger']}")
            st.write(f"**Explanation:** {res['explanation']}")
            st.write(f"**Evidence conflict:** {res['evidence_note']}")

    with tabs[5]:
        st.header("Model Comparison")
        st.info("Performance charts comparing Base Models vs Stacking vs Transformers.")

    with tabs[6]:
        st.header("Evaluation")
        st.info("Reliability diagrams, Risk-Coverage curves, and Confusion Matrices.")

    with tabs[7]:
        st.header("About CAGE-Fake")
        st.write("A robust fake news triage system with fallback capability and rigorous OOF stacking.")

if __name__ == "__main__":
    main()
