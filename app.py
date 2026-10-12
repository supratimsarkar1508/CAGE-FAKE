import sys
import json
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))

from src.preprocessing import TextPreprocessor
from src.nlp_explanation import NLPLinguisticExplainer
from src.utils import load_artifact
from config import (
    TRAINED_MODELS_DIR,
    OUTPUTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    REPORTS_DIR,
)

st.set_page_config(
    page_title="CAGE-Fake | Trustworthy Fake News Triage",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-fake {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-real {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-uncertain {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    """Loads all trained artifacts strictly from disk without retraining."""
    try:
        pipeline = load_artifact(TRAINED_MODELS_DIR / "feature_pipeline.joblib")
        balancer = load_artifact(TRAINED_MODELS_DIR / "balancer.joblib")
        stacking = load_artifact(TRAINED_MODELS_DIR / "stacking_model.joblib")
        calibrator = load_artifact(TRAINED_MODELS_DIR / "calibrator.joblib")
        gate = load_artifact(TRAINED_MODELS_DIR / "uncertainty_gate.joblib")
        explainer = load_artifact(TRAINED_MODELS_DIR / "shap_explainer.joblib")
        return pipeline, balancer, stacking, calibrator, gate, explainer
    except Exception as e:
        return None, None, None, None, None, None


def load_json_safe(filepath: Path):
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    st.markdown('<div class="main-header">🛡️ CAGE-Fake: Trustworthy Fake News Triage</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Calibration-Aware Gating and Evidence-Grounded Reasoning for Fake News Detection</div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "🏠 Home",
        "🔍 Detector",
        "📊 NLP Analysis",
        "🧠 Explanation (SHAP)",
        "⚖️ Uncertainty & Gating",
        "📈 Model Comparison",
        "📉 Evaluation & Reliability",
        "ℹ️ About CAGE-Fake",
    ])

    pipeline, balancer, stacking, calibrator, gate, explainer = load_models()
    preprocessor = TextPreprocessor(use_spacy=True)
    nlp_explainer = NLPLinguisticExplainer()

    # --- TAB 1: HOME ---
    with tabs[0]:
        st.subheader("Welcome to the CAGE-Fake Trustworthy Triage Engine")
        st.markdown(
            """
            **CAGE-Fake** is designed under one central research philosophy: **a model must know when not to trust itself.**
            
            Conventional fake-news classifiers force binary decisions even when faced with novel, ambiguous, or out-of-distribution articles. CAGE-Fake introduces a **Triple-Trigger Uncertainty Gate**:
            1. **Confidence Threshold ($\\tau$)**: Abstains when predicted confidence is below calibrated bounds.
            2. **Disagreement Threshold ($\\delta$)**: Abstains when diverse base models (LightGBM, Random Forest, XGBoost) disagree significantly.
            3. **Evidence Conflict Hook**: Interfaces with verified external evidence if available at runtime.
            """
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Architecture", "4-Stream Unified (10k+ dims)")
        with col2:
            st.metric("Ensemble Engine", "OOF Stacking (LGBM + RF + XGB)")
        with col3:
            st.metric("Calibration", "Platt Scaling (Isolated Calib Split)")

        if not pipeline:
            st.warning("⚠️ Saved model artifacts not found. Please run `python run_pipeline.py` to train and generate models.")
        else:
            st.success("✅ All system artifacts loaded and ready for instantaneous inference.")

    # --- TAB 2: DETECTOR ---
    with tabs[1]:
        st.subheader("Interactive Veracity Verification")
        st.markdown("Paste an article's headline and body text below to analyze its veracity.")

        # Preset examples
        example_choice = st.selectbox(
            "Select an Example or Write Custom Text:",
            [
                "Custom Text",
                "Example 1: High Sensationalism (Likely FAKE)",
                "Example 2: Verified Journalistic News (Likely REAL)",
                "Example 3: Borderline Clickbait (Uncertain / Ambiguous)",
            ],
        )

        default_headline = ""
        default_body = ""

        if example_choice == "Example 1: High Sensationalism (Likely FAKE)":
            default_headline = "BOMBSHELL: Secret Documents Expose Horrific Deep-State Treason Scandal!"
            default_body = "Shocking leaked emails apparently show explosive treasonous activity. Insiders reportedly confirmed the terrifying disaster. Pure corruption exposed!"
        elif example_choice == "Example 2: Verified Journalistic News (Likely REAL)":
            default_headline = "Federal Reserve Announces Benchmark Interest Rate Decision Following Policy Meeting"
            default_body = "The Federal Reserve said Wednesday it decided to keep its benchmark interest rate unchanged, confirming economic projections reported by the central bank committee."
        elif example_choice == "Example 3: Borderline Clickbait (Uncertain / Ambiguous)":
            default_headline = "Could This Unexpected Development Transform Industry Projections Next Quarter?"
            default_body = "Rumors suggest potential shifts might occur. While some analysts confirmed talks, others argued results remain completely unverified and speculative."

        headline = st.text_input("Article Headline / Title:", value=default_headline)
        body = st.text_area("Article Body Text:", value=default_body, height=180)

        if st.button("🚀 Analyze Article", type="primary"):
            if not headline.strip() and not body.strip():
                st.error("Please enter a headline or article body.")
            elif not pipeline:
                st.error("Artifacts missing! Run pipeline first.")
            else:
                with st.spinner("Executing 4-Stream Extraction & Calibrated Gating..."):
                    raw, clean, _ = preprocessor.batch_process([headline], [body])

                    # NLP Linguistic signals
                    nlp_res = nlp_explainer.analyze(raw[0], clean[0])
                    st.session_state["nlp_res"] = nlp_res

                    # Extract features
                    X_sparse = pipeline.transform(clean, raw, return_sparse=True)
                    X_dense = balancer.transform(X_sparse)

                    # Predict
                    p_uncal, dis, probas_dict = stacking.predict_proba(X_dense)
                    p_cal = float(calibrator.calibrate(p_uncal)[0])
                    d = float(dis[0])

                    # Gate evaluation
                    gate_res = gate.evaluate_single(p_cal, d)
                    st.session_state["gate_res"] = gate_res
                    st.session_state["p_cal"] = p_cal
                    st.session_state["p_uncal"] = float(p_uncal[0])
                    st.session_state["disagreement"] = d
                    st.session_state["probas_dict"] = {k: float(v[0]) for k, v in probas_dict.items()}
                    st.session_state["x_dense"] = X_dense[0]

        # Display results if available
        if "gate_res" in st.session_state:
            gres = st.session_state["gate_res"]
            dec = gres["decision"]

            st.markdown("---")
            res_col1, res_col2, res_col3 = st.columns([1.5, 1, 1])

            with res_col1:
                if dec == "UNCERTAIN":
                    st.warning(f"### ⚠️ DECISION: UNCERTAIN (ABSTAINED)\n**{gres['explanation']}**")
                elif dec == "FAKE":
                    st.error(f"### 🚨 PREDICTION: FAKE NEWS\n**Confidence: {gres['confidence']:.1%}**")
                else:
                    st.success(f"### ✅ PREDICTION: REAL NEWS\n**Confidence: {gres['confidence']:.1%}**")

            with res_col2:
                st.metric("Calibrated P(FAKE)", f"{st.session_state['p_cal']:.1%}")
                st.metric("Uncalibrated P(FAKE)", f"{st.session_state['p_uncal']:.1%}")

            with res_col3:
                st.metric("Model Disagreement (std)", f"{st.session_state['disagreement']:.3f}")
                st.metric("Gating Trigger", gres["trigger"].replace("_", " ").title())

            # Base model breakdown
            st.markdown("#### Base Learner Probability Breakdown")
            base_cols = st.columns(3)
            with base_cols[0]:
                st.info(f"**LightGBM P(FAKE)**: {st.session_state['probas_dict']['lgbm']:.1%}")
            with base_cols[1]:
                st.info(f"**Random Forest P(FAKE)**: {st.session_state['probas_dict']['rf']:.1%}")
            with base_cols[2]:
                st.info(f"**XGBoost P(FAKE)**: {st.session_state['probas_dict']['xgb']:.1%}")

    # --- TAB 3: NLP ANALYSIS ---
    with tabs[2]:
        st.subheader("Rule-Based NLP Linguistic Signal Summary")
        st.markdown("*Mathematically computed from verified text metrics — Zero LLM hallucination.*")

        if "nlp_res" in st.session_state:
            nlp = st.session_state["nlp_res"]
            st.markdown("### Linguistic Diagnostic Findings")
            st.markdown(nlp["summary"])

            st.markdown("### Dimension Breakdown")
            sigs = nlp["signals"]
            s_col1, s_col2 = st.columns(2)

            with s_col1:
                st.write("**Sensationalism Density:**", f"{sigs['sensationalism']['score']}% ({sigs['sensationalism']['level']})")
                st.write("**Epistemic Hedging:**", f"{sigs['hedging']['score']}% ({sigs['hedging']['level']})")
                st.write("**Source Attribution Verbs:**", f"{sigs['attribution']['score']}% ({sigs['attribution']['level']})")
                st.write("**Emotional Vocabulary:**", f"{sigs['emotional_language']['score']}% ({sigs['emotional_language']['level']})")
                st.write("**Punctuation Intensity:**", sigs["punctuation_intensity"]["details"])

            with s_col2:
                st.write("**Lexical Diversity (TTR):**", f"{sigs['lexical_diversity']['score']} ({sigs['lexical_diversity']['level']})")
                st.write("**Tense Consistency:**", f"{sigs['tense_consistency']['score']} ({sigs['tense_consistency']['level']})")
                st.write("**Readability Level:**", f"{sigs['readability']['level']} (Avg words/sentence: {sigs['readability']['avg_sentence_len']})")
                st.write("**Quotation Density:**", f"{sigs['quotation_density']['score']}% ({sigs['quotation_density']['level']})")
        else:
            st.info("💡 Run an analysis in the **Detector** tab to view linguistic diagnostics.")

    # --- TAB 4: EXPLANATION ---
    with tabs[3]:
        st.subheader("Global & Local Feature Attributions (SHAP)")

        st.markdown("#### Global Feature Stream Importance")
        stream_fig = FIGURES_DIR / "stream_contribution.png"
        if stream_fig.exists():
            st.image(str(stream_fig), caption="Relative Contribution Across 4 Feature Streams")
        else:
            st.info("Run `python run_pipeline.py` to generate the stream contribution chart.")

        if "x_dense" in st.session_state and explainer:
            st.markdown("#### Local Instance Attribution")
            loc = explainer.explain_local_instance(st.session_state["x_dense"], top_k=5)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Top Features Pushing Towards FAKE:**")
                st.dataframe(pd.DataFrame(loc["top_fake_driving"]))
            with c2:
                st.markdown("**Top Features Pushing Towards REAL:**")
                st.dataframe(pd.DataFrame(loc["top_real_driving"]))

    # --- TAB 5: UNCERTAINTY & GATING ---
    with tabs[4]:
        st.subheader("Triple-Trigger Uncertainty Gating")
        rc_fig = FIGURES_DIR / "risk_coverage_curve.png"
        if rc_fig.exists():
            st.image(str(rc_fig), caption="Risk-Coverage Tradeoff Across Operating Points")

        locked_json = METRICS_DIR / "locked_thresholds.json"
        if locked_json.exists():
            st.markdown("#### Locked Operating Thresholds (Validation Split)")
            st.json(load_json_safe(locked_json))

    # --- TAB 6: MODEL COMPARISON ---
    with tabs[5]:
        st.subheader("Model Comparison: Base Learners vs Stacking vs Baseline")
        comp_csv = METRICS_DIR / "comparison_table.csv"
        if comp_csv.exists():
            comp_df = pd.read_csv(comp_csv)
            st.dataframe(comp_df, use_container_width=True)
        else:
            st.info("Run `python run_pipeline.py` to generate the comparison table.")

    # --- TAB 7: EVALUATION ---
    with tabs[6]:
        st.subheader("Evaluation Artifacts & Reliability Diagrams")
        rel_fig = FIGURES_DIR / "reliability_diagram.png"
        if rel_fig.exists():
            st.image(str(rel_fig), caption="Reliability Diagram: Uncalibrated vs Platt Scaled")

        col_cm1, col_cm2 = st.columns(2)
        cm_fig1 = FIGURES_DIR / "cm_cage_fake.png"
        cm_fig2 = FIGURES_DIR / "cm_cage_fake_accepted.png"
        with col_cm1:
            if cm_fig1.exists():
                st.image(str(cm_fig1), caption="Confusion Matrix (Full Test Set)")
        with col_cm2:
            if cm_fig2.exists():
                st.image(str(cm_fig2), caption="Confusion Matrix (Accepted Selective Subset)")

    # --- TAB 8: ABOUT ---
    with tabs[7]:
        st.subheader("About CAGE-Fake")
        st.markdown(
            """
            **CAGE-Fake**: Calibration-Aware Gating and Evidence-Grounded Reasoning for Trustworthy Fake News Triage.
            
            - **Strict Reproducibility**: Fixed random seed (`42`), deterministic tokenization, stratified 70/10/10/10 partitioning.
            - **No Data Leakage**: All scalers, vectorizers, SVD, and calibration models fitted strictly on designated training/calibration splits.
            - **Modular Interpretability**: Stream-level global attribution + Rule-based linguistic diagnostic summaries.
            """
        )


if __name__ == "__main__":
    main()

