# CAGE-Fake: Calibration-Aware Gating and Evidence-Grounded Reasoning for Trustworthy Fake News Detection

A trustworthy NLP fake-news triage system designed to know when *not* to trust itself. It learns veracity patterns from the dataset (it is NOT evidence-based fact verification).

## Core Philosophy
- **CORRECTNESS & REPRODUCIBILITY**: Fixed seeds, robust error handling, fully deterministic rules.
- **EXPLAINABILITY**: SHAP aggregated per stream + Rule-based Linguistic Signal Summary.
- **TRUSTWORTHINESS**: Triple-trigger Uncertainty Gate (Confidence threshold, Disagreement threshold, Evidence hook).

## Pipeline Steps
1. **Data Inspection**: `data_loader.py` dynamically parses schemas and performs strict 4-way stratified partitioning.
2. **Preprocessing**: `preprocessing.py` maintains dual `raw_text` and `clean_text` representations.
3. **Four Feature Streams**: Lexical (Stream A), Semantic SBERT (Stream B), Stylometric (Stream C), Readability (Stream D).
4. **Balancing**: SMOTE applied safely on a dense TruncatedSVD manifold (`balancing.py`).
5. **OOF Stacking**: Out-of-fold Logistic Regression over LightGBM, Random Forest, XGBoost, and ensemble disagreement (`stacking.py`).
6. **Calibration**: Platt scaling tested on isolated split, computing ECE (`calibration.py`).
7. **Uncertainty Gate**: Selective abstention over $\tau$ and $\delta$ sweeps to maximize precision (`uncertainty.py`).
8. **Explainability**: SHAP attribution and linguistic profiling (`explainability.py`, `nlp_explanation.py`).
9. **Streamlit UI**: `app.py` exposes an interactive triage dashboard.

## Usage
Train the system:
```bash
python experiments/train_cage_fake.py
```
Launch the UI:
```bash
streamlit run app.py
```
