# CAGE-Fake: Calibration-Aware Gating and Evidence-Grounded Reasoning for Trustworthy Fake News Detection

A trustworthy NLP fake-news triage system designed to know when *not* to trust itself. It learns veracity patterns from linguistic and semantic text distributions (it is pattern-based triage, NOT external real-time fact verification).

---

## Core Philosophy
1. **Trustworthiness & Selective Classification**: Knowing when to abstain via a Triple-Trigger Uncertainty Gate ($\tau$ confidence, $\delta$ disagreement, and evidence conflict hook).
2. **Leakage Prevention**: All transformations, TF-IDF vocabularies, scalers, SVD projections, SMOTE balancing, and calibration models are fitted strictly on designated splits (Train or Calibration). Never tune on the test split.
3. **Multi-Stream Linguistic Modeling**: 4 complementary feature streams: Lexical (10k TF-IDF + auxiliary), Semantic (Sentence-BERT 384-d), Stylometric (15 markers including novel epistemic hedge, tense consistency, and source attribution), and Readability (4 cognitive complexity metrics).
4. **Transparent Explainability**: Global stream-level attribution + Local SHAP importance + Rule-based NLP Linguistic Signal Summary (mathematically computed, zero LLM hallucination).

---

## Setup & Installation

### 1. Environment Preparation
```bash
# Activate virtual environment
.\.venv\Scripts\activate   # On Windows
# source .venv/bin/activate # On Linux/macOS

# Install pinned dependencies
pip install -r requirements.txt

# Download required spaCy language model
python -m spacy download en_core_web_sm
```

---

## One-Command Full Pipeline (`run_pipeline.py`)

Run the complete pipeline end-to-end, smoke test on subsets, or resume interrupted stages:

```bash
# Full execution across all stages (CPU-safe)
python run_pipeline.py --skip-transformer

# Quick smoke test on 500 rows
python run_pipeline.py --sample 500 --skip-transformer

# Force complete recomputation (ignoring cached stage outputs)
python run_pipeline.py --force

# Run only a specific stage
python run_pipeline.py --stage features
```

### Pipeline Stages Executed in Order:
`data` $\rightarrow$ `preprocess` $\rightarrow$ `features` $\rightarrow$ `balance` $\rightarrow$ `base_models` $\rightarrow$ `stacking` $\rightarrow$ `calibration` $\rightarrow$ `thresholds` $\rightarrow$ `explain` $\rightarrow$ `transformer` $\rightarrow$ `evaluate` $\rightarrow$ `ablation` $\rightarrow$ `report`.

All stage outputs (`.joblib`, `.npy`, `.json`, `.csv`) are cached in `outputs/cache/` for resume support. Logs and execution timing are recorded in `outputs/pipeline.log`.

---

## Streamlit Interactive Dashboard (`app.py`)

Launch the web triage interface:
```bash
streamlit run app.py
```

### Dashboard Tabs:
1. **🏠 Home**: System overview and active configuration status.
2. **🔍 Detector**: Paste any headline and body text to inspect instantaneous `REAL`, `FAKE`, or `UNCERTAIN` triage decisions.
3. **📊 NLP Analysis**: Rule-based linguistic diagnostic findings (sensationalism, epistemic hedging, source attribution verbs, emotional vocabulary).
4. **🧠 Explanation (SHAP)**: Global 4-stream relative importance percentages and top local driving features.
5. **⚖️ Uncertainty & Gating**: Risk-Coverage curves and locked operating thresholds ($\tau^*, \delta^*$).
6. **📈 Model Comparison**: Benchmark comparison table across Random Forest, XGBoost, LightGBM, and CAGE-Fake.
7. **📉 Evaluation & Reliability**: Reliability diagrams (Platt vs Uncalibrated) and confusion matrices.
8. **ℹ️ About CAGE-Fake**: Architecture details, leakage prevention protocols, and research background.

---

## Repository Structure
```
cage_fake_code/
├── app.py                      # Interactive Streamlit dashboard (8 tabs)
├── config.py                   # Central configuration & hyperparameters
├── run_pipeline.py             # One-command CLI pipeline execution
├── requirements.txt            # Pinned Python package dependencies
├── README.md                   # System documentation & usage instructions
├── data/
│   ├── news.csv                # Corpus dataset (title, text, label)
│   └── sbert_embeddings_cache  # Cached .npy and .joblib SBERT embeddings
├── src/
│   ├── data_loader.py          # Auto-schema detection & stratified 70/10/10/10 split
│   ├── preprocessing.py        # Dual raw/clean text & spaCy sentencizer/tagger
│   ├── lexical_features.py     # Stream A: 10,000 TF-IDF + auxiliary lexical metrics
│   ├── semantic_features.py    # Stream B: Sentence-BERT 384-d embeddings (.npy cache)
│   ├── stylometric_features.py # Stream C: 15 markers (hedge, tense, attribution)
│   ├── readability_features.py # Stream D: 4 readability metrics
│   ├── feature_pipeline.py     # Unified 4-stream concatenator (~10,408 dims)
│   ├── balancing.py            # TruncatedSVD manifold projection + SMOTE
│   ├── base_models.py          # LightGBM, Random Forest, XGBoost ensemble
│   ├── stacking.py             # Leakage-free 5-fold Out-Of-Fold (OOF) meta-learner
│   ├── calibration.py          # Platt scaling on isolated calibration split + ECE
│   ├── uncertainty.py          # Triple-trigger uncertainty gate & Pareto grid sweep
│   ├── explainability.py       # Global stream SHAP & local feature attribution
│   ├── nlp_explanation.py      # Rule-based linguistic signal summary engine
│   ├── transformer_baseline.py # DeBERTa-v3-base fallback baseline
│   ├── evaluation.py           # Selective accuracy, coverage, confusion matrices
│   └── utils.py                # Logging, seeds, and serialization utilities
├── outputs/
│   ├── cache/                  # Intermediate stage serialization for resume
│   ├── figures/                # Reliability diagrams, risk-coverage, and confusion matrices
│   ├── metrics/                # Locked thresholds, comparison table, ablations JSON/CSV
│   ├── reports/                # final_report.md
│   └── pipeline.log            # Execution log with stage timing
└── models/
    └── trained/                # Serialized model joblib artifacts
```

---

## Leakage Prevention Protocol
- **Stratified Partitioning**: The dataset is split into Train (70%), Validation (10%), Calibration (10%), and Test (10%).
- **Strict Isolation**: TF-IDF, Scalers, TruncatedSVD, and SMOTE are fitted **strictly on the Train split**.
- **OOF Stacking**: Meta-features are generated using 5-fold cross-validation on Train so the meta-learner never trains on base-learner memorization.
- **Calibration Split**: Platt scaling parameters are fitted exclusively on the Calibration split.
- **Threshold Locking**: Thresholds $\tau^*$ and $\delta^*$ are selected on Validation/Calibration and locked to `outputs/metrics/locked_thresholds.json` before touching the Test split.

