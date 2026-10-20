# CAGE-Fake: Calibration-Aware Gating and Evidence-Grounded Reasoning for Trustworthy Fake News Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Interface: Terminal CLI](https://img.shields.io/badge/Interface-Terminal%20CLI-brightgreen.svg)]()
[![Reproducible NLP](https://img.shields.io/badge/Evaluation-Reproducible%20Metrics-orange.svg)]()

A trustworthy, production-grade, terminal-based NLP fake-news triage system designed to know when *not* to trust itself. It learns veracity patterns from linguistic and semantic text distributions with triple-trigger uncertainty gating, calibrated stacking, multi-model disagreement measurement, counterfactual explanations, and automated PDF audit reporting.

> **CRITICAL FACT-CHECKING DISCLAIMER:**
> A trained fake-news classifier predicts linguistic and semantic patterns learned from historical datasets; it **does NOT independently verify whether an empirical claim is factually true or false in the real world**. URL extraction, model pattern prediction, and external journalistic fact verification remain strictly separate.

---

## Key Features & Highlights

1. **Terminal-First Workflow**: Zero web dependencies (Streamlit removed). Fully operable from CLI with formatted colored status outputs, progress tracking, and structured tables.
2. **Leakage-Free 4-Way Split**: Real dataset deduplication (6,094 clean articles from `data/news.csv`), followed by a strict 70/10/10/10 stratified split into `Train`, `Validation`, `Calibration`, and `Test` partitions.
3. **Multi-Stream NLP Engineering**:
   - **Stream A (Lexical)**: 10,000-dimensional TF-IDF unigrams & bigrams with sublinear term-frequency scaling.
   - **Stream B (Semantic)**: Sentence-BERT (`all-MiniLM-L6-v2`) 384-dimensional dense semantic embeddings.
   - **Stream C (Stylometric)**: 15 linguistic markers including epistemic hedging (*allegedly, apparently*), tense consistency, attribution verbs (*stated, reported*), sensationalism, and punctuation dynamics.
   - **Stream D (Readability)**: 4 cognitive complexity metrics (Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog, Dale-Chall).
4. **Calibrated Stacked Ensemble**:
   - 4 Base Models: **Random Forest**, **XGBoost**, **LightGBM**, and **Logistic Regression**.
   - 5-Fold Stratified Out-Of-Fold (OOF) training meta-learner.
   - Post-hoc Platt Scaling calibrated on the dedicated calibration split (reducing Stacking ECE to 0.0276, and reducing Random Forest ECE from 0.1488 to 0.0287).
5. **Uncertainty Gating (Confidence & Disagreement Triggers)**:
   - Employs dual active statistical triggers (with an architectural hook for external evidence verification):
     - **Low Calibrated Confidence**: abstains if $\max(p, 1-p) < \tau^*$ ($\tau^* = 0.80$)
     - **High Inter-Model Disagreement**: abstains if $\text{std}(p_{\text{rf}}, p_{\text{xgb}}, p_{\text{lgbm}}, p_{\text{lr}}) > \delta^*$ ($\delta^* = 0.20$)
   - Yields **94.96% selective accuracy** on accepted predictions at 84.59% test coverage (516/610 accepted samples, 26 errors).
6. **Unique Additions**:
   - **Model Disagreement Meter**: Visualizes consensus across all 5 models (RF, XGBoost, LightGBM, Logistic Regression, Calibrated Stack) and flags conflicting predictions.
   - **Counterfactual Explainer**: Identifies influential words (sensational, hedge, attribution tokens) and simulates probability shifts when neutralized.
   - **News Credibility Fingerprint**: Computes a 6-axis radar profile of stylistic indicators.
   - **"Why this prediction?" Synthesizer**: Transparent, rule-grounded natural language rationale.
7. **Article URL Extraction & PDF Reporting**:
   - Scrapes live web articles using **Trafilatura** with an HTTP `requests` network transport fallback when Trafilatura's native fetcher encounters server blocking or timeout issues (requires `trafilatura` for HTML parsing and DOM text extraction).
   - **Never labels a URL as REAL or FAKE if extraction fails** (aborts cleanly with explicit diagnostics).
   - Generates a publication-grade, multi-page PDF audit report with ReportLab.

---

## Repository Structure

```
CAGE-FAKE/
├── run_pipeline.py                 # Unified CLI entrypoint (--train, --eval, --predict)
├── config.py                       # Central hyperparameter & path configurations
├── requirements.txt                # Pinned dependencies (CPU-friendly)
├── README.md                       # Complete documentation & usage guide
├── data/
│   ├── news.csv                    # Dataset (title, text, label)
│   └── sbert_embeddings_cache.npy # SBERT precomputed vector cache
├── src/
│   ├── data_loader.py              # Schema detection, deduplication, 70/10/10/10 split
│   ├── preprocessing.py            # Text normalization, tokenization, POS tagging
│   ├── lexical_features.py         # Stream A: TF-IDF feature extraction
│   ├── semantic_features.py        # Stream B: Sentence-BERT dense embeddings
│   ├── stylometric_features.py     # Stream C: 15 stylometric markers
│   ├── readability_features.py     # Stream D: 4 readability metrics
│   ├── feature_pipeline.py         # Concatenator & pipeline transformer
│   ├── balancing.py                # TruncatedSVD + SMOTE balancer
│   ├── base_models.py              # RF, XGBoost, LightGBM, Logistic Regression
│   ├── stacking.py                 # OOF 5-fold cross-validated stacking meta-learner
│   ├── calibration.py              # Platt scaling & ECE measurement
│   ├── uncertainty.py              # Triple-trigger uncertainty gate & Pareto tuning
│   ├── explainability.py           # Global & local SHAP/feature attributions
│   ├── unique_features.py          # Disagreement Meter, Counterfactual, Fingerprint
│   ├── url_extractor.py            # Robust Trafilatura URL extractor
│   ├── pdf_generator.py            # ReportLab PDF audit report generator
│   └── predict.py                  # End-to-end inference without retraining
├── models/trained/                 # Saved models, vectorizers, calibrators, gates
├── outputs/
│   ├── figures/                    # Reliability diagram, risk-coverage, confusion matrices
│   ├── reports/                    # Generated audit PDFs, markdown and CSV reports
│   └── cache/                      # Split caches and stage checkpoints
└── tests/
    ├── test_data_loader.py         # Schema, deduplication & split tests
    ├── test_features.py            # Multi-stream feature dimension & content tests
    ├── test_uncertainty.py         # Calibration, ECE & gate threshold tests
    └── test_url_extraction.py      # Trafilatura scraping & failure handling tests
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 (CPU-compatible)
- Git

### 2. Setup Virtual Environment
```bash
# Clone the repository (if not already cloned)
cd CAGE-FAKE

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# Upgrade pip and install dependencies
pip install -r requirements.txt

# Download required spaCy language model
python -m spacy download en_core_web_sm
```

---

## CLI Usage

### Phase 1: Train Models Once & Save
Train all 4 base models, the stacking meta-learner, Platt calibrator, uncertainty gate, and feature pipeline on `data/news.csv`:

```bash
python run_pipeline.py --train
```

- Splits data cleanly into 70% Train (4,265 rows), 10% Validation (609 rows), 10% Calibration (610 rows), and 10% Test (610 rows) [Sum: 6,094 clean unique articles].
- Saves all artifacts into `models/trained/`:
  - `feature_pipeline.joblib`
  - `base_ensemble.joblib`
  - `stacking_model.joblib`
  - `calibrator.joblib`
  - `uncertainty_gate.joblib`
  - `locked_thresholds.json`

### Phase 2: Comprehensive Evaluation & Benchmarking
Evaluate the trained models against the held-out test split (610 unseen articles):

```bash
python run_pipeline.py --eval
```

This generates:
- Model comparison table (`outputs/reports/comparison_table.csv`)
- Ablation study table (`outputs/reports/ablation_results.csv`)
- Reliability diagram (`outputs/figures/reliability_diagram.png`)
- Risk-coverage curve (`outputs/figures/risk-coverage_curve.png`)
- Confusion matrices (`outputs/figures/cm_cage_fake.png`, `outputs/figures/cm_cage_fake_accepted.png`)
- Feature stream importance chart (`outputs/figures/stream_contribution.png`)
- Executive report (`outputs/reports/final_report.md`)

#### Benchmark Results (on 610 Unseen Test Samples)
| Model / Pipeline | Accuracy | Macro F1 | Precision | Recall | ECE | Coverage | Selective Acc | Selective Risk |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest (Uncalibrated) | 0.8557 | 0.8552 | 0.8587 | 0.8548 | 0.1488 | 100.0% | 0.8557 | 14.43% |
| XGBoost (Uncalibrated) | 0.8705 | 0.8703 | 0.8709 | 0.8702 | 0.0371 | 100.0% | 0.8705 | 12.95% |
| LightGBM (Uncalibrated) | 0.8787 | 0.8785 | 0.8797 | 0.8782 | 0.0401 | 100.0% | 0.8787 | 12.13% |
| Logistic Regression (Uncalibrated) | 0.8951 | 0.8951 | 0.8951 | 0.8952 | 0.0372 | 100.0% | 0.8951 | 10.49% |
| Uncalibrated Stack | 0.8984 | 0.8983 | 0.8983 | 0.8984 | 0.0372 | 100.0% | 0.8984 | 10.16% |
| **Calibrated Stack (Full Coverage)** | 0.8967 | 0.8967 | 0.8967 | 0.8966 | **0.0276** | 100.0% | 0.8967 | 10.33% |
| **CAGE-Fake (Calibrated + Gated $\tau=0.80, \delta=0.20$)** | **0.8967** | **0.8967** | **0.8967** | **0.8966** | **0.0276** | **84.59%** | **0.9496** | **5.04%** |

*Under uncertainty gating ($\tau=0.80, \delta=0.20$), the pipeline automatically abstains from 15.41% (94/610) of borderline articles, isolating instances where empirical error is 39.36% (over 7.8x higher than the 5.04% risk on accepted predictions), achieving 94.96% selective accuracy on 516 accepted articles.*

---

### Phase 3: Prediction & Audit Reporting

Predict on any text or live article URL **without retraining the pipeline**:

#### 1. Predict from Live Article URL
```bash
python run_pipeline.py --predict --url "https://apnews.com/article/example-story"
```
*If URL extraction fails or the page lacks readable article text, CAGE-Fake aborts immediately with an `ArticleExtractionError` and never guesses a label.*

#### 2. Predict from Pasted Title & Text
```bash
python run_pipeline.py --predict \
  --title "Senate Passes Historic Clean Energy Infrastructure Deal" \
  --text "Congressional leaders announced a bipartisan agreement on comprehensive clean energy investments. According to official committee spokespersons, the legislation confirms targeted incentives for renewable power generation and grid modernization."
```

#### 3. Interactive Prediction Mode
```bash
python run_pipeline.py --predict
```
Prompts the user to enter either a URL or raw headline and body text directly in the console.

---

## Generated Artifacts

Every prediction generates:
1. **Interactive Console Output**:
   - Model Disagreement Meter across all 5 models.
   - Detailed confidence breakdown and uncertainty gate status.
   - Linguistic signal summary.
   - Counterfactual tokens and probability shifts.
   - "Why this prediction?" synthesis.
2. **Visualizations**:
   - `outputs/figures/disagreement_meter.png`: 5-model consensus bar chart.
   - `outputs/figures/credibility_fingerprint.png`: 6-axis radar profile.
3. **Publication-Grade PDF Report**:
   - Saved automatically to `outputs/reports/audit_report_<timestamp>.pdf`.
   - Contains executive summary, formal methodology disclaimer, model consensus table, visual charts, and counterfactual analysis.

---

## Running the Automated Test Suite

Run the full suite of unit and integration tests:

```bash
pytest tests/ -v
```

Test coverage includes:
- `test_data_loader.py`: Schema detection, deduplication, label mapping, split disjointness.
- `test_features.py`: TF-IDF dimensions, SBERT embedding generation, 15 stylometric features, 4 readability indices.
- `test_uncertainty.py`: Platt scaling fit, ECE computation, uncertainty gate behavior, disagreement thresholding.
- `test_url_extraction.py`: Trafilatura live scraping, fallback parsing, missing content rejection.
