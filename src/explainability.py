"""
CAGE-Fake: Explainability and Attribution Module
Global per-stream attribution (A: Lexical, B: Semantic, C: Stylometric, D: Readability),
stream contribution charts, and local human-readable feature attributions.
"""

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SHAP_BACKGROUND_SIZE, SHAP_CACHE_FILE, FIGURES_DIR
from src.utils import setup_logger

logger = setup_logger("explainability")

HUMAN_FEATURE_NAMES = {
    "p_rf": "Random Forest Prediction [P(FAKE)]",
    "p_xgb": "XGBoost Prediction [P(FAKE)]",
    "p_lgbm": "LightGBM Prediction [P(FAKE)]",
    "p_lr": "Logistic Regression Prediction [P(FAKE)]",
    "disagreement": "Inter-Model Base Disagreement (std)",
}

STYLO_READABLE_NAMES = {
    "sensationalism": "Sensational Vocabulary Density",
    "hedging": "Epistemic Hedging / Speculation Density",
    "attribution": "Journalistic Source Attribution Density",
    "punctuation_intensity": "Sensational Punctuation / Shout Intensity",
    "lexical_diversity": "Lexical Diversity (Type-Token Ratio)",
    "readability": "Readability Complexity",
    "tense_consistency": "Tense Consistency (Past vs Present)",
    "named_entities": "Proper Noun / Entity Density",
    "quotations": "Direct Quotation Density",
}


class SHAPExplainer:
    """
    Computes and aggregates attributions across feature streams:
    - Global: Relative importance percentage across Stream A, B, C, D
    - Local: Top FAKE-driving (pushing towards FAKE) and REAL-driving (pushing towards REAL) signals
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        stream_boundaries: Dict[str, Tuple[int, int]],
        background_size: int = SHAP_BACKGROUND_SIZE,
        cache_path: Path = SHAP_CACHE_FILE,
    ):
        self.model = model
        self.feature_names = feature_names
        self.stream_boundaries = stream_boundaries
        self.background_size = background_size
        self.cache_path = cache_path
        self.explainer = None
        self.background_data = None

    def fit(self, X_background: np.ndarray) -> "SHAPExplainer":
        """Initializes SHAP TreeExplainer or LinearExplainer with background sample."""
        try:
            import shap
            n_bg = min(len(X_background), self.background_size)
            indices = np.random.RandomState(42).choice(len(X_background), size=n_bg, replace=False)
            self.background_data = X_background[indices]

            try:
                self.explainer = shap.LinearExplainer(self.model, data=self.background_data)
                logger.info("Initialized shap.LinearExplainer on background data.")
            except Exception:
                try:
                    self.explainer = shap.TreeExplainer(self.model, data=self.background_data)
                    logger.info("Initialized shap.TreeExplainer on background data.")
                except Exception as e:
                    logger.warning(f"Defaulting to coefficient/feature importance attributions: {e}")
                    self.explainer = None
        except ImportError:
            logger.warning("SHAP not installed. Explanations will use model coefficients/importances.")
            self.explainer = None

        return self

    def compute_stream_importance(self, X_sample: np.ndarray) -> Dict[str, float]:
        """
        Calculates percentage contribution of each feature stream (A, B, C, D)
        using aggregated absolute feature importance / coefficients.
        """
        if hasattr(self.model, "feature_importances_"):
            importances = np.abs(self.model.feature_importances_)
        elif hasattr(self.model, "coef_"):
            importances = np.abs(np.asarray(self.model.coef_).flatten())
        else:
            importances = np.ones(len(self.feature_names), dtype=np.float32)

        stream_scores = {}
        total_imp = 0.0

        for stream_name, (start_idx, end_idx) in self.stream_boundaries.items():
            sub_imp = float(np.sum(importances[start_idx:end_idx]))
            stream_scores[stream_name] = sub_imp
            total_imp += sub_imp

        total_imp = max(1e-6, total_imp)
        stream_percentages = {
            k: round((v / total_imp) * 100.0, 2)
            for k, v in stream_scores.items()
        }

        return stream_percentages

    def explain_local_instance(
        self,
        meta_features: np.ndarray,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Computes local feature attribution for the meta-learner instance.
        meta_features: [p_rf, p_xgb, p_lgbm, p_lr, disagreement]
        """
        x_vec = np.asarray(meta_features).flatten()

        if hasattr(self.model, "coef_"):
            coefs = np.asarray(self.model.coef_).flatten()
            if len(coefs) == len(x_vec):
                impacts = (x_vec - 0.5) * coefs
            else:
                impacts = x_vec
        elif hasattr(self.model, "feature_importances_"):
            imp = np.asarray(self.model.feature_importances_).flatten()
            impacts = (x_vec - 0.5) * imp if len(imp) == len(x_vec) else x_vec
        else:
            impacts = x_vec - 0.5

        signals = []
        for i, val in enumerate(x_vec):
            raw_name = self.feature_names[i] if i < len(self.feature_names) else f"meta_feature_{i}"
            readable_name = HUMAN_FEATURE_NAMES.get(raw_name, raw_name)
            impact = float(impacts[i]) if i < len(impacts) else 0.0
            direction = "FAKE" if impact > 0 else "REAL"
            signals.append({
                "id": raw_name,
                "name": readable_name,
                "value": round(float(val), 4),
                "impact": round(impact, 4),
                "direction": direction,
                "explanation": f"Pushes towards {direction} (weight impact = {impact:+.3f})"
            })

        fake_signals = sorted([s for s in signals if s["direction"] == "FAKE"], key=lambda x: abs(x["impact"]), reverse=True)
        real_signals = sorted([s for s in signals if s["direction"] == "REAL"], key=lambda x: abs(x["impact"]), reverse=True)

        return {
            "all_signals": signals,
            "fake_signals": fake_signals[:top_k],
            "real_signals": real_signals[:top_k],
        }

    def plot_stream_contributions(
        self,
        stream_data: Dict[str, float],
        save_path: Optional[Path] = None,
    ) -> Path:
        """
        Generates stream ablation impact chart visualizing the empirical delta in Macro-F1
        when each individual stream is removed from the multi-stream pipeline.
        """
        plt.figure(figsize=(7.5, 4.5))
        labels = [f"Without {k.capitalize()}" for k in stream_data.keys()]
        values = list(stream_data.values())

        # Check if values are deltas (small floats) or percentages
        is_delta = any(v < 1.0 for v in values)
        if is_delta:
            colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in values]
            bars = plt.bar(labels, values, color=colors, edgecolor="black", alpha=0.85)
            for bar in bars:
                height = bar.get_height()
                va = "bottom" if height >= 0 else "top"
                offset = 0.001 if height >= 0 else -0.002
                plt.text(bar.get_x() + bar.get_width() / 2.0, height + offset, f"{height:+.4f}", ha="center", va=va, fontweight="bold", fontsize=9)
            plt.axhline(0, color="black", linestyle="--", linewidth=1.0)
            plt.title("CAGE-Fake: Feature Stream Ablation Impact on Test Set", fontsize=11, fontweight="bold")
            plt.ylabel("Delta Macro-F1 vs. Full Model", fontsize=10)
            y_min = min(values) - 0.005
            y_max = max(values) + 0.005
            plt.ylim([min(-0.015, y_min), max(0.020, y_max)])
        else:
            colors = ["#3498db", "#9b59b6", "#e67e22", "#1abc9c"]
            bars = plt.bar(labels, values, color=colors[:len(labels)], edgecolor="black", alpha=0.85)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1.0, f"{height:.1f}%", ha="center", va="bottom", fontweight="bold")
            plt.title("CAGE-Fake: Feature Stream Relative Importance", fontsize=11, fontweight="bold")
            plt.ylabel("Relative Contribution (%)", fontsize=10)
            plt.ylim([0, max(values) + 14])

        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_path = save_path or (FIGURES_DIR / "stream_contribution.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()

        logger.info(f"Stream ablation plot saved to {out_path}")
        return out_path

    plot_stream_ablations = plot_stream_contributions

