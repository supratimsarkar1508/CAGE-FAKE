"""
CAGE-Fake: Unique Additions & Trustworthiness Intelligence
1. Model Disagreement Meter (separating 4-model gate disagreement from 5-model consensus spread)
2. Counterfactual Explanation (identifying words/signals that shift verdict with signed delta)
3. News Credibility Fingerprint (radar profile of style, sensationalism, attribution, readability with Bayesian length smoothing)
4. "Why this prediction?" Diagnostic Breakdown (strictly aligned with authoritative gate trigger)
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import math
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIGURES_PDF_DIR
from src.utils import setup_logger

logger = setup_logger("unique_features")


class ModelDisagreementMeter:
    """
    Analyzes inter-model consensus and dissent across all 5 models:
    1. Random Forest (Base Model 1)
    2. XGBoost (Base Model 2)
    3. LightGBM (Base Model 3)
    4. Logistic Regression (Base Model 4)
    5. Calibrated Stacked Ensemble (CAGE-Fake Full Pipeline)

    CRITICAL DISTINCTION:
    - Gate Disagreement Metric: std across the 4 base learners: std(p_rf, p_xgb, p_lgbm, p_lr)
    - Five-Model Consensus Spread: std across all 5 models including the Calibrated Stack
    """

    MODEL_NAMES = {
        "rf": "Random Forest",
        "xgb": "XGBoost",
        "lgbm": "LightGBM",
        "lr": "Logistic Regression",
        "stack": "CAGE-Fake Stack",
    }

    @classmethod
    def compute(
        cls,
        probas_dict: Dict[str, float],
        p_calibrated_stack: float,
        base_disagreement_std: float,
        threshold_delta: float = 0.20,
    ) -> Dict[str, Any]:
        """
        Computes consensus statistics across the models.
        """
        def _to_float(v):
            if v is None:
                return 0.5
            return float(np.atleast_1d(v)[0])

        models_data = [
            {"id": "rf", "name": "Random Forest", "prob_fake": _to_float(probas_dict.get("rf", 0.5))},
            {"id": "xgb", "name": "XGBoost", "prob_fake": _to_float(probas_dict.get("xgb", 0.5))},
            {"id": "lgbm", "name": "LightGBM", "prob_fake": _to_float(probas_dict.get("lgbm", 0.5))},
            {"id": "lr", "name": "Logistic Regression", "prob_fake": _to_float(probas_dict.get("lr", 0.5))},
            {"id": "stack", "name": "CAGE-Fake Stack", "prob_fake": _to_float(p_calibrated_stack)},
        ]

        probs_list = [m["prob_fake"] for m in models_data]
        five_model_std = float(np.std(probs_list, ddof=0))
        mean_p = float(np.mean(probs_list))
        min_p = float(np.min(probs_list))
        max_p = float(np.max(probs_list))
        spread = max_p - min_p

        # Model binary votes (>= 0.5 -> FAKE, < 0.5 -> REAL)
        fake_votes = sum(1 for p in probs_list if p >= 0.5)
        real_votes = len(probs_list) - fake_votes

        # Categorize consensus level based on 5-model spread
        if five_model_std < 0.05:
            consensus_level = "Unanimous Consensus"
            level_code = "UNANIMOUS"
            status_desc = "All 5 models exhibit nearly identical probability outputs."
        elif five_model_std < 0.12:
            consensus_level = "Strong Consensus"
            level_code = "STRONG"
            status_desc = "Models strongly agree on the verdict with minor variance."
        elif base_disagreement_std <= threshold_delta:
            consensus_level = "Moderate Agreement"
            level_code = "MODERATE"
            status_desc = "Minor divergence observed, but within acceptable uncertainty limits (std <= delta)."
        else:
            consensus_level = "High Dissent / Conflicting Predictions"
            level_code = "CONFLICT"
            status_desc = f"Base-model disagreement ({base_disagreement_std:.3f}) exceeds safety threshold delta ({threshold_delta:.3f})."

        # Condition checks
        is_vote_split = (fake_votes > 0 and real_votes > 0)
        is_gate_disagreement_triggered = (base_disagreement_std > threshold_delta)

        # Build visual meter string (20 chars) for base disagreement
        meter_len = 20
        fill_count = min(meter_len, int(round((base_disagreement_std / 0.30) * meter_len)))
        meter_bar = "[" + "=" * fill_count + "-" * (meter_len - fill_count) + "]"

        for m in models_data:
            m["vote"] = "FAKE" if m["prob_fake"] >= 0.5 else "REAL"
            m["confidence"] = max(m["prob_fake"], 1.0 - m["prob_fake"])

        return {
            "models": models_data,
            "std_dev": round(five_model_std, 4),
            "gate_base_disagreement": round(base_disagreement_std, 4),
            "five_model_std": round(five_model_std, 4),
            "mean_prob": round(mean_p, 4),
            "spread": round(spread, 4),
            "fake_votes": fake_votes,
            "real_votes": real_votes,
            "consensus_level": consensus_level,
            "level_code": level_code,
            "status_desc": status_desc,
            "is_vote_split": is_vote_split,
            "is_gate_disagreement_triggered": is_gate_disagreement_triggered,
            "meter_bar": meter_bar,
            "threshold_delta": threshold_delta,
        }

    @classmethod
    def plot_meter(cls, disagreement_info: Dict[str, Any], save_path: Path) -> Path:
        """Generates a clean horizontal bar comparison of the 5 models."""
        models = disagreement_info["models"]
        names = [m["name"] for m in models]
        probs = [m["prob_fake"] * 100.0 for m in models]
        colors = ["#e74c3c" if p >= 50 else "#2ecc71" for p in probs]

        plt.figure(figsize=(7, 3.8))
        y_pos = np.arange(len(names))
        bars = plt.barh(y_pos, probs, color=colors, edgecolor="black", height=0.55, alpha=0.85)

        plt.axvline(x=50.0, color="#7f8c8d", linestyle="--", linewidth=1.2, label="Decision Boundary (50%)")

        for bar, p in zip(bars, probs):
            w = bar.get_width()
            text_x = w + 1.5 if w < 85 else w - 12
            text_color = "black" if w < 85 else "white"
            plt.text(text_x, bar.get_y() + bar.get_height() / 2.0, f"{p:.1f}% FAKE", va="center", color=text_color, fontweight="bold", fontsize=9)

        base_std = disagreement_info.get("gate_base_disagreement", 0.0)
        delta = disagreement_info.get("threshold_delta", 0.20)
        plt.yticks(y_pos, names, fontweight="bold", fontsize=9)
        plt.xlabel("Predicted Probability of FAKE (%)", fontweight="bold", fontsize=9)
        plt.xlim(0, 105)
        plt.title(
            f"Five-Model Consensus Breakdown\nBase Disagreement std = {base_std:.3f} (Threshold Delta = {delta:.2f})",
            fontweight="bold", fontsize=10, pad=10
        )
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        pdf_path = save_path.with_suffix(".pdf")
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        try:
            FIGURES_PDF_DIR.mkdir(parents=True, exist_ok=True)
            plt.savefig(FIGURES_PDF_DIR / "02_disagreement_meter.pdf", format="pdf", bbox_inches="tight")
        except Exception:
            pass
        plt.close()
        return save_path


class CounterfactualExplainer:
    """
    Counterfactual Sensitivity Analysis:
    Identifies high-leverage lexical markers (sensational adjectives, speculative hedges,
    and shout punctuation) present in the article, applies targeted neutralization,
    and records the signed probability shift:
        signed_prob_shift = transformed_prob - original_prob
    """

    SENSATIONAL_SYNONYMS = {
        "shocking": "notable",
        "bombshell": "announcement",
        "unbelievable": "unexpected",
        "stunning": "significant",
        "mind-blowing": "substantial",
        "exposed": "disclosed",
        "secretly": "privately",
        "uncovered": "found",
        "horrific": "severe",
        "explosive": "critical",
        "insane": "unusual",
        "conspiracy": "collaboration",
        "scandal": "incident",
        "treason": "violation",
        "apocalypse": "crisis",
        "nightmare": "difficulty",
        "panic": "concern",
        "shocker": "development",
        "massive": "large",
        "devastating": "serious",
    }

    HEDGE_REPLACEMENTS = {
        "allegedly": "according to official records,",
        "reportedly": "in confirmed statements,",
        "purportedly": "in cited testimony,",
        "supposedly": "as documented,",
        "rumor": "verified report",
        "rumored": "confirmed",
        "unconfirmed": "verified",
        "unverified": "cross-referenced",
    }

    @classmethod
    def generate(
        cls,
        clean_text: str,
        raw_text: str,
        original_p_fake: float,
        predict_fn,  # callable: (clean, raw) -> p_fake
        top_k_words: int = 5,
    ) -> Dict[str, Any]:
        """
        Runs counterfactual perturbation:
        1. Identifies high-leverage tokens present in article
        2. Applies neutralizations and records exact changed fragments
        3. Measures change in P(FAKE) through model re-scoring
        """
        tokens = clean_text.lower().split()

        # Check for sensational and hedge hits
        sensational_found = [w for w in set(tokens) if w in cls.SENSATIONAL_SYNONYMS]
        hedge_found = [w for w in set(tokens) if w in cls.HEDGE_REPLACEMENTS]
        excl_count = raw_text.count("!")
        caps_tokens = [tok for tok in raw_text.split() if tok.isupper() and len(tok) > 1 and tok.isalpha()]

        changed_fragments: List[Dict[str, str]] = []
        perturbed_clean = clean_text
        perturbed_raw = raw_text

        # Perturbation 1: Replace sensational words with neutral equivalents
        for word in sensational_found[:top_k_words]:
            replacement = cls.SENSATIONAL_SYNONYMS[word]
            perturbed_clean = re.sub(rf"\b{re.escape(word)}\b", replacement, perturbed_clean, flags=re.IGNORECASE)
            changed_fragments.append({
                "original": word,
                "replacement": replacement,
                "category": "Sensational Vocabulary",
                "description": f"Replaced sensational adjective '{word}' -> '{replacement}'",
            })

        # Perturbation 2: Anchor speculative hedges with attribution phrasing
        for word in hedge_found[:3]:
            replacement = cls.HEDGE_REPLACEMENTS[word]
            perturbed_clean = re.sub(rf"\b{re.escape(word)}\b", replacement, perturbed_clean, flags=re.IGNORECASE)
            changed_fragments.append({
                "original": word,
                "replacement": replacement,
                "category": "Epistemic Hedging",
                "description": f"Grounded speculative hedge '{word}' -> '{replacement}'",
            })

        # Perturbation 3: Strip sensational exclamation points
        if excl_count > 0:
            perturbed_raw = perturbed_raw.replace("!", ".")
            changed_fragments.append({
                "original": "!",
                "replacement": ".",
                "category": "Punctuation Intensity",
                "description": f"Normalized {excl_count} exclamation points to standard periods",
            })

        # Perturbation 4: Normalize shout uppercase words
        if len(caps_tokens) > 0:
            perturbed_raw = " ".join(w.capitalize() if (w.isupper() and len(w) > 1 and w.isalpha()) else w for w in perturbed_raw.split())
            changed_fragments.append({
                "original": ", ".join(caps_tokens[:3]),
                "replacement": ", ".join(w.capitalize() for w in caps_tokens[:3]),
                "category": "Shout Tokens",
                "description": f"Lowercased {len(caps_tokens)} ALL-CAPS words",
            })

        is_modified = len(changed_fragments) > 0

        if not is_modified:
            simulated_p = original_p_fake
            simulation_status = "unmodified"
            summary = (
                "No qualifying sensational words, speculative hedges, or shout punctuation detected in the article text; "
                "no counterfactual perturbations were applied."
            )
        else:
            try:
                simulated_p = float(predict_fn(perturbed_clean, perturbed_raw))
                simulation_status = "model_inference"
                shift = simulated_p - original_p_fake
                orig_verdict = "FAKE" if original_p_fake >= 0.5 else "REAL"
                sim_verdict = "FAKE" if simulated_p >= 0.5 else "REAL"
                verdict_flipped = orig_verdict != sim_verdict

                if verdict_flipped:
                    summary = (
                        f"By neutralizing {len(changed_fragments)} linguistic markers, the predicted probability of FAKE "
                        f"shifted by {shift * 100:+.1f} percentage points (from {original_p_fake:.1%} to {simulated_p:.1%}), "
                        f"flipping the text classifier's verdict from {orig_verdict} to {sim_verdict} "
                        f"(note: the final triage verdict may differ when evidence overrides it)."
                    )
                else:
                    summary = (
                        f"Neutralizing {len(changed_fragments)} linguistic markers resulted in a signed probability shift "
                        f"of {shift * 100:+.1f} percentage points (from {original_p_fake:.1%} to {simulated_p:.1%}), "
                        f"retaining the text classifier's verdict ({orig_verdict}) "
                        f"(note: the final triage verdict may differ when evidence overrides it)."
                    )
            except Exception as e:
                logger.warning(f"Counterfactual simulation re-scoring failed: {e}")
                simulated_p = original_p_fake
                simulation_status = f"failed: {e}"
                summary = "Counterfactual perturbation re-scoring could not be completed; model evaluation failed."

        signed_prob_shift = float(simulated_p - original_p_fake)
        signed_shift_pct = float(signed_prob_shift * 100.0)

        return {
            "original_prob": round(original_p_fake, 4),
            "simulated_prob": round(simulated_p, 4),
            "signed_prob_shift": round(signed_prob_shift, 4),
            "signed_shift_pct": round(signed_shift_pct, 2),
            "is_modified": is_modified,
            "simulation_status": simulation_status,
            "changed_fragments": changed_fragments,
            "sensational_words_found": sensational_found,
            "hedges_found": hedge_found,
            "summary": summary,
            "disclaimer": (
                "Counterfactual sensitivity analysis demonstrates local model sensitivity under lexical perturbation; "
                "it does NOT verify whether empirical claims are factually true in reality."
            ),
        }


class NewsCredibilityFingerprint:
    """
    Computes a multi-dimensional journalistic credibility profile across 6 axes:
    1. Sensationalism Discipline: Avoidance of tabloid/clickbait vocabulary
    2. Source Attribution: Presence of journalistic attribution verbs (stated, reported, confirmed)
    3. Epistemic Grounding: Avoidance of unanchored speculation and rumors
    4. Punctuation Discipline: Restrained use of exclamation marks and all-caps shout words
    5. Readability Structure: Adherence to standard news sentence complexity (15-22 words/sentence)
    6. Lexical Diversity: Healthy vocabulary richness (Type-Token Ratio)

    Length-Aware Normalization:
    Uses Bayesian length smoothing to avoid 0.0 or 100.0 boundary artifacts on short texts.
    """

    DIMENSIONS = [
        "Sensationalism Discipline",
        "Source Attribution",
        "Epistemic Grounding",
        "Punctuation Discipline",
        "Readability Structure",
        "Lexical Diversity",
    ]

    @classmethod
    def compute(cls, signals: Dict[str, Any], word_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Derives normalized 0-100 scores for each dimension (100 = optimal journalistic practice).
        """
        N = word_count if word_count is not None else 100
        # Bayesian length-shrinkage weight: short texts shrink toward neutral baseline (65.0)
        # weight in [0, 1], approaches 1 as N increases
        length_weight = float(N / (N + 40.0))

        # 1. Sensationalism Discipline (Target: 0 sensational words)
        # Raw density is percentage of sensational words (0.0% to 5.0%)
        sens_density = float(signals.get("sensationalism", {}).get("score", 0.0))
        raw_sens = max(10.0, min(95.0, 95.0 - (sens_density * 25.0)))
        score_sens = length_weight * raw_sens + (1.0 - length_weight) * 65.0

        # 2. Source Attribution (Target: 0.5% - 2.0% attribution verbs)
        # Neutral baseline for 0 attribution is 45.0 (unattributed claim, not total failure)
        attr_density = float(signals.get("attribution", {}).get("score", 0.0))
        raw_attr = max(20.0, min(95.0, 45.0 + (attr_density * 25.0)))
        score_attr = length_weight * raw_attr + (1.0 - length_weight) * 55.0

        # 3. Epistemic Grounding (Target: low unverified hedging)
        hedge_density = float(signals.get("hedging", {}).get("score", 0.0))
        raw_hedge = max(15.0, min(95.0, 95.0 - (hedge_density * 20.0)))
        score_hedge = length_weight * raw_hedge + (1.0 - length_weight) * 65.0

        # 4. Punctuation Discipline (Target: few exclamations and shout words)
        punct_score = float(signals.get("punctuation_intensity", {}).get("score", 0.0))
        raw_punct = max(15.0, min(95.0, 95.0 - (punct_score * 20.0)))
        score_punct = length_weight * raw_punct + (1.0 - length_weight) * 70.0

        # 5. Readability Structure (Optimal news sentence length is 15-22 words)
        words_per_sent = float(signals.get("readability", {}).get("words_per_sentence", 18.0))
        if words_per_sent < 15.0:
            dist = 15.0 - words_per_sent
        elif words_per_sent > 22.0:
            dist = words_per_sent - 22.0
        else:
            dist = 0.0
        raw_read = max(25.0, min(95.0, 95.0 - (dist * 2.5)))
        score_read = length_weight * raw_read + (1.0 - length_weight) * 65.0

        # 6. Lexical Diversity (TTR target: 0.45 - 0.70)
        ttr = float(signals.get("lexical_diversity", {}).get("score", 0.5))
        raw_ttr = max(25.0, min(95.0, ttr * 130.0))
        score_ttr = length_weight * raw_ttr + (1.0 - length_weight) * 60.0

        scores = {
            "Sensationalism Discipline": round(score_sens, 1),
            "Source Attribution": round(score_attr, 1),
            "Epistemic Grounding": round(score_hedge, 1),
            "Punctuation Discipline": round(score_punct, 1),
            "Readability Structure": round(score_read, 1),
            "Lexical Diversity": round(score_ttr, 1),
        }

        # Overall Credibility Index (weighted average)
        overall_index = (
            score_sens * 0.25 +
            score_attr * 0.25 +
            score_hedge * 0.20 +
            score_punct * 0.10 +
            score_read * 0.10 +
            score_ttr * 0.10
        )

        grade = "A (Restrained / Professional Copy)" if overall_index >= 80 else (
            "B (Standard News Distribution)" if overall_index >= 65 else (
                "C (Sensational / Speculative)" if overall_index >= 50 else "D (Deceptive / Tabloid Style)"
            )
        )

        return {
            "scores": scores,
            "overall_credibility_index": round(overall_index, 1),
            "credibility_grade": grade,
            "word_count": N,
            "length_weight": round(length_weight, 3),
            "methodology_note": (
                "The News Credibility Fingerprint measures predefined linguistic and stylistic characteristics "
                "characteristic of news copy; it does NOT verify factual truth or real-world source authenticity."
            ),
        }

    @classmethod
    def plot_radar(cls, fingerprint: Dict[str, Any], save_path: Path) -> Path:
        """Plots a 6-axis radar / spider chart of the credibility fingerprint."""
        scores_dict = fingerprint["scores"]
        categories = list(scores_dict.keys())
        values = list(scores_dict.values())

        # Close the loop
        categories += [categories[0]]
        values += [values[0]]

        N = len(categories) - 1
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += [angles[0]]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], categories[:-1], color="#2c3e50", size=9, fontweight="bold")
        ax.set_rlabel_position(0)
        plt.yticks([25, 50, 75, 100], ["25", "50", "75", "100"], color="#7f8c8d", size=8)
        plt.ylim(0, 105)

        # Plot article profile
        ax.plot(angles, values, color="#2980b9", linewidth=2.5, linestyle="solid", label="Article Fingerprint")
        ax.fill(angles, values, color="#3498db", alpha=0.35)

        # Reference baseline (standard journalism benchmark = 70 on all axes)
        ref_vals = [70] * (N + 1)
        ax.plot(angles, ref_vals, color="#7f8c8d", linewidth=1.2, linestyle="--", label="Journalistic Benchmark (70)")

        idx = fingerprint["overall_credibility_index"]
        grade = fingerprint["credibility_grade"]
        plt.title(
            f"News Credibility Fingerprint\nIndex: {idx}/100 - Grade: {grade}",
            size=11, fontweight="bold", y=1.1, pad=14
        )
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        plt.tight_layout()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        pdf_path = save_path.with_suffix(".pdf")
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        try:
            FIGURES_PDF_DIR.mkdir(parents=True, exist_ok=True)
            plt.savefig(FIGURES_PDF_DIR / "01_credibility_radar.pdf", format="pdf", bbox_inches="tight")
        except Exception:
            pass
        plt.close()
        return save_path


def synthesize_why_prediction(
    gate_result: Dict[str, Any],
    disagreement_info: Dict[str, Any],
    credibility_info: Dict[str, Any],
    local_features: Dict[str, Any],
) -> str:
    """
    Synthesizes an authoritative 'Why this prediction?' diagnostic rationale
    based STRICTLY on the single unified prediction result object.
    Never invents or conflates triggers.
    """
    decision = gate_result["decision"]
    trigger = gate_result.get("trigger", "none")
    confidence = gate_result["confidence"]
    disagreement = gate_result["disagreement"]
    tau = gate_result.get("tau", 0.80)
    delta = gate_result.get("delta", 0.20)

    consensus = disagreement_info["consensus_level"]
    cred_index = credibility_info["overall_credibility_index"]
    cred_grade = credibility_info["credibility_grade"]

    lines = []
    lines.append(f"1. Decision & Triage Status: The CAGE-Fake pipeline reached a verdict of **{decision}** with **{confidence:.1%} calibrated confidence**.")

    if decision == "UNCERTAIN":
        if trigger == "both_triggers":
            lines.append(
                f"   - **Gating Trigger (Dual Uncertainty Active)**: Both safety triggers were activated. "
                f"Calibrated confidence ({confidence:.1%}) fell below threshold tau ({tau:.1%}), "
                f"AND base-model disagreement ({disagreement:.3f}) exceeded threshold delta ({delta:.3f}). "
                f"The system abstains to prevent false positives under both statistical ambiguity and ensemble conflict."
            )
        elif trigger == "low_confidence":
            lines.append(
                f"   - **Gating Trigger (Low Confidence)**: Calibrated confidence ({confidence:.1%}) is below threshold tau ({tau:.1%}). "
                f"Base models exhibited acceptable consensus (disagreement = {disagreement:.3f} <= {delta:.3f}). "
                f"The system abstains because the article displays mixed or borderline stylistic distributions."
            )
        elif trigger == "high_disagreement":
            lines.append(
                f"   - **Gating Trigger (High Disagreement)**: Base-model disagreement ({disagreement:.3f}) exceeded threshold delta ({delta:.3f}). "
                f"Although confidence was {confidence:.1%}, the base classifiers sharply diverged in their verdicts. "
                f"The system abstains to ensure decisions require multi-model consensus."
            )
        elif trigger == "evidence_conflict":
            lines.append(
                f"   - **Gating Trigger (Evidence Conflict)**: An active conflict with verified external evidence was detected."
            )
        else:
            lines.append(f"   - **Gating Trigger**: System abstention triggered under safety policy ({trigger}).")
    elif decision == "FAKE":
        lines.append(f"   - **Model Consensus**: {consensus} ({disagreement_info['fake_votes']}/5 models voted FAKE, average P(FAKE)={disagreement_info['mean_prob']:.1%}).")
        top_fake = [f["name"] for f in local_features.get("fake_signals", [])[:3]]
        if top_fake:
            lines.append(f"   - **Driving Deceptive Markers**: {', '.join(top_fake)}.")
        lines.append(f"   - **Credibility Score**: Article credibility index is {cred_index}/100 ({cred_grade}).")
    else:  # REAL
        lines.append(f"   - **Model Consensus**: {consensus} ({disagreement_info['real_votes']}/5 models voted REAL, average P(FAKE)={disagreement_info['mean_prob']:.1%}).")
        top_real = [f["name"] for f in local_features.get("real_signals", [])[:3]]
        if top_real:
            lines.append(f"   - **Grounded Linguistic Anchors**: {', '.join(top_real)}.")
        lines.append(f"   - **Credibility Score**: Journalistic integrity index is {cred_index}/100 ({cred_grade}).")

    return "\n".join(lines)
