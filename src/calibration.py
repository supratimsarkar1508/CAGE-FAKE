"""
CAGE-Fake: Calibration Module
Implements Platt Scaling (Sigmoid) and Isotonic Regression fitted strictly on the isolated CALIBRATION split.
Computes Expected Calibration Error (ECE) and generates reliability diagrams.
"""

from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from config import CALIBRATION_BINS, FIGURES_DIR
from src.utils import setup_logger

logger = setup_logger("calibration")

def compute_ece(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Computes Expected Calibration Error (ECE):
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    where confidence = max(p, 1-p) and accuracy is whether the predicted class matches y_true.
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)

    preds = (probs >= 0.5).astype(int)
    confidences = np.maximum(probs, 1.0 - probs)
    accuracies = (preds == y_true).astype(float)

    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    bin_confs = []
    bin_accs = []
    bin_sizes = []

    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_sizes.append(bin_size)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)
            bin_sizes.append(0)

    bin_details = {
        "bin_accuracies": np.array(bin_accs),
        "bin_confidences": np.array(bin_confs),
        "bin_sizes": np.array(bin_sizes),
        "bin_boundaries": bin_boundaries,
    }

    return float(ece), bin_details

class TemperatureScaling:
    """
    Temperature Scaling: Single scalar parameter T > 0 that rescales logits:
    p_cal = sigmoid(logit / T)
    T is optimized by minimizing cross-entropy (NLL) on the calibration partition.
    """

    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False

    def fit(self, p_uncalibrated: np.ndarray, y_calib: np.ndarray) -> "TemperatureScaling":
        from scipy.optimize import minimize_scalar
        y = np.asarray(y_calib, dtype=float)
        p = np.clip(np.asarray(p_uncalibrated, dtype=float), 1e-6, 1.0 - 1e-6)
        logits = np.log(p / (1.0 - p))

        def nll(temp):
            if temp <= 0:
                return 1e9
            scaled_logits = logits / temp
            prob = 1.0 / (1.0 + np.exp(-scaled_logits))
            prob = np.clip(prob, 1e-7, 1.0 - 1e-7)
            return -np.mean(y * np.log(prob) + (1.0 - y) * np.log(1.0 - prob))

        res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(res.x)
        self.is_fitted = True
        logger.info(f"Temperature scaling fitted: optimal temperature T = {self.temperature:.4f}")
        return self

    def calibrate(self, p_uncalibrated: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TemperatureScaling must be fitted before calling calibrate.")
        p = np.clip(np.asarray(p_uncalibrated, dtype=float), 1e-6, 1.0 - 1e-6)
        logits = np.log(p / (1.0 - p))
        scaled_logits = logits / self.temperature
        p_cal = 1.0 / (1.0 + np.exp(-scaled_logits))
        return np.clip(p_cal.astype(np.float32), 0.0, 1.0)


class ModelCalibrator:
    """
    Post-hoc calibrator fitted exclusively on the CALIBRATION partition.
    Primary method: Platt Scaling (logistic regression on logit/probability).
    Secondary comparison: Temperature Scaling & Isotonic Regression.
    """

    def __init__(self, method: str = "sigmoid"):
        self.method = method
        self.platt_model = LogisticRegression(C=1.0, solver="lbfgs")
        self.temperature_model = TemperatureScaling()
        self.isotonic_model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.is_fitted = False

    def fit(self, p_uncalibrated: np.ndarray, y_calib: np.ndarray) -> "ModelCalibrator":
        """
        Fits calibration mappings on CALIBRATION split predictions.
        p_uncalibrated: 1D array of uncalibrated P(FAKE) probabilities
        y_calib: 1D binary ground truth
        """
        logger.info(f"Fitting calibrator on {len(p_uncalibrated)} calibration instances...")
        p_2d = np.clip(p_uncalibrated, 1e-6, 1.0 - 1e-6).reshape(-1, 1)

        # 1. Platt scaling: log-odds as input feature
        log_odds = np.log(p_2d / (1.0 - p_2d))
        self.platt_model.fit(log_odds, y_calib)

        # 2. Temperature scaling
        self.temperature_model.fit(p_uncalibrated, y_calib)

        # 3. Isotonic regression
        self.isotonic_model.fit(p_uncalibrated, y_calib)

        self.is_fitted = True
        logger.info("Calibrator fitted successfully.")
        return self

    def calibrate(self, p_uncalibrated: np.ndarray, method: str = None) -> np.ndarray:
        """
        Calibrates uncalibrated probabilities.
        Returns p_calibrated (1.0 = FAKE).
        """
        if not self.is_fitted:
            raise RuntimeError("ModelCalibrator must be fitted before calling calibrate.")

        m = method or self.method
        p_clipped = np.clip(p_uncalibrated, 1e-6, 1.0 - 1e-6)

        if m == "identity":
            p_cal = p_clipped
        elif m == "sigmoid":
            log_odds = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
            p_cal = self.platt_model.predict_proba(log_odds)[:, 1]
        elif m in {"temperature", "temp"}:
            p_cal = self.temperature_model.calibrate(p_clipped)
        elif m == "isotonic":
            p_cal = self.isotonic_model.predict(p_clipped)
        else:
            raise ValueError(f"Unknown calibration method '{m}'")

        return np.clip(p_cal.astype(np.float32), 0.0, 1.0)

    def select_method(self, p_val: np.ndarray, y_val: np.ndarray) -> str:
        """
        Evaluates 'identity', 'sigmoid', 'temperature', and 'isotonic' calibration methods
        on validation split predictions and selects the one achieving the lowest ECE.
        Sets self.method to the winning method and returns its name.
        """
        methods = ["identity", "sigmoid", "temperature", "isotonic"]
        best_method = "identity"
        best_ece = float("inf")
        ece_scores = {}

        for m in methods:
            p_cal_candidate = self.calibrate(p_val, method=m)
            ece, _ = compute_ece(y_val, p_cal_candidate)
            ece_scores[m] = ece
            if ece < best_ece:
                best_ece = ece
                best_method = m

        logger.info(
            f"Calibration method selection on validation split: "
            + ", ".join(f"{m}={ece_scores[m]:.4f}" for m in methods)
            + f" => Selected '{best_method}' (lowest ECE: {best_ece:.4f})"
        )
        self.method = best_method
        return best_method


    def plot_reliability_diagram(
        self,
        y_test: np.ndarray,
        p_uncal: np.ndarray,
        p_cal: np.ndarray,
        save_path: Path = None,
    ) -> Path:
        """
        Generates and saves dual reliability diagram (Uncalibrated vs Platt Calibrated).
        """
        ece_uncal, bins_uncal = compute_ece(y_test, p_uncal)
        ece_cal, bins_cal = compute_ece(y_test, p_cal)

        plt.figure(figsize=(9, 5))

        # Reference diagonal (perfect calibration)
        plt.plot([0.5, 1.0], [0.5, 1.0], "k--", label="Perfect Calibration", linewidth=1.5)

        # Uncalibrated curve
        valid_uncal = bins_uncal["bin_sizes"] > 0
        plt.plot(
            bins_uncal["bin_confidences"][valid_uncal],
            bins_uncal["bin_accuracies"][valid_uncal],
            "s-",
            color="#e74c3c",
            label=f"Uncalibrated (ECE = {ece_uncal:.4f})",
            linewidth=2,
        )

        # Calibrated curve
        valid_cal = bins_cal["bin_sizes"] > 0
        plt.plot(
            bins_cal["bin_confidences"][valid_cal],
            bins_cal["bin_accuracies"][valid_cal],
            "o-",
            color="#2ecc71",
            label=f"Platt Calibrated (ECE = {ece_cal:.4f})",
            linewidth=2,
        )

        plt.title("Reliability Diagram: Calibration Curve Comparison", fontsize=12, fontweight="bold")
        plt.xlabel("Confidence", fontsize=11)
        plt.ylabel("Empirical Accuracy", fontsize=11)
        plt.xlim([0.5, 1.0])
        plt.ylim([0.45, 1.05])
        plt.legend(loc="lower right", frameon=True)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_path = save_path or (FIGURES_DIR / "reliability_diagram.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)

        # Also save vector PDF in outputs/figures/pdf/06_reliability_comparison.pdf
        pdf_dir = FIGURES_DIR / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / "06_reliability_comparison.pdf"
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close()

        logger.info(f"Reliability diagram saved to {out_path} and {pdf_path}")
        return out_path


def select_method(
    p_val: np.ndarray,
    y_val: np.ndarray,
    calibrator: Optional[ModelCalibrator] = None,
) -> str:
    """
    Chooses among identity, sigmoid, temperature, isotonic by lowest ECE on the VALIDATION split.
    If calibrator is passed, uses its fitted models and updates its method; otherwise fits a temporary calibrator.
    """
    if calibrator is None:
        calibrator = ModelCalibrator()
        calibrator.fit(p_val, y_val)
    return calibrator.select_method(p_val, y_val)

