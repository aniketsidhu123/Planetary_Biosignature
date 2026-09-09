"""
Calibration metrics for ExoScope models.

Evaluates whether predicted probabilities match observed frequencies,
which matters more than raw accuracy for a scoring system like HBLI.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict
from pathlib import Path

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from src.utils.logging_setup import get_logger

logger = get_logger("evaluation.calibration")


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute calibration quality metrics.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins for calibration curve.

    Returns:
        Dict of calibration metrics.
    """
    # Brier score (lower is better, 0 = perfect)
    brier = brier_score_loss(y_true, y_prob)

    # Log loss
    ll = log_loss(y_true, y_prob)

    # Expected Calibration Error (ECE)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))

    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(fraction_of_positives - mean_predicted_value))

    # Overconfidence ratio: fraction of bins where predicted > observed
    overconfident_bins = np.sum(mean_predicted_value > fraction_of_positives)
    overconfidence_ratio = overconfident_bins / len(fraction_of_positives)

    metrics = {
        "brier_score": float(brier),
        "log_loss": float(ll),
        "expected_calibration_error": float(ece),
        "max_calibration_error": float(mce),
        "overconfidence_ratio": float(overconfidence_ratio),
        "n_bins": n_bins,
    }

    logger.info(
        f"Calibration: Brier={brier:.4f}, ECE={ece:.4f}, MCE={mce:.4f}, "
        f"Overconfidence={overconfidence_ratio:.2f}"
    )

    return metrics


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "ExoScope HBLI",
    n_bins: int = 10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a reliability diagram (calibration curve).

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        model_name: Name for the plot legend.
        n_bins: Number of calibration bins.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [3, 1]})

    # Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)
    ax1.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        color="#6C63FF",
        label=model_name,
        markersize=8,
    )

    # Fill between for calibration gap
    ax1.fill_between(
        mean_predicted_value,
        fraction_of_positives,
        mean_predicted_value,
        alpha=0.15,
        color="#FF6B6B",
        label="Calibration gap",
    )

    ax1.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax1.set_ylabel("Fraction of Positives", fontsize=12)
    ax1.set_title("Reliability Diagram (Calibration Curve)", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)

    # Histogram of predictions
    ax2.hist(y_prob, bins=50, color="#6C63FF", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Predicted Probability", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Distribution of Predictions", fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Calibration plot saved to {save_path}")

    return fig


def generate_calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "ExoScope HBLI",
    save_dir: Optional[str] = None,
) -> str:
    """Generate a full calibration report with metrics and plot.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        model_name: Model name for report.
        save_dir: Directory to save report artifacts.

    Returns:
        Formatted report string.
    """
    metrics = compute_calibration_metrics(y_true, y_prob)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_calibration_curve(
            y_true, y_prob, model_name,
            save_path=str(save_dir / "calibration_curve.png")
        )

    report_lines = [
        f"{'='*60}",
        f"Calibration Report — {model_name}",
        f"{'='*60}",
        f"",
        f"Brier Score:                  {metrics['brier_score']:.4f}  (0 = perfect, 0.25 = random)",
        f"Log Loss:                     {metrics['log_loss']:.4f}",
        f"Expected Calibration Error:   {metrics['expected_calibration_error']:.4f}  (lower = better calibrated)",
        f"Max Calibration Error:        {metrics['max_calibration_error']:.4f}",
        f"Overconfidence Ratio:         {metrics['overconfidence_ratio']:.2f}  (fraction of bins overconfident)",
        f"",
        f"Interpretation:",
    ]

    ece = metrics["expected_calibration_error"]
    if ece < 0.05:
        report_lines.append("  ✅ Well-calibrated — predicted probabilities closely match observed frequencies.")
    elif ece < 0.10:
        report_lines.append("  ⚠️  Moderately calibrated — some deviation between predicted and observed frequencies.")
    else:
        report_lines.append("  ❌ Poorly calibrated — predicted probabilities do not match observed frequencies.")
        report_lines.append("     Consider recalibrating with isotonic regression or Platt scaling.")

    if metrics["overconfidence_ratio"] > 0.7:
        report_lines.append("  ⚠️  Model tends to be overconfident — predicted probabilities are too high.")

    return "\n".join(report_lines)
