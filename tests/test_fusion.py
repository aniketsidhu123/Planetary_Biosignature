"""
Tests for evaluation modules — calibration and negative controls.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCalibration:
    """Tests for calibration metrics."""

    def test_perfect_calibration(self):
        """Perfect predictions should have near-zero calibration error."""
        from src.evaluation.calibration import compute_calibration_metrics

        rng = np.random.default_rng(42)
        n = 500
        y_true = rng.integers(0, 2, n)
        # Near-perfect probabilities
        y_prob = y_true.astype(float) + rng.normal(0, 0.05, n)
        y_prob = np.clip(y_prob, 0.01, 0.99)

        metrics = compute_calibration_metrics(y_true, y_prob)

        assert metrics["brier_score"] < 0.05, "Brier score should be low for good predictions"
        assert metrics["expected_calibration_error"] < 0.15

    def test_random_calibration(self):
        """Random predictions should have ~0.25 Brier score."""
        from src.evaluation.calibration import compute_calibration_metrics

        rng = np.random.default_rng(42)
        n = 1000
        y_true = rng.integers(0, 2, n)
        y_prob = rng.uniform(0, 1, n)  # Random guesses

        metrics = compute_calibration_metrics(y_true, y_prob)

        # Brier score for random should be ~0.25
        assert 0.15 < metrics["brier_score"] < 0.40

    def test_calibration_report_string(self):
        """Report should be a non-empty formatted string."""
        from src.evaluation.calibration import generate_calibration_report

        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 200)
        y_prob = rng.uniform(0, 1, 200)

        report = generate_calibration_report(y_true, y_prob, model_name="Test Model")

        assert len(report) > 50
        assert "Brier" in report
        assert "Calibration" in report


class TestNegativeControls:
    """Tests for negative control definitions."""

    def test_negative_control_data(self):
        """Negative control definitions should have required fields."""
        from src.evaluation.negative_controls import NEGATIVE_CONTROLS, POSITIVE_CONTROLS

        assert len(NEGATIVE_CONTROLS) > 0
        for name, params in NEGATIVE_CONTROLS.items():
            assert "pl_rade" in params, f"{name} missing pl_rade"
            assert "expected_category" in params, f"{name} missing expected_category"
            assert "max_acceptable_score" in params, f"{name} missing max_acceptable_score"
            assert 0 < params["max_acceptable_score"] < 1

        assert len(POSITIVE_CONTROLS) > 0
        for name, params in POSITIVE_CONTROLS.items():
            assert "pl_rade" in params
            assert "min_acceptable_score" in params

    def test_control_report_generation(self):
        """Control report should format correctly."""
        from src.evaluation.negative_controls import generate_control_report, ControlTestResult

        results = [
            ControlTestResult("Moon", "Very Low", 0.05, 0.15, True, True, "PASS"),
            ControlTestResult("Earth", "High", 0.85, 0.65, True, False, "PASS"),
            ControlTestResult("Jupiter", "Very Low", 0.30, 0.10, False, True, "FAIL"),
        ]

        report = generate_control_report(results)
        assert "Moon" in report
        assert "Earth" in report
        assert "Jupiter" in report
        assert "Summary" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
