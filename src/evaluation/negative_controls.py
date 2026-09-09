"""
Negative control testing for ExoScope.

Tests the HBLI system against known-abiotic, known-lifeless targets
to verify the system doesn't over-trigger (false positives are the
historical failure mode in astrobiology).

Control targets:
  - Moon: no atmosphere, no water, no biology → should score near 0
  - Mercury: extreme temperatures, no atmosphere → near 0
  - Gas giants: fundamentally different, non-rocky → very low
  - Known sterile geological formations
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

from src.utils.logging_setup import get_logger

logger = get_logger("evaluation.negative_controls")


# ── Known negative control targets ────────────────────────────────────
# Tabular parameters for known non-habitable bodies
NEGATIVE_CONTROLS = {
    "Moon": {
        "pl_rade": 0.273,       # Earth radii
        "pl_bmasse": 0.0123,    # Earth masses
        "pl_dens": 3.344,       # g/cm³
        "pl_orbeccen": 0.0549,
        "pl_orbsmax": 1.0,      # Orbits Earth at 1 AU
        "pl_insol": 1.0,        # Same as Earth
        "pl_eqt": 250,          # K (varies wildly)
        "st_teff": 5778,        # Sun
        "st_lum": 0.0,          # log(1.0) = 0
        "st_mass": 1.0,
        "expected_category": "Very Low Likelihood",
        "max_acceptable_score": 0.15,
    },
    "Mercury": {
        "pl_rade": 0.383,
        "pl_bmasse": 0.055,
        "pl_dens": 5.427,
        "pl_orbeccen": 0.2056,
        "pl_orbsmax": 0.387,
        "pl_insol": 6.67,
        "pl_eqt": 440,
        "st_teff": 5778,
        "st_lum": 0.0,
        "st_mass": 1.0,
        "expected_category": "Very Low Likelihood",
        "max_acceptable_score": 0.15,
    },
    "Jupiter": {
        "pl_rade": 11.21,
        "pl_bmasse": 317.8,
        "pl_dens": 1.326,
        "pl_orbeccen": 0.0489,
        "pl_orbsmax": 5.203,
        "pl_insol": 0.037,
        "pl_eqt": 110,
        "st_teff": 5778,
        "st_lum": 0.0,
        "st_mass": 1.0,
        "expected_category": "Very Low Likelihood",
        "max_acceptable_score": 0.10,
    },
    "Hot_Jupiter_WASP12b": {
        "pl_rade": 20.73,       # 1.85 Jupiter radii ≈ 20.73 Earth radii
        "pl_bmasse": 460.0,     # 1.45 Jupiter masses
        "pl_dens": 0.267,
        "pl_orbeccen": 0.049,
        "pl_orbsmax": 0.0234,   # Extremely close to star
        "pl_insol": 8000.0,     # Extreme insolation
        "pl_eqt": 2600,         # K — hotter than many stars
        "st_teff": 6300,
        "st_lum": 0.41,
        "st_mass": 1.35,
        "expected_category": "Very Low Likelihood",
        "max_acceptable_score": 0.05,
    },
    "Venus": {
        "pl_rade": 0.950,
        "pl_bmasse": 0.815,
        "pl_dens": 5.243,
        "pl_orbeccen": 0.0068,
        "pl_orbsmax": 0.723,
        "pl_insol": 1.91,
        "pl_eqt": 737,          # Runaway greenhouse
        "st_teff": 5778,
        "st_lum": 0.0,
        "st_mass": 1.0,
        # Venus is an interesting edge case — rocky, right size,
        # but extreme temperature makes it non-habitable
        "expected_category": "Low Likelihood",
        "max_acceptable_score": 0.35,
    },
}

# Positive controls (should score high)
POSITIVE_CONTROLS = {
    "Earth": {
        "pl_rade": 1.0,
        "pl_bmasse": 1.0,
        "pl_dens": 5.514,
        "pl_orbeccen": 0.0167,
        "pl_orbsmax": 1.0,
        "pl_insol": 1.0,
        "pl_eqt": 255,          # Without greenhouse
        "st_teff": 5778,
        "st_lum": 0.0,
        "st_mass": 1.0,
        "expected_category": "High Likelihood",
        "min_acceptable_score": 0.65,
    },
}


@dataclass
class ControlTestResult:
    """Result of a single negative/positive control test."""
    target_name: str
    expected_category: str
    actual_score: float
    max_or_min_score: float     # Max acceptable (neg) or min acceptable (pos)
    passed: bool
    is_negative_control: bool
    details: str


def run_negative_controls(
    model,
    feature_engineering_fn=None,
) -> List[ControlTestResult]:
    """Run all negative control tests.

    Tests the tabular model against known non-habitable targets
    to verify it doesn't produce false positives.

    Args:
        model: Trained TabularHabitabilityModel instance.
        feature_engineering_fn: Optional function to engineer features on control data.

    Returns:
        List of ControlTestResult objects.
    """
    results = []

    # Test negative controls
    for name, params in NEGATIVE_CONTROLS.items():
        result = _test_control(
            model, name, params,
            is_negative=True,
            feature_engineering_fn=feature_engineering_fn,
        )
        results.append(result)

    # Test positive controls
    for name, params in POSITIVE_CONTROLS.items():
        result = _test_control(
            model, name, params,
            is_negative=False,
            feature_engineering_fn=feature_engineering_fn,
        )
        results.append(result)

    # Summary
    n_passed = sum(r.passed for r in results)
    n_total = len(results)
    logger.info(f"Control tests: {n_passed}/{n_total} passed")

    return results


def _test_control(
    model,
    name: str,
    params: dict,
    is_negative: bool,
    feature_engineering_fn=None,
) -> ControlTestResult:
    """Test a single control target.

    Args:
        model: Trained model.
        name: Target name.
        params: Target parameters.
        is_negative: True for negative control, False for positive.
        feature_engineering_fn: Optional feature engineering function.

    Returns:
        ControlTestResult.
    """
    # Extract model features from params
    feature_params = {k: v for k, v in params.items()
                      if k not in ("expected_category", "max_acceptable_score", "min_acceptable_score")}

    # Create single-row DataFrame
    df = pd.DataFrame([feature_params])
    df["pl_name"] = name

    # Engineer features if function provided
    if feature_engineering_fn:
        df = feature_engineering_fn(df)

    # Predict
    try:
        predictions = model.predict(df, n_bootstrap=50)
        score = predictions[0].habitability_score
    except Exception as e:
        logger.warning(f"Prediction failed for {name}: {e}")
        score = 0.5  # Uncertain

    # Check pass/fail
    if is_negative:
        threshold = params.get("max_acceptable_score", 0.20)
        passed = score <= threshold
        details = (
            f"{'✅ PASS' if passed else '❌ FAIL'}: "
            f"Score {score:.4f} {'≤' if passed else '>'} threshold {threshold:.4f}"
        )
    else:
        threshold = params.get("min_acceptable_score", 0.50)
        passed = score >= threshold
        details = (
            f"{'✅ PASS' if passed else '❌ FAIL'}: "
            f"Score {score:.4f} {'≥' if passed else '<'} threshold {threshold:.4f}"
        )

    status = "PASS" if passed else "FAIL"
    logger.info(f"  Control [{name}]: {status} — score={score:.4f}, threshold={threshold:.4f}")

    return ControlTestResult(
        target_name=name,
        expected_category=params.get("expected_category", "Unknown"),
        actual_score=score,
        max_or_min_score=threshold,
        passed=passed,
        is_negative_control=is_negative,
        details=details,
    )


def generate_control_report(results: List[ControlTestResult]) -> str:
    """Generate a formatted negative control test report.

    Args:
        results: List of ControlTestResult objects.

    Returns:
        Formatted report string.
    """
    lines = [
        "═" * 70,
        "  ExoScope — Negative & Positive Control Test Report",
        "═" * 70,
        "",
    ]

    # Negative controls
    neg_results = [r for r in results if r.is_negative_control]
    pos_results = [r for r in results if not r.is_negative_control]

    if neg_results:
        lines.append("── Negative Controls (should score LOW) ──")
        for r in neg_results:
            icon = "✅" if r.passed else "❌"
            lines.append(
                f"  {icon} {r.target_name:20s}  "
                f"Score: {r.actual_score:.4f}  "
                f"Max: {r.max_or_min_score:.4f}  "
                f"Expected: {r.expected_category}"
            )
        lines.append("")

    if pos_results:
        lines.append("── Positive Controls (should score HIGH) ──")
        for r in pos_results:
            icon = "✅" if r.passed else "❌"
            lines.append(
                f"  {icon} {r.target_name:20s}  "
                f"Score: {r.actual_score:.4f}  "
                f"Min: {r.max_or_min_score:.4f}  "
                f"Expected: {r.expected_category}"
            )
        lines.append("")

    # Summary
    n_passed = sum(r.passed for r in results)
    n_total = len(results)
    pass_rate = n_passed / n_total if n_total > 0 else 0

    lines.extend([
        "── Summary ──",
        f"  Total tests:  {n_total}",
        f"  Passed:       {n_passed}",
        f"  Failed:       {n_total - n_passed}",
        f"  Pass rate:    {pass_rate:.0%}",
        "",
    ])

    if pass_rate == 1.0:
        lines.append("  ✅ All control tests passed — model correctly handles known edge cases.")
    elif pass_rate >= 0.8:
        lines.append("  ⚠️  Most controls passed, but some edge cases need investigation.")
    else:
        lines.append("  ❌ Multiple control failures — model may be over-triggering. Review needed.")

    lines.append("═" * 70)
    return "\n".join(lines)
