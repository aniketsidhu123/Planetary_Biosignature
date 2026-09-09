"""
Fusion layer for ExoScope HBLI scoring.

Combines tabular habitability scores with vision-based biosignature
proxy scores into a single composite HBLI (Habitability & Biosignature-
Likelihood Index) with confidence intervals and explainable breakdowns.

The fusion is deliberately kept simple (weighted linear combination)
rather than using a black-box model — this keeps the output explainable,
which is critical for a "how likely is life" claim.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from src.utils.config import get_config
from src.utils.logging_setup import get_logger

logger = get_logger("models.fusion")


@dataclass
class HBLIResult:
    """Complete HBLI scoring result for a target.

    This is the primary output of the ExoScope system — a composite,
    explainable score with confidence bounds and attribution breakdown.
    """
    target_name: str                      # Planet or target identifier
    hbli_score: float                     # Composite score (0.0 - 1.0)
    confidence_lower: float               # Lower bound of confidence interval
    confidence_upper: float               # Upper bound of confidence interval
    confidence_level: float               # CI level (e.g., 0.95)

    # Sub-scores
    tabular_score: float                  # Habitability sub-score (0-1)
    vision_score: float                   # Biosignature proxy sub-score (0-1)

    # Weights used
    tabular_weight: float
    vision_weight: float

    # Explainability
    top_contributing_factors: List[Tuple[str, float, str]]  # [(name, contribution, direction)]
    risk_factors: List[Tuple[str, float, str]]              # Negative contributors

    # Category
    category: str                         # "High", "Moderate", "Low", "Very Low"
    category_description: str             # Human-readable description

    # Disclaimers (always present)
    disclaimers: List[str] = field(default_factory=lambda: [
        "This score represents biosignature/habitability LIKELIHOOD based on proxy indicators validated against Earth analogs.",
        "This is NOT a detection or confirmation of life.",
        "Vision model predictions are based on Earth-biology training data and may not generalize to non-terrestrial biology.",
        "False positives are the dominant historical risk in this field — interpret with appropriate skepticism.",
    ])


class FusionLayer:
    """Weighted fusion of tabular and vision sub-scores into HBLI.

    Implements a transparent, weighted linear combination with
    bootstrap confidence intervals and explainable decomposition.
    """

    def __init__(
        self,
        tabular_weight: Optional[float] = None,
        vision_weight: Optional[float] = None,
    ):
        """Initialize fusion layer.

        Args:
            tabular_weight: Weight for tabular habitability score.
            vision_weight: Weight for vision biosignature score.
        """
        config = get_config()
        fusion_config = config.get_section("fusion")
        weights = fusion_config.get("weights", {})

        self.tabular_weight = tabular_weight or weights.get("tabular", 0.55)
        self.vision_weight = vision_weight or weights.get("vision", 0.45)

        # Normalize weights
        total = self.tabular_weight + self.vision_weight
        self.tabular_weight /= total
        self.vision_weight /= total

        # Confidence settings
        conf_config = fusion_config.get("confidence", {})
        self.ci_method = conf_config.get("method", "bootstrap")
        self.n_bootstrap = conf_config.get("n_bootstrap", 1000)
        self.confidence_level = conf_config.get("confidence_level", 0.95)

        # Score range
        self.score_range = tuple(fusion_config.get("score_range", [0.0, 1.0]))

        logger.info(
            f"Fusion layer initialized: "
            f"tabular_weight={self.tabular_weight:.2f}, "
            f"vision_weight={self.vision_weight:.2f}, "
            f"CI method={self.ci_method}"
        )

    def fuse(
        self,
        target_name: str,
        tabular_score: float,
        tabular_ci: Optional[Tuple[float, float]] = None,
        vision_score: float = 0.0,
        vision_ci: Optional[Tuple[float, float]] = None,
        tabular_features: Optional[Dict[str, float]] = None,
        vision_features: Optional[Dict[str, float]] = None,
    ) -> HBLIResult:
        """Fuse tabular and vision scores into a composite HBLI score.

        Args:
            target_name: Name of the target (planet, tile, etc.).
            tabular_score: Habitability sub-score (0-1).
            tabular_ci: Optional (lower, upper) confidence interval for tabular.
            vision_score: Biosignature proxy sub-score (0-1).
            vision_ci: Optional (lower, upper) confidence interval for vision.
            tabular_features: Dict of tabular feature contributions.
            vision_features: Dict of vision feature contributions.

        Returns:
            Complete HBLIResult with score, CI, and explanations.
        """
        # Compute composite score
        hbli = (
            self.tabular_weight * tabular_score
            + self.vision_weight * vision_score
        )
        hbli = np.clip(hbli, *self.score_range)

        # Compute confidence interval
        ci_lower, ci_upper = self._compute_confidence_interval(
            tabular_score, tabular_ci,
            vision_score, vision_ci,
        )

        # Build attribution breakdown
        top_factors, risk_factors = self._build_attribution(
            tabular_score, vision_score,
            tabular_features or {},
            vision_features or {},
        )

        # Categorize
        category, description = self._categorize(hbli)

        result = HBLIResult(
            target_name=target_name,
            hbli_score=float(hbli),
            confidence_lower=float(ci_lower),
            confidence_upper=float(ci_upper),
            confidence_level=self.confidence_level,
            tabular_score=float(tabular_score),
            vision_score=float(vision_score),
            tabular_weight=self.tabular_weight,
            vision_weight=self.vision_weight,
            top_contributing_factors=top_factors,
            risk_factors=risk_factors,
            category=category,
            category_description=description,
        )

        logger.info(
            f"HBLI for {target_name}: {hbli:.4f} "
            f"[{ci_lower:.4f}, {ci_upper:.4f}] — {category}"
        )

        return result

    def fuse_batch(
        self,
        targets: List[dict],
    ) -> List[HBLIResult]:
        """Fuse scores for multiple targets.

        Args:
            targets: List of dicts with keys matching fuse() parameters.

        Returns:
            List of HBLIResult, sorted by HBLI score descending.
        """
        results = []
        for target in targets:
            result = self.fuse(**target)
            results.append(result)

        # Sort by score (highest first — mission prioritization order)
        results.sort(key=lambda r: r.hbli_score, reverse=True)

        logger.info(
            f"Batch fusion complete: {len(results)} targets, "
            f"top={results[0].target_name if results else 'N/A'} "
            f"({results[0].hbli_score:.4f})"
        )

        return results

    def _compute_confidence_interval(
        self,
        tab_score: float,
        tab_ci: Optional[Tuple[float, float]],
        vis_score: float,
        vis_ci: Optional[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """Compute confidence interval for the fused score.

        Uses bootstrap sampling from component distributions if CIs
        are provided, otherwise uses a heuristic margin.
        """
        alpha = (1.0 - self.confidence_level) / 2.0

        if tab_ci and vis_ci:
            # Bootstrap from component distributions
            rng = np.random.default_rng(42)

            # Approximate component distributions as truncated normals
            tab_std = (tab_ci[1] - tab_ci[0]) / (2 * 1.96)
            vis_std = (vis_ci[1] - vis_ci[0]) / (2 * 1.96)

            tab_samples = rng.normal(tab_score, max(tab_std, 0.01), self.n_bootstrap)
            vis_samples = rng.normal(vis_score, max(vis_std, 0.01), self.n_bootstrap)

            fused_samples = (
                self.tabular_weight * tab_samples
                + self.vision_weight * vis_samples
            )
            fused_samples = np.clip(fused_samples, *self.score_range)

            ci_lower = float(np.percentile(fused_samples, 100 * alpha))
            ci_upper = float(np.percentile(fused_samples, 100 * (1 - alpha)))

        elif tab_ci:
            # Only tabular CI available
            ci_lower = float(self.tabular_weight * tab_ci[0] + self.vision_weight * vis_score)
            ci_upper = float(self.tabular_weight * tab_ci[1] + self.vision_weight * vis_score)

        else:
            # Heuristic margin based on component scores
            fused = self.tabular_weight * tab_score + self.vision_weight * vis_score
            # Wider CI when scores disagree (higher epistemic uncertainty)
            disagreement = abs(tab_score - vis_score)
            margin = 0.05 + 0.15 * disagreement
            ci_lower = float(np.clip(fused - margin, *self.score_range))
            ci_upper = float(np.clip(fused + margin, *self.score_range))

        return ci_lower, ci_upper

    def _build_attribution(
        self,
        tab_score: float,
        vis_score: float,
        tab_features: Dict[str, float],
        vis_features: Dict[str, float],
    ) -> Tuple[List[Tuple[str, float, str]], List[Tuple[str, float, str]]]:
        """Build human-readable attribution breakdown.

        Returns:
            (top_positive_factors, risk_factors) — each is a list of
            (factor_name, contribution_magnitude, direction_description).
        """
        all_contributions = []

        # Tabular contributions
        for feat, value in tab_features.items():
            weighted = value * self.tabular_weight
            direction = "positive" if value > 0 else "negative"
            readable_name = self._feature_to_readable(feat)
            all_contributions.append((readable_name, weighted, direction))

        # Vision contributions
        for feat, value in vis_features.items():
            weighted = value * self.vision_weight
            direction = "detected" if value > 0.3 else "not detected"
            readable_name = self._feature_to_readable(feat)
            all_contributions.append((readable_name, weighted, direction))

        # Add pipeline-level contributions
        all_contributions.append(
            ("Tabular habitability", tab_score * self.tabular_weight, "pipeline score")
        )
        all_contributions.append(
            ("Vision biosignature proxy", vis_score * self.vision_weight, "pipeline score")
        )

        # Sort by absolute contribution
        all_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        top_factors = [c for c in all_contributions if c[1] > 0][:5]
        risk_factors = [c for c in all_contributions if c[1] <= 0][:5]

        return top_factors, risk_factors

    def _categorize(self, score: float) -> Tuple[str, str]:
        """Categorize the HBLI score with a description.

        Args:
            score: HBLI score (0-1).

        Returns:
            (category_label, human_readable_description).
        """
        if score >= 0.75:
            return (
                "High Likelihood",
                "Strong proxy indicators detected across multiple parameters. "
                "This target shows characteristics consistent with habitability "
                "and biosignature presence based on Earth-analog validation. "
                "Recommended for detailed follow-up analysis."
            )
        elif score >= 0.50:
            return (
                "Moderate Likelihood",
                "Several positive proxy indicators detected, but with significant "
                "uncertainty. Some parameters support habitability while others "
                "are inconclusive. May warrant follow-up with additional data."
            )
        elif score >= 0.25:
            return (
                "Low Likelihood",
                "Limited proxy indicators detected. Most parameters do not "
                "strongly support habitability or biosignature presence. "
                "Low priority for follow-up unless new data becomes available."
            )
        else:
            return (
                "Very Low Likelihood",
                "No significant proxy indicators detected. Parameters are "
                "inconsistent with known habitability requirements based on "
                "Earth-analog training data. Not recommended for follow-up."
            )

    @staticmethod
    def _feature_to_readable(feature_name: str) -> str:
        """Convert a feature name to human-readable format."""
        readable_map = {
            "pl_rade": "Planet radius",
            "pl_bmasse": "Planet mass",
            "pl_dens": "Planet density",
            "pl_orbeccen": "Orbital eccentricity",
            "pl_orbsmax": "Orbital distance",
            "pl_insol": "Insolation flux",
            "pl_eqt": "Equilibrium temperature",
            "st_teff": "Stellar temperature",
            "st_lum": "Stellar luminosity",
            "esi": "Earth Similarity Index",
            "hz_ratio": "Habitable zone distance",
            "tidal_lock_likelihood": "Tidal locking risk",
            "stellar_activity": "Stellar activity level",
            "sedimentary_layering": "Sedimentary layering patterns",
            "mineral_water_interaction": "Mineral-water interaction zones",
            "erosion_morphology": "Erosion/fluid-flow morphology",
        }
        return readable_map.get(feature_name, feature_name.replace("_", " ").title())

    def generate_report(self, result: HBLIResult) -> str:
        """Generate a human-readable analysis report.

        Args:
            result: HBLIResult to report on.

        Returns:
            Formatted report string.
        """
        lines = [
            "╔" + "═" * 62 + "╗",
            f"║  HBLI Analysis Report: {result.target_name:>36s}  ║",
            "╠" + "═" * 62 + "╣",
            f"║  Composite HBLI Score: {result.hbli_score:>6.4f}                              ║",
            f"║  {result.confidence_level*100:.0f}% CI: [{result.confidence_lower:.4f}, {result.confidence_upper:.4f}]"
            + " " * (62 - 30 - len(f"{result.confidence_lower:.4f}, {result.confidence_upper:.4f}")) + "║",
            f"║  Category: {result.category:<50s} ║",
            "╠" + "═" * 62 + "╣",
            "║  Sub-Scores:                                                ║",
            f"║    Tabular (weight={result.tabular_weight:.2f}): {result.tabular_score:>6.4f}"
            + " " * (62 - 40) + "║",
            f"║    Vision  (weight={result.vision_weight:.2f}): {result.vision_score:>6.4f}"
            + " " * (62 - 40) + "║",
            "╠" + "═" * 62 + "╣",
        ]

        if result.top_contributing_factors:
            lines.append("║  Top Contributing Factors:                                  ║")
            for name, contrib, direction in result.top_contributing_factors[:5]:
                entry = f"    + {name}: {contrib:+.4f} ({direction})"
                lines.append(f"║  {entry:<60s}║")

        if result.risk_factors:
            lines.append("║  Risk Factors:                                              ║")
            for name, contrib, direction in result.risk_factors[:3]:
                entry = f"    − {name}: {contrib:+.4f} ({direction})"
                lines.append(f"║  {entry:<60s}║")

        lines.extend([
            "╠" + "═" * 62 + "╣",
            "║  ⚠ DISCLAIMERS:                                            ║",
        ])
        for disc in result.disclaimers[:2]:
            # Word-wrap disclaimer
            words = disc.split()
            current_line = "    "
            for word in words:
                if len(current_line) + len(word) + 1 > 58:
                    lines.append(f"║  {current_line:<60s}║")
                    current_line = "    " + word
                else:
                    current_line += " " + word if current_line.strip() else "    " + word
            if current_line.strip():
                lines.append(f"║  {current_line:<60s}║")

        lines.append("╚" + "═" * 62 + "╝")

        return "\n".join(lines)


if __name__ == "__main__":
    fusion = FusionLayer()

    # Demo fusion
    result = fusion.fuse(
        target_name="Kepler-442b",
        tabular_score=0.72,
        tabular_ci=(0.65, 0.79),
        vision_score=0.0,  # No imagery analyzed
        vision_ci=None,
        tabular_features={"esi": 0.84, "hz_ratio": -0.05, "pl_rade": 0.15},
        vision_features={},
    )

    print(fusion.generate_report(result))
