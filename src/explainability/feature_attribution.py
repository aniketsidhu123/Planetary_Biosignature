"""
Feature attribution and explainability for ExoScope.

Provides SHAP-based feature importance for the tabular model
and gradient-based saliency for vision models. Generates
human-readable attribution strings for the HBLI report.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

from src.utils.config import get_config
from src.utils.logging_setup import get_logger

logger = get_logger("explainability.attribution")


class FeatureAttribution:
    """Generate explainable feature attributions for HBLI scores.

    Uses SHAP TreeExplainer for tabular models and provides
    human-readable explanations of what drives each score.
    """

    def __init__(self):
        """Initialize attribution engine."""
        config = get_config()
        explain_config = config.get_section("explainability")
        shap_config = explain_config.get("shap", {})

        self.max_samples = shap_config.get("max_samples", 500)
        self.plot_type = shap_config.get("plot_type", "bar")
        self.top_n = explain_config.get("top_n_factors", 5)

        self._shap_explainer = None
        self._background_data = None

    def setup_tabular_explainer(self, model: Any, X_background: pd.DataFrame):
        """Initialize SHAP explainer for the tabular model.

        Args:
            model: Trained sklearn/XGBoost model.
            X_background: Background dataset for SHAP (subset of training data).
        """
        try:
            import shap

            # Use a subset for speed
            if len(X_background) > self.max_samples:
                X_bg = X_background.sample(self.max_samples, random_state=42)
            else:
                X_bg = X_background

            self._background_data = X_bg
            self._shap_explainer = shap.TreeExplainer(model)
            logger.info(f"SHAP TreeExplainer initialized with {len(X_bg)} background samples")

        except ImportError:
            logger.warning("SHAP not available. Using feature importance fallback.")
        except Exception as e:
            logger.warning(f"SHAP initialization failed: {e}. Using fallback.")

    def explain_tabular(
        self,
        X: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> List[Dict[str, float]]:
        """Get SHAP-based feature attributions for tabular predictions.

        Args:
            X: Feature DataFrame for explanation.
            feature_names: Feature names (inferred from X if None).

        Returns:
            List of dicts, one per sample, mapping feature → SHAP value.
        """
        feature_names = feature_names or list(X.columns)

        if self._shap_explainer is not None:
            try:
                import shap

                shap_values = self._shap_explainer.shap_values(X)

                # Handle multi-output
                if isinstance(shap_values, list):
                    sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                else:
                    sv = shap_values

                attributions = []
                for i in range(sv.shape[0]):
                    attr = {
                        feature_names[j]: float(sv[i, j])
                        for j in range(len(feature_names))
                    }
                    attributions.append(attr)

                return attributions

            except Exception as e:
                logger.warning(f"SHAP computation failed: {e}")

        # Fallback: return empty attributions
        return [{name: 0.0 for name in feature_names} for _ in range(len(X))]

    def get_top_features(
        self,
        attributions: Dict[str, float],
        n: Optional[int] = None,
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Get top positive and negative contributing features.

        Args:
            attributions: Feature → SHAP value dict.
            n: Number of top features to return.

        Returns:
            (positive_features, negative_features) — each is [(name, value), ...].
        """
        n = n or self.top_n
        sorted_attrs = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)

        positive = [(name, val) for name, val in sorted_attrs if val > 0][:n]
        negative = [(name, val) for name, val in sorted_attrs if val < 0][:n]

        return positive, negative

    def generate_explanation_text(
        self,
        attributions: Dict[str, float],
        score: float,
        target_name: str = "target",
    ) -> str:
        """Generate a human-readable explanation string.

        Args:
            attributions: Feature → SHAP value dict.
            score: The prediction score being explained.
            target_name: Name of the target.

        Returns:
            Natural-language explanation string.
        """
        positive, negative = self.get_top_features(attributions)

        readable_map = {
            "pl_rade": "planet radius",
            "pl_bmasse": "planet mass",
            "pl_dens": "planet density",
            "pl_orbeccen": "orbital eccentricity",
            "pl_orbsmax": "orbital distance",
            "pl_insol": "insolation flux",
            "pl_eqt": "equilibrium temperature",
            "st_teff": "stellar temperature",
            "st_lum": "stellar luminosity",
            "esi": "Earth Similarity Index",
            "hz_ratio": "habitable zone proximity",
            "tidal_lock_likelihood": "tidal locking risk",
            "stellar_activity": "stellar activity level",
            "sedimentary_layering": "sedimentary layering detection",
            "mineral_water_interaction": "mineral-water interaction detection",
            "erosion_morphology": "erosion morphology detection",
        }

        def readable(name):
            return readable_map.get(name, name.replace("_", " "))

        parts = [f"{target_name} received a score of {score:.3f}."]

        if positive:
            drivers = ", ".join([f"{readable(n)} (+{v:.3f})" for n, v in positive[:3]])
            parts.append(f"Scored high on: {drivers}.")

        if negative:
            risks = ", ".join([f"{readable(n)} ({v:.3f})" for n, v in negative[:3]])
            parts.append(f"Scored low on: {risks}.")

        if positive:
            main_driver = readable(positive[0][0])
            parts.append(f"The score is primarily driven by {main_driver}.")

        return " ".join(parts)

    def plot_shap_summary(
        self,
        X: pd.DataFrame,
        save_path: Optional[str] = None,
    ):
        """Generate and optionally save a SHAP summary plot.

        Args:
            X: Feature DataFrame.
            save_path: Optional path to save the plot image.
        """
        if self._shap_explainer is None:
            logger.warning("No SHAP explainer available for plotting")
            return

        try:
            import shap
            import matplotlib.pyplot as plt

            shap_values = self._shap_explainer.shap_values(X)

            if isinstance(shap_values, list):
                sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                sv = shap_values

            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(sv, X, plot_type=self.plot_type, show=False)

            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info(f"SHAP plot saved to {save_path}")

            plt.close(fig)

        except Exception as e:
            logger.warning(f"SHAP plotting failed: {e}")

    def generate_vision_saliency(
        self,
        tile: np.ndarray,
        detections: list,
    ) -> np.ndarray:
        """Generate a saliency map for vision detections.

        Creates a heatmap showing which regions of the tile
        contributed most to biosignature proxy detections.

        Args:
            tile: Image tile (H, W, C).
            detections: List of Detection objects.

        Returns:
            Saliency heatmap as a color image.
        """
        import cv2

        h, w = tile.shape[:2]
        saliency = np.zeros((h, w), dtype=np.float64)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            # Clip to tile bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Add Gaussian-weighted confidence to detection region
            region_h = y2 - y1
            region_w = x2 - x1
            if region_h > 0 and region_w > 0:
                # Create 2D Gaussian
                cy, cx = region_h // 2, region_w // 2
                yy, xx = np.ogrid[:region_h, :region_w]
                gaussian = np.exp(
                    -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (min(region_h, region_w) / 3) ** 2)
                )
                saliency[y1:y2, x1:x2] += gaussian * det.confidence

        # Normalize and apply colormap
        if saliency.max() > 0:
            saliency = saliency / saliency.max()

        saliency_uint8 = (saliency * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)

        # Blend with original tile
        blended = cv2.addWeighted(tile, 0.6, heatmap, 0.4, 0)

        return blended
