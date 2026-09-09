"""
Classical computer vision pre-filter for ExoScope.

A cheap, fast pre-filter that runs before any deep-learning model.
Uses edge detection, texture statistics, and color variance to
reject tiles that are featureless (empty terrain, shadow, no-data)
— so YOLO inference only runs on promising tiles.

This is the primary compute-saving mechanism: most tiles in a
planetary image are featureless, and rejecting them early avoids
~90% of unnecessary YOLO inference.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from src.utils.config import get_config
from src.utils.logging_setup import get_logger

logger = get_logger("preprocessing.cv_filter")


@dataclass
class FilterResult:
    """Result from pre-filtering a single tile."""
    passed: bool                    # Whether tile passed all filters
    relevance_score: float          # Combined relevance score (0-1)
    edge_density: float             # Edge pixel ratio
    texture_entropy: float          # Shannon entropy of texture
    texture_contrast: float         # Haralick-style contrast
    color_variance: float           # Cross-channel standard deviation
    details: Dict[str, float]       # Per-filter scores for debugging


class ClassicalCVFilter:
    """Multi-stage classical CV pre-filter.

    Pipeline:
    1. Edge density (Canny) → rejects featureless plains
    2. Texture statistics (entropy, contrast) → rejects uniform surfaces
    3. Color variance → rejects monochrome/shadow tiles
    4. Combined weighted score → final pass/fail decision
    """

    def __init__(
        self,
        canny_low: Optional[int] = None,
        canny_high: Optional[int] = None,
        min_edge_ratio: Optional[float] = None,
        entropy_min: Optional[float] = None,
        contrast_min: Optional[float] = None,
        min_channel_std: Optional[float] = None,
        combined_threshold: Optional[float] = None,
    ):
        """Initialize pre-filter with configurable thresholds.

        Args:
            canny_low: Canny edge detector low threshold.
            canny_high: Canny edge detector high threshold.
            min_edge_ratio: Minimum fraction of edge pixels.
            entropy_min: Minimum Shannon entropy.
            contrast_min: Minimum texture contrast.
            min_channel_std: Minimum color channel std deviation.
            combined_threshold: Minimum combined relevance score.
        """
        config = get_config()
        cv_config = config.get_section("cv_filter")

        edge_cfg = cv_config.get("edge_density", {})
        tex_cfg = cv_config.get("texture", {})
        color_cfg = cv_config.get("color_variance", {})

        self.canny_low = canny_low or edge_cfg.get("canny_low", 50)
        self.canny_high = canny_high or edge_cfg.get("canny_high", 150)
        self.min_edge_ratio = min_edge_ratio or edge_cfg.get("min_edge_ratio", 0.02)
        self.entropy_min = entropy_min or tex_cfg.get("entropy_min", 3.0)
        self.contrast_min = contrast_min or tex_cfg.get("contrast_min", 10.0)
        self.min_channel_std = min_channel_std or color_cfg.get("min_channel_std", 8.0)
        self.combined_threshold = combined_threshold or cv_config.get("combined_threshold", 0.4)

        # Weights for combining sub-scores into overall relevance
        self._weights = {
            "edge": 0.35,
            "entropy": 0.25,
            "contrast": 0.20,
            "color": 0.20,
        }

        logger.info(
            f"CV filter initialized: edge≥{self.min_edge_ratio}, "
            f"entropy≥{self.entropy_min}, contrast≥{self.contrast_min}, "
            f"color_std≥{self.min_channel_std}, combined≥{self.combined_threshold}"
        )

    def filter_tile(self, tile: np.ndarray) -> FilterResult:
        """Run all filters on a single image tile.

        Args:
            tile: Image tile as numpy array (H, W, C) in BGR.

        Returns:
            FilterResult with pass/fail decision and individual scores.
        """
        # Convert to grayscale for edge/texture analysis
        if tile.ndim == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        else:
            gray = tile.copy()

        # 1. Edge density
        edge_density = self._compute_edge_density(gray)
        edge_score = min(edge_density / max(self.min_edge_ratio, 1e-6), 1.0)

        # 2. Texture entropy
        entropy = self._compute_entropy(gray)
        entropy_score = min(entropy / max(self.entropy_min * 2, 1e-6), 1.0)

        # 3. Texture contrast
        contrast = self._compute_contrast(gray)
        contrast_score = min(contrast / max(self.contrast_min * 5, 1e-6), 1.0)

        # 4. Color variance
        color_var = self._compute_color_variance(tile) if tile.ndim == 3 else 0.0
        color_score = min(color_var / max(self.min_channel_std * 5, 1e-6), 1.0)

        # Combined weighted score
        relevance = (
            self._weights["edge"] * edge_score
            + self._weights["entropy"] * entropy_score
            + self._weights["contrast"] * contrast_score
            + self._weights["color"] * color_score
        )

        passed = relevance >= self.combined_threshold

        return FilterResult(
            passed=passed,
            relevance_score=float(relevance),
            edge_density=float(edge_density),
            texture_entropy=float(entropy),
            texture_contrast=float(contrast),
            color_variance=float(color_var),
            details={
                "edge_score": float(edge_score),
                "entropy_score": float(entropy_score),
                "contrast_score": float(contrast_score),
                "color_score": float(color_score),
            },
        )

    def filter_batch(self, tiles: list) -> Tuple[list, list, dict]:
        """Filter a batch of tiles, returning passed and rejected lists.

        Args:
            tiles: List of Tile objects (from tiling module).

        Returns:
            Tuple of (passed_tiles, rejected_tiles, summary_stats).
        """
        passed = []
        rejected = []
        scores = []

        for tile in tiles:
            result = self.filter_tile(tile.image)
            scores.append(result.relevance_score)

            if result.passed:
                passed.append(tile)
            else:
                rejected.append(tile)

        summary = {
            "total": len(tiles),
            "passed": len(passed),
            "rejected": len(rejected),
            "pass_rate": len(passed) / max(len(tiles), 1),
            "mean_relevance": float(np.mean(scores)) if scores else 0.0,
            "median_relevance": float(np.median(scores)) if scores else 0.0,
        }

        logger.info(
            f"Batch filter: {summary['passed']}/{summary['total']} passed "
            f"({summary['pass_rate']:.1%}), "
            f"mean relevance={summary['mean_relevance']:.3f}"
        )

        return passed, rejected, summary

    def _compute_edge_density(self, gray: np.ndarray) -> float:
        """Compute edge pixel density using Canny edge detection.

        Args:
            gray: Grayscale image (H, W).

        Returns:
            Fraction of pixels that are edges (0.0 to 1.0).
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Count edge pixels
        edge_count = np.count_nonzero(edges)
        total_pixels = gray.size

        return edge_count / total_pixels if total_pixels > 0 else 0.0

    def _compute_entropy(self, gray: np.ndarray) -> float:
        """Compute Shannon entropy of pixel intensity distribution.

        Higher entropy = more complex/varied texture.
        Lower entropy = uniform/featureless surface.

        Args:
            gray: Grayscale image (H, W).

        Returns:
            Shannon entropy value (0 to ~8 for 8-bit images).
        """
        # Compute histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()

        # Remove zero bins to avoid log(0)
        hist = hist[hist > 0]

        # Shannon entropy: H = -Σ p(x) * log2(p(x))
        entropy = -np.sum(hist * np.log2(hist))

        return float(entropy)

    def _compute_contrast(self, gray: np.ndarray) -> float:
        """Compute texture contrast using local standard deviation.

        A simplified Haralick-style measure: the mean local standard
        deviation across the image. High contrast = visible features.

        Args:
            gray: Grayscale image (H, W).

        Returns:
            Mean local contrast value.
        """
        # Compute local mean and variance using box filter
        gray_float = gray.astype(np.float64)
        kernel_size = 7

        local_mean = cv2.blur(gray_float, (kernel_size, kernel_size))
        local_sq_mean = cv2.blur(gray_float ** 2, (kernel_size, kernel_size))
        local_variance = local_sq_mean - local_mean ** 2
        local_variance = np.maximum(local_variance, 0)
        local_std = np.sqrt(local_variance)

        return float(np.mean(local_std))

    def _compute_color_variance(self, tile: np.ndarray) -> float:
        """Compute cross-channel color variance.

        Measures how much color variation exists in the tile.
        Monochrome/shadow tiles will have very low values.

        Args:
            tile: Color image (H, W, 3) in BGR.

        Returns:
            Mean standard deviation across color channels.
        """
        if tile.ndim != 3 or tile.shape[2] < 3:
            return 0.0

        # Standard deviation per channel
        channel_stds = [float(np.std(tile[:, :, c])) for c in range(3)]

        return float(np.mean(channel_stds))

    def generate_heatmap(self, tiles: list) -> np.ndarray:
        """Generate a relevance heatmap from filter results.

        Creates a visualization showing which tiles are interesting
        (warm colors) vs. rejected (cool colors).

        Args:
            tiles: List of Tile objects.

        Returns:
            Heatmap as a color image (RGB).
        """
        if not tiles:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        # Find grid dimensions
        max_row = max(t.row_idx for t in tiles) + 1
        max_col = max(t.col_idx for t in tiles) + 1

        heatmap = np.zeros((max_row, max_col), dtype=np.float64)

        for tile in tiles:
            result = self.filter_tile(tile.image)
            heatmap[tile.row_idx, tile.col_idx] = result.relevance_score

        # Scale to larger image for visibility
        scale = 40
        heatmap_large = cv2.resize(
            heatmap, (max_col * scale, max_row * scale),
            interpolation=cv2.INTER_NEAREST,
        )

        # Apply colormap
        heatmap_uint8 = (heatmap_large * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        return cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    # Demo: create a test tile and filter it
    print("Classical CV Pre-Filter Demo")
    print("=" * 50)

    filter_ = ClassicalCVFilter()

    # Create test tiles
    # 1. Random noise (should pass — has texture)
    noise_tile = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    result = filter_.filter_tile(noise_tile)
    print(f"\nRandom noise tile: passed={result.passed}, relevance={result.relevance_score:.3f}")

    # 2. Black tile (should fail — no content)
    black_tile = np.zeros((640, 640, 3), dtype=np.uint8)
    result = filter_.filter_tile(black_tile)
    print(f"Black tile: passed={result.passed}, relevance={result.relevance_score:.3f}")

    # 3. Uniform gray (should fail — featureless)
    gray_tile = np.full((640, 640, 3), 128, dtype=np.uint8)
    result = filter_.filter_tile(gray_tile)
    print(f"Uniform gray tile: passed={result.passed}, relevance={result.relevance_score:.3f}")

    # 4. Gradient (moderate features)
    gradient_tile = np.zeros((640, 640, 3), dtype=np.uint8)
    for i in range(640):
        gradient_tile[i, :, :] = int(255 * i / 640)
    result = filter_.filter_tile(gradient_tile)
    print(f"Gradient tile: passed={result.passed}, relevance={result.relevance_score:.3f}")
