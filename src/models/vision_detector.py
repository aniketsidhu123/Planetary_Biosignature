"""
YOLO specialist detector wrapper for ExoScope.

Manages a stack of specialist YOLO detectors, each trained to
detect a specific biosignature proxy class:
  - sedimentary_layering
  - mineral_water_interaction
  - erosion_morphology

Each detector is a small YOLO model (nano/small variant) that
outputs bounding boxes + confidence scores for its single class.
The wrapper aggregates outputs into a per-tile feature vector
for downstream fusion.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

from src.utils.config import get_config
from src.utils.logging_setup import get_logger

logger = get_logger("models.vision")


@dataclass
class Detection:
    """A single object detection from a YOLO model."""
    class_name: str             # Biosignature proxy class name
    confidence: float           # Detection confidence (0-1)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) bounding box
    area_fraction: float        # Fraction of tile covered by this detection


@dataclass
class TileDetectionResult:
    """All detections for a single image tile."""
    detections: List[Detection]         # Individual detections
    feature_vector: Dict[str, float]    # {class_name: max_confidence}
    overall_score: float                # Aggregated vision biosignature score
    n_detections: int                   # Total detection count
    tile_source: str = ""               # Source tile identifier


@dataclass
class SpecialistModelInfo:
    """Metadata about a specialist detector."""
    name: str
    description: str
    model_path: str
    confidence_threshold: float
    is_loaded: bool = False


class VisionDetectorStack:
    """Stack of specialist YOLO detectors for biosignature proxy detection.

    Manages multiple small YOLO models, one per proxy class.
    Supports batch inference and aggregation into feature vectors.
    """

    def __init__(self, model_dir: Optional[str] = None):
        """Initialize the detector stack.

        Args:
            model_dir: Directory containing trained YOLO model weights.
        """
        config = get_config()
        vision_config = config.get_section("vision")

        self.model_arch = vision_config.get("model_architecture", "yolov8n")
        self.image_size = vision_config.get("image_size", 640)
        self.batch_size = vision_config.get("batch_size", 8)

        # Load specialist class definitions from config
        self.specialists: Dict[str, SpecialistModelInfo] = {}
        self._models: Dict[str, object] = {}

        for cls_config in vision_config.get("specialist_classes", []):
            name = cls_config["name"]
            self.specialists[name] = SpecialistModelInfo(
                name=name,
                description=cls_config.get("description", ""),
                model_path=cls_config.get("model_path", f"models/yolo_{name}.pt"),
                confidence_threshold=cls_config.get("confidence_threshold", 0.25),
            )

        logger.info(
            f"Vision detector stack initialized with {len(self.specialists)} specialist classes: "
            f"{list(self.specialists.keys())}"
        )

    def load_models(self) -> Dict[str, bool]:
        """Load all specialist YOLO models.

        Returns:
            Dict mapping class names to load success status.
        """
        status = {}

        for name, info in self.specialists.items():
            model_path = Path(info.model_path)

            if model_path.exists():
                try:
                    from ultralytics import YOLO
                    self._models[name] = YOLO(str(model_path))
                    info.is_loaded = True
                    status[name] = True
                    logger.info(f"Loaded specialist model: {name} ({model_path})")
                except Exception as e:
                    logger.error(f"Failed to load {name}: {e}")
                    status[name] = False
            else:
                logger.warning(
                    f"Model file not found for {name}: {model_path}. "
                    f"Using simulation mode."
                )
                status[name] = False

        loaded_count = sum(status.values())
        logger.info(f"Loaded {loaded_count}/{len(self.specialists)} specialist models")
        return status

    def detect_tile(self, tile: np.ndarray, tile_id: str = "") -> TileDetectionResult:
        """Run all specialist detectors on a single tile.

        Args:
            tile: Image tile (H, W, C) as numpy array.
            tile_id: Optional identifier for this tile.

        Returns:
            TileDetectionResult with all detections and aggregated scores.
        """
        all_detections = []
        feature_vector = {}

        for name, info in self.specialists.items():
            if name in self._models and info.is_loaded:
                # Run real YOLO inference
                detections = self._run_yolo(
                    self._models[name], tile, name, info.confidence_threshold
                )
            else:
                # Simulation mode: use heuristic-based detection
                detections = self._simulate_detection(tile, name, info.confidence_threshold)

            all_detections.extend(detections)

            # Feature vector: max confidence per class
            if detections:
                feature_vector[name] = max(d.confidence for d in detections)
            else:
                feature_vector[name] = 0.0

        # Overall vision score: weighted mean of per-class max confidences
        if feature_vector:
            overall_score = float(np.mean(list(feature_vector.values())))
        else:
            overall_score = 0.0

        return TileDetectionResult(
            detections=all_detections,
            feature_vector=feature_vector,
            overall_score=overall_score,
            n_detections=len(all_detections),
            tile_source=tile_id,
        )

    def detect_batch(self, tiles: list) -> List[TileDetectionResult]:
        """Run detection on a batch of tiles.

        Args:
            tiles: List of Tile objects (from tiling module).

        Returns:
            List of TileDetectionResult, one per tile.
        """
        results = []

        for i, tile in enumerate(tiles):
            tile_id = f"r{tile.row_idx}_c{tile.col_idx}" if hasattr(tile, "row_idx") else f"tile_{i}"
            tile_img = tile.image if hasattr(tile, "image") else tile

            result = self.detect_tile(tile_img, tile_id)
            results.append(result)

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(tiles)} tiles")

        logger.info(
            f"Batch detection complete: {len(results)} tiles, "
            f"mean score={np.mean([r.overall_score for r in results]):.3f}"
        )
        return results

    def _run_yolo(
        self,
        model,
        tile: np.ndarray,
        class_name: str,
        conf_threshold: float,
    ) -> List[Detection]:
        """Run YOLO inference on a tile.

        Args:
            model: Loaded YOLO model.
            tile: Image tile.
            class_name: Specialist class name.
            conf_threshold: Minimum confidence threshold.

        Returns:
            List of Detection objects.
        """
        try:
            results = model.predict(
                tile,
                imgsz=self.image_size,
                conf=conf_threshold,
                verbose=False,
            )

            detections = []
            tile_area = tile.shape[0] * tile.shape[1]

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        det_area = (x2 - x1) * (y2 - y1)

                        detections.append(Detection(
                            class_name=class_name,
                            confidence=conf,
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            area_fraction=float(det_area / tile_area),
                        ))

            return detections

        except Exception as e:
            logger.warning(f"YOLO inference failed for {class_name}: {e}")
            return []

    def _simulate_detection(
        self,
        tile: np.ndarray,
        class_name: str,
        conf_threshold: float,
    ) -> List[Detection]:
        """Simulate detection using classical CV heuristics.

        Used when trained YOLO models are not available.
        Provides rough estimates based on image features that
        correlate with each biosignature proxy class.

        Args:
            tile: Image tile.
            class_name: Class to simulate detection for.
            conf_threshold: Minimum confidence threshold.

        Returns:
            List of simulated Detection objects.
        """
        import cv2

        if tile.ndim == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        else:
            gray = tile

        h, w = gray.shape[:2]
        detections = []

        if class_name == "sedimentary_layering":
            # Detect horizontal line features (indicative of layering)
            edges = cv2.Canny(gray, 30, 100)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=80, minLineLength=w // 4, maxLineGap=20
            )

            if lines is not None:
                # Filter for roughly horizontal lines
                horizontal_lines = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if angle < 25 or angle > 155:  # Near-horizontal
                        horizontal_lines.append(line[0])

                if len(horizontal_lines) >= 3:
                    # Multiple parallel horizontal lines = layering signal
                    conf = min(0.3 + 0.05 * len(horizontal_lines), 0.85)
                    if conf >= conf_threshold:
                        detections.append(Detection(
                            class_name=class_name,
                            confidence=conf,
                            bbox=(0, 0, w, h),
                            area_fraction=1.0,
                        ))

        elif class_name == "mineral_water_interaction":
            # Detect bright spots / high-saturation regions (mineral deposits)
            if tile.ndim == 3:
                hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
                saturation = hsv[:, :, 1]
                value = hsv[:, :, 2]

                # High-value, varied-saturation regions
                bright_mask = value > 180
                sat_varied = np.std(saturation) > 30

                bright_ratio = np.sum(bright_mask) / bright_mask.size

                if bright_ratio > 0.05 and sat_varied:
                    conf = min(0.2 + bright_ratio * 2, 0.80)
                    if conf >= conf_threshold:
                        # Find bounding box of bright region
                        coords = np.where(bright_mask)
                        if len(coords[0]) > 0:
                            y1, y2 = coords[0].min(), coords[0].max()
                            x1, x2 = coords[1].min(), coords[1].max()
                            det_area = (x2 - x1) * (y2 - y1)
                            detections.append(Detection(
                                class_name=class_name,
                                confidence=conf,
                                bbox=(int(x1), int(y1), int(x2), int(y2)),
                                area_fraction=float(det_area / (h * w)),
                            ))

        elif class_name == "erosion_morphology":
            # Detect dendritic/channel-like features
            edges = cv2.Canny(gray, 40, 120)

            # Look for connected thin structures (channels)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            skeleton = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(
                skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Filter for elongated contours (channel-like)
            elongated = []
            for contour in contours:
                if len(contour) < 10:
                    continue
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width > 0 and height > 0:
                    aspect = max(width, height) / min(width, height)
                    if aspect > 3.0:  # Elongated
                        elongated.append(contour)

            if len(elongated) >= 2:
                conf = min(0.25 + 0.08 * len(elongated), 0.80)
                if conf >= conf_threshold:
                    # Bounding box around all elongated features
                    all_points = np.vstack(elongated)
                    x, y_, bw, bh = cv2.boundingRect(all_points)
                    detections.append(Detection(
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x, y_, x + bw, y_ + bh),
                        area_fraction=float((bw * bh) / (h * w)),
                    ))

        return detections

    def get_class_descriptions(self) -> Dict[str, str]:
        """Get human-readable descriptions of all specialist classes.

        Returns:
            Dict mapping class names to descriptions.
        """
        return {
            name: info.description
            for name, info in self.specialists.items()
        }

    def get_status(self) -> Dict[str, dict]:
        """Get status of all specialist models.

        Returns:
            Dict with per-model status information.
        """
        return {
            name: {
                "description": info.description,
                "model_path": info.model_path,
                "is_loaded": info.is_loaded,
                "confidence_threshold": info.confidence_threshold,
                "mode": "YOLO inference" if info.is_loaded else "CV simulation",
            }
            for name, info in self.specialists.items()
        }

    def visualize_detections(
        self,
        tile: np.ndarray,
        result: TileDetectionResult,
    ) -> np.ndarray:
        """Draw detection bounding boxes on a tile for visualization.

        Args:
            tile: Original tile image.
            result: Detection result for this tile.

        Returns:
            Annotated image with bounding boxes and labels.
        """
        import cv2

        vis = tile.copy()

        # Color map per class
        colors = {
            "sedimentary_layering": (0, 255, 128),    # Green
            "mineral_water_interaction": (255, 128, 0), # Orange
            "erosion_morphology": (0, 128, 255),       # Blue
        }

        for det in result.detections:
            color = colors.get(det.class_name, (255, 255, 255))
            x1, y1, x2, y2 = det.bbox

            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis, (x1, y1 - label_size[1] - 8), (x1 + label_size[0] + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return vis


if __name__ == "__main__":
    print("Vision Detector Stack Demo")
    print("=" * 50)

    stack = VisionDetectorStack()
    status = stack.get_status()

    for name, info in status.items():
        print(f"\n  {name}:")
        print(f"    Description: {info['description']}")
        print(f"    Mode: {info['mode']}")
        print(f"    Threshold: {info['confidence_threshold']}")

    # Test with a synthetic tile
    test_tile = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
    result = stack.detect_tile(test_tile, "test_tile")
    print(f"\nTest tile detections: {result.n_detections}")
    print(f"Feature vector: {result.feature_vector}")
    print(f"Overall score: {result.overall_score:.3f}")
