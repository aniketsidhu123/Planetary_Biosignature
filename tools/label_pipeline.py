"""
Labeling pipeline for ExoScope vision training data.

Provides tools for:
1. Organizing unlabeled image tiles into a labeling workspace
2. Generating YOLO-format annotation files
3. Semi-automated pre-labeling using the classical CV filter
4. Dataset splitting (train/val/test)
5. Integration with labelImg for manual annotation

Usage:
    python tools/label_pipeline.py setup --source data/processed/tiles --output data/labels/workspace
    python tools/label_pipeline.py prelabel --workspace data/labels/workspace
    python tools/label_pipeline.py split --workspace data/labels/workspace --ratio 0.7 0.2 0.1
    python tools/label_pipeline.py export --workspace data/labels/workspace --output data/labels/yolo
"""

import argparse
import shutil
import numpy as np
import cv2
import yaml
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_setup import get_logger

logger = get_logger("tools.labeling")


# ── Biosignature proxy classes ─────────────────────────────────────────
CLASSES = {
    0: "sedimentary_layering",
    1: "mineral_water_interaction",
    2: "erosion_morphology",
}
CLASS_NAMES = list(CLASSES.values())


def setup_workspace(source_dir: str, output_dir: str, max_tiles: int = 500):
    """Set up a labeling workspace from processed tiles.

    Copies tiles into a structured workspace with:
      workspace/
        images/         ← tiles to label
        labels/         ← YOLO-format .txt annotations (initially empty)
        classes.txt     ← class names
        data.yaml       ← YOLO dataset config
        stats.json      ← workspace statistics

    Args:
        source_dir: Directory containing image tiles.
        output_dir: Workspace directory to create.
        max_tiles: Maximum tiles to include.
    """
    source = Path(source_dir)
    workspace = Path(output_dir)

    if not source.exists():
        logger.error(f"Source directory not found: {source}")
        return

    # Create workspace structure
    images_dir = workspace / "images"
    labels_dir = workspace / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    image_files = [
        f for f in source.rglob("*")
        if f.suffix.lower() in extensions
    ]

    if not image_files:
        logger.warning(f"No image files found in {source}")
        # Create sample tiles for development
        logger.info("Generating sample tiles for labeling practice...")
        image_files = _generate_sample_tiles(images_dir, count=min(max_tiles, 20))
    else:
        # Copy tiles to workspace
        selected = image_files[:max_tiles]
        for img_file in selected:
            dest = images_dir / img_file.name
            if not dest.exists():
                shutil.copy2(img_file, dest)
        image_files = list(images_dir.glob("*"))

    # Write classes.txt
    classes_file = workspace / "classes.txt"
    with open(classes_file, "w") as f:
        for cls_name in CLASS_NAMES:
            f.write(f"{cls_name}\n")

    # Write data.yaml (YOLO format)
    data_yaml = {
        "path": str(workspace.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(workspace / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    # Write stats
    stats = {
        "total_images": len(image_files),
        "labeled": 0,
        "unlabeled": len(image_files),
        "classes": CLASS_NAMES,
    }
    with open(workspace / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"Workspace created at {workspace}\n"
        f"  Images: {len(image_files)}\n"
        f"  Classes: {CLASS_NAMES}\n"
        f"  Next: Run 'prelabel' for semi-automated pre-labeling, "
        f"then open in labelImg for refinement."
    )


def prelabel_workspace(workspace_dir: str, confidence_threshold: float = 0.3):
    """Run semi-automated pre-labeling using classical CV heuristics.

    Uses the CV filter and heuristic detection to generate initial
    YOLO-format annotations that can be refined manually in labelImg.

    Args:
        workspace_dir: Path to labeling workspace.
        confidence_threshold: Minimum confidence for pre-labels.
    """
    from src.preprocessing.classical_cv_filter import ClassicalCVFilter
    from src.models.vision_detector import VisionDetectorStack

    workspace = Path(workspace_dir)
    images_dir = workspace / "images"
    labels_dir = workspace / "labels"

    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return

    cv_filter = ClassicalCVFilter()
    detector = VisionDetectorStack()

    image_files = sorted(images_dir.glob("*"))
    image_files = [f for f in image_files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

    stats = {"total": 0, "pre_labeled": 0, "skipped": 0, "per_class": defaultdict(int)}

    for img_path in image_files:
        stats["total"] += 1

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            stats["skipped"] += 1
            continue

        # Check relevance
        filter_result = cv_filter.filter_tile(image)
        if not filter_result.passed:
            stats["skipped"] += 1
            continue

        # Run heuristic detection
        det_result = detector.detect_tile(image, img_path.stem)

        if det_result.n_detections == 0:
            stats["skipped"] += 1
            continue

        # Write YOLO-format annotations
        h, w = image.shape[:2]
        label_path = labels_dir / f"{img_path.stem}.txt"

        with open(label_path, "w") as f:
            for det in det_result.detections:
                if det.confidence < confidence_threshold:
                    continue

                # Get class index
                class_idx = CLASS_NAMES.index(det.class_name) if det.class_name in CLASS_NAMES else -1
                if class_idx < 0:
                    continue

                # Convert to YOLO format: class_id cx cy bw bh (normalized)
                x1, y1, x2, y2 = det.bbox
                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                # Clip to [0, 1]
                cx = np.clip(cx, 0, 1)
                cy = np.clip(cy, 0, 1)
                bw = np.clip(bw, 0, 1)
                bh = np.clip(bh, 0, 1)

                f.write(f"{class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                stats["per_class"][det.class_name] += 1

        stats["pre_labeled"] += 1

    # Update workspace stats
    stats_file = workspace / "stats.json"
    workspace_stats = {
        "total_images": stats["total"],
        "pre_labeled": stats["pre_labeled"],
        "skipped": stats["skipped"],
        "per_class_detections": dict(stats["per_class"]),
        "note": "Pre-labels are heuristic estimates. Review and correct in labelImg.",
    }
    with open(stats_file, "w") as f:
        json.dump(workspace_stats, f, indent=2)

    logger.info(
        f"Pre-labeling complete:\n"
        f"  Total images: {stats['total']}\n"
        f"  Pre-labeled: {stats['pre_labeled']}\n"
        f"  Skipped (featureless): {stats['skipped']}\n"
        f"  Detections per class: {dict(stats['per_class'])}\n"
        f"\n  ⚠️  Pre-labels are heuristic estimates.\n"
        f"  Open in labelImg for manual review and correction:\n"
        f"    labelImg {images_dir} {workspace / 'classes.txt'} {labels_dir}"
    )


def split_dataset(
    workspace_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """Split labeled data into train/val/test sets.

    Creates YOLO-compatible directory structure:
      workspace/images/train/
      workspace/images/val/
      workspace/images/test/
      workspace/labels/train/
      workspace/labels/val/
      workspace/labels/test/

    Args:
        workspace_dir: Path to labeling workspace.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.
    """
    workspace = Path(workspace_dir)
    images_dir = workspace / "images"
    labels_dir = workspace / "labels"

    # Find labeled images (those with corresponding .txt files)
    labeled_images = []
    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                labeled_images.append((img_path, label_path))

    if not labeled_images:
        logger.warning("No labeled images found. Run 'prelabel' first.")
        return

    # Shuffle and split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(labeled_images))

    n_train = int(len(labeled_images) * train_ratio)
    n_val = int(len(labeled_images) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    for split_name, split_indices in splits.items():
        img_split_dir = images_dir / split_name
        lbl_split_dir = labels_dir / split_name
        img_split_dir.mkdir(parents=True, exist_ok=True)
        lbl_split_dir.mkdir(parents=True, exist_ok=True)

        for idx in split_indices:
            img_path, lbl_path = labeled_images[idx]
            shutil.copy2(img_path, img_split_dir / img_path.name)
            shutil.copy2(lbl_path, lbl_split_dir / lbl_path.name)

    logger.info(
        f"Dataset split complete:\n"
        f"  Train: {len(train_idx)} images\n"
        f"  Val:   {len(val_idx)} images\n"
        f"  Test:  {len(test_idx)} images"
    )


def export_yolo_dataset(workspace_dir: str, output_dir: str):
    """Export the labeled workspace as a YOLO-ready dataset.

    Args:
        workspace_dir: Path to labeling workspace.
        output_dir: Path for the exported dataset.
    """
    workspace = Path(workspace_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Copy split directories
    for split in ["train", "val", "test"]:
        for subdir in ["images", "labels"]:
            src = workspace / subdir / split
            dst = output / subdir / split
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

    # Write data.yaml
    data_yaml = {
        "path": str(output.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(output / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    logger.info(f"YOLO dataset exported to {output}")


def _generate_sample_tiles(output_dir: Path, count: int = 20) -> List[Path]:
    """Generate synthetic sample tiles for labeling practice.

    Creates tiles with procedurally generated features that
    approximate the visual characteristics of each class.

    Args:
        output_dir: Directory to save sample tiles.
        count: Number of tiles to generate.

    Returns:
        List of generated file paths.
    """
    rng = np.random.default_rng(42)
    generated = []

    for i in range(count):
        tile = np.zeros((640, 640, 3), dtype=np.uint8)

        # Base terrain texture
        noise = rng.integers(80, 160, (640, 640, 3), dtype=np.uint8)
        tile = noise

        # Add class-specific features to some tiles
        feature_type = i % 4  # 0=plain, 1=layering, 2=mineral, 3=erosion

        if feature_type == 1:
            # Sedimentary layering: horizontal bands
            for y in range(0, 640, rng.integers(30, 80)):
                thickness = rng.integers(3, 15)
                color = rng.integers(40, 200, 3).tolist()
                cv2.rectangle(tile, (0, y), (640, y + thickness), color, -1)
            suffix = "layering"

        elif feature_type == 2:
            # Mineral-water: bright irregular patches
            n_patches = rng.integers(3, 10)
            for _ in range(n_patches):
                cx, cy = rng.integers(50, 590, 2)
                axes = rng.integers(20, 80, 2).tolist()
                angle = rng.integers(0, 180)
                color = (200 + rng.integers(0, 55), 200 + rng.integers(0, 55), 180 + rng.integers(0, 55))
                cv2.ellipse(tile, (int(cx), int(cy)), tuple(axes), int(angle), 0, 360, color, -1)
            suffix = "mineral"

        elif feature_type == 3:
            # Erosion: thin branching lines
            n_channels = rng.integers(2, 6)
            for _ in range(n_channels):
                points = []
                x, y = rng.integers(0, 640), rng.integers(0, 100)
                for _ in range(rng.integers(10, 30)):
                    x += rng.integers(-20, 20)
                    y += rng.integers(5, 25)
                    x = np.clip(x, 0, 639)
                    y = np.clip(y, 0, 639)
                    points.append([int(x), int(y)])
                if len(points) >= 2:
                    pts = np.array(points, dtype=np.int32)
                    color = (60 + rng.integers(0, 40), 50 + rng.integers(0, 30), 40 + rng.integers(0, 30))
                    cv2.polylines(tile, [pts], False, color, rng.integers(2, 6))
            suffix = "erosion"
        else:
            suffix = "plain"

        # Add slight Gaussian blur for realism
        tile = cv2.GaussianBlur(tile, (3, 3), 0.8)

        filename = f"sample_{i:03d}_{suffix}.jpg"
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), tile)
        generated.append(filepath)

    logger.info(f"Generated {len(generated)} sample tiles in {output_dir}")
    return generated


def main():
    """CLI entrypoint for the labeling pipeline."""
    parser = argparse.ArgumentParser(
        description="ExoScope Labeling Pipeline — Manage annotation workflow for vision training data"
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline command")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up labeling workspace")
    setup_parser.add_argument("--source", default="data/processed/tiles", help="Source tiles directory")
    setup_parser.add_argument("--output", default="data/labels/workspace", help="Workspace directory")
    setup_parser.add_argument("--max-tiles", type=int, default=500, help="Max tiles to include")

    # Pre-label command
    prelabel_parser = subparsers.add_parser("prelabel", help="Run semi-automated pre-labeling")
    prelabel_parser.add_argument("--workspace", default="data/labels/workspace", help="Workspace directory")
    prelabel_parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold")

    # Split command
    split_parser = subparsers.add_parser("split", help="Split into train/val/test")
    split_parser.add_argument("--workspace", default="data/labels/workspace", help="Workspace directory")
    split_parser.add_argument("--ratio", nargs=3, type=float, default=[0.7, 0.2, 0.1],
                              help="Train/val/test ratios")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export YOLO dataset")
    export_parser.add_argument("--workspace", default="data/labels/workspace", help="Workspace directory")
    export_parser.add_argument("--output", default="data/labels/yolo", help="Export directory")

    args = parser.parse_args()

    if args.command == "setup":
        setup_workspace(args.source, args.output, args.max_tiles)
    elif args.command == "prelabel":
        prelabel_workspace(args.workspace, args.threshold)
    elif args.command == "split":
        split_dataset(args.workspace, *args.ratio)
    elif args.command == "export":
        export_yolo_dataset(args.workspace, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
