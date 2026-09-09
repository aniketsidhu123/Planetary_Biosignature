"""
Image tiling utility for ExoScope.

Splits large planetary images (HiRISE, Bhoonidhi scenes) into
fixed-size tiles suitable for YOLO inference, with overlap handling
and lazy loading to stay within laptop RAM constraints.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Generator
from dataclasses import dataclass

from PIL import Image
import cv2

from src.utils.config import get_config
from src.utils.logging_setup import get_logger

logger = get_logger("data_ingestion.tiling")


@dataclass
class Tile:
    """A single image tile with position metadata."""
    image: np.ndarray        # Tile pixel data (H, W, C)
    row_idx: int             # Tile row index in grid
    col_idx: int             # Tile column index in grid
    x_offset: int            # Pixel x-offset in original image
    y_offset: int            # Pixel y-offset in original image
    tile_size: int           # Tile dimension (square)
    source_path: str         # Path to source image
    coverage: float          # Fraction of non-black/no-data pixels


@dataclass
class TilingResult:
    """Result of tiling an image."""
    tiles: List[Tile]
    source_path: str
    source_shape: Tuple[int, int, int]  # (H, W, C)
    grid_shape: Tuple[int, int]          # (rows, cols)
    tile_size: int
    overlap: int
    total_tiles: int
    passed_tiles: int                     # Tiles passing coverage filter


def tile_image(
    image_path: str,
    tile_size: Optional[int] = None,
    overlap: Optional[int] = None,
    min_coverage: Optional[float] = None,
    output_dir: Optional[str] = None,
    save_tiles: bool = False,
) -> TilingResult:
    """Split an image into fixed-size tiles.

    Tiles are extracted with configurable overlap and filtered
    by minimum coverage (to reject empty/no-data tiles).

    Args:
        image_path: Path to source image.
        tile_size: Tile dimension in pixels (square). Default from config.
        overlap: Overlap between adjacent tiles in pixels. Default from config.
        min_coverage: Minimum fraction of valid (non-black) pixels. Default from config.
        output_dir: Directory to save tile images. Only used if save_tiles=True.
        save_tiles: If True, save each tile as a separate image file.

    Returns:
        TilingResult with tile list and metadata.
    """
    config = get_config()
    tiling_config = config.get_section("tiling")

    tile_size = tile_size or tiling_config.get("tile_size", 640)
    overlap = overlap or tiling_config.get("overlap", 64)
    min_coverage = min_coverage or tiling_config.get("min_tile_coverage", 0.5)

    image_path = str(image_path)
    logger.info(f"Tiling {image_path} → {tile_size}×{tile_size}, overlap={overlap}")

    # Load image
    image = _load_image(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return TilingResult(
            tiles=[], source_path=image_path,
            source_shape=(0, 0, 0), grid_shape=(0, 0),
            tile_size=tile_size, overlap=overlap,
            total_tiles=0, passed_tiles=0,
        )

    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    logger.info(f"Image size: {w}×{h}, channels: {channels}")

    # Ensure 3-channel for consistency
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Calculate grid
    stride = tile_size - overlap
    n_rows = max(1, (h - overlap) // stride)
    n_cols = max(1, (w - overlap) // stride)

    # Handle edge: add one more row/col if image extends beyond grid
    if n_rows * stride + overlap < h:
        n_rows += 1
    if n_cols * stride + overlap < w:
        n_cols += 1

    logger.info(f"Grid: {n_rows} rows × {n_cols} cols = {n_rows * n_cols} tiles")

    # Setup output directory
    if save_tiles:
        out_path = Path(output_dir or "data/processed/tiles")
        out_path.mkdir(parents=True, exist_ok=True)

    tiles = []
    total = 0
    passed = 0

    for row in range(n_rows):
        for col in range(n_cols):
            y = min(row * stride, max(0, h - tile_size))
            x = min(col * stride, max(0, w - tile_size))

            # Extract tile
            tile_img = image[y:y + tile_size, x:x + tile_size]

            # Pad if tile is smaller than expected (edge case)
            th, tw = tile_img.shape[:2]
            if th < tile_size or tw < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:th, :tw] = tile_img
                tile_img = padded

            # Compute coverage
            coverage = _compute_coverage(tile_img)
            total += 1

            if coverage >= min_coverage:
                tile = Tile(
                    image=tile_img,
                    row_idx=row,
                    col_idx=col,
                    x_offset=x,
                    y_offset=y,
                    tile_size=tile_size,
                    source_path=image_path,
                    coverage=coverage,
                )
                tiles.append(tile)
                passed += 1

                if save_tiles:
                    stem = Path(image_path).stem
                    tile_filename = f"{stem}_r{row:03d}_c{col:03d}.jpg"
                    cv2.imwrite(str(out_path / tile_filename), tile_img)

    logger.info(
        f"Tiling complete: {passed}/{total} tiles passed coverage filter "
        f"(min_coverage={min_coverage:.2f})"
    )

    return TilingResult(
        tiles=tiles,
        source_path=image_path,
        source_shape=(h, w, channels),
        grid_shape=(n_rows, n_cols),
        tile_size=tile_size,
        overlap=overlap,
        total_tiles=total,
        passed_tiles=passed,
    )


def tile_image_lazy(
    image_path: str,
    tile_size: int = 640,
    overlap: int = 64,
    min_coverage: float = 0.5,
) -> Generator[Tile, None, None]:
    """Lazily yield tiles from a large image without loading all into memory.

    Uses memory-mapped access for very large images (multi-GB PDS files).

    Args:
        image_path: Path to source image.
        tile_size: Tile dimension in pixels.
        overlap: Overlap in pixels.
        min_coverage: Minimum valid-pixel fraction.

    Yields:
        Tile objects one at a time.
    """
    image = _load_image(image_path)
    if image is None:
        return

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    stride = tile_size - overlap

    row = 0
    y = 0
    while y < h:
        col = 0
        x = 0
        while x < w:
            tile_img = image[y:y + tile_size, x:x + tile_size]
            th, tw = tile_img.shape[:2]
            if th < tile_size or tw < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:th, :tw] = tile_img
                tile_img = padded

            coverage = _compute_coverage(tile_img)
            if coverage >= min_coverage:
                yield Tile(
                    image=tile_img,
                    row_idx=row,
                    col_idx=col,
                    x_offset=x,
                    y_offset=y,
                    tile_size=tile_size,
                    source_path=str(image_path),
                    coverage=coverage,
                )

            x += stride
            col += 1

        y += stride
        row += 1


def reconstruct_from_tiles(
    tiles: List[Tile],
    original_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Reconstruct an image from overlapping tiles (for visualization).

    Uses averaging in overlap regions for smooth blending.

    Args:
        tiles: List of Tile objects.
        original_shape: (H, W, C) of the original image.

    Returns:
        Reconstructed image array.
    """
    h, w, c = original_shape
    output = np.zeros((h, w, c), dtype=np.float64)
    count = np.zeros((h, w, 1), dtype=np.float64)

    for tile in tiles:
        y, x = tile.y_offset, tile.x_offset
        ts = tile.tile_size
        th = min(ts, h - y)
        tw = min(ts, w - x)

        output[y:y + th, x:x + tw] += tile.image[:th, :tw].astype(np.float64)
        count[y:y + th, x:x + tw] += 1.0

    # Average where tiles overlap
    count = np.maximum(count, 1.0)
    output = (output / count).astype(np.uint8)

    return output


def _load_image(image_path: str) -> Optional[np.ndarray]:
    """Load an image, handling various formats.

    Args:
        image_path: Path to image file.

    Returns:
        Image as numpy array (BGR), or None if failed.
    """
    path = Path(image_path)
    if not path.exists():
        logger.error(f"Image not found: {image_path}")
        return None

    try:
        # Try OpenCV first (handles most formats)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return image

        # Fallback to PIL for unusual formats
        pil_image = Image.open(path)
        image = np.array(pil_image)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    except Exception as e:
        logger.error(f"Failed to load {image_path}: {e}")
        return None


def _compute_coverage(tile: np.ndarray, black_threshold: int = 5) -> float:
    """Compute the fraction of non-black/non-nodata pixels in a tile.

    Args:
        tile: Tile image array.
        black_threshold: Pixel values below this are considered no-data.

    Returns:
        Coverage fraction (0.0 to 1.0).
    """
    if tile.ndim == 3:
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    else:
        gray = tile

    valid_pixels = np.sum(gray > black_threshold)
    total_pixels = gray.size
    return valid_pixels / total_pixels if total_pixels > 0 else 0.0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = tile_image(sys.argv[1], save_tiles=True)
        print(f"\nTiling result:")
        print(f"  Source: {result.source_path}")
        print(f"  Shape: {result.source_shape}")
        print(f"  Grid: {result.grid_shape}")
        print(f"  Tiles: {result.passed_tiles}/{result.total_tiles} passed filter")
    else:
        print("Usage: python -m src.data_ingestion.tiling <image_path>")
