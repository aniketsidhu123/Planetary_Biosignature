"""
Tests for data ingestion modules.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestExoplanetArchive:
    """Tests for NASA Exoplanet Archive data fetcher."""

    def test_synthetic_data_generation(self):
        """Synthetic data should produce valid exoplanet parameters."""
        from src.data_ingestion.exoplanet_archive import _generate_synthetic_data

        columns = ["pl_name", "pl_rade", "pl_bmasse", "pl_dens", "pl_orbeccen",
                    "pl_orbsmax", "st_teff", "st_lum", "pl_insol", "pl_eqt"]
        df = _generate_synthetic_data(columns, n_samples=100)

        assert len(df) == 100
        assert "pl_name" in df.columns
        assert "pl_rade" in df.columns

        # Physical constraints
        assert df["pl_rade"].dropna().min() > 0, "Radius should be positive"
        assert df["pl_orbeccen"].dropna().max() <= 1.0, "Eccentricity should be ≤ 1"
        assert df["st_teff"].dropna().min() > 0, "Temperature should be positive"

    def test_synthetic_data_has_missing_values(self):
        """Synthetic data should include realistic missing values."""
        from src.data_ingestion.exoplanet_archive import _generate_synthetic_data

        columns = ["pl_rade", "pl_bmasse", "pl_dens"]
        df = _generate_synthetic_data(columns, n_samples=200)

        # Should have some missing values (~10%)
        total_nulls = df.isnull().sum().sum()
        assert total_nulls > 0, "Should have some missing values"
        assert total_nulls < len(df) * len(columns), "Shouldn't be all missing"

    def test_data_summary(self):
        """get_data_summary should return valid summary dict."""
        from src.data_ingestion.exoplanet_archive import get_data_summary, _generate_synthetic_data

        columns = ["pl_name", "pl_rade", "pl_eqt", "pl_dens"]
        df = _generate_synthetic_data(columns, n_samples=50)
        summary = get_data_summary(df)

        assert "total_planets" in summary
        assert summary["total_planets"] == 50
        assert "columns" in summary

    def test_fetch_falls_back_to_synthetic(self):
        """fetch_exoplanet_data should fall back to synthetic when archive unavailable."""
        from src.data_ingestion.exoplanet_archive import fetch_exoplanet_data

        # Should work without network access (uses synthetic fallback)
        df = fetch_exoplanet_data(force_refresh=True,
                                  cache_path="data/tabular/test_exoplanets.csv")

        assert len(df) > 0
        assert "pl_rade" in df.columns or len(df.columns) > 0

        # Cleanup
        test_cache = Path("data/tabular/test_exoplanets.csv")
        if test_cache.exists():
            test_cache.unlink()


class TestPDSClient:
    """Tests for NASA PDS imagery client."""

    def test_client_initialization(self):
        """PDS client should initialize with default config."""
        from src.data_ingestion.pds_client import PDSClient

        client = PDSClient(download_dir="data/raw/test_pds")
        assert client.download_dir.exists()
        assert client.rate_limit > 0

    def test_sample_hirise_ids(self):
        """Should return curated HiRISE observation IDs."""
        from src.data_ingestion.pds_client import PDSClient

        client = PDSClient()
        ids = client.get_sample_hirise_ids()

        assert len(ids) > 0
        assert all(isinstance(id_, str) for id_ in ids)
        # All should start with ESP_ or PSP_
        assert all(id_.startswith(("ESP_", "PSP_")) for id_ in ids)


class TestBhoondhiClient:
    """Tests for ISRO Bhoonidhi client."""

    def test_analog_sites_catalog(self):
        """Should have Mars-analog sites defined."""
        from src.data_ingestion.bhoonidhi_client import BhoondhiClient, MARS_ANALOG_SITES

        assert len(MARS_ANALOG_SITES) > 0
        for site_id, site in MARS_ANALOG_SITES.items():
            assert "name" in site
            assert "bbox" in site
            assert "relevance" in site
            assert "analog_features" in site
            assert len(site["bbox"]) == 4

    def test_sample_scenes(self):
        """Should return sample scene metadata."""
        from src.data_ingestion.bhoonidhi_client import BhoondhiClient

        client = BhoondhiClient()
        scenes = client.search_scenes(site_id="rann_of_kutch")

        assert len(scenes) > 0
        assert "scene_id" in scenes[0]

    def test_site_report(self):
        """Should generate readable site report."""
        from src.data_ingestion.bhoonidhi_client import BhoondhiClient

        client = BhoondhiClient()
        report = client.get_site_report()

        assert "Mars-Analog" in report
        assert "Rann of Kutch" in report


class TestTiling:
    """Tests for image tiling utility."""

    def test_tile_image(self):
        """Should tile an image into fixed-size tiles."""
        from src.data_ingestion.tiling import tile_image

        # Create a test image
        test_img = np.random.randint(50, 200, (1280, 1920, 3), dtype=np.uint8)
        test_path = Path("data/processed/test_tile_input.jpg")
        test_path.parent.mkdir(parents=True, exist_ok=True)
        import cv2
        cv2.imwrite(str(test_path), test_img)

        result = tile_image(str(test_path), tile_size=640, overlap=64)

        assert result.total_tiles > 0
        assert result.passed_tiles <= result.total_tiles
        assert result.tile_size == 640
        assert result.grid_shape[0] > 0
        assert result.grid_shape[1] > 0

        # Check tile dimensions
        for tile in result.tiles:
            assert tile.image.shape == (640, 640, 3)
            assert 0.0 <= tile.coverage <= 1.0

        # Cleanup
        test_path.unlink()

    def test_tile_coverage_filter(self):
        """Black tiles should be filtered out by coverage threshold."""
        from src.data_ingestion.tiling import tile_image

        # Create image with large black region
        test_img = np.zeros((1280, 1920, 3), dtype=np.uint8)
        test_img[:640, :640] = 128  # Only top-left quadrant has content

        test_path = Path("data/processed/test_coverage_input.jpg")
        test_path.parent.mkdir(parents=True, exist_ok=True)
        import cv2
        cv2.imwrite(str(test_path), test_img)

        result = tile_image(str(test_path), tile_size=640, min_coverage=0.5)

        # Should filter out mostly-black tiles
        assert result.passed_tiles < result.total_tiles

        test_path.unlink()

    def test_lazy_tiling(self):
        """Lazy tiling should yield tiles one at a time."""
        from src.data_ingestion.tiling import tile_image_lazy

        test_img = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
        test_path = Path("data/processed/test_lazy_input.jpg")
        test_path.parent.mkdir(parents=True, exist_ok=True)
        import cv2
        cv2.imwrite(str(test_path), test_img)

        tiles = list(tile_image_lazy(str(test_path), tile_size=640))
        assert len(tiles) >= 1

        test_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
