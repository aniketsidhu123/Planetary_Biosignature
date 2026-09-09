"""
ISRO Bhoonidhi Earth observation data client.

Provides access to ISRO's Bhoonidhi portal for downloading
Mars-analog terrain imagery (arid/salt-flat regions in India)
for training biosignature proxy detectors.

Note: Requires bhoonidhi-downloader package or falls back to
direct HTTP access for open data scenes.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from src.utils.config import get_config
from src.utils.logging_setup import get_logger

logger = get_logger("data_ingestion.bhoonidhi")


# Mars-analog sites in India with geological relevance
MARS_ANALOG_SITES = {
    "rann_of_kutch": {
        "name": "Rann of Kutch (Great & Little)",
        "bbox": (68.0, 22.5, 72.0, 24.5),
        "relevance": "Salt flats with evaporite deposits — analog for Mars chloride/sulfate deposits",
        "analog_features": ["mineral_water_interaction", "erosion_morphology"],
    },
    "thar_desert": {
        "name": "Thar Desert",
        "bbox": (69.0, 24.0, 72.0, 28.0),
        "relevance": "Arid aeolian/fluvial landscape — analog for Mars dune-channel interactions",
        "analog_features": ["erosion_morphology", "sedimentary_layering"],
    },
    "ladakh_cold_desert": {
        "name": "Ladakh Cold Desert",
        "bbox": (76.0, 32.0, 80.0, 36.0),
        "relevance": "High-altitude cold desert with permafrost — analog for Mars periglacial features",
        "analog_features": ["erosion_morphology", "mineral_water_interaction"],
    },
    "sambhar_lake": {
        "name": "Sambhar Salt Lake",
        "bbox": (74.5, 26.8, 75.2, 27.1),
        "relevance": "Inland saline lake with evaporite crusts — analog for Meridiani Planum sulfates",
        "analog_features": ["mineral_water_interaction", "sedimentary_layering"],
    },
    "spiti_valley": {
        "name": "Spiti Valley",
        "bbox": (77.5, 31.5, 78.5, 33.0),
        "relevance": "Exposed Precambrian-Cambrian sedimentary sequences — analog for layered Mars deposits",
        "analog_features": ["sedimentary_layering"],
    },
}


class BhoondhiClient:
    """Client for accessing ISRO Bhoonidhi Earth observation data.

    Focuses on Mars-analog terrain sites in India for training
    biosignature proxy detection models.
    """

    def __init__(self, download_dir: Optional[str] = None):
        """Initialize Bhoonidhi client.

        Args:
            download_dir: Local directory for downloads.
        """
        config = get_config()
        bhoonidhi_config = config.get_section("data_ingestion").get("bhoonidhi", {})

        self.download_dir = Path(
            download_dir or bhoonidhi_config.get("download_dir", "data/raw/bhoonidhi")
        )
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self._downloader_available = self._check_downloader()
        logger.info(
            f"Bhoonidhi client initialized. "
            f"bhoonidhi-downloader available: {self._downloader_available}"
        )

    def _check_downloader(self) -> bool:
        """Check if bhoonidhi-downloader package is available."""
        try:
            import bhoonidhi_downloader  # noqa: F401
            return True
        except ImportError:
            return False

    def get_analog_sites(self) -> Dict[str, dict]:
        """Get the catalog of Mars-analog sites.

        Returns:
            Dict mapping site IDs to site metadata.
        """
        return MARS_ANALOG_SITES

    def search_scenes(
        self,
        site_id: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        satellite: str = "ResourceSat-2",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_cloud_cover: float = 20.0,
        max_results: int = 10,
    ) -> List[Dict]:
        """Search for available scenes at a Mars-analog site.

        Args:
            site_id: ID from MARS_ANALOG_SITES (e.g., "rann_of_kutch").
            bbox: Custom bounding box (lon_min, lat_min, lon_max, lat_max).
            satellite: ISRO satellite name.
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).
            max_cloud_cover: Maximum cloud cover percentage.
            max_results: Maximum results to return.

        Returns:
            List of scene metadata dicts.
        """
        if site_id and site_id in MARS_ANALOG_SITES:
            bbox = MARS_ANALOG_SITES[site_id]["bbox"]
            site_name = MARS_ANALOG_SITES[site_id]["name"]
            logger.info(f"Searching scenes for {site_name}")
        elif bbox:
            logger.info(f"Searching scenes for bbox: {bbox}")
        else:
            logger.error("Must provide either site_id or bbox")
            return []

        if self._downloader_available:
            return self._search_via_downloader(
                bbox, satellite, start_date, end_date, max_cloud_cover, max_results
            )
        else:
            logger.warning(
                "bhoonidhi-downloader not installed. "
                "Install with: pip install bhoonidhi-downloader\n"
                "Returning sample scene metadata for development."
            )
            return self._get_sample_scenes(site_id or "rann_of_kutch")

    def _search_via_downloader(
        self,
        bbox: Tuple[float, float, float, float],
        satellite: str,
        start_date: Optional[str],
        end_date: Optional[str],
        max_cloud_cover: float,
        max_results: int,
    ) -> List[Dict]:
        """Search using the bhoonidhi-downloader package."""
        try:
            from bhoonidhi_downloader import BhoondhiAPI

            api = BhoondhiAPI()
            results = api.search(
                bbox=bbox,
                satellite=satellite,
                start_date=start_date,
                end_date=end_date,
                cloud_cover_max=max_cloud_cover,
                limit=max_results,
            )
            logger.info(f"Found {len(results)} scenes via Bhoonidhi API")
            return results

        except Exception as e:
            logger.error(f"Bhoonidhi API search failed: {e}")
            return []

    def _get_sample_scenes(self, site_id: str) -> List[Dict]:
        """Return sample scene metadata for development without API access."""
        site = MARS_ANALOG_SITES.get(site_id, list(MARS_ANALOG_SITES.values())[0])
        return [
            {
                "scene_id": f"SAMPLE_{site_id.upper()}_001",
                "satellite": "ResourceSat-2",
                "sensor": "LISS-III",
                "date": "2025-03-15",
                "bbox": site["bbox"],
                "cloud_cover": 5.0,
                "resolution_m": 23.5,
                "status": "sample_metadata_only",
                "analog_relevance": site["relevance"],
            },
            {
                "scene_id": f"SAMPLE_{site_id.upper()}_002",
                "satellite": "CartoSat-3",
                "sensor": "MX",
                "date": "2025-06-22",
                "bbox": site["bbox"],
                "cloud_cover": 8.0,
                "resolution_m": 1.13,
                "status": "sample_metadata_only",
                "analog_relevance": site["relevance"],
            },
        ]

    def download_scene(self, scene_id: str) -> Optional[Path]:
        """Download a scene by ID.

        Args:
            scene_id: Scene identifier from search results.

        Returns:
            Path to downloaded scene, or None if failed.
        """
        if not self._downloader_available:
            logger.warning(
                f"Cannot download {scene_id}: bhoonidhi-downloader not installed"
            )
            return None

        try:
            from bhoonidhi_downloader import BhoondhiAPI

            api = BhoondhiAPI()
            filepath = api.download(
                scene_id=scene_id,
                output_dir=str(self.download_dir),
            )
            logger.info(f"Downloaded scene {scene_id} → {filepath}")
            return Path(filepath)

        except Exception as e:
            logger.error(f"Download failed for {scene_id}: {e}")
            return None

    def get_site_report(self) -> str:
        """Generate a human-readable report of available Mars-analog sites.

        Returns:
            Formatted string report.
        """
        lines = ["Mars-Analog Sites Catalog (India)", "=" * 50, ""]
        for site_id, site in MARS_ANALOG_SITES.items():
            lines.extend([
                f"📍 {site['name']} ({site_id})",
                f"   BBox: {site['bbox']}",
                f"   Relevance: {site['relevance']}",
                f"   Analog features: {', '.join(site['analog_features'])}",
                "",
            ])
        return "\n".join(lines)


if __name__ == "__main__":
    client = BhoondhiClient()
    print(client.get_site_report())
