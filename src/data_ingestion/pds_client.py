"""
NASA Planetary Data System (PDS) imagery client.

Provides functions to search and download HiRISE / Mastcam imagery
from the PDS Imaging Node, with rate limiting and local caching.
"""

import time
import hashlib
import requests
from pathlib import Path
from typing import Optional, List, Dict
from urllib.parse import urljoin

from src.utils.config import get_config
from src.utils.logging_setup import get_logger

logger = get_logger("data_ingestion.pds")


# ── PDS API endpoints ─────────────────────────────────────────────────
PDS_SEARCH_API = "https://pds-imaging.jpl.nasa.gov/solr/pds_archives/select"
PDS_IMAGE_BASE = "https://pds-imaging.jpl.nasa.gov"

# Known HiRISE browse-image collections for Mars
HIRISE_BROWSE_BASE = "https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS"


class PDSClient:
    """Client for searching and downloading NASA PDS imagery.

    Focuses on Mars orbital and surface imagery from HiRISE,
    Mastcam, and Mastcam-Z instruments.
    """

    def __init__(self, download_dir: Optional[str] = None, rate_limit: Optional[float] = None):
        """Initialize PDS client.

        Args:
            download_dir: Local directory for downloaded images.
            rate_limit: Seconds between API requests (be polite to NASA).
        """
        config = get_config()
        pds_config = config.get_section("data_ingestion").get("pds", {})

        self.download_dir = Path(download_dir or pds_config.get("download_dir", "data/raw/pds"))
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self.rate_limit = rate_limit or pds_config.get("rate_limit_seconds", 1.0)
        self._last_request_time = 0.0

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ExoScope/0.1.0 (Planetary Biosignature Research)"
        })

        logger.info(f"PDS client initialized. Download dir: {self.download_dir}")

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def search_images(
        self,
        target: str = "Mars",
        instrument: str = "HiRISE",
        max_results: int = 20,
        keywords: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Search PDS for images matching criteria.

        Args:
            target: Planetary body (e.g., "Mars", "Moon").
            instrument: Instrument name (e.g., "HiRISE", "Mastcam").
            max_results: Maximum number of results to return.
            keywords: Optional keyword filters.

        Returns:
            List of dicts with image metadata (id, url, description).
        """
        self._rate_limit_wait()

        # Build Solr query
        query_parts = [f"target:{target}", f"instrument_id:{instrument}"]
        if keywords:
            keyword_q = " OR ".join(keywords)
            query_parts.append(f"description:({keyword_q})")

        params = {
            "q": " AND ".join(query_parts),
            "rows": max_results,
            "wt": "json",
            "fl": "identifier,title,description,file_ref,product_id",
        }

        try:
            logger.info(f"Searching PDS: target={target}, instrument={instrument}")
            response = self.session.get(PDS_SEARCH_API, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            docs = data.get("response", {}).get("docs", [])
            logger.info(f"Found {len(docs)} results")

            results = []
            for doc in docs:
                results.append({
                    "id": doc.get("identifier", doc.get("product_id", "unknown")),
                    "title": doc.get("title", "Untitled"),
                    "description": doc.get("description", ""),
                    "file_ref": doc.get("file_ref", ""),
                })
            return results

        except requests.RequestException as e:
            logger.error(f"PDS search failed: {e}")
            return []

    def download_image(self, image_url: str, filename: Optional[str] = None) -> Optional[Path]:
        """Download a single image from PDS.

        Args:
            image_url: Full URL to the image file.
            filename: Optional local filename. Auto-generated from URL if None.

        Returns:
            Path to downloaded file, or None if download failed.
        """
        self._rate_limit_wait()

        if not filename:
            # Generate filename from URL hash + extension
            url_hash = hashlib.md5(image_url.encode()).hexdigest()[:12]
            ext = Path(image_url).suffix or ".jpg"
            filename = f"pds_{url_hash}{ext}"

        filepath = self.download_dir / filename

        # Check if already downloaded
        if filepath.exists():
            logger.info(f"Already cached: {filepath}")
            return filepath

        try:
            logger.info(f"Downloading: {image_url}")
            response = self.session.get(image_url, timeout=60, stream=True)
            response.raise_for_status()

            # Stream to disk
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = filepath.stat().st_size
            logger.info(f"Downloaded {file_size / 1024:.1f} KB → {filepath}")
            return filepath

        except requests.RequestException as e:
            logger.error(f"Download failed for {image_url}: {e}")
            if filepath.exists():
                filepath.unlink()
            return None

    def download_hirise_browse(
        self,
        observation_id: str,
        color: bool = True,
    ) -> Optional[Path]:
        """Download a HiRISE browse image by observation ID.

        HiRISE browse images are reduced-resolution JPEGs suitable for
        ML training without requiring full-resolution JP2 processing.

        Args:
            observation_id: HiRISE observation ID (e.g., "ESP_011277_1825").
            color: If True, download color browse; else grayscale.

        Returns:
            Path to downloaded browse image, or None if failed.
        """
        # HiRISE browse URL pattern
        orbit_range = observation_id.split("_")[1][:4]  # e.g., "0112" from "ESP_011277_1825"
        prefix = observation_id[:3]  # "ESP" or "PSP"

        if color:
            browse_path = f"RDR/{prefix}/ORB_{orbit_range}00_{orbit_range}99/{observation_id}/{observation_id}_COLOR.abrowse.jpg"
        else:
            browse_path = f"RDR/{prefix}/ORB_{orbit_range}00_{orbit_range}99/{observation_id}/{observation_id}_RED.abrowse.jpg"

        url = f"{HIRISE_BROWSE_BASE}/{browse_path}"
        return self.download_image(url, filename=f"hirise_{observation_id}_{'color' if color else 'red'}.jpg")

    def get_sample_hirise_ids(self) -> List[str]:
        """Return a curated list of HiRISE observation IDs for Mars-analog features.

        These are selected for containing visible geological features relevant
        to biosignature proxy detection (layering, mineral deposits, channels).

        Returns:
            List of HiRISE observation ID strings.
        """
        return [
            # Jezero Crater delta (layered sediments)
            "ESP_045994_1985",
            "ESP_046060_1985",
            # Mawrth Vallis (clay mineral deposits)
            "ESP_012095_2040",
            "ESP_013740_2045",
            # Nili Fossae (olivine/carbonate — mineral-water)
            "ESP_016060_2010",
            # Valles Marineris layered deposits
            "ESP_019988_1700",
            "ESP_012440_1750",
            # Gale Crater (Curiosity landing site — layered sediments)
            "ESP_012551_1750",
            "ESP_028335_1755",
            # Eberswalde Crater (ancient delta)
            "ESP_016698_1560",
            # Holden Crater (alluvial fans)
            "ESP_014163_1530",
            # Aram Chaos (hematite — mineral-water)
            "ESP_016450_1830",
        ]

    def batch_download(
        self,
        observation_ids: Optional[List[str]] = None,
        max_downloads: int = 10,
        color: bool = True,
    ) -> List[Path]:
        """Download multiple HiRISE browse images.

        Args:
            observation_ids: List of HiRISE IDs. Uses curated defaults if None.
            max_downloads: Maximum number of images to download.
            color: Download color browse images.

        Returns:
            List of paths to successfully downloaded images.
        """
        if observation_ids is None:
            observation_ids = self.get_sample_hirise_ids()

        downloaded = []
        for obs_id in observation_ids[:max_downloads]:
            path = self.download_hirise_browse(obs_id, color=color)
            if path:
                downloaded.append(path)
            else:
                logger.warning(f"Skipping {obs_id} — download failed")

        logger.info(f"Successfully downloaded {len(downloaded)}/{min(max_downloads, len(observation_ids))} images")
        return downloaded


if __name__ == "__main__":
    # CLI: download sample HiRISE imagery
    client = PDSClient()
    print("\nCurated HiRISE observation IDs:")
    for obs_id in client.get_sample_hirise_ids():
        print(f"  • {obs_id}")
    print(f"\nTo batch download, run: client.batch_download(max_downloads=5)")
