"""
NASA Exoplanet Archive data fetcher.

Queries the Exoplanet Archive TAP (Table Access Protocol) service directly
over HTTP, pulling the Planetary Systems Composite Parameters table
(``pscomppars`` — one row per confirmed planet) with the full set of
planetary, stellar and system columns used downstream.

Design notes
------------
* **Direct HTTP over astroquery.** The TAP sync endpoint returns CSV in a
  single request (~2.3 MB / ~7 s for the whole catalog). `astroquery` is
  supported as an optional fallback, but it is not required — it drags in
  astropy/pyvo and was the reason the pipeline used to fail closed.
* **No silent synthetic fallback.** Earlier versions quietly substituted
  generated planets when the network call failed and cached them to the
  real cache path, so the whole pipeline could run on fabricated data
  without anything in the UI saying so. Synthetic data is now opt-in
  (``allow_synthetic=True``), is written to a separate path, and is always
  tagged in the sidecar manifest so callers can warn loudly.
* **Provenance manifest.** Every cache write is accompanied by
  ``<cache>.meta.json`` recording source, row count, query, and fetch time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import requests

from src.utils.config import get_config
from src.utils.logging_setup import get_logger

logger = get_logger("data_ingestion.exoplanet")


# ── TAP service ────────────────────────────────────────────────────────
TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
DEFAULT_TABLE = "pscomppars"
REQUEST_TIMEOUT = 180        # seconds — full catalog is ~2.3 MB
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0          # seconds, doubled per attempt


# ── Column metadata ────────────────────────────────────────────────────
# Every column requested from the archive, mapped to a human-readable label.
# Grouped by domain so the UI can present them coherently.

PLANET_COLUMNS = {
    "pl_name": "Planet Name",
    "hostname": "Host Star",
    "pl_letter": "Planet Letter",
    "pl_rade": "Planet Radius (R⊕)",
    "pl_bmasse": "Planet Mass (M⊕)",
    "pl_bmassprov": "Mass Measurement Provenance",
    "pl_dens": "Planet Density (g/cm³)",
    "pl_orbper": "Orbital Period (days)",
    "pl_orbsmax": "Semi-Major Axis (AU)",
    "pl_orbeccen": "Orbital Eccentricity",
    "pl_insol": "Insolation Flux (S⊕)",
    "pl_eqt": "Equilibrium Temperature (K)",
}

TRANSIT_COLUMNS = {
    "pl_trandep": "Transit Depth (%)",
    "pl_trandur": "Transit Duration (hours)",
    "pl_ratdor": "a/R★ Ratio",
    "pl_ratror": "Rp/R★ Ratio",
    "pl_imppar": "Impact Parameter",
    "pl_rvamp": "Radial Velocity Amplitude (m/s)",
}

STELLAR_COLUMNS = {
    "st_spectype": "Spectral Type",
    "st_teff": "Stellar Effective Temp (K)",
    "st_rad": "Stellar Radius (R☉)",
    "st_mass": "Stellar Mass (M☉)",
    "st_met": "Stellar Metallicity [Fe/H]",
    "st_metratio": "Metallicity Ratio",
    "st_logg": "Stellar Surface Gravity (log g)",
    "st_lum": "Stellar Luminosity (log L☉)",
    "st_age": "Stellar Age (Gyr)",
    "st_dens": "Stellar Density (g/cm³)",
    "st_vsin": "Stellar Rotational Velocity (km/s)",
    "st_rotp": "Stellar Rotation Period (days)",
}

SYSTEM_COLUMNS = {
    "sy_snum": "Number of Stars",
    "sy_pnum": "Number of Planets",
    "sy_mnum": "Number of Moons",
    "sy_dist": "Distance (pc)",
    "sy_plx": "Parallax (mas)",
    "sy_pm": "Proper Motion (mas/yr)",
    "sy_vmag": "V Magnitude",
    "sy_kmag": "Ks Magnitude",
    "sy_gaiamag": "Gaia Magnitude",
    "ra": "Right Ascension (deg)",
    "dec": "Declination (deg)",
}

DISCOVERY_COLUMNS = {
    "discoverymethod": "Discovery Method",
    "disc_year": "Discovery Year",
    "disc_facility": "Discovery Facility",
    "pl_controv_flag": "Controversial Flag",
    "ttv_flag": "Transit Timing Variations",
    "tran_flag": "Detected by Transit",
    "rv_flag": "Detected by Radial Velocity",
}

COLUMN_DESCRIPTIONS = {
    **PLANET_COLUMNS,
    **TRANSIT_COLUMNS,
    **STELLAR_COLUMNS,
    **SYSTEM_COLUMNS,
    **DISCOVERY_COLUMNS,
}

COLUMN_GROUPS = {
    "Planetary": list(PLANET_COLUMNS),
    "Transit / RV Geometry": list(TRANSIT_COLUMNS),
    "Stellar": list(STELLAR_COLUMNS),
    "System & Astrometry": list(SYSTEM_COLUMNS),
    "Discovery": list(DISCOVERY_COLUMNS),
}

# Columns the archive returns as text rather than numbers.
TEXT_COLUMNS = {
    "pl_name", "hostname", "pl_letter", "pl_bmassprov",
    "st_spectype", "st_metratio", "discoverymethod", "disc_facility",
}

ALL_COLUMNS = list(COLUMN_DESCRIPTIONS.keys())


@dataclass
class FetchMetadata:
    """Provenance record for a cached exoplanet table."""

    source: str                 # "nasa_tap", "astroquery", "synthetic", "cache"
    table: str
    n_rows: int
    n_columns: int
    columns: List[str]
    fetched_at: str
    query: str = ""
    is_synthetic: bool = False
    notes: str = ""

    @property
    def fetched_at_display(self) -> str:
        """Human-friendly fetch timestamp."""
        try:
            dt = datetime.fromisoformat(self.fetched_at)
            return dt.strftime("%Y-%m-%d %H:%M UTC")
        except (ValueError, TypeError):
            return self.fetched_at


class ExoplanetArchiveError(RuntimeError):
    """Raised when the NASA Exoplanet Archive cannot be reached or parsed."""


# ── TAP query construction ─────────────────────────────────────────────
def build_tap_query(
    columns: List[str],
    table: str = DEFAULT_TABLE,
    max_rows: Optional[int] = None,
    where: Optional[str] = None,
) -> str:
    """Build an ADQL query for the Exoplanet Archive TAP service.

    Args:
        columns: Column names to select.
        table: Archive table name (default ``pscomppars``).
        max_rows: Optional row cap. Emitted as ``SELECT TOP n`` — note this
            goes immediately after SELECT, which is where ADQL expects it.
        where: Optional ADQL WHERE clause (without the ``WHERE`` keyword).

    Returns:
        The ADQL query string.
    """
    select_clause = "SELECT"
    if max_rows:
        select_clause += f" TOP {int(max_rows)}"

    query = f"{select_clause} {', '.join(columns)} FROM {table}"
    if where:
        query += f" WHERE {where}"
    return query


def query_tap(
    query: str,
    timeout: int = REQUEST_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> pd.DataFrame:
    """Execute an ADQL query against the NASA Exoplanet Archive TAP service.

    Retries on transient network/server errors with exponential backoff.

    Args:
        query: ADQL query string.
        timeout: Per-request timeout in seconds.
        max_retries: Number of attempts before giving up.

    Returns:
        DataFrame of results.

    Raises:
        ExoplanetArchiveError: If all attempts fail or the response is not
            parseable CSV.
    """
    params = {"query": query, "format": "csv"}
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"TAP request (attempt {attempt}/{max_retries}): {query[:120]}...")
            response = requests.get(
                TAP_SYNC_URL,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "ExoScope/0.2 (habitability research pipeline)"},
            )
            response.raise_for_status()

            text = response.text
            # The TAP service reports ADQL errors with HTTP 200 and a plain
            # text body, so check the payload actually looks like our CSV.
            if not text.strip():
                raise ExoplanetArchiveError("TAP service returned an empty response")
            if text.lstrip().lower().startswith(("error", "<!doctype", "<html")):
                raise ExoplanetArchiveError(f"TAP service error: {text[:300]}")

            df = pd.read_csv(StringIO(text))
            if df.empty:
                raise ExoplanetArchiveError("TAP query returned zero rows")

            logger.info(f"TAP returned {len(df)} rows × {len(df.columns)} columns")
            return df

        except (requests.RequestException, pd.errors.ParserError, ExoplanetArchiveError) as exc:
            last_error = exc
            logger.warning(f"TAP attempt {attempt} failed: {exc}")
            if attempt < max_retries:
                backoff = RETRY_BACKOFF * (2 ** (attempt - 1))
                logger.info(f"Retrying in {backoff:.0f}s...")
                time.sleep(backoff)

    raise ExoplanetArchiveError(
        f"NASA Exoplanet Archive unreachable after {max_retries} attempts: {last_error}"
    )


def _fetch_via_astroquery(columns: List[str], table: str) -> pd.DataFrame:
    """Fetch via astroquery as a secondary path.

    Only used when the direct TAP call fails and astroquery is installed.

    Args:
        columns: Column names to select.
        table: Archive table name.

    Returns:
        DataFrame of results.

    Raises:
        ExoplanetArchiveError: If astroquery is missing or the query fails.
    """
    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    except ImportError as exc:
        raise ExoplanetArchiveError(f"astroquery not installed: {exc}") from exc

    try:
        logger.info("Falling back to astroquery client...")
        result = NasaExoplanetArchive.query_criteria(
            table=table, select=", ".join(columns)
        )
        return result.to_pandas()
    except Exception as exc:  # astroquery raises a wide variety of errors
        raise ExoplanetArchiveError(f"astroquery query failed: {exc}") from exc


# ── Public API ─────────────────────────────────────────────────────────
def fetch_exoplanet_data(
    columns: Optional[List[str]] = None,
    cache_path: Optional[str] = None,
    force_refresh: bool = False,
    max_rows: Optional[int] = None,
    allow_synthetic: bool = False,
    return_metadata: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, FetchMetadata]:
    """Fetch the exoplanet catalog from the NASA Exoplanet Archive.

    Downloads the Planetary Systems Composite Parameters table via the TAP
    service and caches it locally as CSV alongside a provenance manifest.

    Args:
        columns: Columns to fetch. None uses the full configured set.
        cache_path: Path to save/load cached CSV. None uses config default.
        force_refresh: Re-download even if a cache exists.
        max_rows: Optional row cap (useful for quick local runs).
        allow_synthetic: If True, fall back to generated data when the
            archive is unreachable. Off by default — a failed fetch should
            be visible, not papered over. Synthetic data is always tagged
            in the returned metadata and written to a separate cache path.
        return_metadata: If True, return ``(df, FetchMetadata)``.

    Returns:
        DataFrame of exoplanet parameters, or ``(df, metadata)`` when
        ``return_metadata`` is set.

    Raises:
        ExoplanetArchiveError: If the archive is unreachable and
            ``allow_synthetic`` is False.
    """
    config = get_config()
    ea_config = config.get_section("data_ingestion").get("exoplanet_archive", {})

    columns = columns or ea_config.get("columns") or ALL_COLUMNS
    table = ea_config.get("table_name", DEFAULT_TABLE)
    max_rows = max_rows if max_rows is not None else ea_config.get("max_rows")
    cache_file = Path(cache_path or ea_config.get("cache_file", "data/tabular/exoplanets.csv"))

    # ── Cache hit ──────────────────────────────────────────────────────
    if not force_refresh and cache_file.exists():
        logger.info(f"Loading cached exoplanet data from {cache_file}")
        df = pd.read_csv(cache_file)
        meta = _load_metadata(cache_file)
        if meta is None:
            meta = FetchMetadata(
                source="cache",
                table=table,
                n_rows=len(df),
                n_columns=len(df.columns),
                columns=list(df.columns),
                fetched_at=datetime.fromtimestamp(
                    cache_file.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                notes="Cache predates provenance tracking; origin unverified.",
            )
        logger.info(f"Loaded {len(df)} exoplanets from cache (source={meta.source})")
        return (df, meta) if return_metadata else df

    # ── Live fetch ─────────────────────────────────────────────────────
    query = build_tap_query(columns, table=table, max_rows=max_rows)
    df: Optional[pd.DataFrame] = None
    source = ""
    fetch_error: Optional[Exception] = None

    try:
        df = query_tap(query)
        source = "nasa_tap"
    except ExoplanetArchiveError as exc:
        fetch_error = exc
        logger.warning(f"Direct TAP fetch failed: {exc}")
        try:
            df = _fetch_via_astroquery(columns, table)
            source = "astroquery"
        except ExoplanetArchiveError as aq_exc:
            fetch_error = aq_exc
            logger.error(f"astroquery fallback also failed: {aq_exc}")

    if df is None:
        if not allow_synthetic:
            raise ExoplanetArchiveError(
                "Could not reach the NASA Exoplanet Archive. "
                "Check your network connection, or pass allow_synthetic=True "
                "to generate development data instead (which is NOT real "
                f"observational data). Underlying error: {fetch_error}"
            )
        logger.warning(
            "⚠ Generating SYNTHETIC exoplanet data — this is not real NASA data "
            "and must not be used for any scientific claim."
        )
        df = _generate_synthetic_data(columns)
        source = "synthetic"
        # Never overwrite the real cache with fabricated rows.
        cache_file = cache_file.with_name(f"{cache_file.stem}_synthetic{cache_file.suffix}")

    # ── Clean, cache, record provenance ────────────────────────────────
    df = _validate_data(df, columns)

    metadata = FetchMetadata(
        source=source,
        table=table,
        n_rows=len(df),
        n_columns=len(df.columns),
        columns=list(df.columns),
        fetched_at=datetime.now(timezone.utc).isoformat(),
        query=query,
        is_synthetic=(source == "synthetic"),
        notes=(
            "SYNTHETIC DEVELOPMENT DATA — not real observations."
            if source == "synthetic"
            else f"NASA Exoplanet Archive {table} via {source}."
        ),
    )

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    _save_metadata(cache_file, metadata)
    logger.info(f"Cached {len(df)} exoplanets to {cache_file} (source={source})")

    return (df, metadata) if return_metadata else df


def _metadata_path(cache_file: Path) -> Path:
    """Return the sidecar manifest path for a cache file."""
    return cache_file.with_suffix(cache_file.suffix + ".meta.json")


def _save_metadata(cache_file: Path, metadata: FetchMetadata) -> None:
    """Write the provenance manifest next to the cached CSV."""
    try:
        _metadata_path(cache_file).write_text(
            json.dumps(asdict(metadata), indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning(f"Could not write metadata manifest: {exc}")


def _load_metadata(cache_file: Path) -> Optional[FetchMetadata]:
    """Read the provenance manifest for a cached CSV, if one exists."""
    meta_file = _metadata_path(cache_file)
    if not meta_file.exists():
        return None
    try:
        payload = json.loads(meta_file.read_text(encoding="utf-8"))
        known = {f for f in FetchMetadata.__dataclass_fields__}
        return FetchMetadata(**{k: v for k, v in payload.items() if k in known})
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        logger.warning(f"Could not read metadata manifest: {exc}")
        return None


def get_cache_metadata(cache_path: Optional[str] = None) -> Optional[FetchMetadata]:
    """Return provenance metadata for the current cache, if available.

    Args:
        cache_path: Path to the cached CSV. None uses the config default.

    Returns:
        FetchMetadata, or None when no manifest exists.
    """
    config = get_config()
    ea_config = config.get_section("data_ingestion").get("exoplanet_archive", {})
    cache_file = Path(cache_path or ea_config.get("cache_file", "data/tabular/exoplanets.csv"))
    return _load_metadata(cache_file)


def _validate_data(df: pd.DataFrame, expected_columns: List[str]) -> pd.DataFrame:
    """Validate fetched data, coerce dtypes, and report quality metrics.

    Args:
        df: Raw DataFrame from the archive.
        expected_columns: Expected column names.

    Returns:
        Cleaned DataFrame.
    """
    logger.info("── Data Quality Report ──")
    logger.info(f"  Total rows: {len(df)}")

    present = [c for c in expected_columns if c in df.columns]
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        logger.warning(f"  Missing columns: {missing}")
    logger.info(f"  Present columns: {len(present)}/{len(expected_columns)}")

    # Coerce numeric columns — the TAP CSV can carry empty strings that
    # would otherwise make an entire column dtype=object.
    for col in df.columns:
        if col not in TEXT_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in present:
        null_pct = 100.0 * df[col].isnull().sum() / max(len(df), 1)
        if null_pct > 0:
            logger.info(f"  {col}: {df[col].isnull().sum()} missing ({null_pct:.1f}%)")

    # Drop rows with no usable numeric measurements at all.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        before = len(df)
        df = df.dropna(subset=list(numeric_cols), how="all")
        if before - len(df) > 0:
            logger.info(f"  Dropped {before - len(df)} rows with all-null numeric features")

    # Drop exact duplicate planet entries, keeping the first.
    if "pl_name" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["pl_name"], keep="first")
        if before - len(df) > 0:
            logger.info(f"  Dropped {before - len(df)} duplicate planet rows")

    logger.info(f"  Final dataset: {len(df)} exoplanets")
    return df.reset_index(drop=True)


# ── Synthetic development data ─────────────────────────────────────────
def _generate_synthetic_data(columns: List[str], n_samples: int = 500) -> pd.DataFrame:
    """Generate synthetic exoplanet data for offline development/testing.

    Produces physically plausible (not uniformly random) parameter
    distributions based on observed population statistics. This is a
    development convenience only — callers must surface the fact that
    the data is fabricated. ``fetch_exoplanet_data`` only reaches this
    path when explicitly asked via ``allow_synthetic=True``.

    Args:
        columns: Column names to generate.
        n_samples: Number of synthetic planets.

    Returns:
        DataFrame with synthetic data.
    """
    logger.info(f"Generating {n_samples} synthetic exoplanets...")
    rng = np.random.default_rng(42)
    data: dict = {}

    if "pl_name" in columns:
        data["pl_name"] = [f"Synth-{i+1:04d}b" for i in range(n_samples)]
    if "hostname" in columns:
        data["hostname"] = [f"Synth-{i+1:04d}" for i in range(n_samples)]
    if "pl_letter" in columns:
        data["pl_letter"] = ["b"] * n_samples

    # Bimodal radius distribution: super-Earths and mini-Neptunes dominate.
    if "pl_rade" in columns:
        data["pl_rade"] = np.concatenate([
            rng.lognormal(np.log(1.5), 0.3, n_samples // 2),
            rng.lognormal(np.log(8.0), 0.6, n_samples - n_samples // 2),
        ])
        rng.shuffle(data["pl_rade"])

    if "pl_bmasse" in columns:
        if "pl_rade" in data:
            # Mass roughly scales with radius^2.06 (Chen & Kipping 2017)
            data["pl_bmasse"] = data["pl_rade"] ** 2.06 * rng.lognormal(0, 0.3, n_samples)
        else:
            data["pl_bmasse"] = rng.lognormal(np.log(5.0), 1.0, n_samples)

    if "pl_dens" in columns:
        if "pl_bmasse" in data and "pl_rade" in data:
            data["pl_dens"] = (data["pl_bmasse"] * 5.51) / (data["pl_rade"] ** 3)
        else:
            data["pl_dens"] = rng.lognormal(np.log(4.0), 0.5, n_samples)

    if "pl_orbeccen" in columns:
        data["pl_orbeccen"] = rng.beta(1.12, 5.0, n_samples)

    if "pl_orbsmax" in columns:
        data["pl_orbsmax"] = rng.lognormal(np.log(0.5), 1.0, n_samples)

    if "pl_orbper" in columns:
        if "pl_orbsmax" in data:
            data["pl_orbper"] = 365.25 * data["pl_orbsmax"] ** 1.5
        else:
            data["pl_orbper"] = rng.lognormal(np.log(30), 1.5, n_samples)

    if "st_teff" in columns:
        data["st_teff"] = rng.normal(5200, 800, n_samples).clip(2500, 10000)

    if "st_lum" in columns:
        if "st_teff" in data:
            data["st_lum"] = 4.0 * np.log10(data["st_teff"] / 5778) + rng.normal(0, 0.2, n_samples)
        else:
            data["st_lum"] = rng.normal(0, 0.5, n_samples)

    if "pl_insol" in columns:
        if "st_lum" in data and "pl_orbsmax" in data:
            data["pl_insol"] = (10 ** data["st_lum"]) / (data["pl_orbsmax"] ** 2)
        else:
            data["pl_insol"] = rng.lognormal(np.log(1.0), 1.5, n_samples)

    if "pl_eqt" in columns:
        if "st_teff" in data and "pl_orbsmax" in data:
            data["pl_eqt"] = data["st_teff"] * (0.25 / data["pl_orbsmax"]) ** 0.5
        else:
            data["pl_eqt"] = rng.normal(400, 200, n_samples).clip(50, 3000)

    if "st_mass" in columns:
        if "st_teff" in data:
            data["st_mass"] = np.clip(
                (data["st_teff"] / 5778) ** 1.6 + rng.normal(0, 0.1, n_samples), 0.1, 5.0
            )
        else:
            data["st_mass"] = rng.lognormal(np.log(0.9), 0.3, n_samples)

    if "st_rad" in columns:
        if "st_mass" in data:
            data["st_rad"] = np.clip(
                data["st_mass"] ** 0.8 + rng.normal(0, 0.05, n_samples), 0.1, 10.0
            )
        else:
            data["st_rad"] = rng.lognormal(np.log(1.0), 0.3, n_samples)

    if "st_age" in columns:
        data["st_age"] = rng.uniform(0.5, 12.0, n_samples)
    if "st_met" in columns:
        data["st_met"] = rng.normal(0.0, 0.2, n_samples)
    if "st_metratio" in columns:
        data["st_metratio"] = ["[Fe/H]"] * n_samples
    if "st_logg" in columns:
        data["st_logg"] = rng.normal(4.4, 0.3, n_samples)
    if "st_dens" in columns:
        data["st_dens"] = rng.lognormal(np.log(1.4), 0.5, n_samples)
    if "st_vsin" in columns:
        data["st_vsin"] = rng.lognormal(np.log(3.0), 0.8, n_samples)
    if "st_rotp" in columns:
        data["st_rotp"] = rng.lognormal(np.log(20.0), 0.6, n_samples)
    if "st_spectype" in columns:
        data["st_spectype"] = rng.choice(["G2 V", "K1 V", "M3 V", "F8 V"], n_samples)

    # System / astrometry
    if "sy_snum" in columns:
        data["sy_snum"] = rng.choice([1, 2, 3], n_samples, p=[0.85, 0.13, 0.02])
    if "sy_pnum" in columns:
        data["sy_pnum"] = rng.choice([1, 2, 3, 4, 5], n_samples, p=[0.55, 0.2, 0.12, 0.08, 0.05])
    if "sy_mnum" in columns:
        data["sy_mnum"] = np.zeros(n_samples, dtype=int)
    if "sy_dist" in columns:
        data["sy_dist"] = rng.lognormal(np.log(250), 0.9, n_samples)
    if "sy_plx" in columns and "sy_dist" in data:
        data["sy_plx"] = 1000.0 / np.clip(data["sy_dist"], 1e-3, None)
    if "sy_pm" in columns:
        data["sy_pm"] = rng.lognormal(np.log(20), 1.0, n_samples)
    if "sy_vmag" in columns:
        data["sy_vmag"] = rng.normal(13.0, 2.0, n_samples)
    if "sy_kmag" in columns:
        data["sy_kmag"] = rng.normal(11.0, 2.0, n_samples)
    if "sy_gaiamag" in columns:
        data["sy_gaiamag"] = rng.normal(12.5, 2.0, n_samples)
    if "ra" in columns:
        data["ra"] = rng.uniform(0, 360, n_samples)
    if "dec" in columns:
        data["dec"] = rng.uniform(-90, 90, n_samples)

    # Transit / RV geometry
    if "pl_trandep" in columns:
        data["pl_trandep"] = rng.lognormal(np.log(0.3), 1.0, n_samples)
    if "pl_trandur" in columns:
        data["pl_trandur"] = rng.lognormal(np.log(3.0), 0.5, n_samples)
    if "pl_ratdor" in columns:
        data["pl_ratdor"] = rng.lognormal(np.log(20.0), 0.8, n_samples)
    if "pl_ratror" in columns:
        data["pl_ratror"] = rng.lognormal(np.log(0.05), 0.6, n_samples)
    if "pl_imppar" in columns:
        data["pl_imppar"] = rng.uniform(0, 1, n_samples)
    if "pl_rvamp" in columns:
        data["pl_rvamp"] = rng.lognormal(np.log(20.0), 1.2, n_samples)

    # Discovery metadata
    if "discoverymethod" in columns:
        data["discoverymethod"] = rng.choice(
            ["Transit", "Radial Velocity", "Microlensing", "Imaging"],
            n_samples, p=[0.75, 0.19, 0.04, 0.02],
        )
    if "disc_year" in columns:
        data["disc_year"] = rng.integers(1995, 2025, n_samples)
    if "disc_facility" in columns:
        data["disc_facility"] = rng.choice(
            ["Kepler", "Transiting Exoplanet Survey Satellite (TESS)", "K2",
             "La Silla Observatory", "W. M. Keck Observatory"], n_samples,
        )
    if "pl_controv_flag" in columns:
        data["pl_controv_flag"] = np.zeros(n_samples, dtype=int)
    if "ttv_flag" in columns:
        data["ttv_flag"] = rng.choice([0, 1], n_samples, p=[0.93, 0.07])
    if "tran_flag" in columns:
        data["tran_flag"] = rng.choice([0, 1], n_samples, p=[0.25, 0.75])
    if "rv_flag" in columns:
        data["rv_flag"] = rng.choice([0, 1], n_samples, p=[0.7, 0.3])

    df = pd.DataFrame(data)

    # Inject ~10% missing values to mimic real archive sparsity.
    for col in df.select_dtypes(include=[np.number]).columns:
        df.loc[rng.random(n_samples) < 0.10, col] = np.nan

    # Guarantee a fraction of habitable-range planets so downstream
    # labelling always produces both classes regardless of sample size.
    n_habitable = max(int(n_samples * 0.15), 3)
    hab_idx = df.index[:n_habitable]
    habitable_ranges = {
        "pl_rade": (0.8, 1.8),
        "pl_eqt": (200, 290),
        "pl_bmasse": (0.5, 5.0),
        "pl_dens": (3.5, 7.0),
        "pl_orbsmax": (0.7, 1.5),
        "pl_insol": (0.3, 2.0),
    }
    for col, (low, high) in habitable_ranges.items():
        if col in df.columns:
            df.loc[hab_idx, col] = rng.uniform(low, high, n_habitable)

    return df


# ── Summaries ──────────────────────────────────────────────────────────
def get_data_summary(df: pd.DataFrame) -> dict:
    """Generate a summary of the exoplanet dataset for display.

    Args:
        df: Exoplanet DataFrame.

    Returns:
        Dict with summary statistics.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    n = max(len(df), 1)

    summary = {
        "total_planets": len(df),
        "columns": list(df.columns),
        "numeric_columns": list(numeric_df.columns),
        "missing_pct": {
            col: round(100.0 * df[col].isnull().sum() / n, 1) for col in df.columns
        },
        "stats": numeric_df.describe().to_dict(),
    }

    # Rough count of potentially habitable planets (radius + temperature proxy)
    if "pl_rade" in df.columns and "pl_eqt" in df.columns:
        habitable_mask = df["pl_rade"].between(0.5, 2.5) & df["pl_eqt"].between(180, 310)
        summary["potentially_habitable"] = int(habitable_mask.sum())
    else:
        summary["potentially_habitable"] = "N/A (missing radius or temperature data)"

    # Discovery breakdown — useful context now that we carry these columns.
    for col, key in (("discoverymethod", "by_method"), ("disc_facility", "by_facility")):
        if col in df.columns:
            summary[key] = df[col].value_counts().head(10).to_dict()

    if "disc_year" in df.columns:
        years = df["disc_year"].dropna()
        if not years.empty:
            summary["year_range"] = (int(years.min()), int(years.max()))

    if "sy_dist" in df.columns:
        dist = df["sy_dist"].dropna()
        if not dist.empty:
            summary["nearest_pc"] = float(dist.min())
            summary["median_dist_pc"] = float(dist.median())

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch the NASA Exoplanet Archive composite planet table."
    )
    parser.add_argument("--refresh", action="store_true", help="Force re-download.")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap the row count.")
    parser.add_argument("--output", default=None, help="Cache path for the CSV.")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Generate fake development data if the archive is unreachable.",
    )
    args = parser.parse_args()

    try:
        df, meta = fetch_exoplanet_data(
            cache_path=args.output,
            force_refresh=args.refresh,
            max_rows=args.max_rows,
            allow_synthetic=args.allow_synthetic,
            return_metadata=True,
        )
    except ExoplanetArchiveError as exc:
        raise SystemExit(f"\n✖ {exc}\n")

    summary = get_data_summary(df)
    banner = "SYNTHETIC (NOT REAL DATA)" if meta.is_synthetic else f"NASA Archive · {meta.table}"

    print(f"\n{'='*64}")
    print(f"Exoplanet Data Summary — {banner}")
    print(f"{'='*64}")
    print(f"Source        : {meta.source}")
    print(f"Fetched       : {meta.fetched_at_display}")
    print(f"Planets       : {summary['total_planets']}")
    print(f"Columns       : {meta.n_columns}")
    print(f"Habitable-ish : {summary['potentially_habitable']}")
    if "year_range" in summary:
        print(f"Discovery span: {summary['year_range'][0]}–{summary['year_range'][1]}")
    if "nearest_pc" in summary:
        print(f"Nearest system: {summary['nearest_pc']:.2f} pc")

    if "by_method" in summary:
        print("\nTop discovery methods:")
        for method, count in list(summary["by_method"].items())[:5]:
            print(f"  {method:<28s} {count:>5d}")

    print("\nColumns with missing data (>0%):")
    for col, pct in sorted(summary["missing_pct"].items(), key=lambda kv: -kv[1]):
        if pct > 0:
            print(f"  {col:<18s} {pct:>5.1f}%")
    print(f"{'='*64}")
