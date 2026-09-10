"""
Session state, data loading and shared labels for the ExoScope dashboard.

Keeps the loading/caching contract in one place so the page modules do not
each re-implement it, and so the data's provenance travels with the data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion.exoplanet_archive import (  # noqa: E402
    COLUMN_DESCRIPTIONS,
    ExoplanetArchiveError,
    FetchMetadata,
    fetch_exoplanet_data,
)
from src.preprocessing.feature_engineering import (  # noqa: E402
    engineer_features,
    get_feature_columns,
)
from app.theme import badge  # noqa: E402


# Readable names for engineered features, layered over the archive's own
# column descriptions.
DERIVED_LABELS = {
    "esi": "Earth Similarity Index",
    "hz_ratio": "HZ distance ratio",
    "hz_position": "Position across the HZ",
    "hz_edge_distance_dex": "Distance from HZ edge (dex)",
    "in_hz_conservative": "In conservative HZ",
    "in_hz_optimistic": "In optimistic HZ",
    "insolation_effective": "Effective insolation",
    "tidal_lock_likelihood": "Tidal locking risk",
    "stellar_activity": "Stellar activity",
    "surface_gravity": "Surface gravity (Earth = 1)",
    "escape_velocity": "Escape velocity (Earth = 1)",
    "orbital_velocity": "Orbital velocity (Earth = 1)",
    "mass_radius_residual": "Mass-radius residual",
    "rocky_likelihood": "Rocky composition likelihood",
    "insolation_swing": "Insolation swing over orbit",
    "spectral_class": "Spectral class",
    "spectral_class_ordinal": "Spectral class (ordinal)",
    "size_category": "Size category",
    "multi_method_confirmed": "Confirmed by transit + RV",
    "is_multiplanet": "Multi-planet system",
    "is_multistar": "Multi-star system",
    "is_nearby": "Within 50 pc",
    "years_since_discovery": "Years since discovery",
    "transit_snr_proxy": "Transit follow-up feasibility",
    "sy_dist_log": "Distance (log pc)",
}

for _boundary, _label in (
    ("recent_venus", "Recent Venus"),
    ("runaway_greenhouse", "Runaway greenhouse"),
    ("moist_greenhouse", "Moist greenhouse"),
    ("maximum_greenhouse", "Maximum greenhouse"),
    ("early_mars", "Early Mars"),
):
    DERIVED_LABELS[f"hz_dist_{_boundary}"] = f"HZ edge — {_label} (AU)"
    DERIVED_LABELS[f"hz_seff_{_boundary}"] = f"HZ flux — {_label} (S⊕)"


def readable_name(column: str) -> str:
    """Return a human-readable label for a raw or engineered column."""
    if column in COLUMN_DESCRIPTIONS:
        return COLUMN_DESCRIPTIONS[column]
    if column in DERIVED_LABELS:
        return DERIVED_LABELS[column]

    # Engineered suffixes and one-hot prefixes.
    if column.endswith("_missing"):
        return f"{readable_name(column[:-8])} — not measured"
    if column.endswith("_log"):
        return f"{readable_name(column[:-4])} (log)"
    if "__" in column:
        source, level = column.split("__", 1)
        pretty_level = level.replace("_", " ").title()
        source_label = {
            "discoverymethod": "Discovered by",
            "disc_facility": "Facility",
            "st_metratio": "Metallicity ratio",
            "pl_bmassprov": "Mass from",
            "spectral": "Host class",
        }.get(source, source.replace("_", " ").title())
        return f"{source_label}: {pretty_level}"
    return column.replace("_", " ").capitalize()


READABLE = {c: readable_name(c) for c in COLUMN_DESCRIPTIONS}
READABLE.update(DERIVED_LABELS)


# ── Data loading ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_catalog(force_refresh: bool = False, allow_synthetic: bool = False):
    """Fetch and engineer the exoplanet catalog.

    Cached across reruns so switching pages does not re-download or
    re-derive 180 features for 6,000+ planets.

    Args:
        force_refresh: Bypass the on-disk cache and re-query NASA.
        allow_synthetic: Permit fabricated data if the archive is down.

    Returns:
        (raw_df, engineered_df, metadata).
    """
    raw, metadata = fetch_exoplanet_data(
        force_refresh=force_refresh,
        allow_synthetic=allow_synthetic,
        return_metadata=True,
    )
    engineered = engineer_features(raw)
    return raw, engineered, metadata


def init_state() -> None:
    """Initialize session state keys used across pages."""
    defaults = {
        "raw_df": None,
        "df": None,
        "meta": None,
        "load_error": None,
        "model": None,
        "train_result": None,
        "catalog_scores": None,
        "selected_planet": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def ensure_data(force_refresh: bool = False, allow_synthetic: bool = False) -> bool:
    """Load the catalog into session state if it is not already there.

    Args:
        force_refresh: Re-query NASA rather than using the disk cache.
        allow_synthetic: Permit fabricated data if the archive is down.

    Returns:
        True when data is available in session state.
    """
    if st.session_state.get("df") is not None and not force_refresh:
        return True

    try:
        with st.spinner("Querying the NASA Exoplanet Archive…"):
            raw, engineered, metadata = load_catalog(force_refresh, allow_synthetic)
        st.session_state.update(
            raw_df=raw, df=engineered, meta=metadata, load_error=None
        )
        # A refreshed catalog invalidates anything derived from the old one.
        if force_refresh:
            st.session_state.update(model=None, train_result=None, catalog_scores=None)
        return True

    except ExoplanetArchiveError as exc:
        st.session_state.load_error = str(exc)
        return False


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model feature columns for an engineered DataFrame."""
    return get_feature_columns(df)["all_features"]


# ── Provenance ─────────────────────────────────────────────────────────
def provenance_banner(meta: FetchMetadata | None) -> None:
    """Render a banner stating exactly where the loaded data came from.

    This is deliberately prominent: an earlier version of this pipeline
    silently substituted generated planets when the archive was
    unreachable, and nothing in the UI distinguished them from real
    observations.
    """
    if meta is None:
        st.markdown(
            '<div class="provenance stale"><span class="dot"></span>'
            "<span>No data loaded yet.</span></div>",
            unsafe_allow_html=True,
        )
        return

    if meta.is_synthetic:
        st.markdown(
            '<div class="provenance synthetic"><span class="dot"></span><span>'
            "<b>SYNTHETIC DATA — not real observations.</b> The NASA Exoplanet "
            "Archive was unreachable, so these planets were generated for "
            "development. Every score on this page is meaningless as science. "
            "Use <b>Refresh from NASA</b> once the connection is restored."
            "</span></div>",
            unsafe_allow_html=True,
        )
        return

    if meta.source == "cache" and "unverified" in meta.notes:
        st.markdown(
            f'<div class="provenance stale"><span class="dot"></span><span>'
            f"Loaded <b>{meta.n_rows:,}</b> planets from a local cache written "
            f"before provenance tracking existed — origin unverified. "
            f"Use <b>Refresh from NASA</b> to re-fetch."
            "</span></div>",
            unsafe_allow_html=True,
        )
        return

    source_label = {
        "nasa_tap": "NASA Exoplanet Archive (TAP)",
        "astroquery": "NASA Exoplanet Archive (astroquery)",
        "cache": "local cache of the NASA Exoplanet Archive",
    }.get(meta.source, meta.source)

    st.markdown(
        f'<div class="provenance live"><span class="dot"></span><span>'
        f"<b>{meta.n_rows:,}</b> confirmed planets × <b>{meta.n_columns}</b> columns "
        f"from the {source_label} — table <code>{meta.table}</code>, "
        f"retrieved {meta.fetched_at_display}."
        "</span></div>",
        unsafe_allow_html=True,
    )
