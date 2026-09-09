"""
Feature engineering for tabular habitability scoring.

Computes the Earth Similarity Index (ESI) and derived features
from NASA Exoplanet Archive parameters.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

from src.utils.config import get_config
from src.utils.logging_setup import get_logger

logger = get_logger("preprocessing.features")


# ── Earth Reference Values ─────────────────────────────────────────────
EARTH_REF = {
    "radius": 1.0,           # Earth radii
    "density": 5.51,          # g/cm³
    "escape_velocity": 1.0,   # Earth units (11.2 km/s)
    "surface_temp": 288.0,    # K
}

# ESI weight exponents (Schulze-Makuch et al., 2011)
ESI_WEIGHTS = {
    "radius": 0.57,
    "density": 1.07,
    "escape_velocity": 0.70,
    "surface_temp": 5.58,
}

# Column mapping: archive column → ESI parameter
ESI_COLUMN_MAP = {
    "radius": "pl_rade",
    "density": "pl_dens",
    "surface_temp": "pl_eqt",
}


def compute_esi(df: pd.DataFrame) -> pd.Series:
    """Compute the Earth Similarity Index for each planet.

    ESI = ∏ᵢ (1 - |xᵢ - x₀ᵢ| / (xᵢ + x₀ᵢ))^(wᵢ/n)

    where xᵢ is the planet value, x₀ᵢ is Earth's value,
    wᵢ is the weight exponent, and n is the number of parameters.

    Args:
        df: DataFrame with exoplanet parameters.

    Returns:
        Series of ESI values (0.0 to 1.0, higher = more Earth-like).
    """
    config = get_config()
    esi_config = config.get_section("esi")

    earth_ref = esi_config.get("earth_reference", EARTH_REF)
    weights = esi_config.get("weight_exponents", ESI_WEIGHTS)

    n_params = 0
    esi = pd.Series(np.ones(len(df)), index=df.index)

    for param_name, col_name in ESI_COLUMN_MAP.items():
        if col_name not in df.columns:
            logger.warning(f"Column {col_name} not in DataFrame, skipping ESI component")
            continue

        x = df[col_name].values.astype(float)
        x0 = earth_ref[param_name]
        w = weights[param_name]

        # ESI sub-index: (1 - |x - x0| / (x + x0))^(w/n)
        # Handle edge cases: negative or zero values
        denominator = np.abs(x) + np.abs(x0)
        denominator = np.where(denominator == 0, 1e-10, denominator)
        sub_index = 1.0 - np.abs(x - x0) / denominator
        sub_index = np.clip(sub_index, 0.0, 1.0)

        esi *= np.power(sub_index, w)
        n_params += 1

    # Handle escape velocity (derived from mass and radius)
    if "pl_bmasse" in df.columns and "pl_rade" in df.columns:
        # v_esc ∝ sqrt(M/R) in Earth units
        mass = df["pl_bmasse"].values.astype(float)
        radius = df["pl_rade"].values.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            v_esc = np.sqrt(np.abs(mass) / np.where(radius == 0, 1e-10, radius))
        v_esc = np.nan_to_num(v_esc, nan=0.0)

        x0_vesc = earth_ref["escape_velocity"]
        w_vesc = weights["escape_velocity"]

        denom = v_esc + x0_vesc
        denom = np.where(denom == 0, 1e-10, denom)
        sub_index = 1.0 - np.abs(v_esc - x0_vesc) / denom
        sub_index = np.clip(sub_index, 0.0, 1.0)

        esi *= np.power(sub_index, w_vesc)
        n_params += 1

    # Normalize by number of parameters
    if n_params > 0:
        esi = np.power(esi, 1.0 / n_params)

    # NaN handling
    esi = esi.fillna(0.0)

    logger.info(
        f"ESI computed for {len(esi)} planets. "
        f"Mean: {esi.mean():.4f}, Median: {esi.median():.4f}, "
        f"Max: {esi.max():.4f}"
    )

    return esi


def compute_habitable_zone_ratio(df: pd.DataFrame) -> pd.Series:
    """Compute the ratio of orbital distance to habitable zone center.

    HZ_ratio = a / sqrt(L), where a = semi-major axis (AU),
    L = stellar luminosity (solar). A ratio near 1.0 means
    the planet is near the center of its star's habitable zone.

    Args:
        df: DataFrame with pl_orbsmax and st_lum columns.

    Returns:
        Series of HZ distance ratios. Values near 1.0 = HZ center.
    """
    if "pl_orbsmax" not in df.columns or "st_lum" not in df.columns:
        logger.warning("Missing columns for HZ ratio computation")
        return pd.Series(np.nan, index=df.index)

    a = df["pl_orbsmax"].values.astype(float)
    log_lum = df["st_lum"].values.astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        luminosity = np.power(10.0, log_lum)
        hz_center = np.sqrt(luminosity)
        hz_ratio = a / np.where(hz_center == 0, 1e-10, hz_center)

    result = pd.Series(hz_ratio, index=df.index)
    result = result.replace([np.inf, -np.inf], np.nan)

    logger.info(
        f"HZ ratio computed. Mean: {result.mean():.4f}, "
        f"Planets in HZ (0.75-1.5): {((result >= 0.75) & (result <= 1.5)).sum()}"
    )

    return result


def compute_tidal_lock_likelihood(df: pd.DataFrame) -> pd.Series:
    """Estimate tidal-lock likelihood based on orbital and stellar parameters.

    Planets close to low-mass stars are more likely tidally locked,
    which reduces surface habitability (extreme day-night contrast).

    Uses the tidal locking timescale proxy:
        t_lock ∝ a⁶ * M_planet / M_star²

    Normalized to [0, 1] where 1 = very likely tidally locked.

    Args:
        df: DataFrame with pl_orbsmax and st_mass columns.

    Returns:
        Series of tidal lock likelihood (0.0 to 1.0).
    """
    if "pl_orbsmax" not in df.columns or "st_mass" not in df.columns:
        logger.warning("Missing columns for tidal lock computation")
        return pd.Series(np.nan, index=df.index)

    a = df["pl_orbsmax"].values.astype(float)
    m_star = df["st_mass"].values.astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Proxy: planets with a < 0.1 AU around M-dwarfs are likely locked
        # Simple sigmoid model based on a/M_star ratio
        lock_param = a / np.where(m_star == 0, 1e-10, m_star ** 0.5)
        # Sigmoid: high probability of lock when lock_param < 0.2
        tidal_likelihood = 1.0 / (1.0 + np.exp(10.0 * (lock_param - 0.2)))

    result = pd.Series(tidal_likelihood, index=df.index)
    result = result.fillna(0.5)  # Uncertain default

    return result


def compute_stellar_activity_proxy(df: pd.DataFrame) -> pd.Series:
    """Estimate stellar activity level from temperature and age.

    Younger, cooler stars (M-dwarfs) tend to be more active
    (frequent flares), which is hostile to surface habitability.

    Args:
        df: DataFrame with st_teff and optional st_age columns.

    Returns:
        Series of stellar activity proxy (0.0 = quiet, 1.0 = very active).
    """
    if "st_teff" not in df.columns:
        logger.warning("Missing st_teff for stellar activity computation")
        return pd.Series(np.nan, index=df.index)

    teff = df["st_teff"].values.astype(float)

    # Cooler stars are more active (M-dwarfs: Teff < 3700 K)
    temp_activity = 1.0 / (1.0 + np.exp(0.005 * (teff - 4000)))

    # Younger stars are more active
    if "st_age" in df.columns:
        age = df["st_age"].values.astype(float)
        age_activity = 1.0 / (1.0 + np.exp(1.0 * (age - 2.0)))  # High activity < 2 Gyr
        activity = 0.6 * temp_activity + 0.4 * age_activity
    else:
        activity = temp_activity

    return pd.Series(activity, index=df.index)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Adds all derived features to the DataFrame and handles
    missing data with principled imputation.

    Args:
        df: Raw exoplanet DataFrame from archive.

    Returns:
        DataFrame with original + engineered features.
    """
    logger.info(f"Engineering features for {len(df)} planets...")

    result = df.copy()

    # Compute derived features
    result["esi"] = compute_esi(df)
    result["hz_ratio"] = compute_habitable_zone_ratio(df)
    result["tidal_lock_likelihood"] = compute_tidal_lock_likelihood(df)
    result["stellar_activity"] = compute_stellar_activity_proxy(df)

    # Log-transform skewed features
    for col in ["pl_orbsmax", "pl_orbper", "pl_insol", "pl_bmasse"]:
        if col in result.columns:
            result[f"{col}_log"] = np.log10(
                result[col].clip(lower=1e-10)
            )

    # Radius categories (useful for classification)
    if "pl_rade" in result.columns:
        result["size_category"] = pd.cut(
            result["pl_rade"],
            bins=[0, 1.0, 1.75, 3.5, 6.0, 15.0, 100.0],
            labels=["sub-Earth", "Earth-size", "super-Earth", "sub-Neptune", "Neptune-size", "Jupiter-size"],
        )

    # Missing data imputation for numeric columns
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        n_missing = result[col].isnull().sum()
        if n_missing > 0:
            # Add missingness indicator
            result[f"{col}_missing"] = result[col].isnull().astype(int)
            # Impute with median
            median_val = result[col].median()
            result[col] = result[col].fillna(median_val)

    engineered_cols = [c for c in result.columns if c not in df.columns]
    logger.info(f"Added {len(engineered_cols)} engineered features: {engineered_cols}")

    return result


def get_feature_columns(df: pd.DataFrame) -> Dict[str, list]:
    """Categorize columns by type for model input.

    Args:
        df: Engineered DataFrame.

    Returns:
        Dict with 'numeric', 'categorical', 'target', 'id' column lists.
    """
    id_cols = ["pl_name", "hostname", "disc_facility"]
    categorical_cols = ["size_category"]

    # Missingness indicator columns
    missing_cols = [c for c in df.columns if c.endswith("_missing")]

    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in id_cols and c not in missing_cols
    ]

    return {
        "numeric": numeric_cols,
        "categorical": [c for c in categorical_cols if c in df.columns],
        "missing_indicators": missing_cols,
        "id": [c for c in id_cols if c in df.columns],
        "all_features": numeric_cols + missing_cols,
    }


if __name__ == "__main__":
    from src.data_ingestion.exoplanet_archive import fetch_exoplanet_data

    df = fetch_exoplanet_data()
    df_eng = engineer_features(df)

    print(f"\n{'='*60}")
    print(f"Feature Engineering Summary")
    print(f"{'='*60}")
    print(f"Original features: {len(df.columns)}")
    print(f"Engineered features: {len(df_eng.columns)}")
    print(f"\nESI statistics:")
    print(df_eng["esi"].describe())
    print(f"\nTop 10 most Earth-like planets:")
    top = df_eng.nlargest(10, "esi")[["pl_name", "esi", "hz_ratio", "tidal_lock_likelihood"]]
    print(top.to_string(index=False))
