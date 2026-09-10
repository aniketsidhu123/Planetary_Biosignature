"""
Feature engineering for tabular habitability scoring.

Turns the raw NASA Exoplanet Archive table into model-ready features.
Everything the archive gives us is used: planetary parameters, stellar
properties, system architecture, transit/RV geometry, and discovery
metadata (which encodes real observational selection effects and is
therefore genuinely informative, not just bookkeeping).

Feature families
----------------
1. **Earth Similarity Index** — Schulze-Makuch et al. (2011).
2. **Habitable zone** — Kopparapu et al. (2014) effective-flux boundaries,
   giving conservative/optimistic HZ edges and a normalised position.
3. **Derived physics** — surface gravity, escape velocity, orbital
   velocity, mass-radius residual, insolation consistency.
4. **Stellar context** — spectral class, metallicity, rotation, activity.
5. **System architecture** — multiplicity, distance, apparent brightness.
6. **Observational metadata** — discovery method/facility/era, detection
   flags, and how the mass was actually measured (a true mass and an
   ``Msini`` lower bound are very different things).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from typing import Optional, Dict, List

from src.utils.config import get_config
from src.utils.logging_setup import get_logger

logger = get_logger("preprocessing.features")


# ── Earth reference values ─────────────────────────────────────────────
EARTH_REF = {
    "radius": 1.0,            # Earth radii
    "density": 5.51,           # g/cm³
    "escape_velocity": 1.0,    # Earth units (11.2 km/s)
    "surface_temp": 255.0,     # K — Earth's EQUILIBRIUM temperature.
                               # The archive gives pl_eqt, so the Earth
                               # reference must also be an equilibrium
                               # temperature (255 K), not the greenhouse-
                               # warmed surface value (288 K).
}

# ESI weight exponents (Schulze-Makuch et al., 2011)
ESI_WEIGHTS = {
    "radius": 0.57,
    "density": 1.07,
    "escape_velocity": 0.70,
    "surface_temp": 5.58,
}

ESI_COLUMN_MAP = {
    "radius": "pl_rade",
    "density": "pl_dens",
    "surface_temp": "pl_eqt",
}

# ── Kopparapu et al. (2014) habitable zone coefficients ────────────────
# Effective stellar flux at each boundary:
#     S_eff = S_eff_sun + a·T + b·T² + c·T³ + d·T⁴,  T = Teff − 5780 K
# Valid for 2600 K ≤ Teff ≤ 7200 K.
HZ_COEFFICIENTS = {
    #  boundary            S_eff_sun,   a,          b,          c,           d
    "recent_venus":      (1.7763, 1.4335e-4, 3.3954e-9, -7.6364e-12, -1.1950e-15),
    "runaway_greenhouse": (1.0385, 1.2456e-4, 1.4612e-8, -7.6345e-12, -1.7511e-15),
    "moist_greenhouse":  (1.0146, 8.1884e-5, 1.9394e-9, -4.3618e-12, -6.8260e-16),
    "maximum_greenhouse": (0.3507, 5.9578e-5, 1.6707e-9, -3.0058e-12, -5.1925e-16),
    "early_mars":        (0.3207, 5.4471e-5, 1.5275e-9, -2.1709e-12, -3.8282e-16),
}

# Conservative HZ = runaway greenhouse (inner) → maximum greenhouse (outer)
# Optimistic HZ  = recent Venus (inner)        → early Mars (outer)
HZ_TEFF_MIN, HZ_TEFF_MAX = 2600.0, 7200.0

# The polynomial is smooth and well-behaved just past its calibrated range,
# so a small extrapolation is allowed rather than discarding the star. This
# matters in practice: TRAPPIST-1 sits at 2566 K, only 34 K below the floor,
# and refusing to solve for it would wrongly report its planets as outside
# the habitable zone. Stars beyond this tolerance get NaN, and every
# extrapolated row is flagged in ``hz_extrapolated``.
HZ_TEFF_TOLERANCE = 300.0

# Main-sequence spectral classes by effective temperature (K).
SPECTRAL_CLASS_BOUNDS = [
    ("O", 30000, np.inf),
    ("B", 10000, 30000),
    ("A", 7500, 10000),
    ("F", 6000, 7500),
    ("G", 5200, 6000),
    ("K", 3700, 5200),
    ("M", 2400, 3700),
    ("L", 0, 2400),
]
SPECTRAL_CLASS_ORDER = {c: i for i, (c, _, _) in enumerate(SPECTRAL_CLASS_BOUNDS)}

# NASA text columns that carry signal and should be encoded numerically.
CATEGORICAL_SOURCES = {
    "discoverymethod": 8,   # keep top-N levels, rest folded into "other"
    "disc_facility": 12,
    "pl_bmassprov": 6,
    "st_metratio": 4,
}

# Columns that are identifiers rather than measurements.
ID_COLUMNS = ["pl_name", "hostname", "pl_letter"]


# ── Earth Similarity Index ─────────────────────────────────────────────
def compute_esi(df: pd.DataFrame) -> pd.Series:
    """Compute the Earth Similarity Index for each planet.

    ESI = ∏ᵢ (1 − |xᵢ − x₀ᵢ| / (xᵢ + x₀ᵢ))^(wᵢ/n)

    where xᵢ is the planet value, x₀ᵢ is Earth's value, wᵢ is the weight
    exponent, and n is the number of parameters available.

    Args:
        df: DataFrame with exoplanet parameters.

    Returns:
        Series of ESI values (0.0 to 1.0, higher = more Earth-like).
    """
    config = get_config()
    esi_config = config.get_section("esi")
    earth_ref = {**EARTH_REF, **esi_config.get("earth_reference", {})}
    weights = {**ESI_WEIGHTS, **esi_config.get("weight_exponents", {})}

    n_params = 0
    esi = pd.Series(np.ones(len(df)), index=df.index, dtype=float)

    for param_name, col_name in ESI_COLUMN_MAP.items():
        if col_name not in df.columns:
            logger.warning(f"Column {col_name} not in DataFrame, skipping ESI component")
            continue

        x = pd.to_numeric(df[col_name], errors="coerce").values.astype(float)
        esi *= np.power(_similarity(x, earth_ref[param_name]), weights[param_name])
        n_params += 1

    # Escape velocity is derived from mass and radius: v_esc ∝ √(M/R).
    if "pl_bmasse" in df.columns and "pl_rade" in df.columns:
        v_esc = _escape_velocity(df)
        esi *= np.power(
            _similarity(v_esc.values, earth_ref["escape_velocity"]),
            weights["escape_velocity"],
        )
        n_params += 1

    if n_params > 0:
        esi = np.power(esi, 1.0 / n_params)

    esi = esi.fillna(0.0)
    logger.info(
        f"ESI computed for {len(esi)} planets. "
        f"Mean: {esi.mean():.4f}, Median: {esi.median():.4f}, Max: {esi.max():.4f}"
    )
    return esi


def _similarity(x: np.ndarray, reference: float) -> np.ndarray:
    """Return the ESI sub-index 1 − |x − x₀| / (x + x₀), clipped to [0, 1]."""
    denominator = np.abs(x) + abs(reference)
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return np.clip(1.0 - np.abs(x - reference) / denominator, 0.0, 1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function, NaN-preserving.

    ``1 / (1 + exp(x))`` overflows for large positive ``x``; the piecewise
    form keeps the exponent negative on both branches.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos, neg = x >= 0, x < 0
    out[pos] = np.exp(-x[pos]) / (1.0 + np.exp(-x[pos]))
    out[neg] = 1.0 / (1.0 + np.exp(x[neg]))
    return np.where(np.isnan(x), np.nan, out)


def _escape_velocity(df: pd.DataFrame) -> pd.Series:
    """Escape velocity in Earth units, from planet mass and radius."""
    mass = pd.to_numeric(df["pl_bmasse"], errors="coerce").values.astype(float)
    radius = pd.to_numeric(df["pl_rade"], errors="coerce").values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_esc = np.sqrt(np.abs(mass) / np.where(radius <= 0, np.nan, radius))
    return pd.Series(v_esc, index=df.index).replace([np.inf, -np.inf], np.nan)


# ── Habitable zone (Kopparapu et al. 2014) ─────────────────────────────
def _seff_boundary(teff: np.ndarray, boundary: str) -> np.ndarray:
    """Effective stellar flux at a named HZ boundary, for given Teff.

    Args:
        teff: Stellar effective temperatures (K).
        boundary: Key into ``HZ_COEFFICIENTS``.

    Returns:
        Effective flux S_eff in Earth units. NaN outside the fit's
        validity range (2600–7200 K).
    """
    s_sun, a, b, c, d = HZ_COEFFICIENTS[boundary]
    # Clamp within the tolerance band so a star just outside the calibrated
    # range is evaluated at the nearest valid temperature instead of being
    # dropped; beyond the band the fit is not trustworthy at all.
    clamped = np.clip(teff, HZ_TEFF_MIN, HZ_TEFF_MAX)
    t = clamped - 5780.0
    seff = s_sun + a * t + b * t**2 + c * t**3 + d * t**4

    usable = (
        (teff >= HZ_TEFF_MIN - HZ_TEFF_TOLERANCE)
        & (teff <= HZ_TEFF_MAX + HZ_TEFF_TOLERANCE)
    )
    return np.where(usable, seff, np.nan)


def _hz_is_extrapolated(teff: np.ndarray) -> np.ndarray:
    """Flag stars whose HZ solution required extrapolating the fit."""
    outside = (teff < HZ_TEFF_MIN) | (teff > HZ_TEFF_MAX)
    within_tolerance = (
        (teff >= HZ_TEFF_MIN - HZ_TEFF_TOLERANCE)
        & (teff <= HZ_TEFF_MAX + HZ_TEFF_TOLERANCE)
    )
    return (outside & within_tolerance).astype(float)


def compute_habitable_zone(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Kopparapu habitable-zone boundaries and the planet's position.

    Produces both the conservative HZ (runaway greenhouse → maximum
    greenhouse) and the optimistic HZ (recent Venus → early Mars), as
    distances in AU and as a normalised position derived from insolation.

    Args:
        df: DataFrame with st_teff, st_lum and (ideally) pl_insol.

    Returns:
        DataFrame of HZ features indexed like ``df``.
    """
    out = pd.DataFrame(index=df.index)

    if "st_teff" not in df.columns:
        logger.warning("Missing st_teff — habitable zone features unavailable")
        return out

    teff = pd.to_numeric(df["st_teff"], errors="coerce").values.astype(float)

    # Luminosity in solar units (archive stores log10 L/L☉).
    if "st_lum" in df.columns:
        log_lum = pd.to_numeric(df["st_lum"], errors="coerce").values.astype(float)
        luminosity = np.power(10.0, log_lum)
    elif "st_rad" in df.columns:
        # Fall back to Stefan-Boltzmann: L ∝ R²·T⁴
        radius = pd.to_numeric(df["st_rad"], errors="coerce").values.astype(float)
        luminosity = (radius**2) * (teff / 5772.0) ** 4
    else:
        luminosity = np.full(len(df), np.nan)

    out["hz_extrapolated"] = _hz_is_extrapolated(teff)

    with np.errstate(divide="ignore", invalid="ignore"):
        for boundary in HZ_COEFFICIENTS:
            seff = _seff_boundary(teff, boundary)
            out[f"hz_seff_{boundary}"] = seff
            # d = √(L / S_eff) in AU
            out[f"hz_dist_{boundary}"] = np.sqrt(luminosity / np.where(seff <= 0, np.nan, seff))

        # Insolation received by the planet. Prefer the archive value, fall
        # back to L/a² when it is missing.
        insol = (
            pd.to_numeric(df["pl_insol"], errors="coerce").values.astype(float)
            if "pl_insol" in df.columns
            else np.full(len(df), np.nan)
        )
        if "pl_orbsmax" in df.columns:
            a = pd.to_numeric(df["pl_orbsmax"], errors="coerce").values.astype(float)
            derived_insol = luminosity / np.where(a <= 0, np.nan, a**2)
            insol = np.where(np.isnan(insol), derived_insol, insol)
        out["insolation_effective"] = insol

        s_inner = out["hz_seff_runaway_greenhouse"].values
        s_outer = out["hz_seff_maximum_greenhouse"].values
        s_inner_opt = out["hz_seff_recent_venus"].values
        s_outer_opt = out["hz_seff_early_mars"].values

        # Normalised position across the conservative HZ in log-flux space:
        # 0 = inner edge, 1 = outer edge, outside [0, 1] = outside the HZ.
        span = np.log10(s_inner) - np.log10(s_outer)
        out["hz_position"] = (np.log10(s_inner) - np.log10(insol)) / np.where(
            np.abs(span) < 1e-12, np.nan, span
        )

        out["in_hz_conservative"] = (
            (insol <= s_inner) & (insol >= s_outer)
        ).astype(float)
        out["in_hz_optimistic"] = (
            (insol <= s_inner_opt) & (insol >= s_outer_opt)
        ).astype(float)

        # Signed distance from the nearest conservative HZ edge, in dex of
        # flux. Negative = inside the HZ, positive = outside.
        too_hot = np.log10(insol) - np.log10(s_inner)
        too_cold = np.log10(s_outer) - np.log10(insol)
        out["hz_edge_distance_dex"] = np.maximum(too_hot, too_cold)

    out = out.replace([np.inf, -np.inf], np.nan)

    n_hz = int(np.nansum(out["in_hz_conservative"].values))
    n_hz_opt = int(np.nansum(out["in_hz_optimistic"].values))
    logger.info(
        f"Habitable zone computed: {n_hz} planets in the conservative HZ, "
        f"{n_hz_opt} in the optimistic HZ"
    )
    return out


def compute_habitable_zone_ratio(df: pd.DataFrame) -> pd.Series:
    """Ratio of orbital distance to the naive habitable-zone centre.

    HZ_ratio = a / √L, where a is the semi-major axis (AU) and L the
    stellar luminosity (solar units). A ratio near 1.0 means the planet
    sits near the centre of its star's habitable zone.

    Retained as a simple, interpretable complement to the full Kopparapu
    treatment in :func:`compute_habitable_zone`.

    Args:
        df: DataFrame with pl_orbsmax and st_lum columns.

    Returns:
        Series of HZ distance ratios. Values near 1.0 = HZ centre.
    """
    if "pl_orbsmax" not in df.columns or "st_lum" not in df.columns:
        logger.warning("Missing columns for HZ ratio computation")
        return pd.Series(np.nan, index=df.index)

    a = pd.to_numeric(df["pl_orbsmax"], errors="coerce").values.astype(float)
    log_lum = pd.to_numeric(df["st_lum"], errors="coerce").values.astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        hz_center = np.sqrt(np.power(10.0, log_lum))
        hz_ratio = a / np.where(hz_center == 0, 1e-10, hz_center)

    result = pd.Series(hz_ratio, index=df.index).replace([np.inf, -np.inf], np.nan)
    logger.info(
        f"HZ ratio computed. Mean: {result.mean():.4f}, "
        f"Planets in HZ (0.75-1.5): {((result >= 0.75) & (result <= 1.5)).sum()}"
    )
    return result


# ── Derived planetary physics ──────────────────────────────────────────
def compute_planetary_physics(df: pd.DataFrame) -> pd.DataFrame:
    """Derive physical quantities not supplied directly by the archive.

    Args:
        df: DataFrame with planetary mass, radius and orbital parameters.

    Returns:
        DataFrame of derived physics features.
    """
    out = pd.DataFrame(index=df.index)
    num = lambda c: pd.to_numeric(df[c], errors="coerce").values.astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        if "pl_bmasse" in df.columns and "pl_rade" in df.columns:
            mass, radius = num("pl_bmasse"), num("pl_rade")
            safe_r = np.where(radius <= 0, np.nan, radius)

            # Surface gravity and escape velocity, both in Earth units.
            out["surface_gravity"] = mass / safe_r**2
            out["escape_velocity"] = np.sqrt(mass / safe_r)

            # Residual against the Chen & Kipping (2017) rocky mass-radius
            # relation: positive = denser/more massive than a rocky planet
            # of this size, negative = puffier (volatile-rich).
            expected_mass = radius**3.7
            out["mass_radius_residual"] = np.log10(
                np.where(mass <= 0, np.nan, mass) / np.where(expected_mass <= 0, np.nan, expected_mass)
            )

            # Rocky-composition likelihood: peaks near Earth's bulk density.
            if "pl_dens" in df.columns:
                dens = num("pl_dens")
                out["rocky_likelihood"] = np.exp(-0.5 * ((dens - 5.51) / 2.0) ** 2)

        if "pl_orbsmax" in df.columns and "st_mass" in df.columns:
            a, m_star = num("pl_orbsmax"), num("st_mass")
            # Orbital velocity in Earth units: v ∝ √(M★/a)
            out["orbital_velocity"] = np.sqrt(
                np.where(m_star <= 0, np.nan, m_star) / np.where(a <= 0, np.nan, a)
            )

        # Eccentricity drives insolation swing across an orbit: the ratio of
        # periastron to apastron flux is ((1+e)/(1−e))².
        if "pl_orbeccen" in df.columns:
            ecc = np.clip(num("pl_orbeccen"), 0.0, 0.99)
            out["insolation_swing"] = ((1.0 + ecc) / (1.0 - ecc)) ** 2

    return out.replace([np.inf, -np.inf], np.nan)


# ── Stellar context ────────────────────────────────────────────────────
def compute_tidal_lock_likelihood(df: pd.DataFrame) -> pd.Series:
    """Estimate tidal-lock likelihood from orbital and stellar parameters.

    Planets close to low-mass stars are more likely tidally locked, which
    reduces surface habitability (extreme day-night contrast).

    Args:
        df: DataFrame with pl_orbsmax and st_mass columns.

    Returns:
        Series of tidal lock likelihood (0.0 to 1.0).
    """
    if "pl_orbsmax" not in df.columns or "st_mass" not in df.columns:
        logger.warning("Missing columns for tidal lock computation")
        return pd.Series(np.nan, index=df.index)

    a = pd.to_numeric(df["pl_orbsmax"], errors="coerce").values.astype(float)
    m_star = pd.to_numeric(df["st_mass"], errors="coerce").values.astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        lock_param = a / np.sqrt(np.where(m_star <= 0, np.nan, m_star))
        tidal_likelihood = _sigmoid(10.0 * (lock_param - 0.2))

    return pd.Series(tidal_likelihood, index=df.index).fillna(0.5)


def compute_stellar_activity_proxy(df: pd.DataFrame) -> pd.Series:
    """Estimate stellar activity from temperature, age and rotation.

    Young, cool, fast-rotating stars flare frequently, which is hostile to
    surface habitability. Rotation period is the strongest single predictor
    of activity, so it is used when the archive supplies it (st_rotp).

    Args:
        df: DataFrame with st_teff and optional st_age / st_rotp columns.

    Returns:
        Series of stellar activity proxy (0.0 = quiet, 1.0 = very active).
    """
    if "st_teff" not in df.columns:
        logger.warning("Missing st_teff for stellar activity computation")
        return pd.Series(np.nan, index=df.index)

    teff = pd.to_numeric(df["st_teff"], errors="coerce").values.astype(float)

    # Cooler stars are more active (M dwarfs: Teff < 3700 K).
    temp_activity = _sigmoid(0.005 * (teff - 4000))
    components = [(temp_activity, 1.0)]

    if "st_age" in df.columns:
        age = pd.to_numeric(df["st_age"], errors="coerce").values.astype(float)
        components.append((_sigmoid(1.0 * (age - 2.0)), 0.7))

    if "st_rotp" in df.columns:
        rotp = pd.to_numeric(df["st_rotp"], errors="coerce").values.astype(float)
        # Fast rotators (P < ~10 d) are magnetically active.
        components.append((_sigmoid(0.3 * (rotp - 12.0)), 1.0))

    # Weighted mean over whichever components are available per row.
    stacked = np.vstack([c for c, _ in components])
    weights = np.array([w for _, w in components])[:, None]
    weighted = np.where(np.isnan(stacked), 0.0, stacked * weights)
    weight_sum = np.where(np.isnan(stacked), 0.0, np.broadcast_to(weights, stacked.shape)).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        activity = weighted.sum(axis=0) / np.where(weight_sum == 0, np.nan, weight_sum)

    return pd.Series(activity, index=df.index)


def derive_spectral_class(df: pd.DataFrame) -> pd.Series:
    """Determine the host star's spectral class (O/B/A/F/G/K/M/L).

    Prefers the archive's ``st_spectype`` string when present (~37% of
    rows) and falls back to binning ``st_teff``, so the feature is
    available for essentially every planet.

    Args:
        df: DataFrame with st_spectype and/or st_teff.

    Returns:
        Series of single-letter spectral classes ("unknown" where neither
        column is usable).
    """
    result = pd.Series("unknown", index=df.index, dtype=object)

    if "st_teff" in df.columns:
        teff = pd.to_numeric(df["st_teff"], errors="coerce")
        for letter, low, high in SPECTRAL_CLASS_BOUNDS:
            result[(teff >= low) & (teff < high)] = letter

    if "st_spectype" in df.columns:
        # Take the leading letter of e.g. "G2 V", "K1IV", "M3.5 Ve".
        declared = (
            df["st_spectype"].astype(str).str.strip().str.upper().str[0]
        )
        valid = declared.isin(SPECTRAL_CLASS_ORDER.keys())
        result[valid] = declared[valid]

    return result


# ── Categorical encoding of NASA text columns ──────────────────────────
def _sanitize_level(value: str) -> str:
    """Make a categorical level safe to use inside a feature name.

    XGBoost rejects feature names containing ``[``, ``]`` or ``<``, which
    appear in real archive values such as the ``[Fe/H]`` metallicity ratio.
    """
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_").lower()
    return cleaned or "unknown"


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the archive's text columns as model-usable numeric features.

    Discovery method and facility encode real observational selection
    effects (TESS finds short-period planets around bright nearby stars;
    microlensing finds distant ones), so they carry information about how
    well-characterised a planet is — worth giving the model explicitly.

    High-cardinality columns are truncated to their most common levels,
    with the tail folded into ``other`` to avoid one-hot explosion.

    Args:
        df: DataFrame with raw archive columns.

    Returns:
        DataFrame of one-hot / ordinal encoded features.
    """
    blocks = []

    for col, top_n in CATEGORICAL_SOURCES.items():
        if col not in df.columns:
            continue
        # Sanitize before counting so levels that differ only in
        # punctuation or case (e.g. "[M/H]" and "[m/H]") merge into one
        # level rather than colliding as duplicate column names later.
        values = (
            df[col].astype(str).str.strip()
            .replace({"nan": "unknown", "": "unknown", "None": "unknown"})
            .map(_sanitize_level)
        )
        top_levels = values.value_counts().head(top_n).index
        folded = values.where(values.isin(top_levels), "other")
        blocks.append(pd.get_dummies(folded, prefix=col, prefix_sep="__", dtype=float))

    # Spectral class: both one-hot and an ordinal "how hot" encoding,
    # since temperature ordering is physically meaningful.
    spectral = derive_spectral_class(df)
    ordinal = spectral.map(SPECTRAL_CLASS_ORDER).astype(float).rename("spectral_class_ordinal")
    blocks.append(ordinal.to_frame())
    blocks.append(
        pd.get_dummies(spectral.map(_sanitize_level), prefix="spectral", prefix_sep="__", dtype=float)
    )

    out = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=df.index)
    out = out.loc[:, ~out.columns.duplicated()]

    logger.info(f"Encoded {len(out.columns)} categorical features from NASA text columns")
    return out


# ── Observational / system context ─────────────────────────────────────
def compute_observational_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features from discovery metadata and system architecture.

    Args:
        df: DataFrame with discovery, system and magnitude columns.

    Returns:
        DataFrame of observational context features.
    """
    out = pd.DataFrame(index=df.index)
    num = lambda c: pd.to_numeric(df[c], errors="coerce")

    if "disc_year" in df.columns:
        year = num("disc_year")
        out["disc_year"] = year
        # Years since discovery — proxy for how much follow-up exists.
        out["years_since_discovery"] = 2026 - year

    for flag in ("pl_controv_flag", "ttv_flag", "tran_flag", "rv_flag"):
        if flag in df.columns:
            out[flag] = num(flag).fillna(0.0)

    # A planet detected by both transit and RV has a true mass and radius,
    # so its bulk density (and hence composition) is actually constrained.
    if "tran_flag" in df.columns and "rv_flag" in df.columns:
        out["multi_method_confirmed"] = (
            (num("tran_flag").fillna(0) > 0) & (num("rv_flag").fillna(0) > 0)
        ).astype(float)

    for col in ("sy_snum", "sy_pnum", "sy_mnum"):
        if col in df.columns:
            out[col] = num(col)

    if "sy_pnum" in df.columns:
        out["is_multiplanet"] = (num("sy_pnum") > 1).astype(float)
    if "sy_snum" in df.columns:
        # Multiple stars destabilise habitable-zone orbits.
        out["is_multistar"] = (num("sy_snum") > 1).astype(float)

    if "sy_dist" in df.columns:
        dist = num("sy_dist")
        out["sy_dist"] = dist
        out["sy_dist_log"] = np.log10(dist.clip(lower=1e-3))
        # Nearby systems are the realistic targets for atmospheric
        # characterisation with JWST-class instruments.
        out["is_nearby"] = (dist < 50).astype(float)

    for col in ("sy_vmag", "sy_kmag", "sy_gaiamag", "sy_plx", "sy_pm", "ra", "dec"):
        if col in df.columns:
            out[col] = num(col)

    # Follow-up feasibility: bright host + deep transit = characterisable.
    if "sy_kmag" in df.columns and "pl_trandep" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["transit_snr_proxy"] = np.log10(
                num("pl_trandep").clip(lower=1e-6)
            ) - 0.2 * num("sy_kmag")

    return out.replace([np.inf, -np.inf], np.nan)


# ── Main pipeline ──────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, impute: bool = False) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Adds every derived feature family to the DataFrame and marks missing
    values with explicit ``<column>_missing`` indicators.

    Args:
        df: Raw exoplanet DataFrame from the archive.
        impute: Replace missing numeric values with the column median.
            Off by default. Imputing here is actively harmful for anything
            that displays the result: a median habitable-zone boundary is
            not a "typical value", it is a *different star's* habitable
            zone, and substituting it silently turns "not measured" into a
            confident wrong answer in the UI. The model does its own
            imputation at fit and predict time, which is where that policy
            belongs.

    Returns:
        DataFrame with original + engineered features.
    """
    logger.info(f"Engineering features for {len(df)} planets across {len(df.columns)} raw columns...")

    result = df.copy()

    # Core habitability indices
    result["esi"] = compute_esi(df)
    result["hz_ratio"] = compute_habitable_zone_ratio(df)
    result["tidal_lock_likelihood"] = compute_tidal_lock_likelihood(df)
    result["stellar_activity"] = compute_stellar_activity_proxy(df)

    # Feature families derived from the wider NASA column set
    for block in (
        compute_habitable_zone(df),
        compute_planetary_physics(df),
        compute_observational_features(df),
        encode_categoricals(df),
    ):
        new_cols = [c for c in block.columns if c not in result.columns]
        result = pd.concat([result, block[new_cols]], axis=1)

    # Spectral class as a readable label for the UI
    result["spectral_class"] = derive_spectral_class(df)

    # Log-transform heavily skewed features
    for col in ["pl_orbsmax", "pl_orbper", "pl_insol", "pl_bmasse", "pl_rade", "pl_dens"]:
        if col in result.columns:
            values = pd.to_numeric(result[col], errors="coerce")
            result[f"{col}_log"] = np.log10(values.clip(lower=1e-10))

    # Radius categories (useful for classification and for the UI)
    if "pl_rade" in result.columns:
        result["size_category"] = pd.cut(
            pd.to_numeric(result["pl_rade"], errors="coerce"),
            bins=[0, 1.0, 1.75, 3.5, 6.0, 15.0, 1e6],
            labels=["sub-Earth", "Earth-size", "super-Earth", "sub-Neptune",
                    "Neptune-size", "Jupiter-size"],
        )

    # Missingness indicators. Built as one block and concatenated once —
    # adding them column by column fragments the frame badly at ~180
    # features. "This was never measured" is real signal in a sparse
    # archive, so the indicators are kept whether or not we impute.
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    incomplete = [c for c in numeric_cols if result[c].isnull().any()]

    if incomplete:
        indicators = pd.DataFrame(
            {f"{col}_missing": result[col].isnull().astype(int) for col in incomplete},
            index=result.index,
        )
        if impute:
            # A column that is entirely NaN medians to NaN, so fall back to
            # 0.0 to guarantee no NaN survives when imputation is requested.
            medians = result[incomplete].median().fillna(0.0)
            result[incomplete] = result[incomplete].fillna(medians)
        result = pd.concat([result, indicators], axis=1).copy()

    engineered_cols = [c for c in result.columns if c not in df.columns]
    logger.info(
        f"Added {len(engineered_cols)} engineered features "
        f"({len(result.columns)} columns total)"
    )
    return result


def get_feature_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> Dict[str, list]:
    """Categorize columns by role for model input.

    Args:
        df: Engineered DataFrame.
        exclude: Feature names to withhold from the model (e.g. columns
            that define the training label).

    Returns:
        Dict with 'numeric', 'categorical', 'missing_indicators', 'id',
        'excluded' and 'all_features' column lists.
    """
    exclude = set(exclude or [])
    id_cols = ID_COLUMNS + ["disc_facility", "discoverymethod", "spectral_class",
                            "st_spectype", "st_metratio", "pl_bmassprov"]

    missing_cols = [c for c in df.columns if c.endswith("_missing")]
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in id_cols and c not in missing_cols and c not in exclude
    ]
    missing_cols = [c for c in missing_cols if c not in exclude]

    return {
        "numeric": numeric_cols,
        "categorical": [c for c in ["size_category"] if c in df.columns],
        "missing_indicators": missing_cols,
        "id": [c for c in id_cols if c in df.columns],
        "excluded": sorted(exclude & set(df.columns)),
        "all_features": numeric_cols + missing_cols,
    }


if __name__ == "__main__":
    from src.data_ingestion.exoplanet_archive import fetch_exoplanet_data

    df = fetch_exoplanet_data()
    df_eng = engineer_features(df)
    col_info = get_feature_columns(df_eng)

    print(f"\n{'='*64}")
    print("Feature Engineering Summary")
    print(f"{'='*64}")
    print(f"Raw NASA columns   : {len(df.columns)}")
    print(f"Engineered columns : {len(df_eng.columns)}")
    print(f"Model features     : {len(col_info['all_features'])}")

    print("\nESI statistics:")
    print(df_eng["esi"].describe().to_string())

    if "in_hz_conservative" in df_eng.columns:
        print(f"\nIn conservative HZ : {int(df_eng['in_hz_conservative'].sum())}")
        print(f"In optimistic HZ   : {int(df_eng['in_hz_optimistic'].sum())}")

    print("\nHost spectral classes:")
    print(df_eng["spectral_class"].value_counts().to_string())

    print("\nTop 10 most Earth-like planets:")
    cols = [c for c in ["pl_name", "esi", "hz_position", "in_hz_conservative",
                        "spectral_class", "sy_dist"] if c in df_eng.columns]
    print(df_eng.nlargest(10, "esi")[cols].to_string(index=False))
