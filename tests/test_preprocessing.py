"""
Tests for preprocessing modules.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFeatureEngineering:
    """Tests for ESI computation and derived features."""

    def _make_sample_df(self, n=50):
        """Create a sample exoplanet DataFrame for testing."""
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "pl_name": [f"Test-{i}b" for i in range(n)],
            "pl_rade": rng.lognormal(np.log(1.5), 0.5, n),
            "pl_bmasse": rng.lognormal(np.log(3.0), 1.0, n),
            "pl_dens": rng.lognormal(np.log(4.0), 0.5, n),
            "pl_orbeccen": rng.beta(1.12, 5.0, n),
            "pl_orbsmax": rng.lognormal(np.log(1.0), 0.8, n),
            "pl_orbper": rng.lognormal(np.log(100), 1.0, n),
            "pl_insol": rng.lognormal(np.log(1.0), 1.0, n),
            "pl_eqt": rng.normal(350, 150, n).clip(50, 2000),
            "st_teff": rng.normal(5200, 800, n).clip(2500, 10000),
            "st_lum": rng.normal(0, 0.5, n),
            "st_mass": rng.lognormal(np.log(0.9), 0.3, n),
            "st_age": rng.uniform(0.5, 12, n),
        })

    def test_esi_computation(self):
        """ESI should produce values in [0, 1]."""
        from src.preprocessing.feature_engineering import compute_esi

        df = self._make_sample_df()
        esi = compute_esi(df)

        assert len(esi) == len(df)
        assert esi.min() >= 0.0
        assert esi.max() <= 1.0
        assert not esi.isnull().all()

    def test_esi_earth_is_high(self):
        """Earth parameters should produce a high ESI (~1.0)."""
        from src.preprocessing.feature_engineering import compute_esi

        earth = pd.DataFrame([{
            "pl_rade": 1.0,
            "pl_bmasse": 1.0,
            "pl_dens": 5.514,
            "pl_eqt": 255,
        }])
        esi = compute_esi(earth)
        assert esi.iloc[0] > 0.7, f"Earth ESI should be high, got {esi.iloc[0]}"

    def test_esi_jupiter_is_low(self):
        """Jupiter parameters should produce a low ESI."""
        from src.preprocessing.feature_engineering import compute_esi

        jupiter = pd.DataFrame([{
            "pl_rade": 11.21,
            "pl_bmasse": 317.8,
            "pl_dens": 1.326,
            "pl_eqt": 110,
        }])
        esi = compute_esi(jupiter)
        assert esi.iloc[0] < 0.3, f"Jupiter ESI should be low, got {esi.iloc[0]}"

    def test_habitable_zone_ratio(self):
        """HZ ratio should be near 1.0 for Earth."""
        from src.preprocessing.feature_engineering import compute_habitable_zone_ratio

        earth = pd.DataFrame([{
            "pl_orbsmax": 1.0,
            "st_lum": 0.0,  # log(1.0) = 0
        }])
        hz = compute_habitable_zone_ratio(earth)
        assert abs(hz.iloc[0] - 1.0) < 0.1, f"Earth HZ ratio should be ~1.0, got {hz.iloc[0]}"

    def test_tidal_lock_likelihood(self):
        """Close-in planets around M-dwarfs should have high tidal lock likelihood."""
        from src.preprocessing.feature_engineering import compute_tidal_lock_likelihood

        hot_planet = pd.DataFrame([{
            "pl_orbsmax": 0.02,  # Very close
            "st_mass": 0.3,      # M-dwarf
        }])
        tidal = compute_tidal_lock_likelihood(hot_planet)
        assert tidal.iloc[0] > 0.5, "Close-in planet should have high tidal lock likelihood"

        far_planet = pd.DataFrame([{
            "pl_orbsmax": 2.0,   # Far out
            "st_mass": 1.0,      # Sun-like
        }])
        tidal_far = compute_tidal_lock_likelihood(far_planet)
        assert tidal_far.iloc[0] < 0.5, "Distant planet should have low tidal lock likelihood"

    def test_engineer_features_adds_columns(self):
        """engineer_features should add derived columns."""
        from src.preprocessing.feature_engineering import engineer_features

        df = self._make_sample_df(30)
        original_cols = set(df.columns)
        df_eng = engineer_features(df)

        new_cols = set(df_eng.columns) - original_cols
        assert "esi" in new_cols
        assert "hz_ratio" in new_cols
        assert "tidal_lock_likelihood" in new_cols
        assert "stellar_activity" in new_cols
        assert len(new_cols) > 4  # Should also have log-transforms, missingness flags etc.

    def test_engineer_features_no_nulls_when_imputing(self):
        """With impute=True, numeric features should have no NaN."""
        from src.preprocessing.feature_engineering import engineer_features

        df = self._make_sample_df(30)
        # Inject some NaN
        df.loc[0, "pl_rade"] = np.nan
        df.loc[1, "pl_eqt"] = np.nan

        df_eng = engineer_features(df, impute=True)

        # All numeric columns should be imputed
        numeric_cols = df_eng.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if not col.endswith("_missing"):
                assert df_eng[col].isnull().sum() == 0, f"Column {col} still has NaN"

    def test_engineer_features_preserves_missing_by_default(self):
        """Engineering must not invent values for unmeasured quantities.

        A median-imputed habitable zone is not a "typical value" — it is a
        *different star's* habitable zone. Substituting one silently turns
        "not measured" into a confident wrong answer wherever the frame is
        displayed, so missingness has to survive engineering. The model
        imputes at fit and predict time instead.
        """
        from src.preprocessing.feature_engineering import engineer_features

        df = self._make_sample_df(40)
        df.loc[df.index[:10], "pl_rade"] = np.nan

        engineered = engineer_features(df)

        assert engineered["pl_rade"].isnull().sum() == 10
        assert engineered["pl_rade_missing"].sum() == 10

    def test_trappist1_planets_land_in_habitable_zone(self):
        """TRAPPIST-1 e/f/g must resolve inside the conservative HZ.

        TRAPPIST-1 is 2566 K — just below the Kopparapu fit's 2600 K floor.
        Discarding stars outside the fit would report these planets as
        outside the habitable zone, contradicting the published result.
        """
        from src.preprocessing.feature_engineering import compute_habitable_zone

        df = pd.DataFrame({
            "pl_name": ["TRAPPIST-1 b", "TRAPPIST-1 e", "TRAPPIST-1 f",
                        "TRAPPIST-1 g", "TRAPPIST-1 h"],
            "st_teff": [2566.0] * 5,
            "st_lum": [-3.25727] * 5,
            "pl_orbsmax": [0.01154, 0.02925, 0.03849, 0.04683, 0.06189],
            "pl_insol": [4.153, 0.646, 0.373, 0.252, 0.144],
        })
        hz = compute_habitable_zone(df)

        assert hz["in_hz_conservative"].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]
        assert hz["hz_extrapolated"].eq(1.0).all(), "Should be flagged as extrapolated"

        # Published conservative HZ for TRAPPIST-1 is roughly 0.025-0.05 AU.
        assert 0.02 < hz["hz_dist_runaway_greenhouse"].iloc[0] < 0.03
        assert 0.04 < hz["hz_dist_maximum_greenhouse"].iloc[0] < 0.06

    def test_hz_extrapolation_flag(self):
        """Only stars outside the calibrated range are flagged as extrapolated."""
        from src.preprocessing.feature_engineering import compute_habitable_zone

        df = pd.DataFrame({
            "st_teff": [5780.0, 2566.0, 1200.0],
            "st_lum": [0.0, -3.25, 0.0],
            "pl_orbsmax": [1.0, 0.03, 1.0],
            "pl_insol": [1.0, 0.6, 1.0],
        })
        hz = compute_habitable_zone(df)

        assert hz["hz_extrapolated"].iloc[0] == 0.0, "In-range star"
        assert hz["hz_extrapolated"].iloc[1] == 1.0, "Just below the floor"
        assert pd.isna(hz["hz_seff_runaway_greenhouse"].iloc[2]), \
            "Far outside the fit — no HZ solution at all"

    def test_kopparapu_hz_matches_solar_system(self):
        """The Sun's habitable zone should bracket Earth at ~1 AU."""
        from src.preprocessing.feature_engineering import compute_habitable_zone

        # A Sun-like star (Teff 5780 K, log L = 0) with an Earth-like orbit.
        df = pd.DataFrame({
            "st_teff": [5780.0],
            "st_lum": [0.0],
            "pl_orbsmax": [1.0],
            "pl_insol": [1.0],
        })
        hz = compute_habitable_zone(df)

        inner = hz["hz_dist_runaway_greenhouse"].iloc[0]
        outer = hz["hz_dist_maximum_greenhouse"].iloc[0]

        # Kopparapu et al. (2014) give ~0.97-1.70 AU for the conservative
        # solar habitable zone.
        assert 0.9 < inner < 1.05, f"Inner edge {inner} AU outside expected range"
        assert 1.6 < outer < 1.8, f"Outer edge {outer} AU outside expected range"
        assert inner < 1.0 < outer, "Earth should sit inside the conservative HZ"
        assert hz["in_hz_conservative"].iloc[0] == 1.0
        assert abs(hz["hz_position"].iloc[0] - 0.0) < 1.0

    def test_hz_undefined_outside_fit_range(self):
        """HZ boundaries are NaN for stars outside the 2600-7200 K fit."""
        from src.preprocessing.feature_engineering import compute_habitable_zone

        df = pd.DataFrame({
            "st_teff": [1500.0, 12000.0, 5500.0],
            "st_lum": [0.0, 0.0, 0.0],
            "pl_orbsmax": [1.0, 1.0, 1.0],
            "pl_insol": [1.0, 1.0, 1.0],
        })
        hz = compute_habitable_zone(df)
        seff = hz["hz_seff_runaway_greenhouse"]

        assert pd.isna(seff.iloc[0]), "Too-cool star should have no HZ solution"
        assert pd.isna(seff.iloc[1]), "Too-hot star should have no HZ solution"
        assert pd.notna(seff.iloc[2]), "In-range star should have an HZ solution"

    def test_hot_planet_is_outside_hz(self):
        """A planet receiving far more flux than Earth is not in the HZ."""
        from src.preprocessing.feature_engineering import compute_habitable_zone

        df = pd.DataFrame({
            "st_teff": [5780.0, 5780.0],
            "st_lum": [0.0, 0.0],
            "pl_orbsmax": [0.05, 1.0],
            "pl_insol": [400.0, 1.0],
        })
        hz = compute_habitable_zone(df)

        assert hz["in_hz_conservative"].iloc[0] == 0.0
        assert hz["in_hz_conservative"].iloc[1] == 1.0
        # hz_position < 0 means interior to the inner edge (too hot).
        assert hz["hz_position"].iloc[0] < 0

    def test_spectral_class_from_temperature(self):
        """Spectral class should be derived from Teff when no type string exists."""
        from src.preprocessing.feature_engineering import derive_spectral_class

        df = pd.DataFrame({"st_teff": [5780.0, 3200.0, 9500.0, 4500.0, np.nan]})
        classes = derive_spectral_class(df)

        assert classes.iloc[0] == "G"
        assert classes.iloc[1] == "M"
        assert classes.iloc[2] == "A"
        assert classes.iloc[3] == "K"
        assert classes.iloc[4] == "unknown"

    def test_spectral_class_prefers_declared_type(self):
        """A declared st_spectype string should win over the Teff binning."""
        from src.preprocessing.feature_engineering import derive_spectral_class

        df = pd.DataFrame({
            "st_teff": [5780.0, 5780.0],
            "st_spectype": ["K1 V", None],
        })
        classes = derive_spectral_class(df)

        assert classes.iloc[0] == "K", "Declared type should override Teff"
        assert classes.iloc[1] == "G", "Missing type should fall back to Teff"

    def test_categorical_encoding_is_model_safe(self):
        """Encoded NASA text columns must be numeric with XGBoost-safe names."""
        import re
        from src.preprocessing.feature_engineering import encode_categoricals

        df = pd.DataFrame({
            "discoverymethod": ["Transit", "Radial Velocity", "Transit"],
            "disc_facility": ["Kepler", "K2", "Kepler"],
            "st_metratio": ["[Fe/H]", "[M/H]", "[m/H]"],
            "st_teff": [5780.0, 3200.0, 6500.0],
        })
        encoded = encode_categoricals(df)

        assert not encoded.empty
        assert encoded.columns.duplicated().sum() == 0
        for name in encoded.columns:
            assert not re.search(r"[\[\]<>]", name), f"Unsafe feature name: {name}"
        assert all(str(d).startswith(("float", "int")) for d in encoded.dtypes)
        # "[M/H]" and "[m/H]" describe the same quantity and should merge.
        assert sum(c.startswith("st_metratio__") for c in encoded.columns) == 2

    def test_escape_velocity_and_gravity(self):
        """Earth-like inputs should give Earth-unit gravity and escape velocity."""
        from src.preprocessing.feature_engineering import compute_planetary_physics

        df = pd.DataFrame({
            "pl_bmasse": [1.0, 4.0],
            "pl_rade": [1.0, 2.0],
            "pl_dens": [5.51, 5.51],
        })
        physics = compute_planetary_physics(df)

        assert abs(physics["surface_gravity"].iloc[0] - 1.0) < 1e-9
        assert abs(physics["escape_velocity"].iloc[0] - 1.0) < 1e-9
        # 4 M⊕ at 2 R⊕: g = 4/4 = 1, v_esc = √(4/2) = √2
        assert abs(physics["surface_gravity"].iloc[1] - 1.0) < 1e-9
        assert abs(physics["escape_velocity"].iloc[1] - np.sqrt(2)) < 1e-9

    def test_engineered_features_have_no_nan_or_inf(self):
        """No NaN or infinity may reach the model feature matrix."""
        from src.preprocessing.feature_engineering import (
            engineer_features, get_feature_columns,
        )

        df = self._make_sample_df(60)
        # Punch holes in the data the way the real archive does.
        df.loc[df.index[:10], "pl_rade"] = np.nan
        df.loc[df.index[5:15], "st_lum"] = np.nan
        df.loc[df.index[20:25], "pl_orbsmax"] = 0.0

        engineered = engineer_features(df, impute=True)
        features = engineered[get_feature_columns(engineered)["all_features"]]

        assert features.isna().sum().sum() == 0
        assert np.isinf(features.to_numpy(dtype=float)).sum() == 0

    def test_get_feature_columns_honours_exclusions(self):
        """Excluded columns must be absent from the model feature list."""
        from src.preprocessing.feature_engineering import (
            engineer_features, get_feature_columns,
        )

        df_eng = engineer_features(self._make_sample_df(30))
        excluded = ["pl_rade", "esi"]
        col_info = get_feature_columns(df_eng, exclude=excluded)

        for col in excluded:
            assert col not in col_info["all_features"]
        assert set(col_info["excluded"]) == set(excluded)

    def test_get_feature_columns(self):
        """get_feature_columns should properly categorize columns."""
        from src.preprocessing.feature_engineering import engineer_features, get_feature_columns

        df = self._make_sample_df(20)
        df_eng = engineer_features(df)
        col_info = get_feature_columns(df_eng)

        assert "numeric" in col_info
        assert "all_features" in col_info
        assert "id" in col_info
        assert len(col_info["all_features"]) > 0


class TestClassicalCVFilter:
    """Tests for classical CV pre-filter."""

    def test_black_tile_rejected(self):
        """All-black tile should be rejected (no features)."""
        from src.preprocessing.classical_cv_filter import ClassicalCVFilter

        cv_filter = ClassicalCVFilter()
        black_tile = np.zeros((640, 640, 3), dtype=np.uint8)
        result = cv_filter.filter_tile(black_tile)

        assert not result.passed, "Black tile should be rejected"
        assert result.relevance_score < 0.2

    def test_uniform_tile_rejected(self):
        """Uniform gray tile should be rejected (featureless)."""
        from src.preprocessing.classical_cv_filter import ClassicalCVFilter

        cv_filter = ClassicalCVFilter()
        gray_tile = np.full((640, 640, 3), 128, dtype=np.uint8)
        result = cv_filter.filter_tile(gray_tile)

        assert not result.passed, "Uniform tile should be rejected"

    def test_noisy_tile_passes(self):
        """Random noise tile should pass (has texture/edges)."""
        from src.preprocessing.classical_cv_filter import ClassicalCVFilter

        cv_filter = ClassicalCVFilter()
        noise_tile = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = cv_filter.filter_tile(noise_tile)

        assert result.passed, "Noise tile should pass (has features)"
        assert result.relevance_score > 0.3

    def test_edge_density(self):
        """Edge density should be higher for structured images."""
        from src.preprocessing.classical_cv_filter import ClassicalCVFilter

        cv_filter = ClassicalCVFilter()

        # Tile with strong edges (checkerboard)
        checker = np.zeros((640, 640, 3), dtype=np.uint8)
        for i in range(0, 640, 40):
            for j in range(0, 640, 40):
                if (i // 40 + j // 40) % 2 == 0:
                    checker[i:i+40, j:j+40] = 200
        result_checker = cv_filter.filter_tile(checker)

        # Plain tile
        result_plain = cv_filter.filter_tile(np.full((640, 640, 3), 128, dtype=np.uint8))

        assert result_checker.edge_density > result_plain.edge_density

    def test_filter_result_fields(self):
        """FilterResult should have all expected fields."""
        from src.preprocessing.classical_cv_filter import ClassicalCVFilter

        cv_filter = ClassicalCVFilter()
        tile = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = cv_filter.filter_tile(tile)

        assert hasattr(result, "passed")
        assert hasattr(result, "relevance_score")
        assert hasattr(result, "edge_density")
        assert hasattr(result, "texture_entropy")
        assert hasattr(result, "texture_contrast")
        assert hasattr(result, "color_variance")
        assert hasattr(result, "details")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
