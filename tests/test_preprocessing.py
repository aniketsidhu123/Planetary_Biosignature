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

    def test_engineer_features_no_nulls_in_numeric(self):
        """After engineering, numeric features should have no NaN (imputed)."""
        from src.preprocessing.feature_engineering import engineer_features

        df = self._make_sample_df(30)
        # Inject some NaN
        df.loc[0, "pl_rade"] = np.nan
        df.loc[1, "pl_eqt"] = np.nan

        df_eng = engineer_features(df)

        # All numeric columns should be imputed
        numeric_cols = df_eng.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if not col.endswith("_missing"):
                assert df_eng[col].isnull().sum() == 0, f"Column {col} still has NaN"

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
