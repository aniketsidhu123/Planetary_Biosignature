"""
Tests for ML models — tabular model, vision detector, and fusion.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTabularModel:
    """Tests for the tabular habitability model."""

    def _make_engineered_df(self, n=100):
        """Create a feature-engineered DataFrame for testing."""
        from src.data_ingestion.exoplanet_archive import _generate_synthetic_data
        from src.preprocessing.feature_engineering import engineer_features

        columns = ["pl_name", "pl_rade", "pl_bmasse", "pl_dens", "pl_orbeccen",
                    "pl_orbsmax", "pl_insol", "pl_eqt", "st_teff", "st_lum",
                    "st_mass", "st_age"]
        df = _generate_synthetic_data(columns, n_samples=n)
        return engineer_features(df)

    def test_create_labels(self):
        """Label creation should produce binary labels."""
        from src.models.tabular_model import TabularHabitabilityModel

        model = TabularHabitabilityModel()
        df = self._make_engineered_df(100)
        labels = model.create_labels(df)

        assert len(labels) == len(df)
        assert set(labels.unique()).issubset({0, 1})
        # Should have both classes
        assert labels.sum() > 0, "Should have some habitable planets"
        assert (labels == 0).sum() > 0, "Should have some non-habitable planets"

    def test_train_xgboost(self):
        """XGBoost model should train without errors."""
        from src.models.tabular_model import TabularHabitabilityModel
        from src.preprocessing.feature_engineering import get_feature_columns

        df = self._make_engineered_df(150)
        col_info = get_feature_columns(df)
        feature_cols = col_info["all_features"]

        model = TabularHabitabilityModel(model_type="xgboost")
        result = model.train(df, feature_cols)

        assert result.model is not None
        assert result.calibrated_model is not None
        assert len(result.feature_names) == len(feature_cols)
        assert "accuracy" in result.cv_scores
        assert result.cv_scores["accuracy"] > 0.0

    def test_train_random_forest(self):
        """Random Forest model should train without errors."""
        from src.models.tabular_model import TabularHabitabilityModel
        from src.preprocessing.feature_engineering import get_feature_columns

        df = self._make_engineered_df(150)
        col_info = get_feature_columns(df)
        feature_cols = col_info["all_features"]

        model = TabularHabitabilityModel(model_type="random_forest")
        result = model.train(df, feature_cols)

        assert result.model is not None
        assert len(result.feature_importances) > 0

    def test_predict_returns_predictions(self):
        """predict() should return HabitabilityPrediction objects."""
        from src.models.tabular_model import TabularHabitabilityModel
        from src.preprocessing.feature_engineering import get_feature_columns

        df = self._make_engineered_df(150)
        col_info = get_feature_columns(df)
        feature_cols = col_info["all_features"]

        model = TabularHabitabilityModel(model_type="xgboost")
        model.train(df, feature_cols)

        # Predict on a subset
        test_df = df.head(5)
        predictions = model.predict(test_df, n_bootstrap=20)

        assert len(predictions) == 5
        for pred in predictions:
            assert 0.0 <= pred.habitability_score <= 1.0
            assert pred.confidence_lower <= pred.habitability_score
            assert pred.habitability_score <= pred.confidence_upper
            assert pred.category in [
                "Potentially Habitable", "Marginally Habitable",
                "Unlikely Habitable", "Non-Habitable"
            ]

    def test_model_save_load(self):
        """Model should save and load correctly."""
        from src.models.tabular_model import TabularHabitabilityModel
        from src.preprocessing.feature_engineering import get_feature_columns

        df = self._make_engineered_df(100)
        col_info = get_feature_columns(df)
        feature_cols = col_info["all_features"]

        # Train and save (auto-saves to models/tabular_habitability.joblib)
        model = TabularHabitabilityModel(model_type="xgboost")
        model.train(df, feature_cols)

        save_path = "models/tabular_habitability.joblib"

        # Load into new instance
        model2 = TabularHabitabilityModel()
        model2.load_model(save_path)

        assert model2._is_trained
        assert model2.feature_names == feature_cols

        # Cleanup
        Path(save_path).unlink(missing_ok=True)


class TestVisionDetector:
    """Tests for the YOLO specialist detector stack."""

    def test_initialization(self):
        """Detector stack should initialize with 3 specialist classes."""
        from src.models.vision_detector import VisionDetectorStack

        stack = VisionDetectorStack()
        assert len(stack.specialists) == 3
        assert "sedimentary_layering" in stack.specialists
        assert "mineral_water_interaction" in stack.specialists
        assert "erosion_morphology" in stack.specialists

    def test_simulation_mode(self):
        """Detector should work in simulation mode (no trained models)."""
        from src.models.vision_detector import VisionDetectorStack

        stack = VisionDetectorStack()
        tile = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
        result = stack.detect_tile(tile, "test")

        assert result.feature_vector is not None
        assert len(result.feature_vector) == 3
        assert isinstance(result.overall_score, float)

    def test_detect_batch(self):
        """Batch detection should process multiple tiles."""
        from src.models.vision_detector import VisionDetectorStack
        from src.data_ingestion.tiling import Tile

        stack = VisionDetectorStack()

        tiles = []
        for i in range(5):
            tile = Tile(
                image=np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8),
                row_idx=i, col_idx=0,
                x_offset=0, y_offset=i * 640,
                tile_size=640, source_path="test",
                coverage=1.0,
            )
            tiles.append(tile)

        results = stack.detect_batch(tiles)
        assert len(results) == 5

    def test_class_descriptions(self):
        """Should return descriptions for all classes."""
        from src.models.vision_detector import VisionDetectorStack

        stack = VisionDetectorStack()
        descriptions = stack.get_class_descriptions()

        assert len(descriptions) == 3
        for name, desc in descriptions.items():
            assert len(desc) > 10, f"Description for {name} is too short"

    def test_visualize_detections(self):
        """Detection visualization should produce a valid image."""
        from src.models.vision_detector import VisionDetectorStack

        stack = VisionDetectorStack()
        tile = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
        result = stack.detect_tile(tile, "test")

        vis = stack.visualize_detections(tile, result)
        assert vis.shape == (640, 640, 3)
        assert vis.dtype == np.uint8


class TestFusionLayer:
    """Tests for the HBLI fusion layer."""

    def test_basic_fusion(self):
        """Fusion should produce valid HBLI score."""
        from src.models.fusion import FusionLayer

        fusion = FusionLayer()
        result = fusion.fuse(
            target_name="Test-1b",
            tabular_score=0.7,
            vision_score=0.5,
        )

        assert 0.0 <= result.hbli_score <= 1.0
        assert result.confidence_lower <= result.hbli_score
        assert result.hbli_score <= result.confidence_upper
        assert result.target_name == "Test-1b"
        assert len(result.disclaimers) > 0

    def test_fusion_weights(self):
        """Fusion should respect configured weights."""
        from src.models.fusion import FusionLayer

        fusion = FusionLayer(tabular_weight=0.8, vision_weight=0.2)

        result = fusion.fuse(
            target_name="Test",
            tabular_score=1.0,
            vision_score=0.0,
        )

        # With weight 0.8 on tabular (=1.0), score should be ~0.8
        assert result.hbli_score > 0.7
        assert result.hbli_score < 0.9

    def test_categorization(self):
        """Scores should be categorized correctly."""
        from src.models.fusion import FusionLayer

        fusion = FusionLayer()

        high = fusion.fuse("high", tabular_score=0.9, vision_score=0.9)
        assert "High" in high.category

        low = fusion.fuse("low", tabular_score=0.1, vision_score=0.1)
        assert "Low" in low.category or "Very" in low.category

    def test_confidence_interval_with_ci(self):
        """CI should narrow when component CIs are provided."""
        from src.models.fusion import FusionLayer

        fusion = FusionLayer()

        # With tight CIs
        result_tight = fusion.fuse(
            target_name="tight",
            tabular_score=0.7,
            tabular_ci=(0.68, 0.72),
            vision_score=0.5,
            vision_ci=(0.48, 0.52),
        )

        ci_width = result_tight.confidence_upper - result_tight.confidence_lower
        assert ci_width > 0, "CI width should be positive"
        assert ci_width < 0.5, "CI should not be extremely wide"

    def test_batch_fusion_sorted(self):
        """Batch fusion should return results sorted by score (descending)."""
        from src.models.fusion import FusionLayer

        fusion = FusionLayer()
        targets = [
            {"target_name": "A", "tabular_score": 0.3, "vision_score": 0.2},
            {"target_name": "B", "tabular_score": 0.9, "vision_score": 0.8},
            {"target_name": "C", "tabular_score": 0.5, "vision_score": 0.5},
        ]

        results = fusion.fuse_batch(targets)
        assert results[0].target_name == "B", "Highest scoring should be first"
        scores = [r.hbli_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_report_generation(self):
        """Report should be a non-empty formatted string."""
        from src.models.fusion import FusionLayer

        fusion = FusionLayer()
        result = fusion.fuse("Test", tabular_score=0.65, vision_score=0.4)
        report = fusion.generate_report(result)

        assert len(report) > 100
        assert "HBLI" in report
        assert "Test" in report
        assert "DISCLAIMER" in report.upper()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
