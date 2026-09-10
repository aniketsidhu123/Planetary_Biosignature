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
        assert "accuracy" in result.cv_scores
        assert result.cv_scores["accuracy"] > 0.0

        # Label-defining columns are withheld by default, so the trained
        # feature set is the candidate set minus those exclusions.
        assert set(result.feature_names) | set(result.excluded_features) == set(feature_cols)
        assert result.excluded_features, "Expected label-defining features to be withheld"
        assert "pl_rade" not in result.feature_names

    def test_train_keeps_all_features_when_leakage_allowed(self):
        """exclude_label_features=False should train on every candidate column."""
        from src.models.tabular_model import TabularHabitabilityModel
        from src.preprocessing.feature_engineering import get_feature_columns

        df = self._make_engineered_df(150)
        feature_cols = get_feature_columns(df)["all_features"]

        model = TabularHabitabilityModel(model_type="xgboost")
        result = model.train(df, feature_cols, exclude_label_features=False)

        assert result.feature_names == feature_cols
        assert result.excluded_features == []

    def test_holdout_metrics_reported(self):
        """Training should report held-out metrics alongside CV metrics."""
        from src.models.tabular_model import TabularHabitabilityModel
        from src.preprocessing.feature_engineering import get_feature_columns

        df = self._make_engineered_df(400)
        feature_cols = get_feature_columns(df)["all_features"]

        model = TabularHabitabilityModel(model_type="random_forest")
        result = model.train(df, feature_cols)

        assert result.holdout_scores, "Expected a held-out evaluation"
        assert 0.0 <= result.holdout_scores["f1"] <= 1.0
        assert 0.0 <= result.holdout_scores["brier"] <= 1.0
        assert result.holdout_scores["n_test"] > 0
        assert result.label_balance["positive"] > 0

    def test_score_catalog_ranks_planets(self):
        """score_catalog should return every planet, ranked by score."""
        from src.models.tabular_model import TabularHabitabilityModel
        from src.preprocessing.feature_engineering import get_feature_columns

        df = self._make_engineered_df(200)
        feature_cols = get_feature_columns(df)["all_features"]

        model = TabularHabitabilityModel(model_type="random_forest")
        model.train(df, feature_cols)
        ranked = model.score_catalog(df)

        assert len(ranked) == len(df)
        scores = ranked["habitability_score"].values
        assert (scores[:-1] >= scores[1:]).all(), "Results should be sorted descending"
        assert (ranked["ci_lower"] <= ranked["habitability_score"]).all()
        assert (ranked["habitability_score"] <= ranked["ci_upper"]).all()

    def test_predict_rejects_missing_features(self):
        """predict() should fail loudly when required features are absent."""
        import pytest as _pytest
        from src.models.tabular_model import TabularHabitabilityModel
        from src.preprocessing.feature_engineering import get_feature_columns

        df = self._make_engineered_df(150)
        feature_cols = get_feature_columns(df)["all_features"]

        model = TabularHabitabilityModel(model_type="random_forest")
        model.train(df, feature_cols)

        raw = df[["pl_name", "pl_rade"]].head(3)
        with _pytest.raises(ValueError, match="missing"):
            model.predict(raw)

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

    def test_confidence_intervals_stay_on_calibrated_scale(self):
        """Intervals must bracket the score without saturating at 0 or 1.

        Deriving the interval from the uncalibrated base learners mixes
        probability scales: a booster with scale_pos_weight set for a rare
        positive class outputs near 1.0 where the calibrated model says
        0.80, which pins every upper bound to 1.0.
        """
        from src.models.tabular_model import TabularHabitabilityModel
        from src.preprocessing.feature_engineering import get_feature_columns

        df = self._make_engineered_df(400)
        feature_cols = get_feature_columns(df)["all_features"]

        model = TabularHabitabilityModel(model_type="xgboost")
        model.train(df, feature_cols)
        ranked = model.score_catalog(df)

        scores = ranked["habitability_score"]
        lower, upper = ranked["ci_lower"], ranked["ci_upper"]

        assert (lower <= scores).all()
        assert (scores <= upper).all()
        assert ((lower >= 0.0) & (upper <= 1.0)).all()
        # A confident prediction should not carry a bound pinned to 1.0.
        confident = ranked[scores > 0.6]
        if not confident.empty:
            assert (confident["ci_upper"] < 1.0).any(), \
                "Every upper bound saturated at 1.0 — scales are mixed"

    def test_calibration_preserves_ranking(self):
        """Sigmoid calibration must not collapse the catalog into few scores."""
        from src.models.tabular_model import TabularHabitabilityModel
        from src.preprocessing.feature_engineering import get_feature_columns

        df = self._make_engineered_df(400)
        feature_cols = get_feature_columns(df)["all_features"]

        model = TabularHabitabilityModel(model_type="xgboost")
        model.config = {**model.config, "calibration_method": "sigmoid"}
        model.train(df, feature_cols)
        ranked = model.score_catalog(df)

        # Isotonic on a tiny positive class ties most of the top of the list
        # at one value, which makes the ranked target table arbitrary.
        assert ranked["habitability_score"].nunique() > len(df) * 0.5

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
        assert model2.feature_names == model.feature_names
        assert model2.excluded_features == model.excluded_features

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
        """Fusion should respect configured weights when both streams ran."""
        from src.models.fusion import FusionLayer

        fusion = FusionLayer(tabular_weight=0.8, vision_weight=0.2)

        result = fusion.fuse(
            target_name="Test",
            tabular_score=1.0,
            vision_score=0.0,
            # Imagery was analysed and scored zero — distinct from imagery
            # never having been analysed, which renormalises the weights.
            has_tabular=True,
            has_vision=True,
        )

        # With weight 0.8 on tabular (=1.0), score should be ~0.8
        assert result.hbli_score > 0.7
        assert result.hbli_score < 0.9

    def test_single_stream_renormalizes_weights(self):
        """A stream that never ran must not drag the composite down."""
        from src.models.fusion import FusionLayer

        fusion = FusionLayer(tabular_weight=0.55, vision_weight=0.45)

        tabular_only = fusion.fuse(
            target_name="Tabular only",
            tabular_score=0.95,
            tabular_ci=(0.90, 0.99),
            vision_score=0.0,
        )
        # Without renormalisation this would cap at 0.55 * 0.95 = 0.5225.
        assert tabular_only.hbli_score == pytest.approx(0.95, abs=1e-6)
        assert tabular_only.tabular_weight == pytest.approx(1.0)
        assert tabular_only.vision_weight == pytest.approx(0.0)

        vision_only = fusion.fuse(
            target_name="Vision only",
            tabular_score=0.0,
            vision_score=0.80,
            vision_features={"sedimentary_layering": 0.8},
        )
        assert vision_only.hbli_score == pytest.approx(0.80, abs=1e-6)
        assert vision_only.vision_weight == pytest.approx(1.0)

    def test_both_streams_use_configured_weights(self):
        """When both streams run, the configured split is applied."""
        from src.models.fusion import FusionLayer

        fusion = FusionLayer(tabular_weight=0.55, vision_weight=0.45)
        result = fusion.fuse(
            target_name="Both",
            tabular_score=0.80,
            tabular_ci=(0.75, 0.85),
            vision_score=0.40,
            vision_ci=(0.35, 0.45),
        )

        assert result.hbli_score == pytest.approx(0.55 * 0.80 + 0.45 * 0.40, abs=1e-6)
        assert result.tabular_weight == pytest.approx(0.55)
        assert result.vision_weight == pytest.approx(0.45)

    def test_absent_stream_not_listed_as_risk_factor(self):
        """An unanalysed stream is absence of evidence, not a negative factor."""
        from src.models.fusion import FusionLayer

        result = FusionLayer().fuse(
            target_name="Tabular only",
            tabular_score=0.9,
            tabular_ci=(0.85, 0.95),
            vision_score=0.0,
        )
        factor_names = [name for name, _, _ in
                        result.top_contributing_factors + result.risk_factors]
        assert not any("Vision" in name for name in factor_names)

    def test_explicit_stream_flags_override_inference(self):
        """has_vision=True keeps a genuine zero-scoring vision result in play."""
        from src.models.fusion import FusionLayer

        fusion = FusionLayer(tabular_weight=0.55, vision_weight=0.45)
        result = fusion.fuse(
            target_name="Barren surface",
            tabular_score=0.90,
            vision_score=0.0,
            has_tabular=True,
            has_vision=True,
        )
        # Imagery WAS analysed and found nothing, so the zero should count.
        assert result.hbli_score == pytest.approx(0.55 * 0.90, abs=1e-6)
        assert result.vision_weight == pytest.approx(0.45)

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
