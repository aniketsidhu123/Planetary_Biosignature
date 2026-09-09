"""
Tabular habitability model — RandomForest / XGBoost classifier.

Trains on NASA Exoplanet Archive parameters to produce a
calibrated habitability probability score, with SHAP-based
feature importance for explainability.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform

import xgboost as xgb

from src.utils.config import get_config
from src.utils.logging_setup import get_logger

logger = get_logger("models.tabular")


@dataclass
class TabularModelResult:
    """Result from tabular model training."""
    model: Any                       # Trained classifier
    scaler: StandardScaler           # Feature scaler
    calibrated_model: Any            # Calibrated version for probabilities
    feature_names: list              # Input feature names
    cv_scores: Dict[str, float]      # Cross-validation metrics
    best_params: Dict[str, Any]      # Best hyperparameters
    feature_importances: Dict[str, float]  # Feature → importance
    training_summary: str            # Human-readable summary


@dataclass
class HabitabilityPrediction:
    """Single planet habitability prediction."""
    planet_name: str
    habitability_score: float        # Calibrated probability (0-1)
    confidence_lower: float          # Lower bound of CI
    confidence_upper: float          # Upper bound of CI
    top_features: list               # [(feature_name, contribution), ...]
    risk_factors: list               # Negative contributors
    category: str                    # "Potentially Habitable", "Unlikely", etc.


class TabularHabitabilityModel:
    """Habitability scoring model using Random Forest or XGBoost.

    Trains on exoplanet physical parameters to predict habitability,
    with calibrated probabilities and SHAP feature attributions.
    """

    def __init__(self, model_type: Optional[str] = None):
        """Initialize the model.

        Args:
            model_type: "random_forest" or "xgboost". Default from config.
        """
        config = get_config()
        tab_config = config.get_section("tabular")

        self.model_type = model_type or tab_config.get("model_type", "xgboost")
        self.config = tab_config
        self.model = None
        self.calibrated_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self._is_trained = False

        logger.info(f"TabularHabitabilityModel initialized (type={self.model_type})")

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create habitability labels from physical parameters.

        Uses a multi-criteria approach based on published habitability
        thresholds. This is a proxy label — real ground truth doesn't
        exist for most exoplanets.

        Criteria for "potentially habitable" (all must be met):
        1. Rocky planet: radius 0.5-2.0 Earth radii
        2. Temperature range: equilibrium temp 180-310 K
        3. Not tidally locked (if estimable)
        4. Moderate stellar activity

        Args:
            df: Engineered DataFrame with ESI and derived features.

        Returns:
            Binary Series (1 = potentially habitable, 0 = unlikely).
        """
        labels = pd.Series(0, index=df.index)

        # Primary criteria
        if "pl_rade" in df.columns and "pl_eqt" in df.columns:
            rocky = df["pl_rade"].between(0.5, 2.5)
            temp_ok = df["pl_eqt"].between(180, 310)
            labels = (rocky & temp_ok).astype(int)
        elif "esi" in df.columns:
            # Fallback: use ESI threshold
            labels = (df["esi"] > 0.6).astype(int)

        # Refine with additional criteria if available
        if "tidal_lock_likelihood" in df.columns:
            # Penalize likely tidally locked planets
            locked = df["tidal_lock_likelihood"] > 0.8
            labels = labels & (~locked).astype(int)

        if "stellar_activity" in df.columns:
            # Penalize planets around very active stars
            active = df["stellar_activity"] > 0.8
            labels = labels & (~active).astype(int)

        n_habitable = labels.sum()
        logger.info(
            f"Labels created: {n_habitable} habitable / "
            f"{len(labels) - n_habitable} non-habitable "
            f"({100.0 * n_habitable / len(labels):.1f}%)"
        )

        return labels

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        label_col: Optional[str] = None,
        labels: Optional[pd.Series] = None,
    ) -> TabularModelResult:
        """Train the habitability model with cross-validation and hyperparameter tuning.

        Args:
            df: Engineered DataFrame.
            feature_cols: List of feature column names to use.
            label_col: Name of label column in df (if labels are in df).
            labels: Explicit label Series (overrides label_col).

        Returns:
            TabularModelResult with trained model and metrics.
        """
        # Prepare features and labels
        X = df[feature_cols].copy()
        if labels is not None:
            y = labels
        elif label_col and label_col in df.columns:
            y = df[label_col]
        else:
            y = self.create_labels(df)

        self.feature_names = list(feature_cols)

        # Handle any remaining NaN
        X = X.fillna(X.median())

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

        logger.info(
            f"Training {self.model_type} on {X_scaled.shape[0]} samples, "
            f"{X_scaled.shape[1]} features"
        )

        # Create base model
        if self.model_type == "random_forest":
            base_model, param_dist = self._create_rf()
        else:
            base_model, param_dist = self._create_xgb(y)

        # Hyperparameter search
        cv = StratifiedKFold(n_splits=min(self.config.get("cv_folds", 5), y.value_counts().min()), shuffle=True, random_state=42)

        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=30,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )

        search.fit(X_scaled, y)

        self.model = search.best_estimator_
        best_params = search.best_params_
        logger.info(f"Best params: {best_params}")

        # Calibrate probabilities
        cal_method = self.config.get("calibration_method", "isotonic")
        self.calibrated_model = CalibratedClassifierCV(
            self.model, method=cal_method, cv=cv
        )
        self.calibrated_model.fit(X_scaled, y)

        # Cross-validation metrics
        cv_metrics = self._compute_cv_metrics(X_scaled, y, cv)

        # Feature importances
        importances = self._get_feature_importances()

        # Save model
        self._save_model()
        self._is_trained = True

        # Build summary
        summary = self._build_training_summary(cv_metrics, importances)

        return TabularModelResult(
            model=self.model,
            scaler=self.scaler,
            calibrated_model=self.calibrated_model,
            feature_names=self.feature_names,
            cv_scores=cv_metrics,
            best_params=best_params,
            feature_importances=importances,
            training_summary=summary,
        )

    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[list] = None,
        n_bootstrap: int = 100,
    ) -> list:
        """Predict habitability scores with confidence intervals.

        Args:
            df: DataFrame with features for prediction.
            feature_cols: Feature columns (uses training features if None).
            n_bootstrap: Number of bootstrap samples for CI estimation.

        Returns:
            List of HabitabilityPrediction objects.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        feature_cols = feature_cols or self.feature_names
        X = df[feature_cols].copy().fillna(df[feature_cols].median())
        X_scaled = self.scaler.transform(X)

        # Get calibrated probabilities
        probas = self.calibrated_model.predict_proba(X_scaled)[:, 1]

        # Bootstrap confidence intervals
        ci_lower, ci_upper = self._bootstrap_ci(X_scaled, n_bootstrap)

        # Feature contributions (via model internals)
        contributions = self._get_contributions(X_scaled)

        predictions = []
        for i in range(len(df)):
            planet_name = df.iloc[i].get("pl_name", f"Planet_{i}")

            # Top positive and negative contributors
            contribs = contributions[i] if i < len(contributions) else {}
            sorted_contribs = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features = [(name, val) for name, val in sorted_contribs if val > 0][:5]
            risk_factors = [(name, val) for name, val in sorted_contribs if val < 0][:5]

            # Categorize
            score = probas[i]
            if score >= 0.7:
                category = "Potentially Habitable"
            elif score >= 0.4:
                category = "Marginally Habitable"
            elif score >= 0.2:
                category = "Unlikely Habitable"
            else:
                category = "Non-Habitable"

            predictions.append(HabitabilityPrediction(
                planet_name=str(planet_name),
                habitability_score=float(score),
                confidence_lower=float(ci_lower[i]),
                confidence_upper=float(ci_upper[i]),
                top_features=top_features,
                risk_factors=risk_factors,
                category=category,
            ))

        return predictions

    def _create_rf(self):
        """Create Random Forest model with hyperparameter distribution."""
        rf_config = self.config.get("random_forest", {})
        model = RandomForestClassifier(
            random_state=42,
            class_weight=rf_config.get("class_weight", "balanced"),
        )
        param_dist = {
            "n_estimators": randint(100, 800),
            "max_depth": randint(4, 20),
            "min_samples_split": randint(2, 10),
            "min_samples_leaf": randint(1, 5),
        }
        return model, param_dist

    def _create_xgb(self, y: pd.Series):
        """Create XGBoost model with hyperparameter distribution."""
        xgb_config = self.config.get("xgboost", {})
        # Compute scale_pos_weight for class imbalance
        n_neg = (y == 0).sum()
        n_pos = max((y == 1).sum(), 1)
        scale_pos = n_neg / n_pos

        model = xgb.XGBClassifier(
            random_state=42,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        param_dist = {
            "n_estimators": randint(100, 800),
            "max_depth": randint(3, 12),
            "learning_rate": uniform(0.01, 0.2),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
        }
        return model, param_dist

    def _compute_cv_metrics(self, X, y, cv) -> Dict[str, float]:
        """Compute cross-validation metrics."""
        metrics = {}
        for metric_name in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
            try:
                scores = cross_val_score(self.model, X, y, cv=cv, scoring=metric_name)
                metrics[metric_name] = float(scores.mean())
                metrics[f"{metric_name}_std"] = float(scores.std())
                logger.info(f"  CV {metric_name}: {scores.mean():.4f} ± {scores.std():.4f}")
            except Exception as e:
                logger.warning(f"Could not compute {metric_name}: {e}")
                metrics[metric_name] = 0.0

        return metrics

    def _get_feature_importances(self) -> Dict[str, float]:
        """Extract feature importances from the trained model."""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            return {}

        imp_dict = {
            name: float(imp)
            for name, imp in zip(self.feature_names, importances)
        }
        # Sort by importance
        return dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))

    def _bootstrap_ci(
        self, X_scaled: np.ndarray, n_bootstrap: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute bootstrap confidence intervals for predictions."""
        config = get_config()
        confidence_level = config.get("fusion.confidence.confidence_level", 0.95)
        alpha = (1.0 - confidence_level) / 2.0

        n_samples = X_scaled.shape[0]

        if hasattr(self.model, "estimators_"):
            # For tree ensembles: use individual tree predictions
            tree_preds = np.array([
                tree.predict_proba(X_scaled)[:, 1]
                if hasattr(tree, "predict_proba")
                else tree.predict(X_scaled).astype(float)
                for tree in (
                    self.model.estimators_[:n_bootstrap]
                    if hasattr(self.model.estimators_[0], "predict_proba")
                    else self.model.estimators_[:n_bootstrap]
                )
            ])
            ci_lower = np.percentile(tree_preds, 100 * alpha, axis=0)
            ci_upper = np.percentile(tree_preds, 100 * (1 - alpha), axis=0)
        else:
            # Fallback: use prediction ± fixed margin
            probas = self.calibrated_model.predict_proba(X_scaled)[:, 1]
            margin = 0.1
            ci_lower = np.clip(probas - margin, 0.0, 1.0)
            ci_upper = np.clip(probas + margin, 0.0, 1.0)

        return ci_lower, ci_upper

    def _get_contributions(self, X_scaled: np.ndarray) -> list:
        """Get per-sample feature contributions."""
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_scaled)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Binary classification: use class 1 values
                sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                sv = shap_values

            contributions = []
            for i in range(sv.shape[0]):
                contrib = {
                    self.feature_names[j]: float(sv[i, j])
                    for j in range(sv.shape[1])
                }
                contributions.append(contrib)
            return contributions

        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}. Using feature importances.")
            # Fallback: return global importances for all samples
            importances = self._get_feature_importances()
            return [importances] * X_scaled.shape[0]

    def _save_model(self):
        """Save trained model to disk."""
        save_path = Path(self.config.get("model_save_path", "models/tabular_habitability.joblib"))
        save_path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model": self.model,
            "calibrated_model": self.calibrated_model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
        }
        joblib.dump(artifact, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, path: Optional[str] = None):
        """Load a previously trained model.

        Args:
            path: Path to saved model. Uses config default if None.
        """
        load_path = Path(path or self.config.get("model_save_path", "models/tabular_habitability.joblib"))
        if not load_path.exists():
            raise FileNotFoundError(f"No saved model at {load_path}")

        artifact = joblib.load(load_path)
        self.model = artifact["model"]
        self.calibrated_model = artifact["calibrated_model"]
        self.scaler = artifact["scaler"]
        self.feature_names = artifact["feature_names"]
        self.model_type = artifact["model_type"]
        self._is_trained = True

        logger.info(f"Model loaded from {load_path}")

    def _build_training_summary(
        self, cv_metrics: Dict[str, float], importances: Dict[str, float]
    ) -> str:
        """Build a human-readable training summary."""
        lines = [
            f"{'='*60}",
            f"Tabular Habitability Model — Training Summary",
            f"{'='*60}",
            f"Model type: {self.model_type}",
            f"Features: {len(self.feature_names)}",
            "",
            "Cross-Validation Metrics:",
        ]

        for metric, value in cv_metrics.items():
            if not metric.endswith("_std"):
                std = cv_metrics.get(f"{metric}_std", 0)
                lines.append(f"  {metric}: {value:.4f} ± {std:.4f}")

        lines.extend(["", "Top 10 Feature Importances:"])
        for i, (feat, imp) in enumerate(list(importances.items())[:10]):
            bar = "█" * int(imp * 50)
            lines.append(f"  {i+1}. {feat:30s} {imp:.4f} {bar}")

        summary = "\n".join(lines)
        logger.info("\n" + summary)
        return summary


if __name__ == "__main__":
    from src.data_ingestion.exoplanet_archive import fetch_exoplanet_data
    from src.preprocessing.feature_engineering import engineer_features, get_feature_columns

    # Fetch and engineer features
    df = fetch_exoplanet_data()
    df_eng = engineer_features(df)

    # Get feature columns
    col_info = get_feature_columns(df_eng)
    feature_cols = col_info["all_features"]

    # Train model
    model = TabularHabitabilityModel()
    result = model.train(df_eng, feature_cols)

    print(result.training_summary)
