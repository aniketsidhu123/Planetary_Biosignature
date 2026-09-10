"""
Tabular habitability model — RandomForest / XGBoost classifier.

Trains on the full NASA Exoplanet Archive feature set (planetary, stellar,
system-architecture, transit geometry and discovery metadata) to produce a
calibrated habitability probability, with SHAP feature attribution.

A note on labels
----------------
There is no ground truth for exoplanet habitability. The training label is
a *rule* derived from published thresholds (rocky radius + Kopparapu
habitable zone). That has an important consequence: any model given the
columns the rule is built from will simply re-derive the rule and score
near-perfectly, which says nothing.

So by default the columns that define the label are withheld from the
feature set (``exclude_label_features=True``). The model must then predict
the habitability rule from the *rest* of the archive — stellar type,
metallicity, system architecture, transit geometry, discovery context — and
the reported metrics measure something real: how well the broader NASA
parameter set reconstructs the habitability criteria, including for planets
whose defining measurements are missing.

Even so, metrics here describe agreement with a proxy rule, not with
biology. See ``docs/parameter_rationale.md``.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field

from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform

from src.utils.config import get_config
from src.utils.logging_setup import get_logger

logger = get_logger("models.tabular")


# XGBoost is imported lazily. On macOS it links against libomp, and a
# missing OpenMP runtime raises at import time — which would otherwise take
# the whole package down even for callers that only want Random Forest.
_XGB_MODULE = None
_XGB_ERROR: Optional[str] = None


def _load_xgboost():
    """Import xgboost on demand.

    Returns:
        The xgboost module, or None if it cannot be loaded.
    """
    global _XGB_MODULE, _XGB_ERROR
    if _XGB_MODULE is None and _XGB_ERROR is None:
        try:
            import xgboost as xgb
            _XGB_MODULE = xgb
        except Exception as exc:  # ImportError, XGBoostError (missing libomp), ...
            _XGB_ERROR = str(exc)
            logger.warning(
                f"XGBoost unavailable ({exc.__class__.__name__}); "
                "falling back to Random Forest. On macOS: brew install libomp"
            )
    return _XGB_MODULE


def xgboost_available() -> bool:
    """Whether XGBoost can be used in this environment."""
    return _load_xgboost() is not None


# Habitability rule thresholds (see docs/parameter_rationale.md).
ROCKY_RADIUS_RANGE = (0.5, 1.8)      # Earth radii — above ~1.8 R⊕ planets
                                      # retain H/He envelopes (Fulton gap)
ROCKY_RADIUS_RANGE_WIDE = (0.5, 2.5)  # fallback when HZ data is unavailable
EQT_RANGE = (180.0, 310.0)            # K, equilibrium temperature fallback

# Columns that define the label, plus their algebraic restatements. These
# are withheld from the feature set by default to avoid trivial leakage.
LABEL_DEFINING_FEATURES = [
    "pl_rade", "pl_rade_log", "pl_rade_missing",
    "pl_eqt", "pl_eqt_missing",
    "pl_insol", "pl_insol_log", "pl_insol_missing",
    "insolation_effective",
    "hz_position", "hz_edge_distance_dex",
    "in_hz_conservative", "in_hz_optimistic",
    "hz_ratio",  # a/√L is algebraically 1/√insolation — the same criterion
    "esi",       # ESI is itself a function of radius and temperature
]

# Residual leakage is unavoidable here and worth stating plainly: the label
# is a deterministic function of quantities the archive measures, so a
# model given semi-major axis and stellar luminosity can always rebuild the
# insolation criterion. Withholding the direct restatements above stops the
# model reading the answer off a single column; it cannot make the task
# leakage-free without deleting the physics that makes the tool useful.


@dataclass
class TabularModelResult:
    """Result from tabular model training."""
    model: Any                              # Trained classifier
    scaler: StandardScaler                  # Feature scaler
    calibrated_model: Any                   # Calibrated version for probabilities
    feature_names: list                     # Input feature names
    cv_scores: Dict[str, float]             # Cross-validation metrics
    best_params: Dict[str, Any]             # Best hyperparameters
    feature_importances: Dict[str, float]   # Feature → importance
    training_summary: str                   # Human-readable summary
    holdout_scores: Dict[str, float] = field(default_factory=dict)
    excluded_features: List[str] = field(default_factory=list)
    label_balance: Dict[str, int] = field(default_factory=dict)
    n_features: int = 0
    n_samples: int = 0


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

    Trains on the engineered NASA Exoplanet Archive feature set to predict
    habitability, with calibrated probabilities and SHAP attributions.
    """

    def __init__(self, model_type: Optional[str] = None):
        """Initialize the model.

        Args:
            model_type: "random_forest" or "xgboost". Default from config.
        """
        config = get_config()
        tab_config = config.get_section("tabular")

        requested = model_type or tab_config.get("model_type", "xgboost")
        if requested == "xgboost" and not xgboost_available():
            logger.warning("Requested XGBoost but it is unavailable — using Random Forest")
            requested = "random_forest"
        self.model_type = requested
        self.config = tab_config
        self.model = None
        self.calibrated_model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.excluded_features: List[str] = []
        self._is_trained = False
        self._explainer = None

        logger.info(f"TabularHabitabilityModel initialized (type={self.model_type})")

    # ── Labelling ──────────────────────────────────────────────────────
    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create habitability labels from physical parameters.

        Uses a multi-criteria rule based on published thresholds. This is a
        proxy label — real ground truth does not exist for exoplanets.

        Criteria for "potentially habitable":
        1. Rocky size: radius within 0.5–1.8 R⊕ (below the Fulton radius
           gap, where planets retain thick H/He envelopes).
        2. Inside the Kopparapu optimistic habitable zone (recent Venus →
           early Mars insolation bounds), falling back to an equilibrium
           temperature window when HZ data is unavailable.
        3. Not overwhelmingly likely to be tidally locked.
        4. Host star not extremely active.

        Args:
            df: Engineered DataFrame with HZ and derived features.

        Returns:
            Binary Series (1 = potentially habitable, 0 = unlikely).
        """
        label_config = self.config.get("labeling", {})
        rule = label_config.get("rule", "optimistic")
        hz_column = "in_hz_conservative" if rule == "conservative" else "in_hz_optimistic"
        radius_range = tuple(label_config.get(
            f"radius_range_{rule}",
            ROCKY_RADIUS_RANGE if rule == "conservative" else ROCKY_RADIUS_RANGE_WIDE,
        ))
        eqt_range = tuple(label_config.get("eqt_range", EQT_RANGE))

        # ── Criterion 1: rocky size ────────────────────────────────────
        if "pl_rade" in df.columns:
            radius = pd.to_numeric(df["pl_rade"], errors="coerce")
            rocky = radius.between(*radius_range).fillna(False)
        elif "esi" in df.columns:
            logger.warning("No radius column — falling back to an ESI threshold label")
            return (pd.to_numeric(df["esi"], errors="coerce") > 0.6).fillna(False).astype(int)
        else:
            raise ValueError("Cannot build labels: no pl_rade or esi column present")

        # ── Criterion 2: energy budget ─────────────────────────────────
        if hz_column in df.columns:
            in_hz = df[hz_column].fillna(0).astype(bool)
            # Planets with no usable HZ solution (host outside the 2600–7200 K
            # Kopparapu fit) fall back to the equilibrium-temperature window.
            if "pl_eqt" in df.columns:
                no_hz_solution = df.get(
                    "hz_seff_runaway_greenhouse", pd.Series(np.nan, index=df.index)
                ).isna()
                eqt_ok = pd.to_numeric(df["pl_eqt"], errors="coerce").between(*eqt_range)
                in_hz = in_hz | (no_hz_solution & eqt_ok.fillna(False))
            energy_ok = in_hz
        elif "pl_eqt" in df.columns:
            energy_ok = pd.to_numeric(df["pl_eqt"], errors="coerce").between(*eqt_range).fillna(False)
        else:
            energy_ok = pd.Series(True, index=df.index)

        labels = rocky & energy_ok

        # ── Criteria 3 & 4: dynamical and stellar environment ──────────
        if "tidal_lock_likelihood" in df.columns:
            labels &= ~(pd.to_numeric(df["tidal_lock_likelihood"], errors="coerce") > 0.9).fillna(False)
        if "stellar_activity" in df.columns:
            labels &= ~(pd.to_numeric(df["stellar_activity"], errors="coerce") > 0.9).fillna(False)
        if label_config.get("exclude_controversial", True) and "pl_controv_flag" in df.columns:
            # Contested detections should not be treated as positive examples.
            labels &= ~(pd.to_numeric(df["pl_controv_flag"], errors="coerce") > 0).fillna(False)

        labels = labels.astype(int)
        n_habitable = int(labels.sum())
        logger.info(
            f"Labels created with the '{rule}' rule "
            f"(radius {radius_range[0]}-{radius_range[1]} R⊕ × {hz_column}): "
            f"{n_habitable} habitable / {len(labels) - n_habitable} non-habitable "
            f"({100.0 * n_habitable / max(len(labels), 1):.1f}%)"
        )
        return labels

    # ── Training ───────────────────────────────────────────────────────
    def train(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        label_col: Optional[str] = None,
        labels: Optional[pd.Series] = None,
        exclude_label_features: bool = True,
        holdout_size: float = 0.25,
        n_iter: Optional[int] = None,
    ) -> TabularModelResult:
        """Train the habitability model with CV and hyperparameter tuning.

        Args:
            df: Engineered DataFrame.
            feature_cols: Candidate feature column names.
            label_col: Name of a label column in df (if labels live in df).
            labels: Explicit label Series (overrides label_col).
            exclude_label_features: Withhold the columns that define the
                label, so the model must learn from the rest of the NASA
                parameter set instead of re-deriving the rule.
            holdout_size: Fraction reserved for held-out evaluation.
            n_iter: Hyperparameter search iterations. Defaults to config.

        Returns:
            TabularModelResult with the trained model and metrics.
        """
        if labels is not None:
            y = labels
        elif label_col and label_col in df.columns:
            y = df[label_col]
        else:
            y = self.create_labels(df)

        # Drop label-defining columns so the task is not trivially leaky.
        self.excluded_features = []
        if exclude_label_features:
            # Withhold each definitional column *and* its missingness
            # indicator — "insolation was not measured" is itself a hint
            # about whether the insolation criterion can be satisfied.
            withheld = set(LABEL_DEFINING_FEATURES) | {
                f"{c}_missing" for c in LABEL_DEFINING_FEATURES
            }
            self.excluded_features = [c for c in feature_cols if c in withheld]
            feature_cols = [c for c in feature_cols if c not in withheld]
            logger.info(
                f"Withheld {len(self.excluded_features)} label-defining features; "
                f"training on {len(feature_cols)} NASA-derived features"
            )

        if not feature_cols:
            raise ValueError("No feature columns left after exclusions")

        class_counts = y.value_counts()
        if len(class_counts) < 2:
            raise ValueError(
                f"Labels contain a single class ({class_counts.to_dict()}); "
                "cannot train a classifier."
            )

        X = df[feature_cols].copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median()).fillna(0.0)
        self.feature_names = list(feature_cols)

        # ── Held-out split for honest evaluation ───────────────────────
        min_class = int(class_counts.min())
        can_holdout = holdout_size > 0 and min_class >= 8
        if can_holdout:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=holdout_size, stratify=y, random_state=42
            )
        else:
            logger.warning(
                f"Minority class has only {min_class} samples — skipping the held-out split"
            )
            X_train, y_train = X, y
            X_test = y_test = None

        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
        )

        logger.info(
            f"Training {self.model_type} on {X_train_scaled.shape[0]} samples × "
            f"{X_train_scaled.shape[1]} features "
            f"({int(y_train.sum())} positive / {int((y_train == 0).sum())} negative)"
        )

        if self.model_type == "random_forest":
            base_model, param_dist = self._create_rf()
        else:
            base_model, param_dist = self._create_xgb(y_train)

        n_folds = int(min(self.config.get("cv_folds", 5), y_train.value_counts().min()))
        n_folds = max(n_folds, 2)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=int(n_iter or self.config.get("search_iterations", 15)),
            cv=cv,
            scoring="average_precision",  # robust under class imbalance
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        search.fit(X_train_scaled, y_train)

        self.model = search.best_estimator_
        best_params = search.best_params_
        logger.info(f"Best params: {best_params}")

        # ── Probability calibration ────────────────────────────────────
        cal_method = self.config.get("calibration_method", "isotonic")
        self.calibrated_model = CalibratedClassifierCV(
            self.model, method=cal_method, cv=cv
        )
        self.calibrated_model.fit(X_train_scaled, y_train)

        cv_metrics = self._compute_cv_metrics(X_train_scaled, y_train, cv)

        holdout_metrics: Dict[str, float] = {}
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            holdout_metrics = self._compute_holdout_metrics(X_test_scaled, y_test)

        importances = self._get_feature_importances()
        self._explainer = None  # invalidate any cached SHAP explainer
        self._is_trained = True
        self._save_model()

        summary = self._build_training_summary(
            cv_metrics, holdout_metrics, importances, len(feature_cols), len(X_train)
        )

        return TabularModelResult(
            model=self.model,
            scaler=self.scaler,
            calibrated_model=self.calibrated_model,
            feature_names=self.feature_names,
            cv_scores=cv_metrics,
            best_params=best_params,
            feature_importances=importances,
            training_summary=summary,
            holdout_scores=holdout_metrics,
            excluded_features=self.excluded_features,
            label_balance={
                "positive": int(y.sum()),
                "negative": int((y == 0).sum()),
            },
            n_features=len(feature_cols),
            n_samples=len(X),
        )

    # ── Inference ──────────────────────────────────────────────────────
    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[list] = None,
        n_bootstrap: int = 100,
        explain: bool = True,
    ) -> list:
        """Predict habitability scores with confidence intervals.

        Args:
            df: DataFrame with features for prediction.
            feature_cols: Feature columns (uses training features if None).
            n_bootstrap: Bootstrap/ensemble samples for CI estimation.
            explain: Compute SHAP attributions. Disable for bulk scoring
                of the whole catalog, where it dominates runtime.

        Returns:
            List of HabitabilityPrediction objects.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() or load_model() first.")

        feature_cols = feature_cols or self.feature_names
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise ValueError(
                f"Input is missing {len(missing)} feature(s) the model was trained on, "
                f"e.g. {missing[:5]}. Re-run engineer_features() on this data."
            )

        X = df[self.feature_names].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median()).fillna(0.0)
        X_scaled = self.scaler.transform(X)

        probas = self.calibrated_model.predict_proba(X_scaled)[:, 1]
        ci_lower, ci_upper = self._bootstrap_ci(X_scaled, probas, n_bootstrap)
        contributions = self._get_contributions(X_scaled) if explain else [{}] * len(X_scaled)

        predictions = []
        for i in range(len(df)):
            row = df.iloc[i]
            planet_name = row["pl_name"] if "pl_name" in df.columns else f"Planet_{i}"

            contribs = contributions[i] if i < len(contributions) else {}
            sorted_contribs = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)
            top_features = [(n, v) for n, v in sorted_contribs if v > 0][:5]
            risk_factors = [(n, v) for n, v in sorted_contribs if v < 0][:5]

            predictions.append(HabitabilityPrediction(
                planet_name=str(planet_name),
                habitability_score=float(probas[i]),
                confidence_lower=float(ci_lower[i]),
                confidence_upper=float(ci_upper[i]),
                top_features=top_features,
                risk_factors=risk_factors,
                category=self._categorize(float(probas[i])),
            ))

        return predictions

    def score_catalog(
        self, df: pd.DataFrame, batch_size: int = 2000
    ) -> pd.DataFrame:
        """Score an entire catalog and return a ranked table.

        Skips SHAP attribution, which makes whole-catalog scoring fast
        enough to run interactively.

        Args:
            df: Engineered DataFrame for every planet to score.
            batch_size: Rows scored per batch.

        Returns:
            DataFrame with planet identifiers, score and CI, sorted by
            score descending.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() or load_model() first.")

        frames = []
        for start in range(0, len(df), batch_size):
            chunk = df.iloc[start:start + batch_size]
            X = chunk[self.feature_names].apply(pd.to_numeric, errors="coerce")
            X = X.fillna(X.median()).fillna(0.0)
            X_scaled = self.scaler.transform(X)

            probas = self.calibrated_model.predict_proba(X_scaled)[:, 1]
            lower, upper = self._bootstrap_ci(X_scaled, probas, n_bootstrap=100)

            out = pd.DataFrame(index=chunk.index)
            for col in ("pl_name", "hostname", "spectral_class", "sy_dist",
                        "pl_rade", "pl_eqt", "esi", "in_hz_conservative",
                        "in_hz_optimistic", "discoverymethod", "disc_year"):
                if col in chunk.columns:
                    out[col] = chunk[col]
            out["habitability_score"] = probas
            out["ci_lower"] = lower
            out["ci_upper"] = upper
            frames.append(out)

        result = pd.concat(frames)
        result["category"] = result["habitability_score"].map(self._categorize)

        # Break score ties on Earth similarity so the ranking is stable and
        # meaningful rather than falling back on row order.
        sort_keys = ["habitability_score"]
        if "esi" in result.columns:
            sort_keys.append("esi")
        return result.sort_values(sort_keys, ascending=False)

    # ── Model construction ─────────────────────────────────────────────
    def _create_rf(self):
        """Create a Random Forest model with its hyperparameter distribution."""
        rf_config = self.config.get("random_forest", {})
        model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight=rf_config.get("class_weight", "balanced"),
        )
        param_dist = {
            "n_estimators": randint(200, 700),
            "max_depth": randint(4, 20),
            "min_samples_split": randint(2, 10),
            "min_samples_leaf": randint(1, 5),
            "max_features": ["sqrt", "log2", 0.3],
        }
        return model, param_dist

    def _create_xgb(self, y: pd.Series):
        """Create an XGBoost model with its hyperparameter distribution."""
        n_neg = int((y == 0).sum())
        n_pos = max(int((y == 1).sum()), 1)

        xgb = _load_xgboost()
        if xgb is None:
            raise RuntimeError(
                f"XGBoost is not usable in this environment ({_XGB_ERROR}). "
                "Install the OpenMP runtime (macOS: brew install libomp) or "
                "use model_type='random_forest'."
            )

        model = xgb.XGBClassifier(
            random_state=42,
            scale_pos_weight=n_neg / n_pos,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )
        param_dist = {
            "n_estimators": randint(150, 600),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.02, 0.2),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.5, 0.5),
            "min_child_weight": randint(1, 8),
            "reg_lambda": uniform(0.5, 3.0),
        }
        return model, param_dist

    # ── Metrics ────────────────────────────────────────────────────────
    def _compute_cv_metrics(self, X, y, cv) -> Dict[str, float]:
        """Compute cross-validation metrics on the training split."""
        metrics: Dict[str, float] = {}
        for metric_name in ["accuracy", "f1", "precision", "recall", "roc_auc", "average_precision"]:
            try:
                scores = cross_val_score(
                    self.model, X, y, cv=cv, scoring=metric_name, n_jobs=-1
                )
                metrics[metric_name] = float(scores.mean())
                metrics[f"{metric_name}_std"] = float(scores.std())
                logger.info(f"  CV {metric_name}: {scores.mean():.4f} ± {scores.std():.4f}")
            except Exception as exc:
                logger.warning(f"Could not compute {metric_name}: {exc}")
                metrics[metric_name] = 0.0
        return metrics

    def _compute_holdout_metrics(self, X_test, y_test) -> Dict[str, float]:
        """Evaluate the calibrated model on the held-out split."""
        probas = self.calibrated_model.predict_proba(X_test)[:, 1]
        preds = (probas >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "brier": float(brier_score_loss(y_test, probas)),
            "n_test": int(len(y_test)),
            "n_positive": int(y_test.sum()),
        }
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, probas))
            metrics["average_precision"] = float(average_precision_score(y_test, probas))
        except ValueError as exc:
            logger.warning(f"Could not compute ranking metrics on holdout: {exc}")

        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

        logger.info(
            f"  Holdout: F1={metrics['f1']:.4f} ROC-AUC={metrics.get('roc_auc', float('nan')):.4f} "
            f"Brier={metrics['brier']:.4f} (n={metrics['n_test']})"
        )
        return metrics

    def _get_feature_importances(self) -> Dict[str, float]:
        """Extract feature importances from the trained model."""
        if not hasattr(self.model, "feature_importances_"):
            return {}
        imp_dict = {
            name: float(imp)
            for name, imp in zip(self.feature_names, self.model.feature_importances_)
        }
        return dict(sorted(imp_dict.items(), key=lambda kv: kv[1], reverse=True))

    # ── Uncertainty ────────────────────────────────────────────────────
    def _bootstrap_ci(
        self, X_scaled: np.ndarray, probas: np.ndarray, n_bootstrap: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute prediction intervals from ensemble member disagreement.

        The members are the per-fold classifiers inside
        ``CalibratedClassifierCV``. This matters: their predictions are on
        the same *calibrated* probability scale as the point estimate.
        Using the raw base learners instead mixes scales — an uncalibrated
        booster with ``scale_pos_weight`` set for a rare positive class
        outputs near 1.0 where the calibrated model says 0.80, which pins
        every upper bound to 1.0 and makes the interval meaningless.

        The spread across folds captures both model variance and
        calibration variance, which is what a prediction interval on a
        calibrated score should reflect.

        Args:
            X_scaled: Scaled feature matrix.
            probas: Calibrated probabilities for the same rows.
            n_bootstrap: Maximum ensemble members to sample.

        Returns:
            (ci_lower, ci_upper) arrays.
        """
        config = get_config()
        confidence_level = config.get("fusion.confidence.confidence_level", 0.95)
        alpha = (1.0 - confidence_level) / 2.0
        z = 1.959964  # two-sided normal quantile at 95%

        member_preds = self._calibrated_member_predictions(X_scaled, n_bootstrap)

        if member_preds is None or member_preds.shape[0] < 2:
            # Last resort: a fixed margin, widened near the decision boundary
            # where the model is least certain.
            margin = 0.05 + 0.15 * (1.0 - np.abs(probas - 0.5) * 2.0)
            return np.clip(probas - margin, 0.0, 1.0), np.clip(probas + margin, 0.0, 1.0)

        if member_preds.shape[0] >= 20:
            ci_lower = np.percentile(member_preds, 100 * alpha, axis=0)
            ci_upper = np.percentile(member_preds, 100 * (1 - alpha), axis=0)
        else:
            # Percentiles of a handful of folds are dominated by the extreme
            # members, so use a normal approximation around their mean.
            spread = z * member_preds.std(axis=0, ddof=1)
            ci_lower = probas - spread
            ci_upper = probas + spread

        # The point estimate must lie inside its own interval.
        ci_lower = np.clip(np.minimum(ci_lower, probas), 0.0, 1.0)
        ci_upper = np.clip(np.maximum(ci_upper, probas), 0.0, 1.0)
        return ci_lower, ci_upper

    def _calibrated_member_predictions(
        self, X_scaled: np.ndarray, n_bootstrap: int
    ) -> Optional[np.ndarray]:
        """Per-member predictions on the calibrated probability scale.

        Args:
            X_scaled: Scaled feature matrix.
            n_bootstrap: Maximum members to evaluate.

        Returns:
            Array of shape (n_members, n_samples), or None if unavailable.
        """
        members = getattr(self.calibrated_model, "calibrated_classifiers_", None)
        if members:
            try:
                return np.array([
                    member.predict_proba(X_scaled)[:, 1]
                    for member in list(members)[:n_bootstrap]
                ])
            except Exception as exc:
                logger.warning(f"Per-fold calibrated intervals unavailable: {exc}")

        # Random Forest exposes genuinely independent trees, which give a
        # usable spread even though they are on the uncalibrated scale.
        if isinstance(self.model, RandomForestClassifier):
            try:
                trees = self.model.estimators_[:n_bootstrap]
                return np.array([t.predict_proba(X_scaled)[:, 1] for t in trees])
            except Exception as exc:
                logger.warning(f"Per-tree intervals unavailable: {exc}")

        return None

    def _get_contributions(self, X_scaled: np.ndarray) -> list:
        """Get per-sample SHAP feature contributions."""
        try:
            import shap

            if self._explainer is None:
                self._explainer = shap.TreeExplainer(self.model)
            shap_values = self._explainer.shap_values(X_scaled)

            sv = np.asarray(
                shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1
                else shap_values[0] if isinstance(shap_values, list)
                else shap_values
            )
            # Multiclass output arrives as (n_samples, n_features, n_classes).
            if sv.ndim == 3:
                sv = sv[:, :, -1]

            return [
                {self.feature_names[j]: float(sv[i, j]) for j in range(sv.shape[1])}
                for i in range(sv.shape[0])
            ]

        except Exception as exc:
            logger.warning(f"SHAP computation failed: {exc}. Using global importances.")
            return [self._get_feature_importances()] * X_scaled.shape[0]

    @staticmethod
    def _categorize(score: float) -> str:
        """Map a probability to a qualitative label."""
        if score >= 0.7:
            return "Potentially Habitable"
        if score >= 0.4:
            return "Marginally Habitable"
        if score >= 0.2:
            return "Unlikely Habitable"
        return "Non-Habitable"

    # ── Persistence ────────────────────────────────────────────────────
    def _save_model(self):
        """Save the trained model to disk."""
        save_path = Path(
            self.config.get("model_save_path", "models/tabular_habitability.joblib")
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "calibrated_model": self.calibrated_model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "excluded_features": self.excluded_features,
            "model_type": self.model_type,
        }, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, path: Optional[str] = None):
        """Load a previously trained model.

        Args:
            path: Path to a saved model. Uses the config default if None.
        """
        load_path = Path(
            path or self.config.get("model_save_path", "models/tabular_habitability.joblib")
        )
        if not load_path.exists():
            raise FileNotFoundError(f"No saved model at {load_path}")

        artifact = joblib.load(load_path)
        self.model = artifact["model"]
        self.calibrated_model = artifact["calibrated_model"]
        self.scaler = artifact["scaler"]
        self.feature_names = artifact["feature_names"]
        self.excluded_features = artifact.get("excluded_features", [])
        self.model_type = artifact["model_type"]
        self._explainer = None
        self._is_trained = True

        logger.info(f"Model loaded from {load_path} ({len(self.feature_names)} features)")

    # ── Reporting ──────────────────────────────────────────────────────
    def _build_training_summary(
        self,
        cv_metrics: Dict[str, float],
        holdout_metrics: Dict[str, float],
        importances: Dict[str, float],
        n_features: int,
        n_samples: int,
    ) -> str:
        """Build a human-readable training summary."""
        lines = [
            "=" * 64,
            "Tabular Habitability Model — Training Summary",
            "=" * 64,
            f"Model type : {self.model_type}",
            f"Features   : {n_features} NASA-derived",
            f"Train rows : {n_samples}",
        ]
        if self.excluded_features:
            lines.append(
                f"Withheld   : {len(self.excluded_features)} label-defining "
                f"({', '.join(self.excluded_features[:4])}…)"
            )

        lines += ["", "Cross-Validation (training split):"]
        for metric in ("accuracy", "f1", "precision", "recall", "roc_auc", "average_precision"):
            if metric in cv_metrics:
                std = cv_metrics.get(f"{metric}_std", 0.0)
                lines.append(f"  {metric:<20s} {cv_metrics[metric]:.4f} ± {std:.4f}")

        if holdout_metrics:
            lines += ["", f"Held-out evaluation (n={holdout_metrics.get('n_test', 0)}, "
                          f"{holdout_metrics.get('n_positive', 0)} positive):"]
            for metric in ("accuracy", "precision", "recall", "f1",
                           "roc_auc", "average_precision", "brier"):
                if metric in holdout_metrics:
                    lines.append(f"  {metric:<20s} {holdout_metrics[metric]:.4f}")
            lines.append(
                f"  confusion (tn/fp/fn/tp): {holdout_metrics.get('tn', 0)}/"
                f"{holdout_metrics.get('fp', 0)}/{holdout_metrics.get('fn', 0)}/"
                f"{holdout_metrics.get('tp', 0)}"
            )

        lines += ["", "Top 15 Feature Importances:"]
        top = list(importances.items())[:15]
        scale = max((imp for _, imp in top), default=1.0) or 1.0
        for i, (feat, imp) in enumerate(top, start=1):
            bar = "█" * int(round(imp / scale * 40))
            lines.append(f"  {i:>2d}. {feat:<32s} {imp:.4f} {bar}")

        lines += [
            "",
            "NOTE: labels are a published-threshold RULE, not observed",
            "habitability. Metrics measure agreement with that rule.",
            "=" * 64,
        ]

        summary = "\n".join(lines)
        headline = holdout_metrics or cv_metrics
        logger.info(
            f"Training complete: {n_features} features, "
            f"F1={headline.get('f1', 0):.4f}, ROC-AUC={headline.get('roc_auc', 0):.4f}"
        )
        return summary


if __name__ == "__main__":
    from src.data_ingestion.exoplanet_archive import fetch_exoplanet_data
    from src.preprocessing.feature_engineering import engineer_features, get_feature_columns

    df = fetch_exoplanet_data()
    df_eng = engineer_features(df)
    feature_cols = get_feature_columns(df_eng)["all_features"]

    model = TabularHabitabilityModel()
    result = model.train(df_eng, feature_cols)
    print(result.training_summary)

    print("\nTop 15 candidates by model score:")
    ranked = model.score_catalog(df_eng)
    cols = [c for c in ["pl_name", "habitability_score", "ci_lower", "ci_upper",
                        "esi", "spectral_class", "sy_dist"] if c in ranked.columns]
    print(ranked.head(15)[cols].to_string(index=False))
