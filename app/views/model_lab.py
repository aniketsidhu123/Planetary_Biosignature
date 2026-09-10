"""
Model Lab — train, evaluate and inspect the tabular habitability model.

Deliberately explicit about what the metrics do and do not mean: the
training label is a rule derived from published thresholds, not observed
habitability, so agreement with that rule is what is being measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app import charts
from app.state import READABLE, feature_columns, readable_name
from app.theme import COLORS, badge, disclaimer, metric_row, panel, section
from src.models.tabular_model import (
    LABEL_DEFINING_FEATURES, TabularHabitabilityModel, xgboost_available,
)


def render() -> None:
    """Render the Model Lab view."""
    df = st.session_state.df

    # ── Training controls ──────────────────────────────────────────────
    section("Training configuration", "the model learns from the full NASA parameter set")

    controls = st.columns([1, 1, 1, 0.9, 1.1])
    with controls[0]:
        options = ["xgboost", "random_forest"] if xgboost_available() else ["random_forest"]
        algorithm = st.selectbox(
            "Algorithm", options,
            help=None if xgboost_available()
            else "XGBoost is unavailable — install the OpenMP runtime "
                 "(macOS: brew install libomp)",
        )
    with controls[1]:
        rule = st.selectbox(
            "Label rule", ["optimistic", "conservative"],
            help="Optimistic: 0.5–2.5 R⊕ inside the recent-Venus → early-Mars "
                 "zone. Conservative: 0.5–1.8 R⊕ inside the runaway → maximum "
                 "greenhouse zone.",
        )
    with controls[2]:
        calibration = st.selectbox(
            "Calibration", ["sigmoid", "isotonic"],
            help="Sigmoid (Platt) is the default: with only ~65 positives, "
                 "isotonic fits a coarse step function that ties most of the "
                 "top of the ranked list at one score.",
        )
    with controls[3]:
        exclude_leaky = st.toggle(
            "Withhold label columns", value=True,
            help="Withhold the columns the label rule is built from, so the "
                 "model must predict habitability from the rest of the "
                 "archive instead of re-deriving the rule.",
        )
    with controls[4]:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        train = st.button("Train model", type="primary", use_container_width=True)

    # Preview the label balance before committing to a training run.
    candidate_features = feature_columns(df)
    preview_model = TabularHabitabilityModel(model_type=algorithm)
    preview_model.config = {**preview_model.config,
                            "labeling": {**preview_model.config.get("labeling", {}),
                                         "rule": rule}}
    labels = preview_model.create_labels(df)
    n_positive = int(labels.sum())

    withheld = (
        [c for c in candidate_features
         if c in set(LABEL_DEFINING_FEATURES) | {f"{f}_missing" for f in LABEL_DEFINING_FEATURES}]
        if exclude_leaky else []
    )

    metric_row([
        ("Planets", f"{len(df):,}"),
        ("Candidate features", f"{len(candidate_features)}"),
        ("Withheld", f"{len(withheld)}", "label-defining"),
        ("Training features", f"{len(candidate_features) - len(withheld)}", "", "primary"),
        ("Positive labels", f"{n_positive}", f"{100 * n_positive / len(df):.2f}% of catalog",
         "accent"),
    ])

    if n_positive < 10:
        st.warning(
            f"Only {n_positive} planets satisfy the *{rule}* rule. Metrics on a "
            "positive class this small are extremely noisy — prefer the "
            "optimistic rule, or read the confidence intervals rather than the "
            "point estimates."
        )

    with st.expander("What the model is actually learning"):
        st.markdown(
            "There is **no ground truth** for exoplanet habitability, so the "
            "training label is a *rule* built from published thresholds "
            "(rocky radius × Kopparapu habitable zone). Two consequences "
            "follow, and both matter for reading the numbers below:\n\n"
            "1. **Metrics measure agreement with that rule, not with biology.** "
            "A held-out F1 of 0.85 means the model reproduces the published "
            "criteria well; it says nothing about whether any planet is "
            "inhabited.\n"
            "2. **Perfect leakage isolation is impossible.** The rule is a "
            "deterministic function of quantities the archive measures, so a "
            "model given semi-major axis and stellar luminosity can always "
            "rebuild the insolation criterion. Withholding the direct "
            "restatements stops the model reading the answer off a single "
            "column; it cannot make the task leakage-free without deleting "
            "the physics that makes the tool useful.\n\n"
            "What the model genuinely adds is **ranking and interpolation**: "
            "a calibrated score for every planet, including the many whose "
            "defining measurements are missing from the archive."
        )
        if withheld:
            st.caption("Withheld columns: " + ", ".join(f"`{c}`" for c in withheld))

    # ── Train ──────────────────────────────────────────────────────────
    if train:
        with st.spinner(f"Training {algorithm} on {len(df):,} planets…"):
            model = TabularHabitabilityModel(model_type=algorithm)
            model.config = {
                **model.config,
                "calibration_method": calibration,
                "labeling": {**model.config.get("labeling", {}), "rule": rule},
            }
            result = model.train(
                df, candidate_features, exclude_label_features=exclude_leaky
            )
        st.session_state.update(model=model, train_result=result, catalog_scores=None)
        # Rerun so the sidebar's model panel reflects the new model — it was
        # drawn before training started and would otherwise stay stale until
        # the next interaction.
        st.session_state.train_notice = (
            f"Trained on {result.n_features} features. "
            f"Held-out F1 {result.holdout_scores.get('f1', float('nan')):.3f}."
        )
        st.rerun()

    notice = st.session_state.pop("train_notice", None)
    if notice:
        st.success(notice)

    result = st.session_state.get("train_result")
    if result is None:
        st.info("Configure the run above and press **Train model**.")
        return

    model = st.session_state.model

    # ── Performance ────────────────────────────────────────────────────
    section("Performance", "held-out evaluation on planets the model never saw")

    holdout = result.holdout_scores
    if holdout:
        metric_row([
            ("Held-out F1", f"{holdout.get('f1', 0):.3f}", "", "accent"),
            ("Precision", f"{holdout.get('precision', 0):.3f}"),
            ("Recall", f"{holdout.get('recall', 0):.3f}"),
            ("ROC-AUC", f"{holdout.get('roc_auc', 0):.3f}"),
            ("Avg precision", f"{holdout.get('average_precision', 0):.3f}"),
            ("Brier score", f"{holdout.get('brier', 0):.4f}", "lower is better"),
        ])
        st.caption(
            f"Evaluated on {holdout.get('n_test', 0):,} held-out planets "
            f"containing {holdout.get('n_positive', 0)} positives. "
            f"Confusion matrix — true negatives {holdout.get('tn', 0):,}, "
            f"false positives {holdout.get('fp', 0)}, "
            f"false negatives {holdout.get('fn', 0)}, "
            f"true positives {holdout.get('tp', 0)}."
        )
        if holdout.get("n_positive", 0) < 20:
            st.caption(
                f"⚠️ With only {holdout.get('n_positive', 0)} positives in the "
                "held-out set, a single misclassification moves F1 by several "
                "points. Treat these as indicative, not precise."
            )

    with st.expander("Cross-validation on the training split"):
        cv = result.cv_scores
        rows = [
            {"Metric": name.replace("_", " ").title(),
             "Mean": f"{cv.get(name, 0):.4f}",
             "Std": f"± {cv.get(f'{name}_std', 0):.4f}"}
            for name in ("accuracy", "f1", "precision", "recall",
                         "roc_auc", "average_precision")
            if name in cv
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Calibration ────────────────────────────────────────────────────
    section("Calibration", "do predicted probabilities match observed frequencies?")

    try:
        labels_full = model.create_labels(df)
        scored = model.score_catalog(df)
        aligned = labels_full.reindex(scored.index)
        left, right = st.columns([1.3, 1])
        with left:
            st.plotly_chart(
                charts.calibration_curve(
                    aligned.to_numpy(dtype=float),
                    scored["habitability_score"].to_numpy(dtype=float),
                ),
                use_container_width=True, config={"displayModeBar": False},
            )
        with right:
            st.markdown(
                '<div class="panel"><h3>Reading this chart</h3>'
                '<p class="note">Each point is a bin of planets sharing a similar '
                "predicted probability. Its height is the fraction of those "
                "planets the label rule actually calls habitable. Points on the "
                "dashed line mean the probabilities can be taken at face value — "
                "planets scored 0.30 are habitable-by-rule about 30% of the "
                "time.<br><br>Points below the line mean over-confidence; above "
                "means under-confidence. Marker size reflects how many planets "
                "fall in each bin, and nearly all of them sit in the lowest bin, "
                "because habitable-zone planets are rare.</p></div>",
                unsafe_allow_html=True,
            )
    except Exception as exc:  # calibration is diagnostic, never fatal
        st.caption(f"Calibration curve unavailable: {exc}")

    # ── Feature importance ─────────────────────────────────────────────
    section("What the model uses", "importance across the NASA-derived feature set")

    left, right = st.columns([1.4, 1])
    with left:
        st.plotly_chart(
            charts.importance_bars(result.feature_importances, readable_name, top_n=18),
            use_container_width=True, config={"displayModeBar": False},
        )
    with right:
        families = _importance_by_family(result.feature_importances)
        with panel("Importance by data family"):
            st.dataframe(
                pd.DataFrame(
                    [{"Family": k, "Share": f"{v:.1%}"} for k, v in families.items()]
                ),
                use_container_width=True, hide_index=True, height=300,
            )
            st.markdown(
                '<p class="note">Every family here comes from the NASA table — '
                "stellar properties, system architecture, transit geometry and "
                "discovery metadata all feed the model, not just planet radius "
                "and temperature.</p>",
                unsafe_allow_html=True,
            )

    with st.expander("Full training summary"):
        st.code(result.training_summary, language="text")

    with st.expander("Best hyperparameters"):
        st.json(result.best_params)


def _importance_by_family(importances: dict) -> dict:
    """Group feature importances into readable data families."""
    families = {
        "Habitable zone": ("hz_",),
        "Planetary": ("pl_",),
        "Stellar": ("st_", "spectral"),
        "System & astrometry": ("sy_", "ra", "dec", "is_multi", "is_nearby"),
        "Discovery metadata": ("disc", "discoverymethod", "tran_flag",
                               "rv_flag", "ttv_flag", "multi_method", "years_since"),
        "Derived physics": ("esi", "surface_gravity", "escape_velocity",
                            "orbital_velocity", "mass_radius", "rocky_",
                            "insolation", "tidal_", "stellar_activity"),
    }

    totals = {name: 0.0 for name in families}
    totals["Other"] = 0.0
    for feature, importance in importances.items():
        for name, prefixes in families.items():
            if feature.startswith(prefixes):
                totals[name] += importance
                break
        else:
            totals["Other"] += importance

    grand = sum(totals.values()) or 1.0
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    return {name: value / grand for name, value in ranked if value > 0}
