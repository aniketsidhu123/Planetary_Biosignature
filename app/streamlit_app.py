"""
ExoScope — Streamlit Web Dashboard

The primary UI for the Habitability & Biosignature-Likelihood Index system.
Provides interactive exploration of exoplanet data, image analysis,
and fused HBLI scoring with explainable breakdowns.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import io
import sys
import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_config
from src.utils.logging_setup import setup_logging
from src.data_ingestion.exoplanet_archive import fetch_exoplanet_data, get_data_summary
from src.preprocessing.feature_engineering import engineer_features, get_feature_columns
from src.preprocessing.classical_cv_filter import ClassicalCVFilter
from src.models.vision_detector import VisionDetectorStack
from src.models.fusion import FusionLayer, HBLIResult
from src.data_ingestion.tiling import tile_image


# ── Page Configuration ─────────────────────────────────────────────────
st.set_page_config(
    page_title="ExoScope — HBLI Dashboard",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    /* Root variables */
    :root {
        --primary: #6C63FF;
        --primary-light: #8B83FF;
        --primary-dark: #4A42CC;
        --accent: #00D4AA;
        --accent-warm: #FF6B6B;
        --bg-dark: #0E1117;
        --bg-card: #1A1D26;
        --bg-card-hover: #22252F;
        --text-primary: #FAFAFA;
        --text-secondary: #9BA1B0;
        --border: #2D3140;
        --gradient-1: linear-gradient(135deg, #6C63FF, #00D4AA);
        --gradient-2: linear-gradient(135deg, #FF6B6B, #FFD93D);
        --gradient-3: linear-gradient(135deg, #00D4AA, #0EA5E9);
    }

    /* Global font */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, rgba(108,99,255,0.15), rgba(0,212,170,0.1));
        border: 1px solid rgba(108,99,255,0.3);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }

    .main-header h1 {
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .main-header p {
        color: var(--text-secondary);
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
        line-height: 1.5;
    }

    /* Metric cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: var(--primary);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(108,99,255,0.15);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-label {
        color: var(--text-secondary);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }

    /* HBLI Score gauge */
    .hbli-gauge {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }

    .hbli-score {
        font-size: 4rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }

    .hbli-high { color: #00D4AA; }
    .hbli-moderate { color: #FFD93D; }
    .hbli-low { color: #FF9F43; }
    .hbli-very-low { color: #FF6B6B; }

    /* Category badge */
    .category-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    .badge-high { background: rgba(0,212,170,0.15); color: #00D4AA; border: 1px solid rgba(0,212,170,0.3); }
    .badge-moderate { background: rgba(255,217,61,0.15); color: #FFD93D; border: 1px solid rgba(255,217,61,0.3); }
    .badge-low { background: rgba(255,159,67,0.15); color: #FF9F43; border: 1px solid rgba(255,159,67,0.3); }
    .badge-very-low { background: rgba(255,107,107,0.15); color: #FF6B6B; border: 1px solid rgba(255,107,107,0.3); }

    /* Disclaimer box */
    .disclaimer-box {
        background: rgba(255,107,107,0.08);
        border: 1px solid rgba(255,107,107,0.25);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        font-size: 0.85rem;
        color: #FF9F9F;
        margin-top: 1rem;
    }

    /* Section headers */
    .section-header {
        color: var(--text-primary);
        font-size: 1.3rem;
        font-weight: 700;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary);
        margin: 1.5rem 0 1rem 0;
        display: inline-block;
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #12151C;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.2rem;
        color: var(--primary-light);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* Plotly chart backgrounds */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ───────────────────────────────────────
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.df_eng = None
    st.session_state.model_trained = False
    st.session_state.tabular_model = None
    st.session_state.hbli_results = []


# ── Helper Functions ───────────────────────────────────────────────────
def get_score_color(score: float) -> str:
    """Return color based on HBLI score."""
    if score >= 0.75:
        return "#00D4AA"
    elif score >= 0.50:
        return "#FFD93D"
    elif score >= 0.25:
        return "#FF9F43"
    return "#FF6B6B"


def get_score_class(score: float) -> str:
    """Return CSS class based on HBLI score."""
    if score >= 0.75:
        return "high"
    elif score >= 0.50:
        return "moderate"
    elif score >= 0.25:
        return "low"
    return "very-low"


def create_gauge_chart(score: float, title: str = "HBLI Score") -> go.Figure:
    """Create a gauge chart for HBLI score display."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        number={"suffix": "%", "font": {"size": 48, "family": "JetBrains Mono", "color": get_score_color(score)}},
        title={"text": title, "font": {"size": 16, "color": "#9BA1B0"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#2D3140"},
            "bar": {"color": get_score_color(score), "thickness": 0.3},
            "bgcolor": "#1A1D26",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "rgba(255,107,107,0.1)"},
                {"range": [25, 50], "color": "rgba(255,159,67,0.1)"},
                {"range": [50, 75], "color": "rgba(255,217,61,0.1)"},
                {"range": [75, 100], "color": "rgba(0,212,170,0.1)"},
            ],
            "threshold": {
                "line": {"color": "#FAFAFA", "width": 2},
                "thickness": 0.75,
                "value": score * 100,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#FAFAFA"},
    )
    return fig


def create_radar_chart(features: dict, earth_baseline: dict) -> go.Figure:
    """Create a radar/spider chart comparing planet features to Earth."""
    categories = list(features.keys())
    values = list(features.values())
    earth_values = [earth_baseline.get(k, 0) for k in categories]

    # Readable names
    readable = {
        "pl_rade": "Radius", "pl_dens": "Density", "pl_orbeccen": "Eccentricity",
        "pl_insol": "Insolation", "pl_eqt": "Temperature", "esi": "ESI",
        "hz_ratio": "HZ Distance", "tidal_lock_likelihood": "Tidal Lock",
    }
    display_categories = [readable.get(c, c) for c in categories]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=display_categories + [display_categories[0]],
        fill="toself",
        fillcolor="rgba(108,99,255,0.15)",
        line=dict(color="#6C63FF", width=2),
        name="Target Planet",
    ))
    fig.add_trace(go.Scatterpolar(
        r=earth_values + [earth_values[0]],
        theta=display_categories + [display_categories[0]],
        fill="toself",
        fillcolor="rgba(0,212,170,0.1)",
        line=dict(color="#00D4AA", width=2, dash="dash"),
        name="Earth Baseline",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2D3140"),
            angularaxis=dict(gridcolor="#2D3140"),
        ),
        showlegend=True,
        height=400,
        margin=dict(l=60, r=60, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#FAFAFA"},
        legend=dict(font=dict(color="#9BA1B0")),
    )
    return fig


def create_feature_importance_chart(importances: dict) -> go.Figure:
    """Create a horizontal bar chart of feature importances."""
    readable = {
        "pl_rade": "Planet Radius", "pl_bmasse": "Planet Mass", "pl_dens": "Density",
        "pl_orbeccen": "Eccentricity", "pl_insol": "Insolation", "pl_eqt": "Temperature",
        "esi": "Earth Similarity", "hz_ratio": "HZ Distance",
        "tidal_lock_likelihood": "Tidal Lock Risk", "stellar_activity": "Stellar Activity",
        "st_teff": "Stellar Temp", "st_lum": "Stellar Luminosity",
    }

    top_n = dict(list(importances.items())[:10])
    names = [readable.get(k, k.replace("_", " ").title()) for k in top_n.keys()]
    values = list(top_n.values())

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker=dict(
            color=values,
            colorscale=[[0, "#6C63FF"], [0.5, "#00D4AA"], [1, "#FFD93D"]],
        ),
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(color="#9BA1B0", size=11),
    ))
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#2D3140", color="#9BA1B0"),
        yaxis=dict(autorange="reversed", color="#FAFAFA"),
        font={"color": "#FAFAFA"},
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🔭 ExoScope")
    st.markdown("*Habitability & Biosignature-Likelihood Index*")
    st.divider()

    analysis_mode = st.radio(
        "Analysis Mode",
        ["🪐 Exoplanet Analysis", "🖼️ Image Analysis", "📊 Dataset Explorer"],
        index=0,
    )

    st.divider()

    # Data loading
    st.markdown("### 📡 Data Source")
    if st.button("Load Exoplanet Data", use_container_width=True, type="primary"):
        with st.spinner("Fetching exoplanet data..."):
            df = fetch_exoplanet_data()
            df_eng = engineer_features(df)
            st.session_state.df = df
            st.session_state.df_eng = df_eng
            st.session_state.data_loaded = True
        st.success(f"✅ Loaded {len(df)} exoplanets")

    if st.session_state.data_loaded:
        st.markdown(f"**{len(st.session_state.df)}** exoplanets loaded")

    st.divider()

    # Model training
    if st.session_state.data_loaded:
        st.markdown("### 🧠 Model")
        model_type = st.selectbox("Algorithm", ["xgboost", "random_forest"])

        if st.button("Train Model", use_container_width=True, type="primary"):
            with st.spinner("Training habitability model..."):
                from src.models.tabular_model import TabularHabitabilityModel

                model = TabularHabitabilityModel(model_type=model_type)
                col_info = get_feature_columns(st.session_state.df_eng)
                feature_cols = col_info["all_features"]
                result = model.train(st.session_state.df_eng, feature_cols)

                st.session_state.tabular_model = model
                st.session_state.train_result = result
                st.session_state.feature_cols = feature_cols
                st.session_state.model_trained = True

            st.success("✅ Model trained!")

    st.divider()
    st.markdown(
        '<div class="disclaimer-box">'
        "⚠️ All scores represent <b>likelihood based on proxy indicators</b> "
        "validated against Earth analogs — not confirmed detection of life."
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main Content ───────────────────────────────────────────────────────

# Header
st.markdown(
    '<div class="main-header">'
    "<h1>ExoScope Dashboard</h1>"
    "<p>Composite, explainable habitability & biosignature-likelihood scoring — "
    "fusing physics-grounded tabular models with vision-based surface analysis.</p>"
    "</div>",
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────────────────
# Mode 1: Exoplanet Analysis
# ────────────────────────────────────────────────────────────────────────
if analysis_mode == "🪐 Exoplanet Analysis":

    if not st.session_state.data_loaded:
        st.info("👈 Click **Load Exoplanet Data** in the sidebar to get started.")
        st.stop()

    df_eng = st.session_state.df_eng

    # Planet selector
    col1, col2 = st.columns([2, 1])
    with col1:
        if "pl_name" in df_eng.columns:
            planet_names = df_eng["pl_name"].dropna().tolist()
            selected_planet = st.selectbox(
                "Select a planet to analyze",
                planet_names,
                index=0,
            )
        else:
            selected_planet = st.selectbox("Select planet index", range(len(df_eng)))

    with col2:
        st.markdown("")
        st.markdown("")
        if st.session_state.model_trained:
            analyze_btn = st.button("🔬 Analyze", type="primary", use_container_width=True)
        else:
            st.warning("Train a model first")
            analyze_btn = False

    # Get selected planet data
    if "pl_name" in df_eng.columns:
        planet_row = df_eng[df_eng["pl_name"] == selected_planet].iloc[0]
    else:
        planet_row = df_eng.iloc[selected_planet]

    # Display planet parameters
    st.markdown('<div class="section-header">📋 Planet Parameters</div>', unsafe_allow_html=True)

    param_cols = st.columns(5)
    params_display = {
        "Radius (R⊕)": ("pl_rade", ""),
        "Mass (M⊕)": ("pl_bmasse", ""),
        "Density": ("pl_dens", "g/cm³"),
        "Temperature": ("pl_eqt", "K"),
        "ESI": ("esi", ""),
    }

    for i, (label, (col_name, unit)) in enumerate(params_display.items()):
        with param_cols[i]:
            val = planet_row.get(col_name, np.nan)
            if pd.notna(val):
                display_val = f"{val:.3f}" if isinstance(val, float) else str(val)
            else:
                display_val = "N/A"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{display_val}</div>'
                f'<div class="metric-label">{label}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    # Analysis results
    if analyze_btn and st.session_state.model_trained:
        model = st.session_state.tabular_model
        planet_df = pd.DataFrame([planet_row])

        predictions = model.predict(planet_df, st.session_state.feature_cols)
        pred = predictions[0]

        # Fuse with vision (no imagery for pure tabular analysis)
        fusion = FusionLayer()
        tab_features = {}
        if pred.top_features:
            tab_features = dict(pred.top_features)

        hbli_result = fusion.fuse(
            target_name=pred.planet_name,
            tabular_score=pred.habitability_score,
            tabular_ci=(pred.confidence_lower, pred.confidence_upper),
            vision_score=0.0,
            tabular_features=tab_features,
        )

        st.session_state.current_hbli = hbli_result

    if "current_hbli" in st.session_state:
        hbli = st.session_state.current_hbli

        st.markdown('<div class="section-header">🎯 HBLI Analysis Result</div>', unsafe_allow_html=True)

        result_cols = st.columns([1.2, 1, 1])

        with result_cols[0]:
            st.plotly_chart(
                create_gauge_chart(hbli.hbli_score, "HBLI Score"),
                use_container_width=True,
            )

        with result_cols[1]:
            score_class = get_score_class(hbli.hbli_score)
            st.markdown(
                f'<div class="hbli-gauge">'
                f'<div class="category-badge badge-{score_class}">{hbli.category}</div>'
                f"<br><br>"
                f'<div style="color: var(--text-secondary); font-size: 0.9rem;">'
                f"{hbli.category_description[:200]}..."
                f"</div>"
                f"<br>"
                f'<div style="font-family: JetBrains Mono; color: #9BA1B0;">'
                f"95% CI: [{hbli.confidence_lower:.4f}, {hbli.confidence_upper:.4f}]"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with result_cols[2]:
            sub_scores = {
                "Tabular Habitability": hbli.tabular_score,
                "Vision Biosignature": hbli.vision_score,
            }
            fig = go.Figure(go.Bar(
                x=list(sub_scores.values()),
                y=list(sub_scores.keys()),
                orientation="h",
                marker=dict(color=["#6C63FF", "#00D4AA"]),
                text=[f"{v:.3f}" for v in sub_scores.values()],
                textposition="outside",
                textfont=dict(color="#FAFAFA"),
            ))
            fig.update_layout(
                title="Sub-Scores",
                height=280,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(range=[0, 1.1], gridcolor="#2D3140", color="#9BA1B0"),
                yaxis=dict(color="#FAFAFA"),
                font={"color": "#FAFAFA"},
            )
            st.plotly_chart(fig, use_container_width=True)

        # Feature attribution
        st.markdown('<div class="section-header">🔍 Feature Attribution</div>', unsafe_allow_html=True)

        attr_cols = st.columns(2)
        with attr_cols[0]:
            st.markdown("**Top Contributing Factors**")
            for name, contrib, direction in hbli.top_contributing_factors[:5]:
                bar_width = min(abs(contrib) * 200, 100)
                st.markdown(
                    f'<div style="margin: 4px 0;">'
                    f'<span style="color: #00D4AA;">▸</span> {name}: '
                    f'<span style="color: #00D4AA; font-family: JetBrains Mono;">+{contrib:.4f}</span> '
                    f'<span style="color: #9BA1B0;">({direction})</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with attr_cols[1]:
            st.markdown("**Risk Factors**")
            if hbli.risk_factors:
                for name, contrib, direction in hbli.risk_factors[:5]:
                    st.markdown(
                        f'<div style="margin: 4px 0;">'
                        f'<span style="color: #FF6B6B;">▸</span> {name}: '
                        f'<span style="color: #FF6B6B; font-family: JetBrains Mono;">{contrib:.4f}</span> '
                        f'<span style="color: #9BA1B0;">({direction})</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("*No significant risk factors identified.*")

        # Radar chart comparison to Earth
        feature_subset = {}
        earth_baseline = {}
        for feat in ["pl_rade", "pl_dens", "pl_orbeccen", "pl_insol", "esi", "hz_ratio"]:
            if feat in planet_row.index:
                val = planet_row[feat]
                if pd.notna(val):
                    # Normalize to [0, 1] range for radar
                    if feat == "esi":
                        feature_subset[feat] = float(val)
                        earth_baseline[feat] = 1.0
                    elif feat == "pl_rade":
                        feature_subset[feat] = min(float(val) / 5.0, 1.0)
                        earth_baseline[feat] = 1.0 / 5.0
                    elif feat == "pl_dens":
                        feature_subset[feat] = min(float(val) / 10.0, 1.0)
                        earth_baseline[feat] = 5.51 / 10.0
                    elif feat == "hz_ratio":
                        feature_subset[feat] = max(0, 1.0 - abs(float(val) - 1.0))
                        earth_baseline[feat] = 1.0
                    else:
                        feature_subset[feat] = min(abs(float(val)), 1.0)
                        earth_baseline[feat] = 0.5

        if feature_subset:
            st.markdown('<div class="section-header">🌍 Earth Comparison</div>', unsafe_allow_html=True)
            st.plotly_chart(
                create_radar_chart(feature_subset, earth_baseline),
                use_container_width=True,
            )


# ────────────────────────────────────────────────────────────────────────
# Mode 2: Image Analysis
# ────────────────────────────────────────────────────────────────────────
elif analysis_mode == "🖼️ Image Analysis":

    st.markdown('<div class="section-header">🖼️ Planetary Image Analysis</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a planetary surface image",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="Upload HiRISE, Mastcam, or Earth-analog imagery for biosignature proxy analysis",
    )

    if uploaded_file:
        # Load image
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # Convert RGB to BGR for OpenCV
        if img_array.ndim == 3 and img_array.shape[2] >= 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        st.image(image, caption=f"Uploaded: {uploaded_file.name} ({img_array.shape[1]}×{img_array.shape[0]})", use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            tile_size = st.slider("Tile Size", 256, 1024, 640, 64)
        with col2:
            confidence_threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.25, 0.05)

        if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
            progress = st.progress(0, "Initializing...")

            # Step 1: Save temp image and tile
            progress.progress(10, "Tiling image...")
            temp_path = Path("data/processed/temp_upload.jpg")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(temp_path), img_bgr)

            tiling_result = tile_image(str(temp_path), tile_size=tile_size)
            st.info(f"📐 Generated **{tiling_result.passed_tiles}** tiles from **{tiling_result.total_tiles}** total (coverage filter applied)")

            # Step 2: Classical CV pre-filter
            progress.progress(30, "Running classical CV pre-filter...")
            cv_filter = ClassicalCVFilter()
            passed_tiles, rejected_tiles, filter_summary = cv_filter.filter_batch(tiling_result.tiles)

            st.success(
                f"🔍 Pre-filter: **{filter_summary['passed']}/{filter_summary['total']}** tiles passed "
                f"({filter_summary['pass_rate']:.0%}) — mean relevance: {filter_summary['mean_relevance']:.3f}"
            )

            # Step 3: Vision detection
            progress.progress(60, "Running biosignature proxy detection...")
            detector = VisionDetectorStack()
            detection_results = detector.detect_batch(passed_tiles)

            # Step 4: Display results
            progress.progress(90, "Generating visualizations...")

            # Detection summary
            total_detections = sum(r.n_detections for r in detection_results)
            mean_score = np.mean([r.overall_score for r in detection_results]) if detection_results else 0

            summary_cols = st.columns(4)
            summaries = [
                ("Total Tiles", str(tiling_result.total_tiles)),
                ("Filtered Tiles", str(filter_summary["passed"])),
                ("Detections", str(total_detections)),
                ("Vision Score", f"{mean_score:.3f}"),
            ]
            for i, (label, value) in enumerate(summaries):
                with summary_cols[i]:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value">{value}</div>'
                        f'<div class="metric-label">{label}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Show top detections
            if detection_results:
                st.markdown("### Top Detection Tiles")
                top_results = sorted(detection_results, key=lambda r: r.overall_score, reverse=True)[:6]

                tile_cols = st.columns(3)
                for idx, result in enumerate(top_results):
                    with tile_cols[idx % 3]:
                        # Find corresponding tile
                        tile = passed_tiles[detection_results.index(result)]
                        vis_img = detector.visualize_detections(tile.image, result)
                        vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                        st.image(vis_rgb, caption=f"Score: {result.overall_score:.3f} | Detections: {result.n_detections}")

                # Per-class detection bar chart
                class_counts = {}
                for r in detection_results:
                    for det in r.detections:
                        class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

                if class_counts:
                    fig = go.Figure(go.Bar(
                        x=list(class_counts.keys()),
                        y=list(class_counts.values()),
                        marker=dict(color=["#00D4AA", "#FFD93D", "#6C63FF"][:len(class_counts)]),
                        text=list(class_counts.values()),
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title="Detections by Proxy Class",
                        height=300,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(color="#FAFAFA"),
                        yaxis=dict(color="#9BA1B0", gridcolor="#2D3140"),
                        font={"color": "#FAFAFA"},
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # HBLI fusion (vision-only, no tabular)
            fusion = FusionLayer()
            hbli = fusion.fuse(
                target_name=uploaded_file.name,
                tabular_score=0.0,
                vision_score=mean_score,
                vision_features={cls: max(
                    (d.confidence for r in detection_results for d in r.detections if d.class_name == cls),
                    default=0.0,
                ) for cls in CLASS_NAMES if any(d.class_name == cls for r in detection_results for d in r.detections)},
            )

            st.divider()
            st.plotly_chart(create_gauge_chart(hbli.hbli_score, "Image HBLI Score"), use_container_width=True)
            st.markdown(
                f'<div class="disclaimer-box">'
                f"Note: This analysis uses vision-only scoring (no tabular data). "
                f"For a complete HBLI assessment, combine with exoplanet parameter analysis."
                f"</div>",
                unsafe_allow_html=True,
            )

            progress.progress(100, "Analysis complete!")

    else:
        st.info("📸 Upload a planetary surface image (HiRISE, Mastcam, or Earth-analog) to begin analysis.")

        # Show example workflow
        with st.expander("📖 How Image Analysis Works"):
            st.markdown("""
            1. **Upload** an image (Mars surface, Earth analog site, or any geological imagery)
            2. **Tiling** — Image is split into 640×640 tiles for efficient processing
            3. **Pre-filter** — Classical CV filter rejects featureless tiles (saves ~90% compute)
            4. **Detection** — YOLO specialist detectors scan for biosignature proxies:
               - 🟢 Sedimentary layering patterns
               - 🟡 Mineral-water interaction zones
               - 🔵 Erosion/fluid-flow morphology
            5. **Scoring** — Detections are aggregated into a vision biosignature score
            """)


# ────────────────────────────────────────────────────────────────────────
# Mode 3: Dataset Explorer
# ────────────────────────────────────────────────────────────────────────
elif analysis_mode == "📊 Dataset Explorer":

    if not st.session_state.data_loaded:
        st.info("👈 Click **Load Exoplanet Data** in the sidebar to get started.")
        st.stop()

    df_eng = st.session_state.df_eng

    st.markdown('<div class="section-header">📊 Exoplanet Dataset Explorer</div>', unsafe_allow_html=True)

    # Summary metrics
    summary = get_data_summary(st.session_state.df)
    metric_cols = st.columns(4)
    metrics = [
        ("Total Exoplanets", summary["total_planets"]),
        ("Potentially Habitable", summary.get("potentially_habitable", "N/A")),
        ("Features", len(df_eng.columns)),
        ("Mean ESI", f"{df_eng['esi'].mean():.3f}" if "esi" in df_eng.columns else "N/A"),
    ]
    for i, (label, value) in enumerate(metrics):
        with metric_cols[i]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{value}</div>'
                f'<div class="metric-label">{label}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("")

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["🌌 Distribution", "🎯 ESI Ranking", "🔬 Feature Correlation", "📋 Data Table"])

    with tab1:
        # Scatter plot: radius vs temperature colored by ESI
        if "pl_rade" in df_eng.columns and "pl_eqt" in df_eng.columns and "esi" in df_eng.columns:
            fig = px.scatter(
                df_eng.dropna(subset=["pl_rade", "pl_eqt", "esi"]),
                x="pl_rade",
                y="pl_eqt",
                color="esi",
                size="esi",
                size_max=15,
                color_continuous_scale="Viridis",
                labels={
                    "pl_rade": "Planet Radius (Earth radii)",
                    "pl_eqt": "Equilibrium Temperature (K)",
                    "esi": "Earth Similarity Index",
                },
                hover_data=["pl_name"] if "pl_name" in df_eng.columns else None,
                title="Exoplanet Population: Radius vs Temperature",
            )

            # Add Earth reference
            fig.add_trace(go.Scatter(
                x=[1.0], y=[255],
                mode="markers+text",
                marker=dict(size=15, color="#00D4AA", symbol="star"),
                text=["🌍 Earth"],
                textposition="top center",
                name="Earth",
                textfont=dict(color="#00D4AA"),
            ))

            # Add habitable zone band
            fig.add_hrect(y0=180, y1=310, fillcolor="rgba(0,212,170,0.08)",
                          line_width=0, annotation_text="Habitable Temperature Range",
                          annotation_position="top left")

            fig.update_layout(
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#2D3140", color="#9BA1B0"),
                yaxis=dict(gridcolor="#2D3140", color="#9BA1B0"),
                font={"color": "#FAFAFA"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Top ESI planets
        if "esi" in df_eng.columns:
            top_esi = df_eng.nlargest(20, "esi")
            name_col = "pl_name" if "pl_name" in top_esi.columns else top_esi.index.astype(str)

            fig = go.Figure(go.Bar(
                x=top_esi["esi"].values,
                y=top_esi["pl_name"].values if "pl_name" in top_esi.columns else list(range(20)),
                orientation="h",
                marker=dict(
                    color=top_esi["esi"].values,
                    colorscale=[[0, "#6C63FF"], [0.5, "#00D4AA"], [1, "#FFD93D"]],
                ),
                text=[f"{v:.4f}" for v in top_esi["esi"]],
                textposition="outside",
                textfont=dict(color="#9BA1B0", size=11),
            ))
            fig.update_layout(
                title="Top 20 Planets by Earth Similarity Index",
                height=600,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(range=[0, 1.1], gridcolor="#2D3140", color="#9BA1B0", title="ESI Score"),
                yaxis=dict(autorange="reversed", color="#FAFAFA"),
                font={"color": "#FAFAFA"},
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Correlation heatmap
        numeric_cols = ["pl_rade", "pl_dens", "pl_orbeccen", "pl_insol", "pl_eqt",
                        "st_teff", "esi", "hz_ratio", "tidal_lock_likelihood", "stellar_activity"]
        available_cols = [c for c in numeric_cols if c in df_eng.columns]

        if available_cols:
            corr_matrix = df_eng[available_cols].corr()
            readable_names = {
                "pl_rade": "Radius", "pl_dens": "Density", "pl_orbeccen": "Eccentricity",
                "pl_insol": "Insolation", "pl_eqt": "Temperature", "st_teff": "Stellar Temp",
                "esi": "ESI", "hz_ratio": "HZ Distance",
                "tidal_lock_likelihood": "Tidal Lock", "stellar_activity": "Stellar Activity",
            }
            display_names = [readable_names.get(c, c) for c in available_cols]

            fig = go.Figure(go.Heatmap(
                z=corr_matrix.values,
                x=display_names,
                y=display_names,
                colorscale="RdBu_r",
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont=dict(size=10),
            ))
            fig.update_layout(
                title="Feature Correlation Matrix",
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#FAFAFA"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        # Data table
        display_cols = [c for c in ["pl_name", "pl_rade", "pl_bmasse", "pl_dens",
                                      "pl_eqt", "esi", "hz_ratio", "tidal_lock_likelihood"]
                        if c in df_eng.columns]
        st.dataframe(
            df_eng[display_cols].sort_values("esi", ascending=False).head(100),
            use_container_width=True,
            height=500,
        )

    # Model results section
    if st.session_state.model_trained and "train_result" in st.session_state:
        st.markdown('<div class="section-header">🧠 Model Performance</div>', unsafe_allow_html=True)

        result = st.session_state.train_result
        metric_cols = st.columns(5)
        cv_metrics = result.cv_scores

        for i, metric in enumerate(["accuracy", "f1", "precision", "recall", "roc_auc"]):
            with metric_cols[i]:
                val = cv_metrics.get(metric, 0)
                std = cv_metrics.get(f"{metric}_std", 0)
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{val:.3f}</div>'
                    f'<div class="metric-label">{metric} ± {std:.3f}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        if result.feature_importances:
            st.plotly_chart(
                create_feature_importance_chart(result.feature_importances),
                use_container_width=True,
            )

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #9BA1B0; font-size: 0.8rem;">'
    "ExoScope v0.1.0 — Habitability & Biosignature-Likelihood Index | "
    "Built with real NASA/ISRO data | "
    "All scores are proxy-based likelihood estimates, not confirmed detections."
    "</div>",
    unsafe_allow_html=True,
)
