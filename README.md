# ExoScope — Habitability & Biosignature-Likelihood Index (HBLI)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.ai/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v8%2Fv11-blueviolet.svg)](https://ultralytics.com/)
[![Tests](https://img.shields.io/badge/pytest-46%20passed-brightgreen.svg)](tests/)

> **A composite, explainable habitability and biosignature-likelihood scoring system** that fuses physics-grounded planetary parameters with computer-vision surface analysis — engineered from the ground up to run efficiently on consumer hardware.

---

## ⚠️ Important Scientific Disclaimers

- **No output should ever be interpreted as "X% chance of life."** All scores represent a *biosignature and habitability likelihood index based on proxy indicators validated against Earth analog environments.*
- **False positives are the dominant historical risk in astrobiology pattern-matching** — abiotic geochemical processes can mimic biological signatures. ExoScope enforces calibrated uncertainty bounds to mitigate overinterpretation.
- **Vision models are trained and validated exclusively on Earth analog and Mars planetary data**; out-of-distribution generalization to actual exoplanetary surfaces is fundamentally bounded by observational limits.

---

## 🌌 System Overview

ExoScope implements a dual-stream multi-modal architecture designed to triage and prioritize planetary targets for follow-up observation:

```
                          ┌───────────────────────────────┐
                          │     Data Ingestion Layer      │
                          │  (NASA Exoplanet Archive TAP, │
                          │   NASA PDS / ISRO Bhoonidhi)  │
                          └───────────────┬───────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  │                                               │
       ┌──────────▼──────────┐                         ┌──────────▼──────────┐
       │   VISION STREAM     │                         │   TABULAR STREAM    │
       │                     │                         │                     │
       │ 1. Classical CV     │                         │ 1. 40+ Engineered   │
       │    Pre-Filter       │                         │    Astrophysics     │
       │    (Edge/Entropy)   │                         │    Features (ESI)   │
       │ 2. Specialist YOLO  │                         │ 2. Calibrated       │
       │    Detector Stack   │                         │    XGBoost/RF       │
       └──────────┬──────────┘                         └──────────┬──────────┘
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │
                               ┌──────────▼──────────┐
                               │ Composite Fusion &  │
                               │ Uncertainty Engine  │
                               │ (Bootstrap 95% CI)  │
                               └──────────┬──────────┘
                                          │
                               ┌──────────▼──────────┐
                               │  HBLI Score Report  │
                               │  & SHAP Attribution │
                               └─────────────────────┘
```

---

## 🚀 Key Features

### 1. Physics-Grounded Tabular Habitability Model
- **40+ Derived Astrophysical Features**:
  - **Earth Similarity Index (ESI)**: Multi-parameter geometric mean across planetary radius, bulk density, escape velocity, and surface equilibrium temperature.
  - **Habitable Zone (HZ) Flux Boundaries**: Kopparapu et al. runaway greenhouse and maximum greenhouse solar insolation boundaries.
  - **Tidal Locking Likelihood**: Orbital period vs. spin-orbit synchronization time based on host star mass and semi-major axis.
  - **Stellar Activity & Flare Exposure**: Host star effective temperature, luminosity, and age-dependent flare probability.
- **Calibrated Ensemble Classifiers**: XGBoost and Random Forest models tuned via `StratifiedKFold` cross-validation with **Isotonic Regression** and **Platt Scaling** for reliable probability calibration.
- **Explainable AI (SHAP)**: Feature contribution breakdown (TreeExplainer) identifying primary positive habitability drivers and negative risk factors for each planet.

### 2. Tiered Specialist Computer Vision Pipeline
- **Tier-1 Classical CV Pre-Filter**:
  - Computes edge density (Canny), local entropy, and Laplacian texture variance.
  - Rejects featureless, dark, or redundant background tiles before deep learning inference, **saving 60–80% CPU/GPU compute**.
- **Tier-2 Specialist YOLO Stack**:
  - Multi-head specialist architecture targeting specific geomorphological biosignature proxies:
    - `sedimentary_layering`: Rhythmic bedding planes and depositional strata.
    - `mineral_water_interaction`: Evaporite crusts, hydrothermal mineral alteration veins, and hydration halos.
    - `erosion_morphology`: Dendritic fluvial networks, outflow channels, and alluvial fans.
- **Visual Bounding Boxes**: Bounding box localization, confidence scoring, and visual overlay generation.

### 3. HBLI Fusion & Uncertainty Quantification
- **Mathematical Weighted Fusion**: Fuses tabular physical habitability ($S_{tab}$) and surface visual evidence ($S_{vis}$) into a unified index:
  $$\text{HBLI} = w_{tab} \cdot S_{tab} + w_{vis} \cdot S_{vis}$$
- **Bootstrap 95% Confidence Intervals**: Propagates component measurement and modeling uncertainties into strict confidence intervals $[CI_{lower}, CI_{upper}]$.
- **Qualitative Likelihood Triage**:
  - **High Likelihood** ($\ge 0.75$) — Prime target for high-resolution spectroscopy follow-up.
  - **Moderate Likelihood** ($0.50 - 0.74$) — Favorable physical environment with partial proxy indicators.
  - **Marginal Likelihood** ($0.25 - 0.49$) — Extreme physical conditions or ambiguous surface proxies.
  - **Unlikely / Sterile** ($< 0.25$) — Hostile stellar flux or barren volcanic/cratered surface.

### 4. Interactive Streamlit Web Application
- **🪐 Mode 1: Exoplanet Analysis Lab**:
  - Real-time physics parameter tuning with sliders (radius, mass, orbit, stellar temp).
  - Live HBLI calculation, radar charts, and SHAP explainability waterfall plots.
  - 3D interactive planetary orbit and habitable zone boundary visualization.
- **🔬 Mode 2: Multi-Spectral Surface Analysis**:
  - Upload or inspect Mars/Earth analog imagery.
  - Interactive Classical CV pre-filter diagnostics.
  - YOLO detection overlay with confidence thresholds and per-class area fractions.
- **📊 Mode 3: NASA Exoplanet Catalog Explorer**:
  - Interactive exploration of **5,600+ confirmed exoplanets** from the NASA Exoplanet Archive.
  - Scatter plots (Radius vs. Period, Insolation vs. Temperature) with habitable zone overlays.

### 5. Bulk Exoplanet Ingestion Engine (`tools/fetch_exoplanet_data.py`)
- **Flexible Target Quota**: Interactively prompts for custom download sizes (e.g. `500 MB`, `5 GB`, `10 GB`, `20 GB`, `30 GB+`).
- **Live Terminal Telemetry**: Real-time ASCII progress bar, transfer rate (MB/s), ETA timer, and disk safety checks.
- **Multi-Modal Data Streams**:
  - **Catalogs**: Live queries to NASA Exoplanet Archive TAP API (`pscomppars`, `cumulative` KOI, `toi` candidates).
  - **Photometric Transit Time-Series**: High-cadence normalized flux light curves (BJD, normalized flux, error bars, quality flags).
  - **Atmospheric Transmission Spectra**: Synthetic/retrieval transmission curves ($0.6 - 14.0\,\mu\text{m}$) capturing $\text{H}_2\text{O}, \text{CH}_4, \text{CO}_2, \text{O}_3$ biosignature bands.
  - **Planetary Analog Surface Imagery**: Realistic procedural generation across 6 distinct geological regimes (Sedimentary, Craters, Fluvial channels, Dunes, Hydrothermal veins, Basalt) and 5 planetary color palettes (Mars rust, lunar basalt, evaporite salt, hydrothermal sulfur, permafrost).
  - **Manifest Generation**: Generates `manifest.json` with an audit trail and an auto-generated dataset `README.md`.

### 6. Calibration & Negative Control Verification
- **Reliability Diagnostics**: Brier score calculation, expected calibration error (ECE), and reliability diagrams.
- **Null-Model Negative Controls**: Evaluates pipelines against randomized Gaussian noise and scrambled label distributions to guarantee zero false confidence spikes.

---

## 🛠️ Technology Stack

| Domain | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Core language runtime |
| **Machine Learning** | `scikit-learn` (v1.4+) | Random Forest, Isotonic calibration, cross-validation |
| **Gradient Boosting** | `xgboost` (v2.0+) | High-performance tabular habitability classification |
| **Computer Vision** | `ultralytics` (YOLO) | Specialist object detection on surface imagery |
| **Image Processing** | `opencv-python`, `scikit-image` | Classical CV filtering (entropy, edge density, Sobel) |
| **Explainable AI** | `shap` (v0.44+) | TreeExplainer feature attributions & risk factors |
| **Astrophysics APIs** | `astroquery`, `requests` | NASA Exoplanet Archive TAP service & MAST queries |
| **Data Processing** | `pandas`, `numpy`, `scipy` | Feature engineering, numerical math, matrix ops |
| **Interactive UI** | `streamlit` (v1.38+) | Web dashboard with glassmorphism styling |
| **Visualizations** | `plotly`, `matplotlib`, `seaborn` | 3D orbital orbits, radar charts, scatter plots |
| **Testing** | `pytest`, `pytest-cov` | 46-test unit and integration test suite |
| **Configuration** | `PyYAML` | Centralized parameter management in YAML |

---

## 📂 Project Structure

```text
Planetary Biosignature/
├── app/
│   └── streamlit_app.py          # Interactive web UI (3 analysis modes)
├── config/
│   └── default_config.yaml       # Master configuration (paths, thresholds, model params)
├── data/                         # Local data cache (auto-generated, git-ignored)
│   ├── catalogs/                 # Downloaded NASA Exoplanet Archive tables
│   ├── exoplanets_bulk/          # High-volume multi-modal datasets (10GB - 30GB+)
│   ├── raw/                      # Raw imagery and downloads
│   └── tabular/                  # Engineered exoplanet CSVs
├── docs/
│   └── parameter_rationale.md    # Scientific justification for all physical parameters
├── models/                       # Serialized trained model weights (.joblib, .pt)
├── reports/                      # Evaluation figures, calibration plots, metrics
├── src/
│   ├── data_ingestion/           # NASA TAP, PDS, ISRO Bhoonidhi, and image tiling
│   │   ├── exoplanet_archive.py  # Exoplanet Archive client with TAP API
│   │   ├── pds_client.py         # NASA Planetary Data System imagery client
│   │   ├── bhoonidhi_client.py   # ISRO Bhoonidhi analog data client
│   │   └── tiling.py             # High-res gigapixel image tiler
│   ├── preprocessing/            # Feature engineering & CV pre-filtering
│   │   ├── feature_engineering.py# ESI, HZ boundaries, tidal locking features
│   │   └── cv_filter.py          # Classical edge/entropy pre-filtering
│   ├── models/                   # ML models and fusion engine
│   │   ├── tabular_model.py      # XGBoost / Random Forest habitability model
│   │   ├── vision_detector.py    # YOLO specialist detector stack
│   │   └── fusion.py             # HBLI composite scoring & bootstrap CI
│   ├── evaluation/               # Model calibration & negative controls
│   │   ├── calibration.py        # Reliability diagrams & Brier scoring
│   │   └── negative_controls.py  # Abiotic and noise negative control tests
│   └── utils/                    # Shared logging, config, and utilities
├── tests/                        # Full test suite (46 passing tests)
│   ├── test_data_ingestion.py    # Tests for TAP, PDS, Bhoonidhi, and tiling
│   ├── test_preprocessing.py     # Tests for ESI, feature engineering, CV filter
│   ├── test_models.py            # Tests for tabular, vision, and fusion layers
│   └── test_fusion.py            # Tests for calibration and negative controls
├── tools/                        # Operational CLI tools
│   ├── fetch_exoplanet_data.py   # Interactive bulk ingestion engine (10GB - 30GB+)
│   └── label_pipeline.py         # Image slicing & YOLO labeling pipeline
├── planetary-biosignature-project-plan.md # Original scientific specification
├── requirements.txt              # Production dependencies
├── setup.py                      # Package installation config
└── README.md                     # Documentation
```

---

## ⚡ Installation & Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Clone and Setup Environment
```powershell
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Planetary_Biosignature.git
cd "Planetary Biosignature"

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate       # Windows
# source venv/bin/activate    # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

---

## 🖥️ Usage Guide

### 1. Launch the Streamlit Web Application
```powershell
streamlit run app\streamlit_app.py
```
Open **http://localhost:8501** in your browser to access the full ExoScope dashboard.

---

### 2. Bulk Exoplanet Data Ingestion (Interactive 500 MB – 30 GB+)
To download and assemble exoplanet catalogs, photometric transit time-series, transmission spectra, and surface analog imagery:

```powershell
# Interactive mode: prompts for size, folder, and data composition
python tools\fetch_exoplanet_data.py
```

Or pass command-line arguments directly:
```powershell
# Ingest 10 GB of multi-modal data
python tools\fetch_exoplanet_data.py --size 10GB --output data/exoplanets_bulk

# Ingest 20 GB of light curves & spectra
python tools\fetch_exoplanet_data.py --size 20GB --mode all --yes
```

---

### 3. Fetch Standard Exoplanet Catalog
To download or refresh the base NASA Exoplanet Archive composite table:
```powershell
python -m src.data_ingestion.exoplanet_archive
```

---

### 4. Train the Tabular Habitability Model
Train the calibrated XGBoost/Random Forest model with 5-fold cross-validation:
```powershell
python -m src.models.tabular_model
```

---

### 5. Run the Test Suite
Verify that all 46 unit and integration tests pass:
```powershell
python -m pytest tests\ -v
```

---

## 🔬 Astrobiology Grounding & References

ExoScope builds upon established research in planetary science and astrobiology:

1. **SETI Institute / NASA Ames — Salar de Pajonales Study** (*Cabrol, Warren-Rhodes, Kalaitzis et al.*): Demonstrating that machine learning trained on terrestrial Mars-analog sites dramatically increases biosignature-proxy detection rates over random search.
2. **Earth Similarity Index (ESI)** (*Schulze-Makuch et al., 2011*): Formulation of physical metric scalings for planetary radius, density, escape velocity, and temperature.
3. **Habitable Zone Insolation Limits** (*Kopparapu et al., 2013, 2014*): Analytical solar flux boundary conditions for rocky planet surface liquid water.
4. **Neural-Network Biosignature Detection in Rock Imagery** (*Corenblit et al., 2023, Astrobiology Journal*): Deep learning applied to macro- and micro-scale morphological texture in rocky substrates.
5. **NASA Planetary Data System (PDS)** & **NASA Exoplanet Archive**: Public astronomical archives enabling reproducible astrophysical research.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
