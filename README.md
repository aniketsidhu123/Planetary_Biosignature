# ExoScope — Habitability & Biosignature-Likelihood Index (HBLI)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.ai/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v8%2Fv11-blueviolet.svg)](https://ultralytics.com/)
[![Tests](https://img.shields.io/badge/pytest-71%20passed-brightgreen.svg)](tests/)

> **A composite, explainable habitability and biosignature-likelihood scoring system** that fuses physics-grounded planetary parameters with computer-vision surface analysis — engineered to run efficiently on consumer hardware.

---

## ⚠️ Read this first

- **No output should ever be read as "X% chance of life."** Every score is a *habitability and biosignature likelihood index built from proxy indicators*, validated against Earth analog environments.
- **The training label is a rule, not an observation.** There is no ground truth for exoplanet habitability. The model is trained against published threshold criteria (rocky radius × Kopparapu habitable zone), so its accuracy measures *agreement with that rule* — never agreement with biology.
- **False positives are the dominant historical risk in astrobiology pattern-matching.** Abiotic geochemistry mimics biological signatures. ExoScope reports calibrated uncertainty bounds to blunt overinterpretation.
- **Vision models are trained on Earth-analog and Mars data only.** Generalization to real exoplanet surfaces is bounded by observational limits, and the shipped repository contains **no trained YOLO weights** — the detector stack runs in clearly-labelled simulation mode until you train and supply them.
- **ExoScope never silently substitutes fake data.** If the NASA archive is unreachable, it stops and says so. Generated development data is opt-in, written to a separate cache path, and flagged in the UI everywhere it appears.

---

## 🌌 System overview

```
                          ┌───────────────────────────────┐
                          │     Data Ingestion Layer      │
                          │  NASA Exoplanet Archive TAP   │
                          │  6,360 planets × 48 columns   │
                          └───────────────┬───────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  │                                               │
       ┌──────────▼──────────┐                         ┌──────────▼──────────┐
       │   TABULAR STREAM    │                         │   VISION STREAM     │
       │                     │                         │                     │
       │ 1. ~180 engineered  │                         │ 1. Classical CV     │
       │    features from    │                         │    pre-filter       │
       │    every NASA column│                         │    (edge/entropy)   │
       │ 2. Calibrated       │                         │ 2. Specialist YOLO  │
       │    XGBoost / RF     │                         │    detector stack   │
       └──────────┬──────────┘                         └──────────┬──────────┘
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │
                               ┌──────────▼──────────┐
                               │ Composite Fusion &  │
                               │ Uncertainty Engine  │
                               │  (weights adapt to  │
                               │  available streams) │
                               └──────────┬──────────┘
                                          │
                               ┌──────────▼──────────┐
                               │  HBLI Score Report  │
                               │  & SHAP Attribution │
                               └─────────────────────┘
```

---

## 🚀 Key features

### 1. Direct NASA Exoplanet Archive ingestion

- **Queries the TAP service over plain HTTP** — the full `pscomppars` composite table (one row per confirmed planet) arrives as CSV in a single ~7 s request. `astroquery` is supported as a fallback but is not required.
- **48 columns, not 16.** Planetary parameters, transit/RV geometry, stellar properties, system architecture, astrometry and discovery metadata — the whole set is fetched and fed downstream.
- **Provenance manifest.** Every cache write is accompanied by `<cache>.meta.json` recording source, row count, query and fetch time. The dashboard shows this in a banner so you always know whether you are looking at real observations.
- **Fails loudly.** A failed fetch raises `ExoplanetArchiveError`. Synthetic development data requires `allow_synthetic=True`, is written to a separate path, and is tagged as synthetic in the metadata.
- **Retries with backoff**, validated response parsing, `TOP n` row caps, and dtype coercion for the archive's sparse columns.

### 2. Physics-grounded feature engineering (~180 features)

| Family | What it contributes |
|---|---|
| **Habitable zone** | Kopparapu et al. (2014) effective-flux boundaries — recent Venus, runaway greenhouse, moist greenhouse, maximum greenhouse, early Mars — as AU distances, flux limits, a normalised position across the zone, and conservative/optimistic membership flags. |
| **Earth Similarity Index** | Schulze-Makuch et al. (2011), over radius, density, escape velocity and equilibrium temperature. |
| **Derived physics** | Surface gravity, escape velocity, orbital velocity, mass-radius residual against the rocky relation, rocky-composition likelihood, insolation swing across an eccentric orbit. |
| **Stellar context** | Spectral class (from `st_spectype` where available, else binned `st_teff`), metallicity, surface gravity, rotation period, and an activity proxy that uses rotation when the archive supplies it. |
| **System architecture** | Star/planet/moon counts, multiplicity flags, distance, parallax, proper motion, apparent magnitudes. |
| **Observational metadata** | Discovery method, facility and era, detection flags, and mass provenance — these encode real selection effects and how well-characterised a planet actually is. |
| **Missingness** | Explicit `<column>_missing` indicators. The archive is sparse, and "never measured" is genuine signal. |

Feature engineering **does not impute by default**. A median habitable-zone boundary is not a typical value — it is a *different star's* habitable zone, and substituting one turns "not measured" into a confident wrong answer in the UI. The model imputes at fit and predict time, where that policy belongs.

### 3. Calibrated habitability model over the NASA feature set

- **XGBoost or Random Forest**, tuned with `RandomizedSearchCV` on average precision (robust under the severe class imbalance — ~65 positives in 6,360 planets).
- **Held-out evaluation** on planets the model never saw, alongside cross-validation, with the confusion matrix and a warning when the positive count is too small for the metrics to be precise.
- **Leakage handling.** The columns that define the label rule are withheld by default, so the model must predict habitability from the *rest* of the archive. Residual leakage is unavoidable and stated plainly: the rule is a deterministic function of measured physics, so a model given semi-major axis and stellar luminosity can always rebuild the insolation criterion.
- **Platt (sigmoid) calibration by default.** Isotonic regression fits a step function that, with this few positives, collapses 6,360 planets into ~137 distinct scores — tying the whole top of the ranked list. Sigmoid keeps 6,300+ distinct scores at an equal or better Brier score.
- **Prediction intervals on the calibrated scale**, derived from the per-fold calibrated classifiers rather than the raw base learners (mixing those scales pins every upper bound to 1.0).
- **SHAP attribution** per prediction, plus importance grouped by data family.

### 4. Tiered specialist computer vision

- **Tier 1 — classical CV pre-filter**: Canny edge density, local Shannon entropy, and colour variance reject featureless or no-data tiles before any deep model runs.
- **Tier 2 — specialist YOLO stack**: one detector per geomorphological proxy class — `sedimentary_layering`, `mineral_water_interaction`, `erosion_morphology` — with bounding-box overlays and per-class confidences.
- Runs in **clearly-labelled simulation mode** when no trained weights are present, so the pipeline is demonstrable without pretending to evidential weight.

### 5. HBLI fusion with adaptive weighting

$$\text{HBLI} = w_{tab} \cdot S_{tab} + w_{vis} \cdot S_{vis}$$

Weights **renormalise over the streams that actually ran**. A tabular-only analysis previously capped at 0.55 no matter how Earth-like the planet — an unanalyzed stream was scored as zero, which reads as evidence *against* habitability rather than absence of evidence. A stream that ran and found nothing still counts against the score; a stream that never ran does not.

Bootstrap confidence intervals propagate component uncertainty, and the qualitative triage bands are:

| Band | Score | Meaning |
|---|---|---|
| **High Likelihood** | ≥ 0.75 | Prime target for follow-up spectroscopy |
| **Moderate Likelihood** | 0.50 – 0.74 | Favourable environment, partial proxy support |
| **Low Likelihood** | 0.25 – 0.49 | Extreme conditions or ambiguous proxies |
| **Very Low Likelihood** | < 0.25 | Hostile flux or barren surface |

### 6. Streamlit dashboard

- **Target Analysis** — score any planet, with SHAP attribution, a Kopparapu habitable-zone diagram, an Earth-similarity radar, and the planet's position in the population.
- **Catalog Explorer** — configurable population scatter, model-ranked target shortlist with CSV export, discovery history by method and facility, correlation matrix, and the raw NASA table.
- **Model Lab** — train, evaluate, and inspect: held-out metrics, a reliability diagram, feature importance by data family, and an explicit account of what the metrics do and do not mean.
- **Surface Imagery** — upload imagery and run tiling → pre-filter → detection → fused score.

---

## 📂 Project structure

```text
Planetary_Biosignature/
├── app/
│   ├── streamlit_app.py          # Dashboard entry point and page layout
│   ├── theme.py                  # Design tokens, stylesheet, UI primitives
│   ├── charts.py                 # Plotly figure builders
│   ├── state.py                  # Data loading, caching, provenance banner
│   └── views/
│       ├── model_lab.py          # Train / evaluate / inspect
│       └── image_analysis.py     # Vision pipeline view
├── config/
│   └── default_config.yaml       # Columns, thresholds, model and label rules
├── data/tabular/                 # Cached NASA table + .meta.json manifest
├── docs/
│   └── parameter_rationale.md    # Scientific justification for parameters
├── models/                       # Serialized weights (.joblib, .pt)
├── src/
│   ├── data_ingestion/
│   │   ├── exoplanet_archive.py  # NASA TAP client, provenance, validation
│   │   ├── pds_client.py         # NASA PDS imagery client
│   │   ├── bhoonidhi_client.py   # ISRO Bhoonidhi analog data client
│   │   └── tiling.py             # High-res image tiler
│   ├── preprocessing/
│   │   ├── feature_engineering.py# ESI, Kopparapu HZ, derived physics, encoding
│   │   └── classical_cv_filter.py# Edge/entropy/variance pre-filter
│   ├── models/
│   │   ├── tabular_model.py      # Calibrated XGBoost / RF + SHAP
│   │   ├── vision_detector.py    # YOLO specialist stack
│   │   └── fusion.py             # HBLI composite scoring
│   ├── evaluation/               # Calibration and negative controls
│   └── utils/                    # Config and logging
├── tests/                        # 71 unit and integration tests
├── tools/
│   ├── fetch_exoplanet_data.py   # Bulk multi-modal ingestion CLI
│   └── label_pipeline.py         # Image slicing & YOLO labeling
├── requirements.txt              # Runtime dependencies
└── requirements-optional.txt     # astroquery, labelImg, onnxruntime
```

---

## ⚡ Installation

```bash
git clone https://github.com/nishikadhankhar/Planetary_Biosignature.git
cd Planetary_Biosignature

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .\.venv\Scripts\Activate       # Windows

pip install -r requirements.txt
```

On macOS, XGBoost needs the OpenMP runtime. Without it ExoScope logs a warning and falls back to Random Forest:

```bash
brew install libomp
```

---

## 🖥️ Usage

### Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

Then open **http://localhost:8501**. The catalog loads automatically on first run.

### Fetch the NASA catalog from the command line

```bash
python -m src.data_ingestion.exoplanet_archive --refresh
```

Useful flags: `--max-rows N` to cap the download, `--output PATH` to choose a cache location, `--allow-synthetic` to generate development data when the archive is unreachable.

### Train the habitability model

```bash
python -m src.models.tabular_model
```

Prints cross-validation and held-out metrics, feature importances, and the top-ranked candidate planets.

### Inspect the engineered features

```bash
python -m src.preprocessing.feature_engineering
```

### Run the test suite

```bash
python -m pytest tests/ -v
```

---

## 🔬 Scientific grounding

1. **Habitable zone boundaries** — *Kopparapu et al. (2013, 2014)*: effective stellar flux limits for surface liquid water, valid for 2600–7200 K hosts. ExoScope extrapolates up to 300 K beyond that range and flags any row where it does — TRAPPIST-1 sits at 2566 K, and discarding it would wrongly place TRAPPIST-1 e/f/g outside the habitable zone.
2. **Earth Similarity Index** — *Schulze-Makuch et al. (2011)*: physical metric scalings for radius, density, escape velocity and temperature.
3. **Radius valley** — *Fulton et al. (2017)*: the ~1.8 R⊕ gap above which planets retain H/He envelopes, used as the conservative rocky-planet ceiling.
4. **Mass-radius relation** — *Chen & Kipping (2017)*, used for the composition residual feature.
5. **SETI Institute / NASA Ames — Salar de Pajonales** (*Cabrol, Warren-Rhodes, Kalaitzis et al.*): machine learning on terrestrial Mars-analog sites raises biosignature-proxy detection rates over random search.
6. **NASA Exoplanet Archive** — the `pscomppars` composite parameters table.

Sanity check: on the current catalog the pipeline places **177 planets in the conservative habitable zone** and **283 in the optimistic zone**, tracking the Planetary Habitability Laboratory catalogs, and independently reproduces the accepted result that TRAPPIST-1 e, f and g fall inside the conservative zone.

---

## 📄 License

MIT — see [LICENSE](LICENSE).
