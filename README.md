# ExoScope — Habitability & Biosignature-Likelihood Index (HBLI)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A composite, explainable habitability and biosignature-likelihood scoring system** that fuses vision-based surface analysis with physics-grounded tabular modeling — optimized to run on consumer hardware.

---

## ⚠️ Important Disclaimers

- **No output should ever be interpreted as "X% chance of life."** All scores represent *biosignature/habitability likelihood based on proxy indicators validated against Earth analogs.*
- False positives are the dominant historical risk in astrobiology pattern-matching — this system reports calibrated uncertainty to mitigate overinterpretation.
- Vision models are trained/validated exclusively on Earth analog imagery; out-of-distribution performance on actual Mars/exoplanet data is fundamentally uncertain.

---

## What It Does

1. **Ingests** public NASA/ISRO planetary data — imagery, spectra, and tabular parameters.
2. **Vision Pipeline**: Runs a stack of specialist YOLO detectors to identify surface/geological features associated with biosignature proxies (sedimentary layering, mineral-water interaction zones, erosion/fluid-flow morphology).
3. **Tabular Pipeline**: Runs RandomForest/XGBoost on physical parameters (radius, density, orbital eccentricity, stellar temperature, etc.) to compute a physics-grounded habitability sub-score.
4. **Fusion Layer**: Combines both pipelines into a single **HBLI score** with confidence intervals and human-readable feature attributions.
5. **Runs efficiently** on a consumer laptop (CPU-first, optional GPU acceleration).

---

## Architecture

```
┌──────────────────────────┐
│   Data Ingestion Layer   │
│ (PDS / Bhoonidhi /       │
│  Exoplanet Archive)      │
└────────────┬─────────────┘
             │
   ┌─────────┴──────────┐
   │                     │
┌──▼──────────┐   ┌──────▼──────────┐
│ IMAGE       │   │ TABULAR/SPECTRAL│
│ PIPELINE    │   │ PIPELINE        │
│             │   │                 │
│ 1. CV       │   │ RF / XGBoost   │
│    pre-     │   │ on physical    │
│    filter   │   │ params (ESI)   │
│ 2. YOLO     │   └────────┬───────┘
│    stack    │            │
└──────┬──────┘            │
       └────────┬──────────┘
                │
     ┌──────────▼──────────┐
     │  Fusion / Scoring   │
     │  (weighted, with    │
     │  confidence band)   │
     └──────────┬──────────┘
                │
     ┌──────────▼──────────┐
     │  HBLI Score +       │
     │  Explainability     │
     │  Report             │
     └─────────────────────┘
```

---

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd "Planetary Biosignature"

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Streamlit Web UI
```bash
streamlit run app/streamlit_app.py
```

### CLI — Fetch Exoplanet Data
```bash
python -m src.data_ingestion.exoplanet_archive
```

### CLI — Run Tabular Model
```bash
python -m src.models.tabular_model --input data/tabular/exoplanets.csv
```

---

## Data Sources

| Source | Type | Access |
|--------|------|--------|
| [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu) | Tabular (5800+ exoplanets) | `astroquery` API |
| [NASA PDS](https://pds.nasa.gov) | HiRISE / Mastcam imagery | HTTP download |
| [ISRO Bhoonidhi](https://bhoonidhi.nrsc.gov.in) | Earth observation (Mars analogs) | `bhoonidhi-downloader` |

---

## Prior Art & References

- SETI Institute / NASA Ames — Salar de Pajonales biosignature-AI study (Cabrol, Warren-Rhodes, Kalaitzis et al.)
- NASA Exoplanet Archive habitability classification (RF/XGBoost baselines)
- Habitable Worlds Observatory prep work (Bayesian CNNs with uncertainty quantification)
- Corenblit et al. (2023, Astrobiology Journal) — neural-network biosignature detection in rock imagery

---

## License

MIT License — see [LICENSE](LICENSE) for details.
