# Planetary Habitability & Biosignature-Likelihood Estimator
### Project Plan v1.0

---

## 0. Read This First: Reframing the Goal

Your instinct — multi-parameter analysis, per-parameter specialist models, efficiency on consumer hardware — is sound engineering. The one thing to adjust is the *output claim*. No current system (NASA, ISRO, ESA, SETI) scans images/video and outputs "% chance of life." What exists is:

- **Habitability scoring** from planetary/orbital/atmospheric parameters (tabular data) — e.g. the Earth Similarity Index, ML classifiers on the NASA Exoplanet Archive.
- **Biosignature-*pattern* detection in imagery** — trained on **known Earth biology** (microbial mats, pigment patterns, mineral-biology associations in Mars-analog sites like Chile's Salar de Pajonales), used to flag *where a rover or telescope should look next* — not to confirm life exists.
- **Spectral biosignature-gas prediction** from atmospheric spectra (JWST-style), for mission target prioritization.

All three are **triage/prioritization tools with calibrated uncertainty**, not detectors. This project should be framed the same way: a **Habitability & Biosignature-Likelihood Index (HBLI)** — a composite, explainable score with confidence bounds, built from real published parameters and imagery, validated against known ground truth on Earth. This is honest, it's genuinely useful (mirrors real mission-prioritization workflows), and it's a much stronger portfolio/research piece than an unfalsifiable "life %" number.

---

## 1. Project Vision (Reframed)

**Name suggestion:** *ExoScope* or *HBLI — Habitability & Biosignature-Likelihood Index*

**What it does:**
1. Ingests public NASA/ISRO planetary data — imagery, spectra, and tabular parameters.
2. Runs a **stack of specialist vision models** (your YOLO idea) to detect specific surface/geological/textural features associated with habitability or biosignature proxies (not "life" directly — e.g. mineral-water interaction patterns, layered sediment structures, potential microbial-mat-like textures, erosion/fluid-flow morphology).
3. Runs a **tabular/spectral model** on published planetary parameters (orbital, atmospheric composition, stellar flux, density) to compute a physics-grounded habitability sub-score.
4. **Fuses** both into a single composite index with an uncertainty band, plus a human-readable breakdown ("scored high on X, low on Y, driven mainly by Z").
5. Runs efficiently on a consumer laptop (CPU or low-end GPU), via small quantized models and a tiered classical-CV pre-filter.

**What it explicitly does NOT claim:** confirmed detection of life. Every output should carry the framing "likelihood/similarity score based on proxy indicators observed on Earth" — because that's what the underlying science actually supports, and because overclaiming here is a real, well-documented failure mode in astrobiology (many "biosignature-like" patterns turn out to be abiotic).

---

## 2. Real-World Precedents (so you can cite prior art)

- **SETI Institute / NASA Ames — Salar de Pajonales study** (Cabrol, Warren-Rhodes, Kalaitzis et al.): AI/ML trained on drone + orbital imagery of a Mars-analog site achieved an 87.5% biosignature-detection rate vs <10% for random search, cutting search area by ~97%. This is your closest real precedent for the *vision* half of the project.
- **NASA Exoplanet Archive habitability classification**: Random Forest / XGBoost models classify exoplanets as habitable/non-habitable from ~12 physical parameters (orbital period, density, eccentricity, stellar temperature, etc.), building on the Earth Similarity Index. This is your closest precedent for the *tabular* half.
- **Habitable Worlds Observatory prep work**: Bayesian CNNs and transformer models predicting biosignature-gas flux from reflected-light spectra, used purely for observation-time prioritization, with explicit uncertainty quantification — a good template for how to present your fused score.
- **Corenblit et al. (2023, Astrobiology Journal)**: neural-network analysis of rock imagery for biosignature detection — direct precedent for the "scan images for biosignature-adjacent texture" piece.

None of these systems output a single "% life" number. They all output **ranked, uncertainty-bounded likelihood scores used for prioritization** — that's the credible target for your project too.

---

## 3. Data Sources (verified, currently accessible)

**NASA:**
- **NASA Planetary Data System (PDS)** — pds.nasa.gov — the core archive: imaging (Cartography & Imaging Sciences node), geosciences, atmospheres. Recently redesigned; new releases weekly (e.g. Mars Reconnaissance Orbiter imagery drops regularly).
- **NASA Planetary Data Ecosystem (PDE)** — science.data.nasa.gov — unified hub linking PDS + Astromaterials Data System + Science Discovery Engine search.
- **NASA Exoplanet Archive** — tabular parameters for ~5,800+ confirmed exoplanets (radius, mass, density, orbit, stellar properties) — this is your tabular-model training source.
- **Mars mission imagery**: HiRISE (Mars Reconnaissance Orbiter), Curiosity/Perseverance Mastcam/Mastcam-Z — all archived in PDS Imaging node.
- **Kepler/TESS light curves** — for a stretch-goal transit-detection module.

**ISRO:**
- **Bhoonidhi** (bhoonidhi.nrsc.gov.in) — ISRO/NRSC's open data portal: ResourceSat, CartoSat, EOS, NISAR (joint NASA-ISRO radar mission, launched July 2025), plus mirrored Landsat-8/Sentinel data. Has a public API (contact NRSC for API access) and an open-source Python SDK/CLI (`bhoonidhi-downloader` on PyPI) for programmatic search and download by bounding box, date, satellite/sensor.
- **ISRO Science Data Archive (ISDA)** — planetary science mission data (Chandrayaan, Mangalyaan).
- Note the Bhoonidhi EULA: free-to-use "open data" scenes can be published/built on with attribution ("ISRO-IRS"); priced/on-order scenes require going through the portal; no commercial resale of raw data.

**Practical note:** you don't need agency-scale data volume. A curated few hundred–few thousand labeled image tiles per feature class, plus the full (small) tabular exoplanet archive, is enough for a working v1 on a laptop.

---

## 4. System Architecture

```
                    ┌─────────────────────────┐
                    │   Data Ingestion Layer   │
                    │ (PDS / PDE / Bhoonidhi   │
                    │  / Exoplanet Archive)     │
                    └───────────┬──────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │                                     │
    ┌─────────▼─────────┐               ┌───────────▼───────────┐
    │  IMAGE PIPELINE     │               │  TABULAR/SPECTRAL     │
    │                     │               │  PIPELINE              │
    │ 1. Classical-CV     │               │  RandomForest/XGBoost  │
    │    pre-filter        │              │  on physical params    │
    │    (cheap, rules out │              │  (ESI-style features)  │
    │    empty/irrelevant  │              └───────────┬────────────┘
    │    tiles)             │                          │
    │ 2. Stack of small     │                          │
    │    YOLO detectors      │                          │
    │    (per-parameter:     │                          │
    │    sediment layering,  │                          │
    │    mineral-water        │                          │
    │    interaction, erosion │                          │
    │    morphology, texture   │                          │
    │    anomalies)             │                          │
    └─────────┬───────────┘                          │
              │                                        │
              └───────────────┬────────────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │   Fusion / Scoring     │
                    │   Layer (weighted,      │
                    │   explainable, with      │
                    │   confidence interval)   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  HBLI Score + Report   │
                    │  (per-target, ranked,  │
                    │   with feature          │
                    │   attributions)          │
                    └────────────────────────┘
```

---

## 5. Model Stack

**Vision — stack of specialist detectors, per your original idea:**
- Base architecture: **YOLO26** (Ultralytics, released Jan 2026) — purpose-built as an edge-first detector, NMS-free inference, ~43% faster on CPU than the previous generation, exports cleanly to ONNX/TFLite/OpenVINO. This is a better fit than older YOLO versions for your consumer-hardware constraint.
- Train/fine-tune **one small model per parameter class** rather than one large multi-class model — matches your "specialized per parameter" idea and keeps each model tiny (YOLO26-n scale, a few MB).
- Candidate parameter classes to start with (grounded in real geobiology signal classes, not sci-fi ones): sedimentary layering patterns, mineral-water interaction zones (evaporite/gypsum-type formations), erosion/fluid-flow morphology, surface texture anomalies vs. surrounding terrain, color/pigment-band anomalies (proxy for pigmented microbial communities in the Salar de Pajonales precedent).
- Use a **cheap classical-CV pre-filter** (edge detection / simple texture statistics) before running any YOLO model, so you only run deep models on tiles that pass a basic relevance threshold — this is where most of your compute savings will actually come from, more than model choice alone.

**Tabular/spectral — habitability sub-score:**
- **Random Forest or XGBoost** on physical parameters (radius, density, eccentricity, stellar temperature/luminosity, orbital semi-major axis, insolation) — mirrors the published ESI-based classification approach, is lightweight, interpretable, and trains in seconds on a laptop.
- Optional stretch goal: a small CNN/1D-transformer on spectral data if you later add JWST-style spectra — but this is heavier and can be a v2 feature, not a v1 requirement.

**Fusion layer:**
- Simple weighted composite (like the published "ATA score" approach) rather than a black-box combiner — this keeps the output explainable, which matters enormously for a "how likely is life" claim.
- Output a **score + confidence interval + top contributing factors**, never a bare percentage.

---

## 6. Hardware Efficiency Strategy (your core differentiator)

Be precise about the claim here: you won't beat agency infrastructure on **raw throughput** (they process petabytes across a mission's lifetime) — but you can absolutely build something dramatically more **efficient per-analysis-unit**, which is the practically useful framing for a laptop-based tool.

Concrete techniques:
1. **Model size**: use YOLO26's smallest variants (n/s), fine-tuned, not trained from scratch — transfer learning cuts both training and inference compute drastically.
2. **Quantization**: export to ONNX Runtime or TFLite with INT8 post-training quantization — typically 2-4x inference speedup, ~4x memory reduction, small accuracy cost.
3. **Tiered pipeline**: classical CV pre-filter → cheap → only escalate promising tiles to YOLO models → only escalate high-scoring tiles to the (heavier) fusion/explanation step. Most images never reach the expensive stage.
4. **Tiling + lazy loading**: process large PDS images in tiles, stream from disk, never load a full multi-GB image into memory at once.
5. **Batch processing on CPU with ONNX Runtime** (has strong CPU kernels) rather than assuming GPU availability — makes the tool actually usable on a normal laptop, not just a gaming laptop.
6. **Dataset curation over dataset scale**: a well-curated few-hundred-image training set per class will outperform a huge noisy one and keeps training itself laptop-feasible.

---

## 7. Validation Strategy (the part most similar projects skip — don't skip it)

You cannot validate "is there life" against ground truth on Mars or an exoplanet — nobody has that ground truth. So validate what you *can*:
- **Vision models**: validate against Earth analog sites with known outcomes — e.g. the published Salar de Pajonales dataset/results, or other terrestrial biosignature imagery, where "biosignature present / absent" is independently confirmed.
- **Tabular model**: validate against the NASA Exoplanet Archive's own labeled habitable/non-habitable classifications and cross-check against the published ESI-based benchmark papers (Random Forest/XGBoost baselines already exist to compare against).
- **Report calibration, not just accuracy**: for a score like this, a well-calibrated confidence interval matters more than raw accuracy — show it's honest about what it doesn't know.
- **Negative controls**: explicitly test known-abiotic, known-lifeless targets (e.g. the Moon, known sterile geological formations) to check the system doesn't over-trigger — false positives are the historical failure mode in this field.

---

## 8. Build Roadmap

| Phase | Focus | Deliverable |
|---|---|---|
| 0 | Literature review + finalize parameter list | Short doc listing chosen proxy signals, cited to real astrobiology/remote-sensing papers |
| 1 | Data pipeline | Scripts to pull/cache PDS, Bhoonidhi, and Exoplanet Archive data; tiling utility |
| 2 | Tabular habitability model | Trained RF/XGBoost model + ESI-style benchmark comparison |
| 3 | Vision stack v1 | 2-3 fine-tuned YOLO26-n specialist detectors on curated Earth-analog imagery |
| 4 | Fusion + explainability layer | Weighted scorer with confidence interval + feature attribution |
| 5 | Hardware optimization pass | ONNX/INT8 export, tiered pipeline, benchmark on a real consumer laptop |
| 6 | Validation | Run against Salar de Pajonales-style ground truth + negative controls; publish calibration report |
| 7 | UI/packaging | Simple Streamlit/Gradio app; batch CLI for processing PDS/Bhoonidhi image sets |

---

## 9. Suggested Tech Stack

- **Vision**: Ultralytics YOLO26 (PyTorch), OpenCV for classical pre-filtering
- **Tabular ML**: scikit-learn, XGBoost
- **Optimization/export**: ONNX Runtime, TensorFlow Lite
- **Data access**: `bhoonidhi-downloader` (PyPI) for ISRO data, `astroquery` for NASA Exoplanet Archive, direct PDS4 tooling for NASA imagery
- **UI**: Streamlit or Gradio (fast to build, laptop-friendly)
- **Experiment tracking**: simple MLflow or even just structured logging, given laptop scale

---

## 10. Risks & Honest Limitations (state these in your own docs/README)

- No output should ever be phrased as "X% chance of life" — always "biosignature/habitability *likelihood* score based on proxy indicators, validated against Earth analogs."
- False positives are the dominant historical risk in this field — pattern-matching on Earth biology can flag abiotic look-alikes.
- The vision models can only be trained/validated on Earth analog imagery — there's no ground truth for other planets, so out-of-distribution performance on real Mars/exoplanet data is fundamentally uncertain and should be reported as such.
- "Reduced hardware usage vs. space agencies" is true in the sense of efficient per-analysis compute on curated data — it's not a claim of matching agency-scale throughput or scientific authority.

---

## 11. Key Resources

- NASA Planetary Data System: https://pds.nasa.gov
- NASA Planetary Data Ecosystem: https://science.data.nasa.gov
- NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu
- ISRO Bhoonidhi portal: https://bhoonidhi.nrsc.gov.in
- `bhoonidhi-downloader` (PyPI): programmatic ISRO data access
- Ultralytics YOLO26 docs: https://docs.ultralytics.com
- SETI Institute / NASA Ames Salar de Pajonales biosignature-AI study (Cabrol, Warren-Rhodes, Kalaitzis et al.)
- NASA Astrobiology "AI Astrobiology: Life Detection & Biosignatures" resource page: https://www.nasa.gov/a-i-astrobiology-life-detection-biosignatures/
