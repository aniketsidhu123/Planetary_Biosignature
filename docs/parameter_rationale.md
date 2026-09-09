# Parameter Rationale & Literature Grounding
## ExoScope / HBLI v1.0

---

## 1. Vision Proxy Signal Classes

### 1.1 Sedimentary Layering Patterns
**What it detects:** Horizontally or sub-horizontally stratified rock/soil layers visible in orbital or rover imagery, indicative of depositional processes (lacustrine, fluvial, aeolian, or chemical precipitation).

**Why it matters for biosignatures:**
- On Earth, sedimentary layering in lacustrine/marine environments is strongly associated with microbial mat preservation (stromatolites, MISS — microbially induced sedimentary structures).
- Mars's Jezero Crater delta (Perseverance target) was selected precisely because its layered sedimentary structure indicates a past aqueous environment — the most favorable context for biosignature preservation (Farley et al., 2020, *Space Science Reviews*).
- The Salar de Pajonales study (Warren-Rhodes et al., 2023) demonstrated that layered evaporite-sediment interfaces were the highest-probability locations for detectable biosignatures.

**Detection approach:** Fine-grained edge detection for sub-parallel linear features, trained on HiRISE cutouts of known layered deposits (Mawrth Vallis, Jezero, Gale Crater Murray Formation).

---

### 1.2 Mineral-Water Interaction Zones
**What it detects:** Surface textures and color signatures consistent with evaporite deposits (gypsum, halite, jarosite), hydrated mineral formations, and chemical weathering patterns that require sustained liquid water.

**Why it matters for biosignatures:**
- Liquid water is the single strongest habitability prerequisite across all current astrobiology frameworks (Cockell et al., 2016, *Astrobiology*).
- Evaporite minerals are excellent biosignature-preservation media — halite inclusions on Earth routinely preserve viable microorganisms for 10⁴–10⁵ years, and potentially much longer (Schreder-Gomes et al., 2022, *Geology*).
- Mars CRISM data has identified widespread phyllosilicates and sulfates indicating past water-rock interaction (Ehlmann & Edwards, 2014, *Annual Review of Earth and Planetary Sciences*).
- The ESA ExoMars Rosalind Franklin rover's primary target selection criterion is subsurface access to clay-mineral-bearing units — directly analogous to what this detector seeks in surface imagery.

**Detection approach:** Color-band anomaly detection (spectral ratios if multi-band data available), texture classification of crystalline vs. amorphous surfaces, trained on terrestrial analog sites (Salar de Pajonales evaporites, Atacama gypsum fields, Pilbara hydrothermal cherts).

---

### 1.3 Erosion / Fluid-Flow Morphology
**What it detects:** Dendritic channel networks, streamlined islands, scour marks, alluvial fan morphology, and other landforms requiring fluid (water or potentially other solvent) transport.

**Why it matters for biosignatures:**
- Fluid-flow morphology is direct evidence of past or present liquid transport — the single most important habitability indicator after atmospheric composition.
- On Mars, the discovery of valley networks and outflow channels (Carr & Head, 2010, *Earth and Planetary Science Letters*) fundamentally changed habitability assessments from "unlikely" to "plausible past habitability."
- Perseverance's Jezero Crater was selected because orbital imagery revealed a clear delta morphology — fluid-flow features visible from orbit directly drove a $2.7B mission's landing site selection.
- On Titan, Cassini-Huygens imaged methane river channels and lakes — fluid-flow morphology in a non-water solvent, expanding the habitability parameter space.

**Detection approach:** Morphological segmentation of channel-like features (thin, branching, low-elevation relative to surroundings), trained on Mars valley network imagery (MOLA/HRSC DEMs, HiRISE) and terrestrial analogs (desert arroyos, dry lakebeds).

---

## 2. Tabular Features — Habitability Sub-Score

### 2.1 Feature Set (from NASA Exoplanet Archive)

| Feature | Column | Units | Relevance |
|---------|--------|-------|-----------|
| Planet radius | `pl_rade` | Earth radii | Size filtering — rocky planets (0.5–2.0 R⊕) vs. gas giants |
| Planet mass | `pl_bmasse` | Earth masses | Atmosphere retention capability |
| Planet density | `pl_dens` | g/cm³ | Composition proxy — rocky vs. volatile-rich |
| Orbital eccentricity | `pl_orbeccen` | — | Climate stability; high eccentricity → extreme temperature swings |
| Semi-major axis | `pl_orbsmax` | AU | Distance from star → insolation level |
| Insolation flux | `pl_insol` | Earth flux | Direct energy input; habitable zone proxy |
| Equilibrium temperature | `pl_eqt` | K | Surface temperature estimate (no atmosphere) |
| Stellar effective temp | `st_teff` | K | Star type → UV environment, HZ width |
| Stellar luminosity | `st_lum` | log(L☉) | Energy output driving planetary climate |
| Stellar mass | `st_mass` | M☉ | Main-sequence lifetime → time for life to evolve |

### 2.2 Derived Features

| Derived Feature | Formula / Logic | Reference |
|----------------|-----------------|-----------|
| **Earth Similarity Index (ESI)** | Geometric mean of sub-indices for radius, density, escape velocity, surface temperature, each weighted by empirically determined exponents | Schulze-Makuch et al. (2011), *Astrobiology* |
| **Habitable Zone Distance Ratio** | `pl_orbsmax / sqrt(10^st_lum)` — ratio of actual orbital distance to estimated HZ center | Kopparapu et al. (2013), *ApJ* |
| **Tidal Lock Likelihood** | Proxy based on `pl_orbsmax` and `st_mass` — planets too close to low-mass stars are likely tidally locked, reducing habitability | Barnes (2017), *Celestial Mechanics* |
| **Stellar Activity Proxy** | Function of `st_teff` and `st_age` — younger, cooler stars tend to be more active (flares), hostile to surface life | Shields et al. (2016), *Physics Reports* |

### 2.3 ESI Calculation

The Earth Similarity Index is computed as:

```
ESI = ∏ᵢ (1 - |xᵢ - xᵢ₀| / (xᵢ + xᵢ₀))^(wᵢ/n)
```

Where:
- `xᵢ` = planet's value for parameter i
- `xᵢ₀` = Earth's reference value for parameter i
- `wᵢ` = weight exponent for parameter i
- `n` = number of parameters

Reference values and exponents (Schulze-Makuch et al., 2011):
- Radius: x₀ = 1.0 R⊕, w = 0.57
- Density: x₀ = 5.51 g/cm³, w = 1.07
- Escape velocity: x₀ = 1.0 (Earth units), w = 0.70
- Surface temperature: x₀ = 288 K, w = 5.58

---

## 3. Precedent Studies

### 3.1 Salar de Pajonales (Warren-Rhodes, Cabrol, Kalaitzis et al.)
- **What:** AI/ML system trained on drone + orbital imagery of a Mars-analog salt flat in Chile's Atacama Desert.
- **Result:** 87.5% biosignature-detection rate vs. <10% for random search; reduced search area by ~97%.
- **Relevance:** Direct precedent for the vision pipeline — demonstrates that ML on orbital-resolution imagery can meaningfully guide biosignature search on Earth analogs.

### 3.2 NASA Exoplanet Archive Classification
- **What:** Random Forest / XGBoost classifiers trained on ~12 physical parameters to classify exoplanets as habitable/non-habitable.
- **Relevance:** Direct precedent for the tabular pipeline — published baselines exist for benchmarking.

### 3.3 Habitable Worlds Observatory Prep Work
- **What:** Bayesian CNNs and transformer models predicting biosignature-gas flux from reflected-light spectra, with explicit uncertainty quantification.
- **Relevance:** Template for how to present fused scores — always with calibrated confidence bounds.

### 3.4 Corenblit et al. (2023)
- **What:** Neural-network analysis of rock imagery for biosignature detection patterns.
- **Relevance:** Direct precedent for texture-based biosignature proxy detection from imagery.

---

## 4. Key Negative Controls

To prevent the historical failure mode of over-detection, the system must be tested against known-abiotic targets:
- **Lunar surface** — no atmosphere, no water, no biology ever → should score near 0.
- **Mercury surface** — similar to Moon, extreme temperatures → should score near 0.
- **Known abiotic geological formations** — e.g., purely volcanic basalt plains, impact melt sheets → should not trigger biosignature proxies.
- **Gas giant atmospheres** — fundamentally different environment → tabular model should score very low.
