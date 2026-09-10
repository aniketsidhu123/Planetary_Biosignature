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

### 2.1 Feature Set (from the NASA Exoplanet Archive)

All 48 columns below are pulled from the `pscomppars` composite table and
fed to the feature engineering layer.

**Planetary parameters**

| Feature | Column | Units | Relevance |
|---------|--------|-------|-----------|
| Planet radius | `pl_rade` | R⊕ | Size filtering — rocky planets vs. gas giants |
| Planet mass | `pl_bmasse` | M⊕ | Atmosphere retention capability |
| Mass provenance | `pl_bmassprov` | — | Whether the mass is a true mass or an `Msini` lower bound |
| Planet density | `pl_dens` | g/cm³ | Composition proxy — rocky vs. volatile-rich |
| Orbital period | `pl_orbper` | days | Orbital dynamics; pairs with mass for stability |
| Semi-major axis | `pl_orbsmax` | AU | Distance from star → insolation level |
| Eccentricity | `pl_orbeccen` | — | Climate stability; high e → extreme temperature swings |
| Insolation flux | `pl_insol` | S⊕ | Direct energy input; habitable zone criterion |
| Equilibrium temperature | `pl_eqt` | K | Surface temperature estimate (airless) |

**Transit / RV geometry**

| Feature | Column | Relevance |
|---------|--------|-----------|
| Transit depth, duration | `pl_trandep`, `pl_trandur` | Follow-up feasibility for atmospheric characterisation |
| a/R★, Rp/R★, impact parameter | `pl_ratdor`, `pl_ratror`, `pl_imppar` | Geometry constraining the orbit and radius measurement |
| RV semi-amplitude | `pl_rvamp` | Mass measurement quality |

**Stellar parameters**

| Feature | Column | Relevance |
|---------|--------|-----------|
| Spectral type | `st_spectype` | Direct host classification where the archive supplies it |
| Effective temperature | `st_teff` | Star type → UV environment, HZ width and position |
| Radius, mass | `st_rad`, `st_mass` | Luminosity, main-sequence lifetime → time for life to evolve |
| Luminosity | `st_lum` | Energy output driving planetary climate |
| Metallicity | `st_met`, `st_metratio` | Planet-formation efficiency; correlates with rocky planet occurrence |
| Surface gravity, density | `st_logg`, `st_dens` | Evolutionary state — main sequence vs. evolved |
| Age | `st_age` | Time available for life; activity declines with age |
| Rotation | `st_rotp`, `st_vsin` | Strongest single predictor of magnetic activity and flare rate |

**System architecture, astrometry and discovery metadata**

| Feature | Column | Relevance |
|---------|--------|-----------|
| Star / planet / moon counts | `sy_snum`, `sy_pnum`, `sy_mnum` | Multiple stars destabilise habitable-zone orbits |
| Distance, parallax, proper motion | `sy_dist`, `sy_plx`, `sy_pm` | Whether the target is reachable for characterisation |
| Magnitudes | `sy_vmag`, `sy_kmag`, `sy_gaiamag` | Host brightness → achievable spectroscopic SNR |
| Coordinates | `ra`, `dec` | Sky position |
| Discovery method / year / facility | `discoverymethod`, `disc_year`, `disc_facility` | Encodes observational selection effects, and how well characterised a planet is |
| Detection flags | `tran_flag`, `rv_flag`, `ttv_flag` | A transit **and** RV detection gives both radius and true mass, so bulk density is genuinely constrained |
| Controversial flag | `pl_controv_flag` | Contested detections are excluded from positive labels |

### 2.2 Derived Features

| Derived Feature | Formula / Logic | Reference |
|----------------|-----------------|-----------|
| **Earth Similarity Index (ESI)** | Weighted geometric mean of sub-indices for radius, density, escape velocity and equilibrium temperature | Schulze-Makuch et al. (2011), *Astrobiology* |
| **Kopparapu HZ boundaries** | `S_eff = S_eff☉ + aT + bT² + cT³ + dT⁴` with `T = Teff − 5780 K`, then `d = √(L / S_eff)`, for five boundaries: recent Venus, runaway greenhouse, moist greenhouse, maximum greenhouse, early Mars | Kopparapu et al. (2014), *ApJL* |
| **HZ position** | `(log S_inner − log S) / (log S_inner − log S_outer)` across the conservative zone: 0 = inner edge, 1 = outer edge, outside [0,1] = outside the HZ | derived |
| **HZ distance ratio** | `pl_orbsmax / √(10^st_lum)` — the simple interpretable complement to the full treatment above | Kopparapu et al. (2013), *ApJ* |
| **Surface gravity, escape velocity** | `g = M/R²`, `v_esc = √(M/R)`, both in Earth units | derived |
| **Mass-radius residual** | `log₁₀(M / R^3.7)` — positive means denser than a rocky planet of that size, negative means volatile-rich | Chen & Kipping (2017), *ApJ* |
| **Insolation swing** | `((1+e)/(1−e))²` — the periastron-to-apastron flux ratio over an eccentric orbit | derived |
| **Tidal Lock Likelihood** | Sigmoid over `pl_orbsmax / √st_mass` — planets close to low-mass stars are likely locked, reducing habitability | Barnes (2017), *Celestial Mechanics* |
| **Stellar Activity Proxy** | Weighted blend of `st_teff`, `st_age` and `st_rotp` — young, cool, fast-rotating stars flare frequently | Shields et al. (2016), *Physics Reports* |
| **Transit SNR proxy** | `log₁₀(transit depth) − 0.2 × Ks magnitude` — bright host plus deep transit means characterisable | derived |
| **Missingness indicators** | `<column>_missing` per sparse column — in an archive this sparse, "never measured" is real signal | derived |

### 2.3 ESI Calculation

```
ESI = ∏ᵢ (1 - |xᵢ - xᵢ₀| / (xᵢ + xᵢ₀))^(wᵢ/n)
```

Where `xᵢ` is the planet's value, `xᵢ₀` Earth's reference value, `wᵢ` the
weight exponent, and `n` the number of available parameters.

Reference values and exponents (Schulze-Makuch et al., 2011):
- Radius: x₀ = 1.0 R⊕, w = 0.57
- Density: x₀ = 5.51 g/cm³, w = 1.07
- Escape velocity: x₀ = 1.0 (Earth units), w = 0.70
- Equilibrium temperature: x₀ = **255 K**, w = 5.58

**Note on the temperature reference.** The original formulation uses Earth's
*surface* temperature of 288 K. The archive supplies `pl_eqt`, an
*equilibrium* temperature that excludes greenhouse warming, so the Earth
reference must be Earth's equilibrium temperature of 255 K for the
comparison to be like-for-like. Using 288 K against `pl_eqt` systematically
penalises genuinely Earth-like planets.

### 2.4 Habitable Zone Validity Range

The Kopparapu polynomial fit is calibrated for **2600 K ≤ Teff ≤ 7200 K**.
ExoScope evaluates hosts up to 300 K beyond that band by clamping to the
nearest calibrated temperature, and flags every such row in
`hz_extrapolated`. Beyond the tolerance no habitable zone is computed at all
and the columns stay null rather than being filled with a plausible-looking
number.

This tolerance is not cosmetic. TRAPPIST-1 has Teff = 2566 K — 34 K below
the floor. Refusing to solve for it would report TRAPPIST-1 e, f and g as
outside the habitable zone, contradicting the published result. With the
tolerance applied, the computed conservative zone spans 0.025–0.050 AU and
correctly contains exactly those three planets.

### 2.5 Habitability Label Definition

There is no ground truth for exoplanet habitability, so the model's training
label is a rule assembled from published thresholds. Two variants are
available in `config/default_config.yaml`:

| Rule | Radius range | Habitable zone | Count on the current catalog |
|------|--------------|----------------|------------------------------|
| **Conservative** | 0.5 – 1.8 R⊕ | Runaway greenhouse → maximum greenhouse | ~21 |
| **Optimistic** (default) | 0.5 – 2.5 R⊕ | Recent Venus → early Mars | ~65 |

The 1.8 R⊕ conservative ceiling is the radius valley identified by
**Fulton et al. (2017)**, above which planets predominantly retain H/He
envelopes and lack a surface. Both variants additionally exclude planets
that are near-certainly tidally locked, hosts that are extremely active, and
detections flagged as controversial.

These counts track the Planetary Habitability Laboratory's conservative and
optimistic catalogs, which is a useful independent check on the physics.

**Consequences for interpretation.** Because the label is a deterministic
function of measured physical quantities, any model given those quantities
re-derives the rule rather than learning anything. ExoScope therefore
withholds the label-defining columns from the feature set by default, so the
model must predict the criteria from the rest of the archive. Residual
leakage remains unavoidable — semi-major axis and stellar luminosity
reconstruct insolation — so reported metrics should be read as *agreement
with the published rule*, not as evidence about biology. What the model
genuinely contributes is calibrated ranking across the whole catalog,
including the many planets whose defining measurements are missing.

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
