# Query-Anywhere InSAR Displacement Decomposition — Complete Reference

**Full thesis project documentation**. Use this to brief Claude Code, your team, or your own memory between coding sessions.

---

## Table of Contents

1. [Project Objective & Deliverable](#1-project-objective--deliverable)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Core Design Decisions (with Rationale)](#3-core-design-decisions-with-rationale)
4. [Data Acquisition & Sampling Strategy](#4-data-acquisition--sampling-strategy)
5. [Model Architecture](#5-model-architecture)
6. [Uncertainty Quantification](#6-uncertainty-quantification)
7. [Training Procedure](#7-training-procedure)
8. [Validation & Baseline](#8-validation--baseline)
9. [Implementation Stack](#9-implementation-stack)
10. [Known Issues & Caveats](#10-known-issues--caveats)
11. [Thesis Positioning & Novelty](#11-thesis-positioning--novelty)
12. [Coding Conventions](#12-coding-conventions)

---

## 1. Project Objective & Deliverable

**Given any user-specified lat/lon in Europe, return:**
- **East–West** ground-displacement time series with calibrated uncertainty (aleatoric + epistemic)
- **Vertical** ground-displacement time series with calibrated uncertainty (aleatoric + epistemic)
- **Time coverage**: full available EGMS period (typically 2015–present), at ~6–12-day temporal resolution

**Learned from:**
- EGMS L2b LOS (ascending + descending orbits), at PS (persistent scatterer) locations
- Globally transferable open predictors (AlphaEarth embeddings + DEM + hydrogeology)
- A deliberately diverse European panel of Sentinel-1 coverage

**No training labels needed.** The model is self-supervised on LOS interpolation; GNSS is reserved for final validation only.

---

## 2. Pipeline Overview

### Two decoupled stages

```
Stage 1: LEARNED SPATIAL INTERPOLATION (your contribution)
  Input: query lat/lon, surrounding LOS measurements (asc + desc), environmental context
  Method: Graph-attention network per orbit, inductive kriging
  Output: predicted LOS time series (+ uncertainty) at query point, per orbit

Stage 2: CLASSICAL GEOMETRIC DECOMPOSITION (closed-form)
  Input: LOS_asc & LOS_desc at query point, viewing geometry at query (incidence + heading)
  Method: Linear 2×2 trigonometric inversion
  Output: EW & vertical time series (uncertainty propagated linearly)
```

**Why this split?**
- Isolates the hard, learned part (interpolation) from the solved part (geometry)
- Makes the baseline a fair swap (same decomposition, different interpolator)
- Keeps the model from learning something it doesn't need to (projection geometry)
- Lets uncertainty propagate cleanly in closed form through both stages

---

## 3. Core Design Decisions (with Rationale)

### 3.1 Interpolate-then-decompose, not learned decomposition

**Decision:** Predict LOS first, *then* apply classical 2×2 decomposition.

**Rejected alternative:** Train the network to directly output EW/vertical from LOS + geometry.

**Why:** The classical inversion is exact, stable, and well-understood. Asking a neural net to learn this mapping wastes capacity and makes it harder to reason about what the model learned vs. what it got from geometry. The baseline can use the same decomposition, so any performance difference is purely interpolation quality. Also, LOS prediction is a larger, more generic task (useful even if you later deploy with a different decomposition or for a different application), whereas learned EW/vertical is narrowly fit to this one pipeline.

### 3.2 Inductive spatial kriging via masked-target training

**Decision:** During training, **randomly mask which points are "known" vs "target" every step**, rebuilding the neighbourhood graph each time.

**Why:** This is the one non-obvious mechanism that makes "query anywhere" actually work. If you train on a fixed graph with fixed targets, an expressive GNN can quietly memorize that specific spatial arrangement. At inference, you query a new neighbourhood layout and the model fails to generalize. Masked-target training forces the network to learn *a transferable function* — "given this local neighbourhood configuration, predict the centre" — rather than "when I see this exact set of points, the answer is…". This idea comes from IGNNK (Inductive Graph Neural Network Kriging, Wu et al. AAAI 2021) and is the reason the model transfers across Europe.

**Practical implication:** Your training loop doesn't just load a static batch of graphs. Each epoch, for each burst, you (a) pick a random subset of points as "known", (b) build the graph from the known set, (c) hold out the rest and predict them. Masking rate typically ~20–30% holdout.

### 3.3 Position is relative-only; never feed absolute coordinates

**Decision:** Target-to-neighbour distance and direction (unit vectors) go on **edges**. No absolute lat/lon on any node.

**Why:** Absolute coordinates are a memorization pathway. A model that sees absolute position can learn "region X has motion Y" and overfit to geography. The moment you query a new area, it generalizes poorly. Relative geometry is translation-invariant — your model learns a *local* function that doesn't care whether it's applied in Spain or Poland. This is essential for "globally transferable."

**Checked alternative:** Fourier encodings or positional embeddings of *relative* offset. Not necessary; target-centring on edges already achieves it.

> ⚠️ This ban covers node/edge features only. Learned predictors (AlphaEarth) may re-introduce absolute position implicitly — see §10.9.

### 3.4 Time lives in node features, not graph structure

**Decision:** Each node's LOS time series is **encoded to a fixed-length vector** (node feature). Temporal edges between adjacent acquisitions were considered and rejected.

**Rejected alternative:** Build a spatiotemporal graph with nodes = (location, time) and edges both spatial (between locations) and temporal (between consecutive epochs). Reference: the ST-GAT InSAR imputation paper.

**Why:** Spatiotemporal edges are powerful for *gap-filling in a fixed area*, but they cost a lot and buy you little for query-anywhere. Your EGMS series are essentially complete (no large temporal gaps), and deformation evolves near-synchronously across a subsidence bowl. A full spatiotemporal graph blows up to `k·T` nodes per sample (k neighbours, T epochs), multiplied by ensemble size and subgraph resampling every training step — that's 10–100× more computation for a benefit you don't need. Also, spatiotemporal papers typically feed absolute coordinates as node features (for temporal edge messaging), which would kill your transfer claim. Temporal structure enters *inside* the encoder, not the graph.

### 3.5 Temporal basis coefficients (not raw time series)

**Decision:** Fit each LOS series to a small basis (linear trend + annual/semiannual sinusoids + ~3 cubic-spline knots) → `p ≈ 8–15` coefficients. The network operates in coefficient space.

**Rejected alternative A:** Feed the raw `T`-length series as a node feature, run a 1D-CNN temporal encoder.

**Rejected alternative B:** Predict on a fixed common epoch grid for all points.

**Why this is better (three payoffs):**
1. **Dimensionality:** Instead of hundreds of raw time steps, you have ~10 coefficients. Smaller, faster, less noisy.
2. **Asc/desc date mismatch dissolves.** Ascending and descending orbits acquire on different dates. If you force both onto one epoch grid via interpolation, you're guessing what happened on dates one orbit didn't observe. With basis coefficients, fit each orbit's series on its *own* native epochs, then reconstruct both onto a common grid via `series = Φ @ coeffs`. Clean, no guessing.
3. **Uncertainty propagates linearly.** The heteroscedastic head predicts coefficient mean and variance; reconstruction and decomposition are both linear, so aleatoric and epistemic variances pass through unchanged. Your final EW/vertical intervals are traceable all the way back to the learned uncertainty.

**How to fit the basis:** Use `scipy.optimize.lstsq` or `numpy.linalg.lstsq` with a design matrix `Φ (T_raw × p)` built from the basis functions. Fit once per point in preprocessing, cache the coefficients.

### 3.6 L2b input (not L2a)

**Decision:** Use EGMS **L2b (calibrated)** LOS, not L2a (basic).

**Why:** L2a is relative to a local reference per burst/processing unit. Asc and desc have *different* reference frames. Inverting a 2×2 on LOS from two different datums contaminates both the EW and vertical outputs with an arbitrary offset/ramp. L2b is tied to a GNSS-calibrated model, so asc and desc share a compatible reference. The decomposition output is much more trustworthy.

### 3.6a Current prototype scope (interim simplifications)

**Status: active working scope, not a permanent architecture change.** To get a first end-to-end pipeline running, several of the above mechanisms are deliberately narrowed. The fuller versions they defer are kept everywhere else in this document, not deleted — the prototype choices below are a strict subset of them, chosen so the later, fuller version reuses the same code rather than requiring a rewrite.

- **Target: EGMS's own `mean_velocity` (mean LOS velocity over the full 2019–2023 window), not the full time series or §3.5's basis coefficients.** Decided 2026-10-20; this replaces the earlier single-year self-fit. It is already a column in the preprocessed csvs, so no velocity-fitting step exists anywhere in the pipeline, and it spans the same window as the GNSS validation trends (2019.0–2024.0), so the comparison is like for like.
- **Covariate: AlphaEarth embedding; which year(s) is open (§10.16).** With a 5-year target, a single year's embedding no longer matches the target window by construction. Options: one representative year, or an embedding pooled over 2019–2023 (§5.3's `AnnualEncoder` idea becomes relevant again).
- **Node features, simplified accordingly:** `[velocity (1), is_query (1), incidence_angle (1), dist_to_coherent (1), encoded GSE static vector]` — no `p`-dimensional basis coefficients. Output head is scalar `(mu, logvar)` instead of `ℝ^p`.
- **Uncertainty propagation (§6.3/6.4), scalar version:** no basis-reconstruction step (`Φ @ coeffs`) needed, since there's no coefficient vector. The asc/desc decomposition (§6.4) applies the same `G⁻¹` once to `(mu_v, var_v)` per orbit, instead of looping over `T` epochs.
- **Deferred, revisit once this works end-to-end:** full multi-year time series target; reprojection loss (§7.2 — parked, not on the near-term roadmap); MSE-warmup before switching to NLL (§7.4 — pending literature justification, see §10.12); GSE per-channel normalization and the compression-MLP choice (§5.4 — both flagged for supervisor discussion, not finalized, see §10.13).

### 3.7 Viewing geometry is derived analytically, never learned

**Decision:** Build the LOS unit vector at a query point from the per-point `incidence_angle` + `track_angle` that every provider ships, via a low-order (planar) fit per **(track, sub-swath)**. Do *not* add the look vector as a GNN output, and do not depend on provider-specific metadata files.

For right-looking SAR (Sentinel-1 and effectively all civilian systems):
```
los_up    = cos(incidence)
look_az   = track_angle - 90                    # azimuth of the ground→satellite direction
los_east  = sin(incidence) * sin(look_az)
los_north = sin(incidence) * cos(look_az)
```
Verified against EGMS's own `los_*` columns: agreement to 6e-4. The asc/desc sign flip in `los_east` (negative ≈ 259° vs positive ≈ 100° look azimuth) is what makes `G` well-conditioned.

**Why analytic, not learned:**
- **It's exact and sparsity-proof.** Measured over 800k points of a full burst, incidence is planar to a 0.019° residual (gradient 0.067°/km). A per-track plane pools the whole scene, so local point density is irrelevant — a query in a sparse region still resolves the look vector to ~0.02%. Even naively copying the nearest point's vector from 10 km away costs only 0.75%. Sparsity that would wreck *displacement* interpolation barely registers here, because geometry is a smooth deterministic function of orbit position while displacement is short-scale and noisy.
- **It keeps uncertainty propagation closed-form.** §6.4's `G⁻¹ Σ G⁻ᵀ` is linear *because `G` is known exactly*. A predicted `G` carries its own variance, the propagation stops being linear, and the aleatoric/epistemic split no longer survives to the EW/Up intervals.
- **It preserves the §3.1 separation** of the learned part from the solved part, keeping the baseline comparison clean.
- Cost was *not* the reason — extra head outputs are negligible. Three independently predicted components would also violate the unit-norm constraint, requiring a renormalization step anyway.

**Practical rules:** interpolate the Cartesian components `(los_east, los_north, los_up)`, never the angles (`track_angle ≈ 349°` sits next to the 0/360 wrap), then renormalize. Never fit or interpolate across different tracks or sub-swaths — IW1/IW2/IW3 occupy distinct incidence bands (~29–36°, 34–41°, 39–46°) with genuine discontinuities at their boundaries.

---

## 4. Data Acquisition & Sampling Strategy

### 4.1 Data sources

| Source | Use | Format | Notes |
|---|---|---|---|
| EGMS L2b LOS | Training | CSV per burst (lat, lon, time series) | Both orbits; separate files |
| AlphaEarth Foundations embeddings (GSE) | **Sole covariate source, for now** | Analysis-ready 64-dim annual rasters, int8-quantised | Google Cloud Storage, anonymous HTTPS, "provider-pays" egress, CC-BY 4.0. Downloaded **cropped to each study area's achieved bbox via windowed COG reads**, not full tiles — a full tile is ~4.3 GB (8192×8192 px × 64 bands), a typical region's window is ~200 MB. De-quantise with `x = ((v/127.5)**2) * np.sign(v)` (unit-length output, verified). |
| ESA WorldCover (v200, 2021) | **Sampling stratification only — not a model covariate.** Used for §4.3 Step 3's land-cover-stratified target sampling. | 10 m GeoTIFF, EPSG:4326, tiled per 3°×3° | Also a proper COG (confirmed: internally tiled, 6-level overview pyramid) — same windowed-read approach as AlphaEarth applies, and the savings are proportionally similar (~95 MB full tile vs. ~2 MB for a typical region's window). |
| DEM, Hydrogeology / Aquifer maps | **Deprioritised for now** | — | Originally planned as predictors (see git history for the original table). Not currently being pursued; AlphaEarth is the only covariate source until/unless this is revisited. |
| Viewing geometry (incidence, track angle) | Decomposition | Per-point columns in the LOS input itself | **No separate raster/metadata dependency** — derived analytically per (track, sub-swath), see §3.7 |
| GNSS (EPN / EUREF / national CORS, via Nevada Geodetic Laboratory) | Validation only | `.tenv3` time series, EU-fixed (plate-fixed Eurasia) frame | Never training. Reserve validation regions with GNSS. Use the `IGS20/tenv3/EU/` tree, not `IGS14` — both resolve but are different reference-frame realisations; IGS20 is current. |

### 4.2 Coordinate system & distances

**Decision (2026-10-20): every model-facing distance and bearing (angular selection, k-NN, edge features, `dist_to_coherent`) is computed geodesically from latitude/longitude** (`pyproj.Geod(ellps="WGS84").inv`, as in the tutorial notebook), not in EPSG:3035. This keeps the pipeline identical inside and outside Europe, which the transferability claim of the thesis depends on (see below and §10.7). The EPSG:3035 columns (`easting`/`northing`) stay in the preprocessed csvs and may still be used for purely internal geometry that never reaches the model: AOI clip, edge buffer, spatial indexing.

**Resolved (Section 10.7): the CRS choice itself doesn't matter for global transfer.** Because the architecture only ever consumes *relative* distance + unit-direction on edges (Section 3.3) — never absolute position — there is no need for one continent-spanning projected CRS. What matters is only that each pairwise distance/bearing is computed with a method that's metrically valid for that specific pair of points, e.g. geodesic distance/azimuth (`pyproj.Geod`, CRS-agnostic, valid anywhere) or a per-query local azimuthal-equidistant projection centred on the target. Outside Europe, swap EPSG:3035 for one of those; no architecture change needed. Raster predictors (DEM/AEF/hydrogeology, Section 4.1) still need *some* locally-appropriate grid per region, but that's a data-engineering detail, not a modelling constraint.

### 4.3 Sampling strategy: handpicked + random with stratification

**Step 1: Handpicked known-process regions (occupy buffer, reserve test subset).**

Choose one representative region per major deformation process; hold *at least one* instance per process as test-only. These are your anchor points for demonstrating process coverage and transferability.

**Recommended handpicked regions:**

| Process | Region | Justification | Rates | GNSS |
|---|---|---|---|---|
| Groundwater | Alto Guadalentín (Lorca), Murcia, Spain | Fastest in Europe; 20+ yrs Sentinel-1 + DGPS campaigns | 10–13 cm/yr | 33 stations |
| Mining + rebound | Upper Silesian Coal Basin, Poland | Active mining (subsidence to 43 cm/yr) + post-mine flooding (uplift); EGMS-studied | ±10–40 cm/yr | Corner reflectors, GNSS campaigns |
| Peat (low-coherence) | Green Heart, western Netherlands | Oxidation+compaction; severe decorrelation — stress test for uncertainty | ~1 cm/yr | National GNSS; sparse coherence |
| Coastal/delta | Po Delta / Ravenna / Venice, Italy | Mixed drivers (compaction, GIA, fluids); 30 yrs data; internal rate contrast | −30 to +5 mm/yr | PODELNET (multiple stations) |
| Groundwater (optional 2nd) | Firenze–Prato–Pistoia, Tuscany, Italy | LOS→EW/Vert decomposition precedent; different geology from Spain | ~0.5–2 cm/yr | Regional GNSS |
| Stable baseline (optional) | Fennoscandian Shield (e.g., central Sweden) | Slow GIA uplift, high coherence; prevents "only subsidence" bias | +0.5–2 mm/yr | IGS continuous |

**Reserve some handpicked regions entirely for test** (never train on them). This directly tests: "does the model generalize to an unseen mining area / unseen peat region / etc.?"

**Step 2: Min-distance random draw for the background.**

Over the rest of Europe, place **random burst-anchor points** with a **minimum separation** equal to or greater than your **neighbourhood radius** (see Section 4.4). This ensures training targets from different anchors don't share neighbour points, maintaining sample independence.

**Step 3: Within-burst candidate pool construction (angular-balanced k-NN, sequential, no point reuse).**

Runs per burst csv (one orbit direction's file), train-split regions only — val is excluded entirely, since Tier 2 validation (§8.1) queries the model directly at GNSS station locations, not through a cached training-style pool; dev/cal may get their own pools later for their own separate uses, not as part of this step. Consumes the already quality-filtered EGMS output from preprocessing (`rmse`/`temporal_coherence`/`amplitude_dispersion`/`mp_type` thresholds) — filtering happens upstream, once, not redone here; this step should never need to re-examine a point it later has to discard for quality reasons.

Within each selected burst csv, build the pool of mini-graph anchors as follows — this supersedes the simpler uniform-radius/min-distance idea originally sketched for this step (see §4.4 for what that radius concept is still used for):

0. **Clip to the actual area of interest first**, separately from step 2's edge buffer below. EGMS downloads the *entire winning relative orbit* (§3.7/download design), which is routinely far larger than the original query bbox/`large_bbox` — especially for the handpicked real-subsidence regions, where most of a burst's million+ points can sit well outside the area actually of interest. Skipping this clip means wasting candidate-pool slots (and anchor budget, point 1 below) on irrelevant points. Distinct purpose and distinct (much larger) scale from step 2's edge buffer — conceptually two different clips, even if implemented in one function.
1. **Cap the number of seeds drawn from this burst.** Fully consuming every available point would yield dozens to hundreds of millions of training examples pooled across all bursts. The budget is the capacity formula plus the region weighting, see §4.9 and §4.10.
2. Exclude a buffer margin along the burst's own (post-step-0) boundary (`buffer_m`, 1000 m) so every seed has points available in every direction. Margin points can still be pool neighbours; only seeds are restricted to the centre.
3. **Seed selection (separate step, §4.9):** grid share + land-cover stratified share with a global minimum distance. Land cover is used here only to pick a diverse set of seeds. It is **not stored with the pools** and never a model feature; GSE embeddings remain the only covariate (§4.1). Land cover is sampled again later at the query points of the val regions, to evaluate in which land-cover types the model performs best (§8.4). The seed step returns the burst table with a `seed` flag column (1 = seed).
4. **Pool construction (separate step, after seed selection, decided 2026-10-21):** go through the seeds (random order as the neutral default; earlier seeds get the better neighbours). For each seed, select its neighbours by **angular-balanced selection** (split the neighbourhood into angular sectors, probably 8, take the nearest point(s) per sector) from points that are **not seeds** and **not already assigned to an earlier seed** (no point reuse between pools). Every selected point gets the seed's `pid` in a new column `seed_pid`; the seed carries its own pid. One pool = all rows with the same `seed_pid`. The pool is built burst by burst, so pids only have to be unique within a burst, which they are.
5. **Prototype rule (decided 2026-10-21, kept simple until the whole pipeline runs end to end):** angular-balanced selection of the **25 nearest** free points within a **maximum distance of 2 km** from the seed; a pool with **fewer than 7 points** is rejected. The more elaborate pool construction and the sparsification augmentation stay open (§10.11). `k_nearest` (25) is the working pool size the capacity formula assumes. Graphs don't need a fixed size (PyG batches variable-size graphs natively). **The minimum pool size must leave room for the training-time dropout (noted 2026-10-21):** the number of points a pool keeps after the per-epoch sparsification (random subset, exclusion radius around the query, dropped sectors; §7.1, §10.11) must never fall below the minimum graph size (at least the query plus k = 3 neighbours, better a few more). So the pool minimum has to be higher than the graph minimum, roughly `min_pool ≥ min_graph / smallest kept fraction` plus a margin for the exclusion radius. Example: a minimum graph of 5 and subsets down to 60% already need pools of ≥ 9 points; with an exclusion radius removing further points near the query, more. In addition, the sampler itself should never drop below `min_graph` (clamp the drop count per draw), so a single unlucky draw cannot produce a degenerate graph.
6. On acceptance the seed's pool is fixed; its points are unavailable for all later seeds.

This replaces reserving a conservative worst-case min-distance around every anchor (which wastes points in dense areas) with directly tracking which points are already claimed — denser sampling, more training examples per burst for the same budget, same no-shared-points-between-graphs guarantee.

**Sparsity augmentation — open, not yet decided.** The variable-graph-size support above is also what a future "simulate sparser regions" augmentation (dropping neighbours, possibly with its own point-level min-distance) would need. Tracked as open, see §10.11.

**Pipeline staging note:** the actual stratified *sampling* happens later than Steps 1–2, at graph-generation/training-sampling time — not during the raw EGMS acquisition step (`01_download_data.py` only downloads full-burst products for each anchor; it doesn't yet pick individual target points). The land-cover *data* itself is now downloaded during acquisition, though (`download_esa_worldcover`, cropped per study area via windowed COG reads) — so by the time Step 3 is actually implemented, the WorldCover tiles it needs will already be sitting in each region's folder. Don't read Step 3's absence from the acquisition script as a gap in that script — only the sampling logic is scoped for later, not the data itself.

**Why this hybrid?** Handpicking ensures rare-but-important processes (mining, peat) are represented and held out for testing. Random draw with stratification ensures broad coverage and fair representation of conditions. Together, you sample the "condition space," not just the "area space."

### 4.4 Neighbourhood radius and burst-edge buffer

Compute the typical radius a k-NN search reaches, as before:
```python
radii = [distance from target to k-th neighbour, for each training point]
neighbourhood_radius = np.percentile(radii, 95)  # conservative upper bound
```

**Superseded use:** this radius was originally meant to also set a global `min_distance ≥ 1.5×` between anchors at the point level (§4.3 Step 3). That use is now superseded by Step 3's sequential, no-reuse candidate pool construction, which tracks shared points directly instead of reserving a conservative worst-case radius around every anchor — denser sampling, same independence guarantee.

**Remaining use:** sizing the burst-edge buffer (§4.3 Step 3.2) — the margin near a burst's boundary within which no anchor may land, so every accepted anchor gets angular-balanced coverage in every direction.

Step 2's region-level min-distance (between randomly sampled *regions* across Europe, before any single burst is even downloaded) is a different, coarser pipeline stage and is unaffected by this.

### 4.5 Training / calibration / test split

**Entire bursts or entire regions**, never random points. Assign bursts/regions to disjoint groups:
- **Train:** ~50–60% of area/bursts → compute normalization stats, train model
- **Calibration:** ~15–20% → unused during training; used for conformal quantile (Section 6.4)
- **Test:** ~20–30% → held-out; measure final performance

The split is **geographic**: train blocks are spatially far from test blocks, so a test point's neighbours are in train, not in test. This prevents "leakage" where a test sample's ground truth is hidden from the model but its neighbourhood is known.

**Current implementation uses four groups, not three** — `01_download_data.py`'s `set` column is train / dev / cal / val (0/1/2/3), mapping onto this section as: `cal` = this section's **Calibration**, `val` = this section's **Test**. `dev` is an addition beyond this original plan — a fourth, separately held-out group, kept deliberately distinct from `cal` so that whatever `dev` ends up used for (e.g. iteration/sanity-checking during pipeline development) can never leak into the conformal-calibration guarantee the way it would if the two were merged (see §6.5's train/calibrate/test independence argument — the same reasoning applies a level up, to the splits themselves). `val`/test is drawn only from the handpicked real-subsidence regions, never the random background points, since GNSS validation (§8.1 Tier 2) needs a real named region with actual nearby GNSS stations, not an arbitrary anchor.

### 4.6 Global normalization (on train split only)

Compute standardization statistics (mean, std) for all features **across the entire train split**, treating all bursts as one pool:
- LOS coefficients: mean/std per coefficient
- Predictors (AEF, DEM, hydro, etc.): mean/std per channel
- Distances: mean/std

Apply these same scalers uniformly at inference. **Never compute stats on test or cal data.**

### 4.7 Handling EGMS L2b CSV structure

EGMS L2b CSVs actually have columns like (verified against real downloaded files, not the product spec):
```
pid, mp_type, latitude, longitude, easting, northing, height, height_wgs84, line, pixel,
rmse, temporal_coherence, amplitude_dispersion, incidence_angle, track_angle,
los_east, los_north, los_up, mean_velocity, mean_velocity_std, acceleration, acceleration_std,
seasonality, seasonality_std, 20190101, 20190107, 20190113, ..., 20231224
```
Date columns are plain 8-digit `YYYYMMDD` strings, **not** prefixed with `t_`. `rmse`/`temporal_coherence`/`amplitude_dispersion`/`mp_type` are the per-point quality fields to filter/flag on before basis-fitting (§3.5); `incidence_angle`/`track_angle`/`los_*` are the §3.7 viewing-geometry columns, already present, no separate raster needed.

**Loading:**
```python
import pandas as pd, numpy as np
df = pd.read_csv("burst_id.csv")
date_cols = [c for c in df.columns if c.isdigit() and len(c) == 8]
epochs_raw = [pd.to_datetime(c, format="%Y%m%d") for c in date_cols]
los = df[date_cols].values  # (N_points, T_raw), cumulative displacement in mm
xy = np.c_[df["longitude"], df["latitude"]]  # or use easting/northing (EPSG:3035) directly, already provided
```

Then project to EPSG:3035 and fit the temporal basis.

### 4.8 Overlap between neighbouring bursts (measured 2026-10-21, judged negligible)

**Decision: the overlap between neighbouring bursts is not handled (no de-duplication across bursts).** Measured with `test/burst_overlap.py` over all 1,091 downloaded bursts (convex hull of each preprocessed csv, EPSG:3035, pairs within the same region and orbit direction; tables in `test/figs/burst_overlap/`).

| Pair type | pairs | overlap depth, km (p10 / median / p90) | share of a burst's outer edge inside the other burst (median) |
|---|---|---|---|
| same orbit, along-track neighbour (next burst, same subswath) | 871 | 1.26 / 1.62 / 2.25 | 41% |
| same orbit, across-track (adjacent subswath) | 399 | 1.80 / 2.06 / 2.59 | 7% |
| same orbit, diagonal | 381 | 1.66 / 2.01 / 2.55 | 5% |
| different relative orbit | 191 | 2.2 / 11.5 / 15.9 | 34% |

- "Depth" is the short side of the minimum rotated rectangle around the overlap, i.e. how far one burst reaches into its neighbour. Along-track overlaps are long strips (median ~84 km, ~8% of a burst's area); across-track overlaps are short (~13 km, <2% of the area). Ascending and descending behave the same (along-track median 1.58 / 1.65 km, across-track 2.05 / 2.07 km).
- **Edge margin (§4.3 Step 3.2, `buffer_m` = 1000 m):** the strips are only slightly deeper than the margin. For bursts with a same-orbit neighbour (1,087 of 1,091), a median of 41% of the margin zone (p10 16%, p90 65%) is covered by the centre of a neighbour.
- **Are the overlapping points the same?** Only partly. Pids differ per burst (0% match), so pids cannot be used to find duplicates. Matched by nearest coordinates in 12 random same-orbit pairs: 6 pairs had 94–100% of the points within 1 m of a partner (velocity difference 0.0 mm/yr); the other 6 had only 5–17% within 1 m, median nearest distance 4–14 m, velocity differences 0.4–3.1 mm/yr (different pixel grid, so different measurements). The identical cases were mostly along-track neighbours of the same subswath.
- **Why this is not a concern:** the 1000 m margin already keeps seeds out of most of the overlap strips, and the remaining variability across bursts and regions is large. A seed pair or pool in the overlap is a negligible share of the data. Removing duplicates would need a distance tolerance (~5–15 m) and would not remove truly identical measurements reliably, so it is not worth the complexity.


### 4.9 Seed selection within a burst (implemented in s03, tested 2026-10-20/21)

A **seed** is the starting point of one candidate pool, so one seed = one training graph (§7.1). Seed selection per accepted burst, as implemented in `s03` (`select_area_and_buffer_burst_edges`, `find_n_seed_points`, `select_center_nodes`):

1. **Overlap and rejection.** Burst hull ∩ AOI (EPSG:3035). A burst is rejected if the overlap is < 4 km² (`MIN_AREA`) or holds < 10,000 points (`MIN_PTS`).
2. **Edge margin.** Points within `buffer_m` (default 1000 m) of the overlap's edge are flagged `in_center = 0`; they can be pool neighbours but never seeds. `n_center` = points outside the margin.
3. **Capacity.** `n_seeds = (n_center // (POOL_FACTOR * k_nearest)) * density`. `POOL_FACTOR` (so far hardcoded as 4, to become a parameter) reserves room so that every seed can find its neighbours without point reuse. Working values: `k_nearest` = 25, `density` = 0.66 (must be in (0, 1]).
4. **Placement** (`select_center_nodes`): 2/3 of the seeds spread over a 2 × 2 km grid with the same quota per occupied cell (random within the cell), 1/3 stratified over six WorldCover classes (Built-up, Grassland, Tree cover, Shrubland, Bare/sparse, Cropland) with weight = class size ^ `strat_exponent` (0.5; 0 = equal per class, 1 = proportional). In both phases every seed must keep `min_dist_m` (one global value, default 100 m) to all seeds chosen so far (spatial hash). Shortfalls are not redistributed; intended vs real counts are appended to the s03 log.

**Test results** (`test/seed_selection_test.py`, 15 accepted train bursts, k 25, density 0.66 unless stated; share of intended seeds that survive the minimum distance):

| min distance | density | mean surviving | worst burst |
|---|---|---|---|
| 0 m | 0.66 | 99.4% | 97.9% |
| 50 m | 0.66 | 98.4% | 94.2% |
| **100 m** | 0.66 | 97.2% | 89.2% |
| 200 m | 0.66 | 89.8% | 37.7% |
| 300 m | 0.66 | 82.4% | 17.4% |
| 100 m | 1.00 | 95.4% | 78.8% |
| 100 m | 0.30 | 98.9% | 95.0% |

Losses come from dense bursts: a 2 × 2 km cell holds at most ~(2000 / min_dist)² seeds. Equal shares per land-cover class (exponent 0) lose most in the stratified phase (rare classes run out). Choice: 100 m, density ≤ 0.66, exponent 0.5–1.0.

**Land cover of EGMS points** (`test/landcover_distribution.py`, 10 random bursts, 2.65 M points): Built-up 44.6%, Grassland 22.0%, Tree cover 17.2%, Shrubland 6.3%, Bare 4.8%, Cropland 1.9%, Moss/lichen 1.9%, Water 0.9%, Snow 0.4%, Wetland 0.04%. Cropland, water and wetland are nearly absent, which is why the stratification weights small classes up but not to equal shares.

**Full train split, per-burst capacity only** (`test/seed_total_estimate.py`, all 470 train bursts, min distance 100 m, buffer 1000 m): 178 bursts (37.9%) rejected (124 for both area and points, 53 for points only, 1 for area only; 31% of all train points). The 292 accepted bursts give 576,177 intended and **538,790 real seeds** (93.5%; dense bursts 92%, others 96%). Very unequal: train_9 and train_13 alone give 58% of all graphs, the 9 handpicked regions only 20%, the median accepted burst 526 seeds. This motivated the region weighting in §4.10.

### 4.10 Region weighting and the share of real deformation (decided 2026-10-21)

**Goal.** The handpicked real-subsidence (real-sub) regions must carry enough weight. Target split about 50/50 between real-sub and random train regions, rather more real-sub than less; not strict. Uplift counts as well: what matters is that the training data is not dominated by flat background.

**Schema** (to be implemented when s03 is reworked):
1. Train regions only (`set == 0`); random regions are the `train_N` directories, every other train region is real-sub. No extra flag needed.
2. Real-sub regions are processed first: every accepted burst gets its capacity (§4.9, from `n_center`, i.e. edge margin excluded). Their sum is `S`.
3. Random regions get the target `R = RATIO * S` (RATIO 1.0 = 50/50, adjustable). `R` is spread by **water-filling**: the same level per random region, capped at the region's capacity, the unused remainder passed on to regions that still have room. `R` is never exceeded; small regions simply give what they have. Equal shares per region are not possible because many small regions cannot hold enough data.
4. Within a region, its allocation is split over the bursts in proportion to their `n_center`.

Rejected alternatives: per-burst capacity only (two random regions dominate), an equal budget per region in points instead of seeds (unit error: the capacity always won, nothing changed), equal budget per region without redistribution (total halves), proportional allocation of the random half (train_9 + train_13 = 74% of the random half).

**Result of the schema** (`test/sampling_schema_velocity.py`, run for real, k 25, density 0.66, pool factor 4, min distance 100 m, buffer 1000 m): real-sub 110,460 seeds (49.7%), random 111,749 (50.3%), total ~222 k graphs. Water level 9,977 seeds per random region; 12 of 17 random regions are below it and give everything they have.

**mean_velocity at the seeds** (LOS, mm/yr):

| set | median | std | abs(v) > 2 | abs(v) > 5 | abs(v) > 10 | v < −5 |
|---|---|---|---|---|---|---|
| real-sub seeds | −0.50 | 6.34 | 34.2% | 14.3% | 7.1% | 8.0% |
| random seeds | −0.40 | 5.07 | 28.7% | 8.3% | 3.1% | 2.8% |

Real-sub regions carry clearly more deformation (strong subsidence ~3× as common), but two thirds of their seeds are stable background (abs(v) < 2): the AOIs are much larger than the actual deformation. Per region (share abs(v) > 5): Tagebau NRW 23.7% (strongest subsidence), Napoli 18.4% (mostly uplift, Campi Flegrei), Larisa 14.2%, Cottbus 6.2%, Murcia 5.8% (AOI mostly outside the Lorca subsidence), Northern England 2.1% and Austria 1.6% (practically stable). **Samos** (93.7%, mean +12.7 mm/yr over the whole region) and the random region **train_3** (mean +6) show region-wide offsets, plausibly co-seismic steps (Samos M7.0, Oct 2020) or reference offsets inside the 2019–2023 linear velocity, not local deformation. Conclusion: raising the real-sub share only helps proportionally; tight AOIs around the actual deformation are the stronger lever.

**Tight AOIs and new regions (in progress, 2026-10-21).** Method: EGMS L3 ORTHO-UP (mean vertical velocity, 100 m, 100 km tiles; the GeoTIFF in each zip) for all 743 EEA tiles of the EGMS coverage (`test/l3_scan_download.py`). Scan (`test/l3_subsidence_scan.py`): a cell deforms if abs(v) ≥ 5 mm/yr and at least half of its 5 × 5 neighbourhood deforms in the same direction; 3 × 3 closing, 8-connected clusters ≥ 2 km². Each cluster is compared with a ring 1–5 km around it (contrast = median abs(v − ring median)); region-wide offsets have low contrast and are excluded. Candidate AOI = deforming cells + 1000 m buffer, holes filled, overlapping clusters merged, labelled with known sites from the literature (Lorca, Granada, Valle Umbra, Pistoia/Prato, Po delta, Thessaloniki, LGOM, Bełchatów, Ostrava-Karviná, Groningen, ...) and flagged near 2019–2023 earthquakes/volcanic unrest. `test/build_new_study_areas.py` then tightens the existing train regions to their deforming parts (drops regions without), adds new regions (contrast ≥ 4 mm/yr, core ≥ 3 km², AOI ≥ 20 km², ≥ 25 km from val regions), keeps val regions unchanged (GNSS validation), and writes an s01-compatible `real_subsidence_pointbased.gpkg` (rectangle = AOI bounding box + 1 km) plus `real_subsidence_aoi.gpkg` with the tight polygons, which s03 must use as AOI instead of the rectangle. **Scan results (2026-10-21):** 741 tiles (2 without product), 6,559 deforming clusters ≥ 2 km², 1,668 candidate areas. The top of the list is dominated by Fennoscandia (Sweden, Finland, Norway: post-glacial uplift and many isolated strong uplift clusters that look like artefacts); these are excluded, as are areas > 1,500 km² (regional trends) and earthquake-flagged areas without a known site (Samos, Arkalochori). Known sites are found reliably (Lorca, Thessaly, Rhineland, Bologna, Campi Flegrei, Etna, Ostrava-Karviná, Upper Silesia, Lusatia, LGOM, Prato, Bełchatów, Granada, Thessaloniki), plus plausible unlabelled ones (e.g. Catalan potash mines near Súria/Sallent, Turów, Puertollano, Estonian oil shale, UK/Irish peat and coalfields, Foggia, Modena). 69 candidates pass outside the old train rectangles and val zones, 62 groups after merging within 15 km, the 40 largest by deformation volume are kept.

Existing train regions: tightened Napoli (to Campi Flegrei, uplift), Larisa, Tagebau NRW, Murcia (to Lorca), Cottbus, Northern England; dropped Samos (region-wide offset, Samos earthquake), Sisak (Petrinja earthquake) and Austria (no local deformation). Northern England's tight AOI is too small: all its bursts fail the 4 km² / 10k points rule. Murcia's AOI is cut at the old rectangle although the subsidence continues west. Val regions unchanged.

**Estimate with the tight AOIs** (k 25, density 0.66, pool factor 4, 50/50): real-sub 14,782 seeds, of which 71.7% have abs(LOS velocity) > 5 mm/yr (previously 14.3%); whole training set 39.9% (previously 11.3%). But the total drops to ~29.6 k graphs (previously ~222 k), because the tight AOIs are small and the capacity formula allows only (n_center / 100) × 0.66 seeds. New regions are estimated from L3 (18.4 L2b points per L3 cell, vertical × cos 38° as LOS). Open: how to recover data volume (seeds only inside the tight AOI but pool neighbours from the surrounding burst; seed count limited by the minimum distance instead of the capacity formula; RATIO > 1; all 62 groups). Files: `test/new_study_areas/`.

**Revised build (2026-10-21, decisions):** earthquake/volcanic offsets are kept (the model must recover them, and they matter for later time-series work); Fennoscandian post-glacial uplift is kept as 5 windows of 30 × 30 km where the L3 points are densest (≥ 150 km apart), the small isolated Nordic clusters stay excluded; all candidate groups are taken (no cap); existing regions are not clipped to their old rectangle. The study-area file now holds **polygons**: `real_subsidence_polygonbased.gpkg` (location, set, source, known_site, event_flag, Polygon in EPSG:4326) with region polygon = simplified convex hull of (tight AOI + 2 km); s03 uses `buffer_m` = 2000 m, so seeds lie inside roughly the hull of the tight AOI and pool points can come from the 2 km ring. Val regions keep their rectangles (as polygons). The EGMS search API accepts arbitrary polygons (tested), but only searches their lon/lat bounding box (see s01 input format below). Result: 76 train regions (8 existing tightened incl. Samos and Sisak, Austria dropped; 63 from the scan; 5 GIA windows), median polygon 115 km². Estimate (k 25, density 0.66, pool factor 4, 50/50): real-sub 62,155 seeds with 33.7% abs(LOS velocity) > 5 mm/yr (18.8% below −5, 14.8% above +5); random 62,155; total ~124 k graphs with 20.9% > 5 mm/yr (previous schema: 222 k, 11.3%). The GIA windows give ~35% of the real-sub seeds; Sisak's deforming area (2 km²) is too small for any burst to pass.

**Scan v2 (2026-10-22), after the Crete check:** v1 only kept cells ≥ 5 mm/yr with local support plus 1 km, which missed the weaker parts of deformation fields (e.g. the uplift lobe of the Arkalochori earthquake dipole, visible in the EGMS viewer). v2 (`test/l3_subsidence_scan_v2.py`, developed on known sites in `test/aoi_refine_dev.py`): regional background removed (40 km median on 2 km blocks), normalised Gaussian smoothing (300 m), hysteresis per sign (strong ≥ 5 mm/yr growing over connected cells ≥ 2 mm/yr), weak regions of either sign within 3 km of a core attached (dipoles, halos), closing 1 km, holes filled, ≥ 3 km². Result: Crete now covers both lobes; Thessaly the whole earthquake field. Build with v2: 121 train regions (8 existing incl. Sisak, Austria dropped; 108 scan regions; 5 GIA windows from v1), polygons total ~48,500 km² (median 130 km²). Estimate (same parameters, 50/50): real-sub 151,865 seeds (33.2% abs(v) > 5, 18.1% > 10; 11.1% below −5, 22.0% above +5), total ~304 k graphs, 20.8% abs(v) > 5 overall. Seed shares of the real-sub half: existing regions 34%, other scan regions 27%, **Iceland 24%**, GIA windows 15%. Problems: the Iceland regions are mostly relative patches inside the strong regional glacial uplift (long convex hulls over the highlands), not local deformation; convex hulls of far-apart parts inflate some polygons; v2's background removal did not remove the Fennoscandian uplift (excluded by the country rule anyway); broad weak fields (UK, Ireland, ~−4 mm/yr) are now included; Sisak still gives 0 seeds. Only the vertical component is scanned; horizontal (E-W) motion is not seen.

**s01 input format (decided 2026-10-22).** The regions are now picked by hand in QGIS (on the 250 m L3 vertical / east-west mosaics and the GNSS station gpkg) and stored in `assets/real_subsidence_polygonbased.gpkg` (tracked in git): any CRS (converted to EPSG:3035 when read), one single-part polygon per region = the core area for seeds, columns `location` and `set`. s01 adds `clip_buffer_m` = `CLIP_BUFFER_M` (5000 m, module constant) and queries EGMS with core + buffer; right after unzipping, every burst csv is clipped in place to core + 5 km (rows only, bursts never deleted, so reruns don't download them again). The raw bursts are therefore gone, but the pool buffer of s03 (`--buffer-m`, default 2000 m) can still vary up to 5 km; s03 refuses larger values. **The EGMS search only uses the min/max lon/lat of the query polygon** (tested 2026-10-22: an L-shape and a thin diagonal strip give the same hits as their lon/lat envelope, and only 22 of 30 resp. 24 of 44 hits actually intersect the shape; vertex count up to 18k makes no difference). Therefore `pull_egms` checks every returned burst against the actual query area (core + buffer, EPSG:3035) and picks the relative orbit with the largest overlap; densifying the query ring is not needed. The minimum-overlap check (≥ 4 km² and ≥ 10,000 points) is done in s03 (`select_area_and_buffer_burst_edges`), so s01 always downloads something and every region folder exists. Random regions are handled the same way (square core of half side `egms_rand_extension`). The exclusion buffer for the random sampling is `real_sub_buffer_value + egms_rand_extension·√2 + 2·CLIP_BUFFER_M` around the real-sub cores, so `real_sub_buffer_value` is the minimum gap between real-sub and random query areas; overlapping query areas would put the same points into train and dev/cal (leakage). For the same reason, handpicked regions of different sets must be ≥ 10 km apart and `mindist_rnd` ≥ 2·(e·√2 + 5 km). Git only holds what is needed to start the process (this gpkg, the coverage geojson); `study_areas_io` writes and reads the region table only as gitignored `study_areas_gdf_local*` files.

---

## 5. Model Architecture

### 5.1 Overview

```
                Node features (all nodes)
                ├─ LOS coefficients (p)         [neighbours only; zeroed at target]
                ├─ is_query flag (1)
                ├─ Neighbour incidence angle (1) [per-node viewing angle]
                ├─ Distance to nearest coherent (1) [uncertainty driver]
                ├─ Annual predictors (Y × F_annual → encoded to d_enc)
                └─ Static predictors (n_static → MLP to d_static)
                       ▼
            Concatenate to h⁰ ∈ ℝ^{n_nodes × d}

                Message passing (spatial)
                Each node exchanges with angular-balanced k-NN
                Edge features: [distance, unit_direction_x, unit_direction_y]
                
                ├─ GATv2Conv(d, d/heads, heads=4, edge_dim=3) × L
                └─ LayerNorm between hops
                       ▼
            h^L ∈ ℝ^{n_nodes × d}

                Readout (single query node per graph)
                μ ∈ ℝ^p, logσ² ∈ ℝ^p
                (mean and log-variance of LOS coefficients)
```

### 5.2 Temporal encoding (basis coefficients)

**Input:** Coefficients directly (no encoding needed). They're already low-dimensional.

**Why this is simpler than an encoder:** The basis is fit in preprocessing; the network sees `p` numbers, not `T`. If you need temporal structure, it's already in the coefficients (trend captured in the linear term, seasonality in the sinusoids, etc.). The network doesn't re-learn it.

**Alternative** (if basis doesn't fit your series shapes): 1D-CNN on raw series.
```python
class TemporalEncoder(nn.Module):
    def __init__(self, T, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1))
        self.proj = nn.Linear(64, d)
    def forward(self, los):  # (N, T)
        h = self.net(los.unsqueeze(1)).squeeze(-1)
        return self.proj(h)
```

### 5.3 Annual predictor encoding

**Prototype note (§3.6a):** the target is the 2019–2023 mean velocity, so whether one year's embedding or a pooled one is used is open (§10.16). Either way a single (64,) vector per point goes directly into §5.4's `StaticEncoder`; this `AnnualEncoder` is only needed if several years are fed in separately.

```python
class AnnualEncoder(nn.Module):
    def __init__(self, n_annual, d=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_annual, 64), nn.GELU(),
            nn.AdaptiveAvgPool1d(1))  # pool over years
        self.proj = nn.Linear(64, d)
    def forward(self, annual):  # (N, Y, n_annual)
        # average over years, then project
        h = annual.mean(dim=1)  # (N, n_annual)
        return self.proj(self.enc(h))
```

Alternatives: Transformer over the year axis, or a small 1D-CNN over time.

### 5.4 Static predictor MLP

Compresses the raw GSE vector before it's concatenated with the other node features — rationale: (1) dimensional balance, 64 raw dims would otherwise dwarf the handful of other features (is_query, incidence, distance, velocity); (2) lets the network learn a task-specific projection of the embedding rather than consuming it raw at full width; (3) matches the same group → small-encoder → common-width pattern used for every other feature group, so adding more covariate groups later doesn't require reshaping the fuse layer.

**Open, pending supervisor discussion (§10.13):** whether to compress at all, and whether/how to normalize the raw GSE vector beforehand — it's already roughly unit-scale post de-quantization and may encode meaningful relative geometry across its 64 dims (cosine-similarity-like structure), which per-channel standardization could distort. Not finalized.

```python
class StaticEncoder(nn.Module):
    def __init__(self, n_static, d=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_static, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, d))
    def forward(self, static):  # (N, n_static)
        return self.net(static)
```

### 5.5 Spatial message passing (full model)

```python
import torch, torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

class InSARGNN(nn.Module):
    def __init__(self, p_los, n_annual, n_static, d=96, hops=2, heads=4):
        super().__init__()
        self.p_los = p_los
        
        # Static encoders (annual + static predictors; LOS is already coefficients)
        self.enc_annual = AnnualEncoder(n_annual, d // 2)
        self.enc_static = StaticEncoder(n_static, d // 2)
        
        # Node features: [los (p), is_query (1), incidence (1), dist_pc (1)] + encoded
        n_input = p_los + 3 + d  # + encoded annual/static = d total
        self.fuse = nn.Linear(n_input, d)
        
        # Message passing
        self.convs = nn.ModuleList([
            GATv2Conv(d, d // heads, heads=heads, edge_dim=3)
            for _ in range(hops)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(hops)])
        
        # Output head (heteroscedastic)
        self.head_mean   = nn.Linear(d, p_los)
        self.head_logvar = nn.Linear(d, p_los)
    
    def forward(self, data):
        # data.x = [los (p), is_query (1), incidence (1), dist_pc (1), annual encoded, static encoded]
        # Split and encode annual/static
        los = data.x[:, :self.p_los]
        static_feat = data.x[:, self.p_los:]
        # (assuming static_feat already has encoded dims; alternative: encode here)
        
        # Fuse all node features
        h = self.fuse(data.x)
        
        # Message passing
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + nn.functional.gelu(
                conv(h, data.edge_index, data.edge_attr)))
        
        # Readout at target node (query_mask tells us which node is the query)
        h_query = h[data.query_mask]  # (batch_size, d)
        
        # Heteroscedastic head
        mu = self.head_mean(h_query)           # (batch_size, p_los)
        logvar = self.head_logvar(h_query).clamp(-8, 8)  # (batch_size, p_los)
        
        return mu, logvar
```

**Alternative layer choices (not adopted, worth benchmarking against).** GATv2Conv is a reasonable fit for the edge/node feature situation here (attention lets the model learn a context-adaptive, edge-feature-informed neighbour weighting instead of a hand-fixed kernel), but it is not the only one, and it is not obviously the best one:

- **`PointTransformerConv`** — designed specifically for point clouds with explicit relative-position encoding, which is arguably a closer match to this problem's actual shape (an irregular point cloud in continuous space) than a generic attention-based GNN layer. Worth trying as a direct comparison to GATv2Conv, not just a footnote.
- **`NNConv` / edge-conditioned convolution** — edge features generate a full per-edge weight *matrix* applied to the neighbour's features, rather than a scalar attention weight. More expressive than GATv2Conv, more parameters, typically less stable to train with limited data.
- **`TransformerConv`** (PyG's graph transformer layer) — also supports `edge_dim`, closely related to GATv2Conv but with a different normalization/query-key-value structure; a cheap ablation if GATv2Conv underperforms.

None of these change §3.3's relative-only-position constraint — all three still take `[distance, unit_direction]`-style edge features rather than absolute coordinates. What they're *not* a fit for: diffusion convolution (IGNNK's own layer, built for directed/asymmetric reachability like one-way roads) has no analogue need here, since an InSAR point cloud has no directional-flow structure to model — pulling it in would be solving a problem this data doesn't have.

### 5.6 Graph construction per query

```python
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from scipy.spatial import cKDTree

def build_query_graph(target_xy, target_annual, target_static, target_dpc,
                      neighbour_xy, neighbour_los_coeffs, neighbour_annual, 
                      neighbour_static, neighbour_incidence, k=24):
    """
    Build a single Data object for one query.
    
    target_* : scalars/vectors for the query location
    neighbour_* : arrays of shape (k, ...)
    """
    n_nodes = 1 + len(neighbour_xy)
    d = neighbour_annual.shape[-1] if neighbour_annual.ndim > 1 else 1
    
    # Node features
    is_query = torch.zeros((n_nodes, 1))
    is_query[0] = 1.0
    
    los_coeffs = torch.zeros((n_nodes, neighbour_los_coeffs.shape[1]))
    los_coeffs[1:] = torch.tensor(neighbour_los_coeffs, dtype=torch.float32)
    
    incidence = torch.zeros((n_nodes, 1))
    incidence[1:] = torch.tensor(neighbour_incidence, dtype=torch.float32).unsqueeze(1)
    
    dpc = torch.zeros((n_nodes, 1))
    dpc[0, 0] = target_dpc
    dpc[1:, 0] = torch.tensor(
        np.linalg.norm(neighbour_xy - target_xy, axis=1),
        dtype=torch.float32)
    
    annual = torch.zeros((n_nodes, d))
    annual[0] = torch.tensor(target_annual, dtype=torch.float32)
    annual[1:] = torch.tensor(neighbour_annual, dtype=torch.float32)
    
    static = torch.zeros((n_nodes, len(target_static)))
    static[0] = torch.tensor(target_static, dtype=torch.float32)
    static[1:] = torch.tensor(neighbour_static, dtype=torch.float32)
    
    x = torch.cat([is_query, los_coeffs, incidence, dpc, annual, static], dim=1)
    
    # Edges (spatial KNN among all nodes, relative positions)
    pos = torch.tensor(neighbour_xy - target_xy, dtype=torch.float32)  # centre on target
    pos_target = torch.zeros((1, 2), dtype=torch.float32)
    pos = torch.cat([pos_target, pos], dim=0)
    
    edge_index = knn_graph(pos, k=min(k, n_nodes - 1), loop=False)
    
    # Edge features: distance + unit direction
    src, dst = edge_index
    dvec = pos[dst] - pos[src]
    dist = torch.norm(dvec, dim=1, keepdim=True)
    unit = dvec / (dist + 1e-6)
    edge_attr = torch.cat([dist, unit], dim=1)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    data.query_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.query_mask[0] = True
    
    return data
```

---

## 6. Uncertainty Quantification

### 6.1 Aleatoric uncertainty (heteroscedastic head)

The network predicts per-coefficient variance alongside the mean, trained with Gaussian NLL:

```python
import torch.nn.functional as F

def loss_supervised(mu, logvar, y_coeffs):
    """
    Gaussian negative log-likelihood on coefficients.
    mu, logvar: (batch, p_los)
    y_coeffs: (batch, p_los)  [labels from basis-fitted LOS]
    """
    var = torch.exp(logvar)
    return F.gaussian_nll_loss(mu, y_coeffs, var, reduction='mean')
```

This variance captures **irreducible noise** in the LOS measurements and unmodelled dynamics. It does **not** shrink with more data (that's epistemic).

### 6.2 Epistemic uncertainty (deep ensemble)

Train **M** independent copies of the model (M ≈ 5), each with different:
- Random seed
- Bootstrap sample of the training points

At inference, run all M members and compute:
```python
means = [mu_m for each member m]          # (M, p_los)
vars_aleatoric = [var_m for each member] # (M, p_los), from heteroscedastic head
mu_ensemble = torch.stack(means).mean(dim=0)  # (p_los)
var_aleatoric_mean = torch.stack(vars_aleatoric).mean(dim=0)  # (p_los)
var_epistemic = torch.stack(means).var(dim=0)  # (p_los) — disagreement across members
var_total = var_aleatoric_mean + var_epistemic
```

**Epistemic** variance inflates when:
- Sparse data nearby (high neighbourhood radius)
- Novel covariate combinations (e.g., a land-cover class rare in training)
- Extrapolation regions

This is the honest "I'm unsure where you are" signal.

### 6.3 Propagation through basis reconstruction

**Prototype note (§3.6a):** not needed for now — with a scalar mean-velocity target there's no coefficient vector to reconstruct. Skip straight to §6.4, applied to `(mu_v, var_v)` directly. Revisit this section once the target moves to multi-year/full time series.

Basis coefficients → LOS series:
```
LOS(t) = Φ(t) @ coeffs,  where Φ ∈ ℝ^{T × p}
```

Both mean and variances are linear:
```python
# Φ is fixed (basis matrix), computed in preprocessing
Phi = torch.tensor(basis_matrix, dtype=torch.float32)  # (T_raw, p_los)

mu_los = Phi @ mu_coeffs  # (T_raw,)
var_los_aleatoric = Phi @ diag(var_aleatoric) @ Phi.T  # full covariance, but diagonal suffices
var_los_epistemic = Phi @ diag(var_epistemic) @ Phi.T
```

### 6.4 Propagation through decomposition

**Prototype note (§3.6a):** for the scalar mean-velocity target, `G⁻¹` is applied once to `(mu_v_asc, var_v_asc)` / `(mu_v_desc, var_v_desc)` — the scalar case of the same formula below, no per-epoch loop. Run it twice (once for aleatoric variances, once for epistemic) to keep the split alive into the final EW/Up numbers.

Decomposition at the query point:
```
[EW; Up] = G⁻¹ @ [LOS_asc; LOS_desc]

where G = [[e_asc_E, e_asc_U],
            [e_desc_E, e_desc_U]]

G⁻¹ ∈ ℝ^{2×2}
```

Variance propagation (Jacobian rule):
```python
# G_inv: the inverse matrix, same for all time steps
G_inv = torch.linalg.inv(torch.tensor(G, dtype=torch.float32))

# Per-epoch (independent)
sigma2_los = torch.cat([var_los_asc, var_los_desc])  # (2, T_raw)
cov_ew_up = G_inv @ diag(sigma2_los) @ G_inv.T  # (2, 2)
# Extract diagonals for EW and Up
var_ew_total = cov_ew_up[0, 0]
var_up_total = cov_ew_up[1, 1]
```

Apply this to **aleatoric and epistemic separately**, so the split survives into the final product.

### 6.5 Calibration via split-conformal prediction

**Purpose:** Ensure your intervals actually contain the truth at the claimed coverage level.

**Procedure (after training):**
1. On the held-out **calibration blocks**, run the trained model + ensemble on all coherent L2 points.
2. Collect residuals: `residuals = |y_true - mu_pred| / sqrt(var_total + ε)`
3. Compute the `(1 - α)`-quantile of residuals, call it `q` (e.g., α = 0.1 → 90% coverage target, q = quantile at 0.9).
4. At inference, scale intervals: `CI = [mu - q*sigma, mu + q*sigma]`

**Distance-binned calibration — flagged as a major planned addition, not just an option.** A single global `q` only guarantees *marginal* coverage: it necessarily trades away accuracy in both directions — needlessly wide intervals in dense, well-supported areas, and (the dangerous failure mode) falsely narrow intervals in sparse or far-extrapolated areas, exactly where "query anywhere" is being trusted most. Binning `q` by `distance_to_nearest_coherent` fixes this by giving each distance regime its own calibrated interval width, closer to true *conditional* coverage instead of only marginal coverage — directly analogous to how classical kriging variance already grows with distance from data, just derived empirically instead of from an assumed covariance model. Defer implementing this until the base single-`q` pipeline (train → calibrate → test) is working end to end, then revisit — the two later parameters to decide then are bin edges and how many calibration points land in each bin (too many bins against a limited calibration set gives noisy per-bin quantiles).
```python
# Bin calibration residuals by distance_to_nearest_coherent
dist_bins = [0, 500, 1000, 2000, 5000, np.inf]  # metres
q_per_bin = {}
for lo, hi in zip(dist_bins[:-1], dist_bins[1:]):
    mask = (dist >= lo) & (dist < hi)
    q_per_bin[(lo, hi)] = torch.quantile(residuals[mask], 1 - alpha)

# At inference, apply the q for the query's distance bin
```

This adapts coverage: points far from data get wider intervals.

**Verify on test:** Measure empirical coverage on the untouched test blocks.

---

## 7. Training Procedure

### 7.1 Data loading and graph generation

**Not a star graph.** The mini-graph is not a fixed anchor with spokes to its neighbours. It is an interconnected network over a random subset of a candidate pool: every node is connected to its k nearest nodes in that subset (e.g. k=3), and any node can be the masked query. This is what makes it GNN kriging rather than weighted averaging around one centre: information reaches the query over several hops, through the neighbours' own neighbours.

**Two levels: a fixed pool on disk, a fresh graph every draw.**

- **Persisted once (s03), offline:** the §4.3 Step 3 candidate pools. Each pool is a seed point plus the points gathered around it by angular selection (the seed is just where the pool starts, not a graph centre). Per pool point: `pid`, `seed_pid` (the pool it belongs to), lat/lon, `mean_velocity` (EGMS's own 2019–2023 field, no fitting needed, §3.6a), incidence angle and GSE embedding. Land cover is not stored (§4.3 Step 3.3). This is the expensive part (angular selection, GSE sampling) and is never recomputed during training. **GSE is sampled once, here, per burst and only at seed and pool points, from a transient windowed COG read; only the per-point 64-dim int8 vectors are kept and no raster tiles are stored (§10.16).**
- **Rebuilt fresh on every `__getitem__` call:**
  1. draw a random subset of the pool's points,
  2. pick one point of the subset as the query and hide its velocity,
  3. compute all pairwise geodesic distances and bearings of the subset from lat/lon (`pyproj.Geod`, as in the tutorial notebook, §4.2) and connect every node to its k nearest nodes (symmetrised),
  4. take the edge features (distance + unit direction of the bearing) from those same matrices, plus the query's `dist_to_coherent` (distance to its nearest unmasked node), and normalise them (see below).

  The subset, the graph shape and the query all change between epochs. None of this may be cached, or that variation disappears.

**Storage: one table per burst csv, decided 2026-10-21.** The pools of a burst are written to one Parquet table next to that burst's data (columnar, compressed, inspectable with polars). Working per burst file means the `pid`/`seed_pid` values never have to be combined across files, where they are not guaranteed unique. File count = number of accepted bursts (~300 for the current train split), which scratch handles. One row per pool point:

| column | type | meaning |
|---|---|---|
| `pid` | str | EGMS point id |
| `seed_pid` | str | pid of the pool's seed; one pool = all rows with the same value |
| `lat`, `lon` | float64 | WGS84, only used for pairwise distances/bearings |
| `mean_velocity` | float32 | raw mm/yr, 2019–2023 (§3.6a) |
| `incidence_angle` | float32 | raw degrees |
| `gse_0` ... `gse_63` | int8 | raw quantised embedding, de-quantised on read; year handling open (§10.16) |

**Static information is attached in the same per-burst step:** GSE is sampled only at the points that are a seed or a pool member (not at the other points of the burst), from a transient windowed read. **Edge features are not stored**; they are computed at training time from the drawn subset (below).

When a burst's Dataset is created, its table is sorted once by `seed_pid` and turned into flat tensors plus an offset array `pool_ptr` (pool `p` = rows `pool_ptr[p]:pool_ptr[p+1]`). The offset array exists only in memory while training runs: fast slicing, variable pool sizes, no padding.

**Edge features: computed on the fly from the subset's pairwise geodesic matrices, not precomputed per pool.** The edge set depends on the subset drawn anyway, and precomputing would store n² values per pool for no real saving. The unit direction is bounded in [−1, 1] and needs no scaling. **Distance does need normalising, and its mean/std can't be read off stored values because the edges only exist per draw. They don't have to be exact:** run the same sampling function offline over a random sample of train pools (a few thousand draws give hundreds of thousands of edges, far more than a stable mean/std needs), store the result as a separate stats entry, and recompute only when the sampling parameters (k, subset fraction) change. Distances are heavy-tailed (dense urban pools vs sparse rural ones), so standardise `log1p(distance)` rather than raw metres; same treatment for `dist_to_coherent`. Do not normalise per graph (e.g. by the graph's median edge length): that removes the absolute scale, which is what tells the model how far the nearest evidence is.

```python
class PoolDataset(torch.utils.data.Dataset):
    """One per burst table. One item = one candidate pool."""
    def __init__(self, table_path, k=3, subset_frac=(0.6, 0.9), stats=None):
        t = pl.read_parquet(table_path).sort("seed_pid")
        sizes = t.group_by("seed_pid", maintain_order=True).len()["len"].to_numpy()
        gse = t.select([f"gse_{i}" for i in range(64)]).to_numpy()
        self.b = {"lat": torch.from_numpy(t["lat"].to_numpy()), "lon": torch.from_numpy(t["lon"].to_numpy()),
                  "velocity": torch.from_numpy(t["mean_velocity"].to_numpy()),
                  "incidence": torch.from_numpy(t["incidence_angle"].to_numpy()), "gse": torch.from_numpy(gse),
                  "pool_ptr": torch.from_numpy(np.concatenate([[0], np.cumsum(sizes)]))}
        self.k, self.subset_frac, self.stats = k, subset_frac, stats

    def __len__(self):
        return len(self.b["pool_ptr"]) - 1

    def __getitem__(self, p):
        start, end = self.b["pool_ptr"][p], self.b["pool_ptr"][p + 1]
        n = end - start
        m = max(self.k + 2, int(n * random.uniform(*self.subset_frac)))
        idx = start + torch.randperm(n)[:m]          # FRESH subset every call
        q = random.randrange(m)                      # FRESH query every call

        lat, lon = self.b["lat"][idx].numpy(), self.b["lon"][idx].numpy()
        ii, jj = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
        az, _, dist = Geod(ellps="WGS84").inv(lon[ii.ravel()], lat[ii.ravel()],   # all ordered pairs i -> j
                                              lon[jj.ravel()], lat[jj.ravel()])
        dist = torch.tensor(dist, dtype=torch.float32).view(m, m)                 # metres
        az = torch.tensor(np.radians(az), dtype=torch.float32).view(m, m)         # bearing i -> j, clockwise from north

        d = dist.clone(); d.fill_diagonal_(float("inf"))
        nbr = d.topk(self.k, largest=False).indices                # (m, k) nearest, however far they are
        src = nbr.reshape(-1)
        dst = torch.arange(m).repeat_interleave(self.k)
        edge_index = to_undirected(torch.stack([src, dst]))       # plain torch, no torch_cluster needed
        s, t = edge_index                                         # attributes looked up after symmetrising,
        edge_attr = torch.stack([torch.log1p(dist[s, t]),         # so each direction gets its own bearing
                                 torch.sin(az[s, t]), torch.cos(az[s, t])], dim=1)  # distance standardised with self.stats

        vel = self.b["velocity"][idx].clone()
        y = vel[q].clone()
        vel[q] = 0.0                                              # hide the target
        is_query = torch.zeros(m); is_query[q] = 1.0
        # build x from [vel, is_query, incidence, dist_to_coherent, gse], normalised with self.stats
        ...
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                    query_mask=is_query.bool())

full_dataset = torch.utils.data.ConcatDataset([PoolDataset(f) for f in train_burst_tables])
loader = torch_geometric.loader.DataLoader(full_dataset, batch_size=64, shuffle=True)
```

**Connectivity with k=3.** Every node gets its 3 nearest neighbours however far away they are, so the query always has 3 incoming edges and never sees an empty neighbourhood. That does not make the graph a single connected component: any group of ≥4 points whose three nearest neighbours all lie inside the group is cut off from the rest, no matter how far away the rest is. Harmless for the prediction itself, since nodes outside the query's component cannot influence it, but multi-hop information only flows within the query's component. One more reason to keep the subset large enough (§10.15).

**Edge queries (known concern, kept as is for now, §10.17).** A random query near the edge of a pool has neighbours on one side only; the angular balance around the seed doesn't hold for it, so predicting it is extrapolation, not interpolation. Since most of a pool's points lie further out than the seed, many queries are at least partly like this. Kept as is (uniform choice of query): it is a realistic "query anywhere" situation.

**Why the graph actually varies per epoch — mechanism, not magic.** `shuffle=True` only randomizes the *order* pools are visited in. The subset, graph and query vary because `__getitem__(p)` draws fresh randomness **every time it's called**, and `DataLoader` calls it again every epoch without caching the return value — same persisted pool, new draw each time.

**Caveat needing actual code, not automatic:** with `num_workers > 0`, forked/spawned worker processes can inherit identical RNG state, producing duplicated "random" masks across workers in the same epoch. Needs an explicit `worker_init_fn` that reseeds each worker (e.g. from `torch.utils.data.get_worker_info().seed`). Not an issue at `num_workers=0`.

**Normalization placement:** stats are stored separately and applied inside `__getitem__` on read; normalized values are never baked into the cached raw tensors, so the stats can change without refetching anything. Velocity and incidence stats are exact (computed by `s02` from the filtered train csvs, `assets/egms_norm_stats.json`). Edge-distance and `dist_to_coherent` stats are sampled estimates (see above). GSE handling per §5.4/§10.13.

`ConcatDataset` is what combines the per-burst datasets into the one dataset a single `DataLoader` wraps — each `__getitem__` only ever draws points from one pool, and a pool never spans bursts or orbits, so batches can freely mix graphs from different bursts without ever mixing points *within* one graph across bursts.

### 7.2 Loss function

```python
def loss_total(mu, logvar, y, data, model, lambda_reproj=0.2):
    # Supervised NLL on the target node
    var = torch.exp(logvar)
    loss_nll = F.gaussian_nll_loss(mu, y, var, reduction='mean')
    
    # Optional: reprojection loss (LOS consistency at neighbour locations)
    # Run the head at all nodes, reproject to LOS, match observed LOS
    # loss_reproj = ...  [see tutorial Section 7.2]
    
    return loss_nll  # + lambda_reproj * loss_reproj
```

**Status: parked, not on the near-term roadmap (§3.6a).** The idea is kept for later — run the head on the *unmasked* neighbour nodes too, and compare each to its own already-known value, as extra (lower-weighted) supervision on the same shared weights, not just the one masked query node. Not circular despite reusing the known value as both input and target: residual/skip connections mean it isn't a pure identity mapping. Likely redundant with §10.10's multi-node masking if that's adopted later.

### 7.3 Training loop (pseudocode)

```python
model = InSARGNN(...)
ensemble_models = [InSARGNN(...) for _ in range(M)]
optimizers = [torch.optim.AdamW(m.parameters(), lr=3e-4) for m in ensemble_models]

for member_id, (model, opt) in enumerate(zip(ensemble_models, optimizers)):
    # Bootstrap sample WITH replacement, same size as the full train pool.
    # NOT a disjoint 1/M split: each member sees ~63% of the unique training
    # points on average (some omitted, some duplicated), which is what
    # creates between-member disagreement (the epistemic signal) without
    # starving any one member down to a small fixed slice.
    # Decision: keep bootstrapping (confirmed) -- needs a literature citation
    # before writeup, see §10.12.
    train_bursts_sample = random.choices(train_bursts, k=len(train_bursts))
    
    for epoch in range(EPOCHS):
        for graph in DataLoader(InSARBurstDataset(train_bursts_sample), batch_size=64):
            opt.zero_grad()
            
            mu, logvar = model(graph)
            loss = loss_total(mu, logvar, graph.y, graph, model)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        
        # Validation on cal blocks
        val_loss = evaluate(model, cal_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"model_m{member_id}.pt")
        
        if early_stop_count > patience:
            break
```

### 7.4 Warmup strategy

**Status: under consideration, not yet adopted — pending literature justification (§10.12).** The idea: train with MSE on the mean only for a few epochs before switching to Gaussian NLL, since an uncalibrated variance head early on can destabilize both heads at once. Keep in mind, don't implement yet without evidence this is worth the added complexity.

---

## 8. Validation & Baseline

### 8.1 Two-tier validation

**Tier 1: LOS level (abundant, isolates interpolator)**
- Held-out LOS points (not used for training) within train/cal/test blocks
- Metrics per orbit: RMSE, MAE, NRMSE, Pearson r (for each orbit separately)
- Reports the interpolation quality directly, independent of decomposition

**Tier 2: Product level (sparse, real objective)**
- Decomposed EW/vertical at GNSS station locations in the reserved overlap regions
- Metrics per component (EW, vertical): bias ± STD, RMSE, MAE, velocity difference (mm/yr), Pearson r
- Report as distributions across stations (median, IQR, per-stratum)
- Also PICP, MPIW, CRPS for uncertainty validation

### 8.2 Baseline: regression-kriging (RK) with external drift

```python
from sklearn.linear_model import LinearRegression
from pykrige.ok import OrdinaryKriging  # or gstools
from pykrige.rk import RegressionKriging

# For each orbit, each time step:
# 1. Fit linear model: LOS ~ (DEM + AEF + hydro + ...)
# 2. Fit variogram on residuals
# 3. At query, use RK to interpolate

# Decompose identically to your model's output
```

**Fairness:** RK has access to the same predictors as your model, operates on the same training points, uses the same decomposition. The only variable is the interpolator (RK's linear + kriging vs. your GNN).

If RK matches your model on both accuracy and calibration, the simpler model wins. If your model beats RK on both, you have a clear contribution.

### 8.3 Metrics in detail

**Deterministic (per component, per station):**
- `bias = mean(pred - truth)` [mm/yr for rates; mm for series]
- `std_residuals = std(pred - truth)`
- `RMSE = sqrt(mean((pred - truth)²))`
- `MAE = mean(|pred - truth|)`
- `velocity_diff = velocity_pred - velocity_truth` [mm/yr, from linear fit]
- `r = Pearson_correlation(pred, truth)`

**Probabilistic (per component, per station, per epoch or epoch bin):**
- `PICP = fraction_of_epochs_where_truth_in_CI` [target: ~90% for 90% CI]
- `MPIW = mean(CI_width)` [should be tight, but not at cost of coverage]
- `CRPS = continuous_ranked_probability_score(truth, mean, std)`

**Reliability diagram:** Plot empirical coverage vs nominal coverage across quantile levels (0.1, 0.2, …, 0.9). Should follow the diagonal.

### 8.4 Reporting

- Per-stratum: report medians and IQR across stations in that stratum
- Spatial map of residuals / coverage by location (reveals regional bias)
- Plot: velocity (true vs pred) for each station
- Plot: time series at a few representative GNSS stations

---

## 9. Implementation Stack

| Task | Library | Notes |
|---|---|---|
| **Geodata** | rasterio, rioxarray, geopandas, pyproj | Raster I/O, reprojection, spatial operations |
| **Numerics** | numpy, scipy (linalg, optimize, spatial) | Linear algebra, KD-trees |
| **Deep learning** | PyTorch, PyTorch Geometric | Model, message passing, batching |
| **Data** | pandas, polars (optional) | CSV/tabular ops |
| **Baselines** | pykrige, gstools, scikit-learn | Kriging, regression, CV |
| **Calibration** | scipy.stats, custom or MAPIE | Conformal quantiles |
| **Viz** | matplotlib, cartopy | Plots, maps |
| **Coordinate ops** | EPSG:3035 (pyproj) | All distances in metres |

---

## 10. Known Issues & Caveats

### 10.1 L2b datum consistency

L2b is calibrated, but verify that your asc/desc products share a common reference before decomposing. If they don't, the decomposition output is biased. Check against GNSS in the overlap region.

### 10.2 N–S insensitivity

The 2×2 decomposition from asc/desc LOS cannot constrain north–south motion (you're solving for 2 unknowns [EW, Up] from 2 equations; N–S is a third). Near-N–S-moving landslides will have wrong EW estimates. Document this limitation.

### 10.3 Reference frame at inference

When a user queries a new location, the query graph is built from a *different* set of neighbours than were in training. If the training region had a datum offset, that offset doesn't transfer — you're not memorizing it. This is good for transfer but means absolute vertical velocity will be off by an unknown constant if the two regions have different reference frames. Mention that queries within a region are internally consistent, but cross-region comparisons may have datum shifts.

### 10.4 Low coherence (peat, forest, etc.)

EGMS sparse/missing in vegetated areas. Your model's uncertainty (epistemic variance) should inflate there due to sparse neighbours. Verify this happens and report it. The peat region is a good stress test.

### 10.5 Non-probabilistic sample

Handpicking makes this a diverse-coverage panel, not a random probability sample. You cannot claim "representativeness in the population-statistics sense," only "we deliberately covered these processes." Be honest about this in the methods.

### 10.6 Temporal resolution and AEF

AEF and other annual rasters are at annual cadence. They capture *yearly* context, not fine-temporal variations. The model uses them as background; temporal detail comes from the LOS series itself. If a sudden event (e.g., a large rainfall or water-management change) happens mid-year, the model won't capture it unless the LOS responds.

### 10.7 Coordinate system for worldwide transfer — resolved

The single-CRS concern raised in Section 4.2 turned out to be a non-issue: the architecture only ever consumes *relative* distance + unit-direction on edges (Section 3.3), never absolute position, so no single global CRS is required. Requirement going forward: compute each pairwise distance/bearing with a locally metric-valid method — geodesic (`pyproj.Geod`) or a per-query local azimuthal-equidistant projection — rather than assuming EPSG:3035 outside Europe. This affects Section 5.5/5.6 edge-feature construction and Section 4.4 sampling-radius geometry only; raster predictor alignment (Section 4.1) is a separate data-engineering concern, not an architectural one.

### 10.8 OPEN — temporal representation for irregular / non-EGMS series

**Status: moot for now.** With the §3.6a prototype scoped to a scalar mean-velocity target, no temporal basis fitting is happening at all — this only becomes relevant again once the target moves back to multi-year/full time series. Shelved until then, not resolved.

**Original concern, unchanged below:**

The Section 3.5 global basis-coefficient fit (one set of ~8–15 coefficients per node over the whole record) assumes a reasonably long, stationary, densely-sampled series — true for EGMS, not guaranteed for worldwide transfer (shorter tracks, gaps, regime changes mid-record e.g. mine flooding, pumping changes).

Being considered: a **windowed/layered variant** — chunk time into blocks (e.g. quarterly/yearly, possibly with sliding overlap as suggested by supervisor), fit local basis coefficients per window (reusing the same `lstsq` machinery), run the spatial GNN per window with shared weights, and link windows with a lightweight temporal model (small GRU/attention over the window axis) rather than full per-acquisition spatiotemporal edges (which Section 3.4 already rejected on cost grounds). Overlapping windows would need overlap-add-style blending at reconstruction. This is effectively the Section 3.4 rejected alternative revived at coarse (window) resolution instead of per-acquisition resolution, which changes the cost profile substantially (`k·num_windows` vs `k·T`).

**Unresolved fork:** whether seasonal terms (annual/semiannual) are fit once globally per node with only trend/level windowed, vs. fit fully locally per window (which would force windows ≥ ~1 year to constrain a seasonal cycle at all). Needs a decision before window length/stride can be fixed.

---

### 10.9 OPEN — predictors may smuggle absolute location back in

**Status: point of concern, not resolved.**

Section 3.3 bans absolute coordinates from node/edge features because they are a memorization pathway. That guarantee is architectural only — it constrains what the *graph* carries, not what the *predictors* carry. AlphaEarth embeddings are produced by a model that takes coordinates as an input, so the 64-dim vectors may encode absolute position implicitly. If they do, a geographic fingerprint reaches the network through the predictor channel, and the model can still learn "region X behaves like Y" without ever seeing a lat/lon column. This would weaken the transferability claim in Section 11.2 while leaving Section 3.3 formally satisfied — i.e. the failure would be invisible to the design as written.

Applies to any pretrained/learned embedding, not only AEF; physically-grounded predictors (DEM, slope, hydrogeology) are not exposed to this in the same way, since they are direct measurements rather than model outputs.

**How to test (cheap, do before scaling):**
- **Position probe:** regress lat/lon from the embeddings alone (ridge or small MLP) on the train blocks. High held-out R² = absolute position is linearly recoverable, and the network can recover it too.
- **Geographic ablation:** train with and without AEF, compare degradation on *geographically disjoint* test blocks against in-distribution held-out points. A model leaning on a location fingerprint degrades much harder on the former.
- **Analogue check:** compare embedding similarity between physically similar but distant sites (e.g. two subsidence bowls in different countries). If proximity dominates similarity over land-cover/process, that is the fingerprint.

**Mitigations if confirmed:** drop AEF and keep only physically-grounded predictors; or aggregate/normalize embeddings per burst so only *local contrast* survives; or adversarially penalize position-decodability during training. Note the first is the cleanest and costs the least thesis-narrative complexity.

### 10.10 OPEN — masked-target scope: single query node vs. multi-node local subgraph

**Status: design question, not resolved. Current plan (§5.6/§7.1) uses the single-query-node version; this is a candidate refinement, not a correction.**

IGNNK (§3.2's source) doesn't just predict one masked node per training sample — within each sampled subgraph it computes reconstruction loss over *every* node, the ones left "observed" included, not only the masked ones (Wu et al. 2021, Loss Function section; citing Hamilton, Ying & Leskovec 2017 for the underlying argument). Your current design (§5.6, §7.1) only ever supervises the one designated query node per graph; neighbour dropout removes some points from the visible context but extracts no gradient from them. Adopting IGNNK's fuller scheme here would mean: build one local subgraph of several nearby points, randomly designate a *subset* of them (not always exactly one) as masked each training step, and take a loss over some or all of the graph's nodes rather than a single readout.

**Why the observed-node loss isn't circular, even though it looks trivial at first glance.** A node's own input isn't thrown away before message passing — only the *masked* nodes get their input zeroed; observed nodes keep their real values as input, and most architectures in this family include a residual/self-loop path (IGNNK's own Eq. 2 explicitly adds the previous layer's representation back in), so reconstructing an observed node's own value is a mostly easy task, close to learned identity. That's fine — it isn't meant to be a hard prediction task. Its value is elsewhere: (1) it's essentially free extra supervision per graph built — `n_o + n_m` loss terms instead of `n_m` — which matters when graph construction itself is a real cost at burst scale; (2) per Hamilton et al.'s argument, it shapes the *same shared* message-passing weights from every node's own vantage point, not only from the vantage point of whichever nodes happen to get drawn as masked targets in a given sample; (3) because which nodes land in the observed vs. masked set is redrawn randomly every training iteration (same mechanism as §3.2's core trick), no node is "always observed" — over the course of training essentially every point takes a turn in both roles. The easy, near-identity signal and the hard, genuinely-predictive signal are two different training pressures on the same weights, not two separate objectives.

**What adopting this for real would change here, concretely:**
- **Mechanically cheap.** `h[data.query_mask]` already returns however many rows are `True` — if `query_mask` marks several nodes instead of one, the model architecture (§5.5) needs no change at all, only the graph-construction step (§5.6/§7.1) does.
- **Loss function (§7.2):** straightforward extension — the same Gaussian NLL, summed or averaged over however many masked nodes are in the graph, rather than evaluated at one row.
- **Uncertainty/calibration (§6.5) — the part that needs real care, not a mechanical change.** Predictions for multiple masked nodes drawn from the same graph share the same random neighbourhood draw and the same message-passing computation — they are correlated, not independent. Split-conformal's coverage guarantee relies on exchangeability of calibration residuals; tightly correlated within-graph residuals shrink the *effective* calibration sample size and can distort coverage if treated as if they were as informative as that many independent draws. Likely mitigation: decouple training-time masking (use several masked nodes per graph, for efficiency) from calibration-time sampling (draw at most one calibration residual per graph, even if training used several) — keeps §6.5's guarantee clean regardless of what the training-side ablation shows.
- **Possible mismatch with the actual inference distribution, worth naming rather than assuming away.** At inference, "query anywhere" most likely means one arbitrary point queried against a neighbourhood of *real, observed* EGMS points — never neighbours that are themselves simultaneously unknown. Multi-masking trains on a strictly more general (harder) configuration than that — some neighbours being masked-and-uncertain during training, never at inference. Plausibly a net positive (same logic as dropout improving robustness despite not matching the test-time computation graph exactly), but it is a trained-distribution vs. deployed-distribution mismatch, not a pure win, and should be evaluated as such rather than assumed to help.

**Paths to explore, roughly in order of cost:**
1. Implement as an ablation alongside the current single-query design — same architecture, only the graph-sampling/masking step and loss aggregation differ — and compare Tier-1 LOS metrics (§8.1) between the two.
2. If it helps, retire the currently-optional reprojection-loss idea (§7.2) — multi-node masking is a more systematic, built-in version of the same consistency-forcing intuition, so keeping both would likely be redundant.
3. Decide the local-subgraph sampling shape: still a k-NN neighbourhood around a rough centre (closest to what §5.6 already builds), or something closer to IGNNK's own arbitrary-subset-of-indices sampling restricted to a bounded local radius. The former is a smaller change from the current plan; the latter more faithfully reproduces IGNNK's actual algorithm but has less obvious justification at this data density.
4. If adopted, implement the calibration-decoupling mitigation above *before* running any §6.5 calibration numbers, not after — retrofitting it later would mean redoing calibration.

### 10.11 OPEN — candidate-pool sparsity augmentation

**Status: open, postponed until the prototype runs end to end (2026-10-21).** Problem: with nearest-neighbour pools the pool shape follows the point density (about 30 m across in cities, several km in rural areas). PS points concentrate in cities, so the model would mostly practise easy short-range interpolation, while the goal includes interpolating across large gaps (up to 2 km). Proposal to revisit:
- **Pool construction by distance rings:** within the 2 km radius, rings (e.g. 0–100, 100–300, 300–800, 800–2000 m) × 8 sectors; per cell the nearest free point. Pools of similar extent regardless of density; density shows in which rings are filled. To test: more seeds fail the minimum because pools claim points over ~12 km².
- **Training-time sparsification (fresh per draw):** random subset with a randomly drawn drop rate (DropNode / IGNNK random subgraphs, PointNet++ random input dropout); an **exclusion radius** around the query (all points within r removed, r e.g. log-uniform 0–1.5 km), which sets the gap size directly; occasionally dropping whole sectors (one-sided gaps; sparingly, see §10.17). A mix of these, or a curriculum from dense to sparse.
- **Target the gap distribution:** log the histogram of `dist_to_coherent` of the training queries per epoch and tune the probabilities to a target (e.g. roughly log-uniform 10 m–2 km); evaluate errors per distance bin (fits §6.5).
- Caveat (KITS, Xu et al.): masking inside dense data can differ from real test-time gaps; the exclusion radius addresses that. The pool minimum must leave room for the dropout (§4.3 Step 3.5).
- Sources: IGNNK (Wu et al., AAAI 2021, arXiv 2006.07527), GRAND / DropNode (Feng et al., arXiv 2005.11079), DropEdge (Rong et al., arXiv 1907.10903), graph augmentation survey (arXiv 2202.08871), KITS (arXiv 2311.02565), PointNet++ (Qi et al. 2017).

Original note: §4.3 Step 3's angular-balanced candidate pool construction deliberately supports variable-size graphs so a future "simulate sparser regions" augmentation (dropping neighbours, possibly with its own point-level min-distance) can reuse the same mechanism. The augmentation method itself — how aggressively to sparsify, whether it varies by land-cover stratum, whether it changes per epoch or is baked into the pool — is undetermined. Motivation: EGMS point clouds are usually dense, so without this the model may never see realistically sparse neighbourhoods at train time, which §10.4 already flags as a case where epistemic uncertainty should inflate.

### 10.12 OPEN — literature support needed: MSE warmup and bootstrap ensembling

**Status: decisions made pragmatically, literature review pending before thesis writeup.**

- **Bootstrap ensembling (§7.3): keep it (confirmed).** Need a citation establishing bootstrap-resampled deep ensembles as a valid epistemic-uncertainty estimator (standard practice, but cite it) before writing this up.
- **MSE-then-NLL warmup (§7.4): not yet adopted.** Need literature evidence this measurably helps heteroscedastic-head training before committing to it — currently just "keep in mind," not implemented.

### 10.13 OPEN — GSE normalization and compression, pending supervisor discussion

**Status: idea sketched, not finalized — needs supervisor input before implementing.**

Two linked questions on how the AlphaEarth embedding enters the model (§5.3/§5.4):
- **Compression:** project a 64-dim embedding down via a small MLP before concatenating with other node features (rationale: dimensional balance, task-specific projection — see §5.4), vs. feeding it raw. Leaning toward compressing, not committed.
- **Normalization:** the embedding is already roughly unit-scale post de-quantization and may encode meaningful relative geometry across its 64 dims; standardizing each dimension independently (the default treatment for every other feature group, §4.6) could distort that. Leaning toward skipping per-channel normalization for GSE specifically (or applying one shared scalar at most), unlike velocity/incidence/distance which get ordinary independent normalization. Not committed either way.

### 10.14 OPEN — per-burst anchor budget for candidate pool construction

**Status: decided 2026-10-21, see §4.9 (per-burst capacity) and §4.10 (region weighting).** The text below is the original problem statement.

§4.3 Step 3's exhaustive version (consume every available point via sequential no-reuse k-NN) would produce on the order of dozens to hundreds of millions of training examples pooled across all bursts — not tractable to store or train on. A pre-determined per-burst budget (point 1 of that step) is required, not optional. Undecided: the actual formula -- fixed count per burst, density-based (target spacing, scaled by burst area), proportional to available post-clip/post-buffer point count, or something else. Whichever is chosen directly determines how balanced dataset composition is across bursts of very different size/density, so it's not a cosmetic parameter.

### 10.15 OPEN — pool and graph parameters

**Status: starting values only, to be tuned empirically.** Seed-selection parameters are settled for now (k_nearest 25, density 0.66, pool factor 4, min distance 100 m, buffer 1000 m, §4.9). Angular sectors for the pool: probably 8. Pool size (~20 points assumed), subset fraction per draw, k (3 assumed), minimum neighbours below which a seed is rejected (§4.3 Step 3.5): all undecided. Two behaviours to keep in mind rather than solve now: (1) k=3 does not guarantee one connected component (§7.1), (2) queries near a pool's edge are extrapolation cases (§7.1); whether to restrict or reweight them is open, and implementing that control is non-trivial. Seed spatial spread (land-cover stratification alone won't do it, since PS points are rare in forest/fields) needs a grid or minimum spacing on top, with per-class caps rather than equal counts per class.

### 10.16 OPEN — GSE year(s) and sampling strategy

**Status: working plan, not verified.** (1) *Which year(s):* the target is the 2019–2023 mean velocity (§3.6a), so a single year's embedding doesn't match it by construction. Options: one representative year (64 B/point), or a pooled embedding over 2019–2023 (average then renormalise, or all five years kept and pooled in the model: 320 B/point). Interacts with §10.13. (2) *Sampling:* GSE is not stored as rasters (full tiles ~4.3 GB, regional windows ~200 MB per year). `s03` samples it once per region from a transient windowed COG read at the pool points and keeps only the int8 vectors. Sampling per `__getitem__` during training is **not** viable: remote range requests per draw would stall the data loader, and compute nodes may not have outbound HTTPS. As of 2026-10-20 the GSE download is not implemented in `s01` (only the tile index exists), so this is a prerequisite for `s03`'s feature step.

---

---

### 10.17 CONCERN — extrapolation queries at the pool edge

**Status: accepted for now, possible reason for poor performance later (noted 2026-10-21).** Pools are built angular-balanced around their seed (§4.3 Step 3), but at training time any point of the drawn subset can be the query (§7.1). Points on the rim of a pool have all their neighbours on one side, so a large share of the training queries is extrapolation rather than interpolation. Consequences to watch: (1) the model may learn a blurred mix of both tasks and interpolate worse than it could; (2) validation errors may be dominated by such cases without it being visible. If performance is poor, check this first: record the query's largest angular gap to its neighbours per draw, break errors down by it, and then restrict queries to inner points or reweight the query choice.

## 11. Thesis Positioning & Novelty

### 11.1 Related work

**Inductive kriging family:** IGNNK (Wu et al., AAAI 2021), KCN (kriging convolutional nets), INCREASE (heterogeneous spatial relations). These papers prove the training trick (masked targets) and the inductive principle.

**InSAR + GNN precedent:** A 2026 paper (Springer) uses a DGCNN to predict 3D GNSS-like displacements at InSAR locations, weakly supervised by sparse GNSS. Close, but theirs is (a) supervised by GNSS; (b) outputs at InSAR locations, not arbitrary; (c) doesn't decompose. Your work is self-supervised, query-anywhere, and explicitly learns LOS→EW/vertical.

**ST-GAT InSAR imputation (2025):** Spatiotemporal graph for gap-filling within a region. Rejected for your task because (a) you don't have large temporal gaps; (b) absolute coordinates in their setup kill transfer; (c) 10× cost, 1× benefit for query-anywhere.

**Other decomposition work:** Most assume decomposition happens at existing PS and then interpolates the products. Your interpolate-then-decompose is mechanically simpler and isolates the learned step.

### 11.2 Your novelty

1. **LOS→EW/vertical as inductive kriging.** Combining self-supervised LOS interpolation (abundant labels) with classical decomposition (solved problem) is conceptually clean and separates learning from geometry.
2. **Globally transferable predictors.** Using open, boundary-agnostic features (AlphaEarth + DEM + hydrogeology) + relative-position-only architecture means the model doesn't memorize regions.
3. **Calibrated aleatoric/epistemic uncertainty.** Most interpolators give point estimates or hand-wavy variances. Your deep ensemble + heteroscedastic head + split-conformal calibration + distance-binned adaptation is explicit and validated.
4. **Query-anywhere.** Strict inductive design (masked-target training, relative positions) actually achieves generalization to unseen geographies, with honest uncertainty that inflates where data is sparse.

### 11.3 Thesis one-liner

> *A graph-attention network for spatial interpolation of InSAR line-of-sight displacement, from which east–west and vertical components are recovered via classical decomposition. Trained self-supervised on EGMS L2b data, using globally transferable open environmental predictors. The model generalizes to arbitrary locations across Europe with calibrated, decomposed aleatoric and epistemic uncertainty.*

---

## 12. Coding Conventions

How the pipeline code is written, so that every script looks and behaves the same. New code follows this; deviations are fixed, not copied.

### 12.1 Tooling and running

- Python only through **uv** (`module load uv`), with the project venv in `scripts/python/.venv`; never `pip` directly.
- Pipeline steps run from `scripts/python/`. Anything longer than a few minutes or resource-heavy goes through SLURM, never on the login node.
- `scripts/slurm/NN_<step>.slurm` holds the pipeline steps only. Test and exploration scripts, with their slurm files, live in `test/`.
- Code stays lean: write complete, working code, but add a function, argument or column only when it is actually needed.

### 12.2 Script layout

- File name `sNN_<verb>_<object>.py`. The first lines are a banner with the same name: `# sNN_<verb>_<object>`.
- Fixed top-level sections, each with the double banner (`#====#` between `####` lines):
  1. **Import Statements**, followed by `# 1.1) Constants`
  2. **Helper Functions**, split into numbered sub-sections (`#=== Section n: ... ===#`) per processing step. Within a sub-section, helpers are ordered by first call, and a helper's own helpers come right after it.
  3. **Processing**: one function per processing step (e.g. `download_egms`, `preprocess_gnss`).
  4. **Main**: `main(...)` plus the `if __name__ == '__main__':` block with argparse.
- Values that several scripts need are **module constants** in `UPPER_SNAKE_CASE` in the script that defines them (e.g. `s01.CLIP_BUFFER_M`). Later scripts import them via `import s01_download_data as s01`; they never copy the number.
- Functions and variables are `snake_case`. Counts start with `n_`, file paths end with `_path`, distances end with `_m` and areas with `_m2`.

### 12.3 Docstrings and comments

Every function has a docstring in this form:

```
    """
    What the function does, in one to a few sentences (including why, if not obvious).

    Arguments:
        name -- type, meaning (unit)

    Returns:
        name -- type, meaning      (or: None -- what is written to disk)
    """
```

Inline comments explain the *why* of each step. They are short, lowercase sentences above the line they explain, as dense as in the existing scripts.

### 12.4 Command-line arguments and parameter logs

- Every argument is written as `parser.add_argument("--kebab-case", dest="snake_case", type=..., required=True | default=..., help="meaning (unit)")`.
- Validity checks (value ranges, `--buffer-m` ≤ `s01.CLIP_BUFFER_M`, mutually exclusive options) use `parser.error(...)` right after `parse_args()`.
- Every step records its arguments with `s01.write_param_log("<script name>", s01.PARAM_LOG_DIR, vars(args))` (inside s01 without the prefix). The logs go to `assets/param_logs/` (gitignored); the folder is created if missing.
- In the slurm scripts the arguments are a bash array `ARGS=( ... )`, one argument per line with a comment, followed by `uv run <script>.py "${ARGS[@]}"`.

### 12.5 CRS, units and geometry

- **EPSG:3035** for everything metric: geometries, distances, areas, buffers, clipping.
- **EPSG:4326** only at the interfaces that need it (EGMS search, WorldCover index, NGL station list). Transformers are always built with `always_xy=True`, and coordinate lists are `[lon, lat]`.
- Distances in metres, areas in m², velocities in mm/yr.
- Distances for the model's edge features are geodesic (§4.2).

### 12.6 Data layout and file names

- **Project tree:**
  - `<project-dir>/data/{train,dev,cal,val}/<location>/{ascending,descending}/<burst>.csv`
  - GNSS series and WorldCover tiles sit directly in `<location>/`.
- **File names:**
  - `PART_<file>` for an unfinished download;
  - `<burst>_PREPROCESSED.csv` for the s02 output next to the (clipped) raw csv;
  - `worldcover_merged.tif`;
  - `<Sta>_gnss_time_series.tenv3`.
- **The region table** is read and written only through `s01.study_areas_io`. It writes and reads only the gitignored `assets/study_areas_gdf_local.gpkg` + `_local_complex.json`; the region table is a run result and never goes into git.
- **Other `assets/` files:**
  - Inputs, tracked in git: `real_subsidence_polygonbased.gpkg` (handpicked regions) and `egms_coverage_countries.geojson`.
  - Secret: `token.jwt` (gitignored).
  - Generated or downloaded on the fly: `egms_norm_stats.json`, `failed_egms_downloads.csv`, `gnss_station_list.txt`, `esa_world_cover_index.geojson`.

### 12.7 Column names

**Region table (`study_areas_gdf`)**, one row per region:

| Column | Set by | Meaning |
|---|---|---|
| `geometry` | input gpkg / s01 | core polygon (seeds), EPSG:3035 |
| `location` | input gpkg / s01 (random: `train_n`, `dev_n`, `cal_n`) | unique region name = folder name |
| `set` | input gpkg / s01 | 0 = train, 1 = dev, 2 = cal, 3 = val |
| `clip_buffer_m` | s01 | ring around the core that was downloaded and kept (= `CLIP_BUFFER_M`) |
| `egms_query_bbox` | s01 | `[lon, lat]` ring of core + `clip_buffer_m`, the EGMS search polygon |
| `region_path` | s01 | `data/<set>/<location>/` |
| `gnss_station_codes`, `gnss_station_paths` | s01 (val only) | NGL stations inside the core and their downloaded series |
| `egms_union_bbox`, `wc_tile_urls` | s01 | `[lon, lat]` extent of all clipped bursts, WorldCover tiles |
| `wc_merged_path` | s02 | merged WorldCover raster |
| `gnss_station_codes_filtered`, `gnss_station_paths_filtered`, `gnss_velocities`, `gnss_filter_params` | s02 (val only) | stations passing the quality filter and their velocities |
| `n_points_region`, `n_seed_points_per_region`, `train_file` | s03 (train only) | point count, seed budget, output file |

List or dict columns go into the `_complex.json`. A new list column must be added to `complex_cols` in `study_areas_io`. `uid` is internal to `study_areas_io` and only used to match the two files.

**EGMS point tables:**
- The original EGMS column names are kept (`pid`, `latitude`, `longitude`, `easting`, `northing`, `incidence_angle`, `track_angle`, `los_east`, `los_north`, `los_up`, `mean_velocity`, `temporal_coherence`, `amplitude_dispersion`, ...). The preprocessed csvs keep `s02.EGMS_KEEP_COLS`.
- Floats are read as Float64 via `s02.EGMS_FLOAT_COLS`, and `pid` as String.
- `pid` is only unique within a burst, so a point is identified by burst + `pid`.
- Columns that s03 adds: `in_center` (int8, 1 = outside the edge margin, may be a seed), `seed` (int8, 1 = seed), and `seed_pid` in the pool tables (planned, §7.1).
- Flags are int8 0/1.

**GNSS:** the velocity keys are `ve`, `vn`, `vu` (mm/yr, east/north/up).

---

## Quick Reference Checklist

**Done:**
- [x] Handpicked + random regions selected, EGMS/GNSS/WorldCover downloaded per region (`s01_download_data.py`), resumable
- [x] Train / dev / cal / val split assigned at region level

**Next up (§3.6a prototype scope):**
- [ ] Per-burst candidate pool: land-cover-stratified anchors, angular-balanced k-NN, sequential no-reuse sampling, edge buffer (§4.3 Step 3 / §4.4)
- [x] Target is EGMS's own `mean_velocity` (2019–2023), a column in the preprocessed csvs, no fitting step (§3.6a)
- [ ] GSE sampled once at candidate points only from a transient windowed read, vectors kept, no rasters stored; year choice decided (§10.16)
- [ ] Candidate pools (lat/lon, velocity, incidence, GSE vectors, land cover, pool offsets) written to a few shard files, one per region (§7.1)
- [ ] `PoolDataset` + `ConcatDataset` + `DataLoader(shuffle=True)` wired up, subset/graph/query drawn fresh in `__getitem__` (§7.1)
- [ ] Normalization applied at read time: velocity/incidence stats from s02 (done), edge-distance and `dist_to_coherent` stats estimated from sampled draws (§4.6, §7.1)
- [ ] Model architecture implemented (scalar head: `(mu_v, logvar_v)`, no basis coefficients yet) (§5, §3.6a)
- [ ] Masked-target training loop working, bootstrap ensemble × M copies (§7.3)
- [ ] Heteroscedastic NLL loss (§6.1) — MSE warmup still pending literature decision (§10.12)
- [ ] Uncertainty propagation: scalar `G⁻¹` decomposition on `(mu_v, var_v)` per orbit (§6.4, §3.6a)
- [ ] Conformal calibration quantile computed on cal blocks (§6.5)
- [ ] LOS validation: held-out points, per-orbit metrics (§8.1 Tier 1)
- [ ] Product validation: GNSS comparison, decomposition quality (§8.1 Tier 2)
- [ ] Baseline (regression-kriging) implemented and compared (§8.2)
- [ ] Per-stratum performance analysis, spatial maps and time series plots (§8.4)

**Deferred until the scalar mean-velocity prototype works end-to-end:**
- [ ] Full multi-year temporal basis fit (§3.5), reconstruction propagation (§6.3)
- [ ] Reprojection loss (§7.2, parked)
- [ ] Sparsity augmentation (§10.11), distance-binned calibration (§6.5), multi-node masking (§10.10)
- [ ] GSE normalization/compression finalization — pending supervisor input (§10.13)

---

**End of reference document.**
