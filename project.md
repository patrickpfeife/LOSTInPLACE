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
| DEM (e.g., GEBCO / SRTM30) | Predictors | GeoTIFF, EPSG:3035 unified grid | Slope derived from DEM |
| Hydrogeology / Aquifer maps | Predictors | GeoTIFF or polygon (proximity) | BGR / state geological surveys; EU WFD data |
| AlphaEarth Foundations embeddings | Predictors | Analysis-ready 64-dim annual rasters | Google Cloud Storage / Earth Engine; CC-BY 4.0; "provider-pays" egress |
| Viewing geometry (incidence, track angle) | Decomposition | Per-point columns in the LOS input itself | **No separate raster/metadata dependency** — derived analytically per (track, sub-swath), see §3.7 |
| GNSS (EPN / EUREF / national CORS) | Validation only | Time series, velocity + epoch series | Never training. Reserve validation regions with GNSS. |

### 4.2 Coordinate system & distances

For the **European training panel**, EPSG:3035 (ETRS89-LAEA) is fine for distance/neighbour operations. Distances are in metres. Convert to/from lat/lon only at I/O edges (user query, output metadata).

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

**Step 3: Within-burst stratified sampling.**

Within each selected burst, sample **target points stratified by dominant land-cover class** (use CORINE or ESA WorldCover). Draw roughly equal numbers per class per burst, again respecting the minimum-distance constraint. This avoids area-proportional bias and ensures the model sees forests, urban, agricultural, shrub, etc., equally.

**Why this hybrid?** Handpicking ensures rare-but-important processes (mining, peat) are represented and held out for testing. Random draw with stratification ensures broad coverage and fair representation of conditions. Together, you sample the "condition space," not just the "area space."

### 4.4 Neighbourhood radius and min-distance

Compute the typical radius your k-NN search reaches:
```python
# e.g., after selecting k=24 angular-balanced neighbours
radii = [distance from target to k-th neighbour, for each training point]
neighbourhood_radius = np.percentile(radii, 95)  # conservative upper bound
```

Set **min_distance ≥ 1.5 × neighbourhood_radius** to guarantee targets don't share neighbours.

### 4.5 Training / calibration / test split

**Entire bursts or entire regions**, never random points. Assign bursts/regions to disjoint groups:
- **Train:** ~50–60% of area/bursts → compute normalization stats, train model
- **Calibration:** ~15–20% → unused during training; used for conformal quantile (Section 6.4)
- **Test:** ~20–30% → held-out; measure final performance

The split is **geographic**: train blocks are spatially far from test blocks, so a test point's neighbours are in train, not in test. This prevents "leakage" where a test sample's ground truth is hidden from the model but its neighbourhood is known.

### 4.6 Global normalization (on train split only)

Compute standardization statistics (mean, std) for all features **across the entire train split**, treating all bursts as one pool:
- LOS coefficients: mean/std per coefficient
- Predictors (AEF, DEM, hydro, etc.): mean/std per channel
- Distances: mean/std

Apply these same scalers uniformly at inference. **Never compute stats on test or cal data.**

### 4.7 Handling EGMS L2b CSV structure

EGMS CSVs typically have columns like:
```
latitude, longitude, t_YYYY_MM_DD_1, t_YYYY_MM_DD_2, ..., t_YYYY_MM_DD_N
```

**Loading:**
```python
import pandas as pd, numpy as np
df = pd.read_csv("burst_id.csv")
epochs_raw = [parse_date(col) for col in df.columns if col.startswith("t_")]
los = df[[f"t_{e}" for e in epochs_raw]].values  # (N_points, T_raw)
xy = np.c_[df["longitude"], df["latitude"]]
```

Then project to EPSG:3035 and fit the temporal basis.

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

**Distance-binned calibration** (optional, stronger):
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

```python
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data

class InSARLOSGraphDataset(InMemoryDataset):
    """
    Pre-generated graphs from all bursts, masked-target training.
    Each sample: query node + k neighbours, from one burst.
    """
    def __init__(self, list_of_data_objects):
        super().__init__(None)
        self.data, self.slices = self.collate(list_of_data_objects)

# Training:
# For each epoch:
#   For each burst:
#     For each point in the burst (sample some or all):
#       Randomly select k neighbours from the same burst
#       Randomly mask some neighbouring points (don't predict them)
#       Build the query graph (target = this point, neighbours = unmasked)
#       Append to batch
#
# This is more flexible than pre-generating all graphs.

class InSARBurstDataset(torch.utils.data.IterableDataset):
    def __init__(self, burst_files, k=24, mask_rate=0.3, burst_sample_rate=0.2):
        self.bursts = [torch.load(f) for f in burst_files]  # pre-cached tensors
        self.k = k
        self.mask_rate = mask_rate
        self.burst_sample_rate = burst_sample_rate
    
    def __iter__(self):
        while True:
            for burst in self.bursts:
                # Sample points in this burst as targets
                n_points = burst['los_coeffs'].shape[0]
                n_targets = max(1, int(n_points * self.burst_sample_rate))
                target_idx = np.random.choice(n_points, n_targets, replace=False)
                
                for t_idx in target_idx:
                    # Find k neighbours (excluding self, + min-distance constraint)
                    neighbours = self.find_neighbours(burst, t_idx, k=self.k)
                    
                    # Mask some neighbours (simulation of unknown data)
                    mask = np.random.rand(len(neighbours)) > self.mask_rate
                    known_idx = neighbours[mask]
                    target_neighbours = neighbours[~mask]
                    
                    # Build graph (only known_idx inform the target)
                    graph = build_query_graph(
                        target_idx, known_idx, burst, ...)
                    yield graph
```

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

### 7.3 Training loop (pseudocode)

```python
model = InSARGNN(...)
ensemble_models = [InSARGNN(...) for _ in range(M)]
optimizers = [torch.optim.AdamW(m.parameters(), lr=3e-4) for m in ensemble_models]

for member_id, (model, opt) in enumerate(zip(ensemble_models, optimizers)):
    # Bootstrap sample of training bursts
    train_bursts_sample = random.sample(train_bursts, size=len(train_bursts))
    
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

Optionally, train with MSE on the mean only for a few epochs before switching to Gaussian NLL. This stabilizes variance head learning.

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

**Status: point of concern, not decided.**

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

---

---

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

## Quick Reference Checklist

- [ ] EGMS L2b bursts downloaded, per-burst CSVs cached as Parquet/torch
- [ ] Temporal basis fitted and cached (8–15 coefficients per point)
- [ ] Look-vector fit per (track, sub-swath) from incidence + track angle (§3.7)
- [ ] AlphaEarth / DEM / hydrogeology rasters in EPSG:3035, stacked
- [ ] Handpicked regions chosen, test subsets reserved
- [ ] Min-distance random draw placed, stratified by land cover
- [ ] Train / calibration / test split assigned at burst level
- [ ] Global normalization stats computed on train split
- [ ] Model architecture implemented, ensemble × M copies
- [ ] Masked-target subgraph training loop working
- [ ] Heteroscedastic NLL loss + (optional) reprojection loss
- [ ] Uncertainty propagation: Φ @ coeffs, G⁻¹ @ LOS
- [ ] Conformal calibration quantile computed on cal blocks
- [ ] LOS validation: held-out points, per-orbit metrics
- [ ] Product validation: GNSS comparison, decomposition quality
- [ ] Baseline (regression-kriging) implemented and compared
- [ ] Per-stratum performance analysis
- [ ] Spatial maps and time series plots for key regions

---

**End of reference document.**
