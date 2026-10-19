# Pipeline Scripts

Reference for every step in `python/`, what it needs as input, what it produces, and how to launch it. Full design rationale lives in [`../project.md`](../project.md); this page only documents the scripts as they exist, including what's still unimplemented.

Every step runs with [uv](https://docs.astral.sh/uv/) from `scripts/python/` (`module load uv` on the cluster). Anything that takes more than a few minutes or needs real memory/CPU goes through SLURM via the matching script in `scripts/slurm/` (submit from `scripts/slurm/`; adjust `--partition`/`--account` to your cluster) — never run those directly on the login node.

---

## Step 1 — Download Data

```bash
uv run s01_download_data.py --projectdir PROJECTDIR --real_sub_buffer_value=150000 --n_rnd=35 --mindist_rnd=150000 --maxit_rnd=200
```

| Argument | Example | Description |
|---|---|---|
| `--projectdir` | `/dss/.../LOSTInPLACE` | Root of the project's data tree (needs tens of GB free); `data/{train,dev,cal,val}/` is created under it |
| `--real_sub_buffer_value` | `150000` | Buffer radius in metres around each handpicked real-subsidence point excluded from random sampling |
| `--n_rnd` | `35` | Number of extra random study areas to sample across the EGMS coverage area |
| `--mindist_rnd` | `150000` | Minimum distance in metres required between those random areas |
| `--maxit_rnd` | `200` | Upper bound on sampling attempts for the random areas |

**Input:** `assets/real_subsidence_pointbased.gpkg` (handpicked regions), `assets/token.jwt` (EGMS API service-account key), `assets/egms_coverage_countries.geojson`.

**Output:** EGMS L2b CSVs (one relative orbit per direction per region) under `data/{train,dev,cal,val}/<location>/{ascending,descending}/`; GNSS time series for val-region stations; ESA WorldCover tiles per region; `assets/study_areas_gdf.gpkg` + `assets/study_areas_gdf_complex.json` (the region table everything downstream reads).

Downloads, in order: EGMS products for the handpicked regions, EGMS for the randomly sampled regions (train/dev/cal only), GNSS station time series (val regions only), ESA WorldCover tiles. Needs outbound HTTPS. Run via `scripts/slurm/01_download_data.slurm`.

---

## Step 2 — Preprocessing

```bash
uv run s02_preprocessing.py [--no-egms] [--no-gnss] [--no-worldcover]
```

| Argument | Description |
|---|---|
| `--no-egms` | Skip the EGMS preprocessing step |
| `--no-gnss` | Skip the GNSS preprocessing step |
| `--no-worldcover` | Skip the WorldCover tile merge |

All three run by default; each is independently skippable.

**Input:** `assets/study_areas_gdf` (from Step 1).

**Output:** `assets/study_areas_gdf` updated in place with `gnss_station_codes_filtered`/`gnss_station_paths_filtered`/`gnss_velocities`/`gnss_filter_params` (val regions) and `wc_merged_path` (one merged WorldCover GeoTIFF per region).

Currently implemented:
- **GNSS** (`preprocess_gnss`): drops val-region stations with <80% temporal coverage or a gap >120 days over 2019.0–2024.0, despikes the survivors (local median/MAD + Mann-Whitney test, see `remove_outlier`) and fits an east/north/up linear trend per station.
- **WorldCover** (`preprocess_esa10mwc`): merges each region's individual WorldCover tiles into one `worldcover_merged.tif`, safe to rerun.

**Not yet implemented:** `preprocess_egms` is a stub. Per `project.md` §3.5/§4.6, this is where EGMS quality filtering (`rmse`/`temporal_coherence`/`amplitude_dispersion`/`mp_type` thresholds) and per-point temporal basis-coefficient fitting belong, ahead of Step 3.

Run via `scripts/slurm/02_preprocessing.slurm`.

---

## Step 3 — Build Training Dataset

**Status: design only, not yet implemented** (`s03_build_training_dataset.py` is pseudocode/skeleton). Documented here so the intended shape is in one place; see `project.md` §4.3–§4.4 and §10.14 for the full rationale and open questions.

**Input:** per-burst EGMS CSVs already quality-filtered by Step 2's `preprocess_egms`, for train-split regions only (`set == 0`). Val regions are excluded entirely — Tier 2 validation (§8.1) queries the model directly at GNSS station locations, not through a cached pool.

**Output (planned):** one cached tensor bundle per burst csv — coordinates, per-year velocity, GSE embedding, land-cover class, and precomputed angular-balanced neighbour indices for every selected anchor, saved as `.pt`. The masked PyG `Data` graph itself is rebuilt fresh at `__getitem__` time every epoch, not cached, since the mask must vary per epoch (§7.1).

Planned per-burst steps:
1. Clip to the actual area of interest (EGMS downloads the whole winning relative orbit, far larger than the original query bbox), then buffer in from the burst's edges so every anchor has full neighbourhood coverage.
2. Cap the number of anchors drawn from this burst to a pre-determined budget — exhaustive consumption would yield on the order of 10⁸ examples pooled across all bursts (§10.14, formula still open).
3. Select anchors stratified by dominant land-cover class, for a spatially/class-diverse sample — land cover is never a model feature, only used here and optionally for per-class evaluation breakdown (§8.4).
4. Build each anchor's candidate pool via angular-balanced k-NN (sector-wise nearest unused point), sequentially, with no point reused across graphs.
5. Attach node/edge features (GSE embedding per point, edge distance + direction) and save the tensor bundle.

No SLURM script or CLI arguments exist yet.
