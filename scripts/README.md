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
# thresholds given explicitly
uv run s02_preprocessing.py --project-dir PROJECTDIR --egms-coherence 0.55 --egms-ampdisp-max 0.58
# or thresholds derived from a sample of the data
uv run s02_preprocessing.py --project-dir PROJECTDIR --egms-compute-thresholds
```

| Argument | Default | Description |
|---|---|---|
| `--project-dir` | required | Project root on scratch (same as Step 1's `--projectdir`); the parameter log goes to `PROJECTDIR/param_log/` |
| `--no-egms` | runs | Skip the EGMS preprocessing |
| `--no-gnss` | runs | Skip the GNSS preprocessing |
| `--no-worldcover` | runs | Skip the WorldCover tile merge |
| `--egms-coherence` | – | Minimum `temporal_coherence` to keep a point. Required unless `--egms-compute-thresholds` |
| `--egms-ampdisp-max` | – | Maximum `amplitude_dispersion` to keep a point. Required unless `--egms-compute-thresholds` |
| `--egms-compute-thresholds` | off | Derive both thresholds as quantiles instead. Can't be combined with the two values above |
| `--egms-coherence-q` | `0.05` | Quantile used for the coherence threshold |
| `--egms-ampdisp-q` | `0.95` | Quantile used for the amplitude dispersion threshold |
| `--egms-sample-every` | `20` | For the quantiles, use every n-th row of each csv |
| `--egms-file-every` | `3` | For the quantiles, use every n-th csv |

All three steps run by default; each is independently skippable. Flags are switches without a value (`--no-gnss`, not `--no-gnss=True`).

**Input:** `assets/study_areas_gdf` and the raw EGMS csvs from Step 1.

**Output:**
- `<burst>_PREPROCESSED.csv` next to every raw burst csv: only points passing both thresholds, only the columns `pid`, `latitude`, `longitude`, `easting`, `northing`, `incidence_angle`, `track_angle`, `los_east`, `los_north`, `los_up`, `mean_velocity`. Raw csvs stay untouched; dropped columns can be joined back via `pid`.
- `assets/egms_norm_stats.json` (gitignored): mean/std of `mean_velocity` and `incidence_angle`, computed from the preprocessed csvs of **train regions only**, plus the thresholds used. The csvs keep raw values; normalization is applied later at read time.
- `assets/study_areas_gdf` updated with `gnss_station_codes_filtered`/`gnss_station_paths_filtered`/`gnss_velocities`/`gnss_filter_params` (val regions) and `wc_merged_path`.
- `PROJECTDIR/param_log/s02_preprocessing_log_<timestamp>.log`: all arguments of the run.

What each part does:
- **EGMS** (`preprocess_egms`): keeps points with `temporal_coherence >= --egms-coherence` and `amplitude_dispersion <= --egms-ampdisp-max` (`rmse` isn't used: it correlates at r≈−0.93 with coherence). Computed thresholds are pooled over all splits and printed to the job output.
- **GNSS** (`preprocess_gnss`): drops val-region stations with <80% temporal coverage or a gap >120 days over 2019.0–2024.0, despikes the survivors (local median/MAD + Mann-Whitney test, see `remove_outlier`) and fits an east/north/up linear trend per station (`ve`/`vn`/`vu`, mm/yr).
- **WorldCover** (`preprocess_esa10mwc`): merges each region's individual WorldCover tiles into one `worldcover_merged.tif`, safe to rerun.

Run via `scripts/slurm/02_preprocessing.slurm`. The raw EGMS csvs total ~1.1 TB, so the EGMS part reads a lot of data; check `--time` before submitting.

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
