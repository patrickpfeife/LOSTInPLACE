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

**Status: design only, not yet implemented** (`s03_build_training_dataset.py` is a skeleton). Documented here so the intended shape is in one place; see `project.md` §4.3–§4.4, §7.1 and §10.14–§10.16 for rationale and open questions.

**Input:** per-burst `*_PREPROCESSED.csv` files from Step 2, for train-split regions only (`set == 0`). Val regions are excluded: Tier 2 validation (§8.1) queries the model at GNSS station locations, not through candidate pools.

**Output (planned):** a few large shard files, one per region, each holding many **candidate pools** as flat arrays plus an offset array (`pool_ptr`): `lat`/`lon` (float64), `mean_velocity` (float32), `incidence` (float32), `gse` (int8, 64 per point), `land_cover` (uint8, metadata only), and the burst each pool came from. A pool never mixes bursts or orbits. No graph is stored.

**What happens at training time, not in this step:** every `__getitem__` draws a random subset of one pool, picks a random query point in it and hides its velocity, computes all pairwise geodesic distances/bearings from lat/lon, connects every node to its k nearest nodes (k=3 assumed) and builds the edge features. So the graph is an interconnected network, not a star around a fixed centre, and the subset, the graph and the query change every epoch (§7.1).

Planned per-burst steps:
1. Clip to the area of interest (EGMS downloads the whole winning relative orbit), then cut a buffer off the burst edges so every seed has neighbours in all directions.
2. Cap the number of seeds drawn from the burst to a pre-determined budget (§10.14, formula open).
3. Pick seeds stratified by land cover, with a grid or minimum spacing for spatial spread. Land cover is never a model feature.
4. For each seed, gather its pool by angular selection (about 8 sectors, nearest unused point per sector), sequentially in one loop with no point reused across pools.
5. Sample GSE at the pool points from a transient windowed read per region and keep only the int8 vectors (§10.16). Write the region's shard.

No SLURM script or CLI arguments exist yet.
