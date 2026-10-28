# Pipeline Scripts

Reference for every step in `python/`, what it needs as input, what it produces, and how to launch it. Full design rationale lives in [`../project.md`](../project.md); this page only documents the scripts as they exist, including what's still unimplemented.

Every step runs with [uv](https://docs.astral.sh/uv/) from `scripts/python/` (`module load uv` on the cluster). Anything that takes more than a few minutes or needs real memory/CPU goes through SLURM via the matching script in `scripts/slurm/` (submit from `scripts/slurm/`; adjust `--partition`/`--account` to your cluster) — never run those directly on the login node.

---

## Step 1 — Download Data

```bash
uv run s01_download_data.py --project-dir PROJECTDIR --real-sub-buffer-value=150000 --n-rnd=35 --mindist-rnd=150000 --maxit-rnd=200 --egms-rand-extension=15000
```

| Argument | Example | Description |
|---|---|---|
| `--project-dir` | `/dss/.../LOSTInPLACE` | Root of the project's data tree (needs a few hundred GB free); `data/{train,dev,cal,val}/` and `checkpoints/` (model weights of the later steps) are created under it |
| `--real-sub-buffer-value` | `150000` | Minimum gap in metres between the query areas of the real-subsidence regions and those of the random regions |
| `--n-rnd` | `35` | Number of random regions to sample across the EGMS coverage area |
| `--mindist-rnd` | `150000` | Minimum distance in metres between the random region centres; keep it ≥ 2 · (`egms-rand-extension` · √2 + `CLIP_BUFFER_M`) so their query areas can't overlap |
| `--maxit-rnd` | `200` | Upper bound on sampling attempts for the random regions |
| `--egms-rand-extension` | `15000` | Half side length in metres of the square core of a random region (15000 → 30 × 30 km) |

**Regions:** every region is a **core polygon** (seeds) plus a ring of `CLIP_BUFFER_M` = 5000 m (module constant in s01, stored per region as `clip_buffer_m`). EGMS is queried with core + ring. The EGMS search only uses the min/max lon/lat of the query polygon, so every returned burst is checked against the actual query area; per orbit direction, all intersecting bursts of the relative orbit with the largest overlap are downloaded. Right after unzipping, every burst csv is **clipped in place** to its region's core + ring (rows only, all columns kept unchanged); bursts are never deleted, the minimum-overlap check follows in s03. s03 only uses the overlap of each burst with the core (edge margin `--buffer-m` inside it); the ring is downloaded as a reserve but not used for now. Random regions are squares around random points, sampled outside a zone around the real-subsidence cores so that no query areas overlap. GNSS stations are taken from the core of the val regions only.

**Input:** `assets/real_subsidence_polygonbased.gpkg` (handpicked regions, tracked in git), `assets/token.jwt` (EGMS API service-account key), `assets/egms_coverage_countries.geojson`.

The real-subsidence gpkg may be in any CRS (s01 converts it to EPSG:3035 when reading) and holds one single-part polygon per region (the core area where seeds are placed; the `CLIP_BUFFER_M` ring is added by s01 for the EGMS query and the clipping), with the columns `location` (unique name) and `set` (0 = train, 1 = dev, 2 = cal, 3 = val). The tracked `assets/real_subsidence_pointbased.gpkg` is the original point-based version and is not read anymore.

**Output:** EGMS L2b CSVs clipped to core + `CLIP_BUFFER_M` (one relative orbit per direction per region) under `data/{train,dev,cal,val}/<location>/{ascending,descending}/`; GNSS time series for val-region stations; ESA WorldCover tiles per region; `assets/study_areas_gdf_local.gpkg` + `assets/study_areas_gdf_local_complex.json` (the region table everything downstream reads). `study_areas_io` writes and reads only these gitignored `_local` files; the region table is a run result, git only holds the inputs needed to start (the handpicked gpkg and the coverage geojson).

Downloads, in order: EGMS products for the handpicked regions, EGMS for the randomly sampled regions (train/dev/cal only), GNSS station time series (val regions only), ESA WorldCover tiles. Needs outbound HTTPS. Run via `scripts/slurm/01_download_data.slurm` (the last run took ~7 h for 71 regions, 875 bursts, 270 GB after clipping). At the end of a successful run the Markdown log `assets/param_logs/s01_download_data_<timestamp>.md` is written (run times, arguments, regions and bursts per set, failed downloads, GNSS stations per val region).

---

## Step 2 — Preprocessing

```bash
# thresholds given explicitly
uv run s02_preprocessing.py --egms-coherence 0.55 --egms-ampdisp-max 0.58
# or thresholds derived from a sample of the data
uv run s02_preprocessing.py --egms-compute-thresholds
```

| Argument | Default | Description |
|---|---|---|
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

**Input:** the region table (`assets/study_areas_gdf_local*`) and the clipped EGMS csvs from Step 1.

**Output:**
- `<burst>_PREPROCESSED.csv` next to every raw burst csv: only points passing both thresholds, only the columns `pid`, `latitude`, `longitude`, `easting`, `northing`, `incidence_angle`, `track_angle`, `los_east`, `los_north`, `los_up`, `mean_velocity`. Raw csvs stay untouched; dropped columns can be joined back via `pid`.
- `assets/egms_norm_stats.json` (gitignored): mean/std of `mean_velocity` and `incidence_angle`, computed from the preprocessed csvs of **train regions only**, plus the thresholds used. The csvs keep raw values; normalization is applied later at read time.
- the region table updated with `gnss_station_codes_filtered`/`gnss_station_paths_filtered`/`gnss_velocities`/`gnss_filter_params` (val regions) and `wc_merged_path`.
- `assets/param_logs/s02_preprocessing_<timestamp>.md` (gitignored), written at the end of a successful run: run times, arguments, the thresholds used, preprocessed bursts per set, the normalization stats, GNSS stations per val region before/after the quality filter, merged WorldCover files.

What each part does:
- **EGMS** (`preprocess_egms`): keeps points with `temporal_coherence >= --egms-coherence` and `amplitude_dispersion <= --egms-ampdisp-max` (`rmse` isn't used: it correlates at r≈−0.93 with coherence). Computed thresholds are pooled over all splits, printed to the job output and written to the log.
- **GNSS** (`preprocess_gnss`): drops val-region stations with <80% temporal coverage or a gap >120 days over 2019.0–2024.0, despikes the survivors (local median/MAD + Mann-Whitney test, see `remove_outlier`) and fits an east/north/up linear trend per station (`ve`/`vn`/`vu`, mm/yr).
- **WorldCover** (`preprocess_esa10mwc`): merges each region's individual WorldCover tiles into one `worldcover_merged.tif`, safe to rerun.

Run via `scripts/slurm/02_preprocessing.slurm` (the last run took ~40 min on the clipped bursts).

---

## Step 3 — Build Training Dataset

```bash
uv run s03_build_training_dataset.py --k-nearest=25 --density=0.66 --min-dist=50 --buffer-m=2000 --max-dist=2000
```

| Argument | Default | Description |
|---|---|---|
| `--k-nearest` | required | Number of pool points per seed; also sets the seed capacity `n_center // (4 · k) · density` |
| `--density` | required | Share (0, 1] of the burst capacity used for seeds |
| `--min-dist` | `50` | Minimum distance in metres between any two seeds |
| `--buffer-m` | `2000` | Edge margin in metres inside the overlap of burst and core: no seeds there, but its points can be pool points |
| `--max-dist` | `2000` | Maximum distance in metres of a pool point from its seed |

**Input:** the region table and the `*_PREPROCESSED.csv` files from Step 2, for the train, dev and cal regions (val regions only need their GNSS stations); the AlphaEarth satellite embedding (GSE, year `GSE_YEAR` = 2025) read directly from the open bucket. The tile index (0.8 GB csv) is downloaded once and kept as `assets/gse_index.parquet` (gitignored).

**What happens, separately for each set (train, dev, cal):**
1. **Budget:** per region the seed-eligible points are counted (burst hull ∩ core, bursts with ≥ 4 km² and ≥ 1,000 points, shrunk by the edge margin). Every region of the set gets the same seed budget, the seed capacity of an average region of the set; large regions are capped, small ones give everything they have.
2. **Per burst:** the overlap with the core is kept (all its points can be pool points), `in_center` marks the points outside the edge margin; the burst gets min(budget · its share of the region's `in_center` points, its own capacity) seeds.
3. **Seeds** (`select_seed_nodes`): 2/3 on a 2 × 2 km grid, 1/3 stratified over the WorldCover classes, with a minimum distance between all seeds. Land cover is never a model feature.
4. **Pools** (`select_candidate_pool`): seeds in random order, each gets up to 25 points within `--max-dist`, angular-weighted (8 sectors, nearest free point per sector first); no point is used twice and seeds never end up in another seed's pool. The minimum pool size is applied at training time.
5. **GRAPH table:** only the pool points (seed included) are kept and saved per burst; then the GSE is sampled per region: every tile touching the core is opened once and every needed 10 × 10 km block is read once for all bursts of the region (nodata = −128 is not accepted, so another tile can still fill the point).

**Output:**
- `<burst>_GRAPH.parquet` next to every burst csv that got seeds: `pid`, `seed_pid` (pid of the pool's seed), `seed`, `latitude`, `longitude`, `easting`, `northing`, `incidence_angle`, `mean_velocity` (raw values, normalized at training time with `egms_norm_stats.json`), `gse_0` … `gse_63` (int8, raw quantised, de-quantised at training time with `x = ((v/127.5)**2) · sign(v)`). One file per burst, because `pid` is only unique within a burst.
- the region table updated with `n_center_region`, `n_seed_points_per_region` and `graph_file_list`.
- `assets/param_logs/s03_build_training_dataset_<timestamp>.md` (gitignored): run times, arguments, seeds and pool points per set, the budget calculation per set, seeds per region (capacity, budget, capped or not), the seed placement (grid vs land cover) and the points without GSE.

**What happens at training time, not in this step:** every `__getitem__` draws a random subset of one pool, picks a random query point in it and hides its velocity, computes all pairwise geodesic distances/bearings from lat/lon, connects every node to its k nearest nodes (k=3 assumed) and builds the edge features. So the graph is an interconnected network, not a star around a fixed centre, and the subset, the graph and the query change every epoch (§7.1).

Run via `scripts/slurm/03_build_training_dataset.slurm`. The GSE reads go over the network (~50–70 GB for all cores, ~3 s per block).
