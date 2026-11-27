#######################################
# s06b_baseline_comparison

"""
Evaluation of trained s06 runs on dev, on the CPU after the GPU training: the baselines (IDW, neighbour mean, ordinary
kriging) on exactly the same dev graphs as the GNN, cached once per dataset and recomputed only if the dev data or the
settings changed, GNN vs baselines overall, per region, per velocity and per distance range, the graph statistics of
train and dev, and one markdown report for one run or all runs of a grid.
Run via scripts/slurm/submit_train_eval.sh (after s06) or scripts/slurm/06b_baseline_comparison.slurm.
"""

##############################################################################
#============================================================================#
# 1) Import Statements
#============================================================================#
##############################################################################

import argparse
import hashlib
import json
import os
import numpy as np
import polars as pl
import torch
import matplotlib
matplotlib.use("Agg")      # no display on the compute nodes, figures are only saved
import gstools as gs

import s01_download_data as s01
import s06_train_model as s06
import xx_plotting as xx

from pathlib import Path
from datetime import datetime
from collections import defaultdict

################################################
# 1.1) Constants
################################################

# automatic variogram fit of the ordinary kriging baseline (compute_variograms, project.md §7.5): the largest lag is
# the VARIO_PERCENTILE of the query -> visible point distances of a dev region x orbit times VARIO_LAG_FACTOR, the bin
# edges are 0 plus VARIO_N_BINS geometrically spaced edges from VARIO_FIRST_EDGE_M (fine bins at short lags), and bins
# with fewer than VARIO_MIN_PAIRS point pairs are dropped. The fit uses at most VARIO_SAMPLING_SIZE random points of
# the group (the number of pairs grows with n^2), the fitted range is bounded by VARIO_LEN_FACTOR x the largest lag
VARIO_PERCENTILE = 95
VARIO_LAG_FACTOR = 1.5
VARIO_FIRST_EDGE_M = 10
VARIO_N_BINS = 50
VARIO_MIN_PAIRS = 30
VARIO_SAMPLING_SIZE = 5000
VARIO_LEN_FACTOR = 1.5

# the draw settings that decide which dev graphs are built (saved in every s06 checkpoint); all evaluated runs must share them
DRAW_KEYS = ["subset_min", "subset_max", "min_graph_size", "k_graph"]

# the compared methods: display name -> (prediction column, sigma column or None). The GNN comes first, so the
# report tables take their columns (incl. inside_90) from its row (s01.table_to_md uses the keys of the first row)
RESULT_RECIPE = {"GNN": ("mu_mm", "sigma_mm"),
                 "IDW1": ("idw_1", None),
                 "IDW2": ("idw_2", None),
                 "Neighbour Mean": ("neighbour_mean", None),
                 "Ordinary Kriging": ("ok_mu", "ok_sigma")}

##############################################################################
#============================================================================#
# 2) Helper Functions
#============================================================================#
##############################################################################

################################################
#========= Section 1: Baseline Helpers ========#
################################################


def graph_stats(pool, idx, q):
    """
    Describes one drawn graph: its size and connectivity, the distances from the query to the visible points and the
    velocities of the visible points around the query. The keys are the columns xx.GRAPH_STAT_LABELS expects
    (graph-statistics and win-rate figures).

    Arguments:
        pool -- dict, the pool (see load_pools)
        idx -- numpy int array (m,), positions of the subset's points in the pool (see PoolDataset.draw_subset_method)
        q -- int, position of the query within idx

    Returns:
        stats -- dict: n_nodes, mean_degree, q_degree (number of edges into the query), span_m (largest distance in the
            graph), q_nn_m / q_median_m (nearest / median distance query -> visible points), edge_median_m (median edge
            length), query_vel, query_abs_vel, vel_std / vel_range (std / range of the visible velocities) and
            q_dev_from_mean (|query velocity - mean of the visible velocities|); distances in m, velocities in mm/yr
    """
    # the same distances and the same Delaunay graph the GNN gets (s06.build_graph)
    dist, _ = s06.pairwise_geodesic(pool["lat"][idx], pool["lon"][idx])
    edges = s06.delaunay_edges(pool["x"][idx], pool["y"][idx]).numpy()     # every edge in both directions
    # positions of the visible points (all but the query), their raw velocities and their distances to the query
    vis = np.delete(np.arange(len(idx)), q)
    v, d_q = pool["vel"][idx], dist[q, vis]
    # every edge is stored in both directions, so n_edges / n_nodes is the mean degree and the edges ending in q are
    # the query's degree. For train this is one draw per pool, not every graph the model sees over the epochs (the
    # subset is redrawn every epoch), but the redraws come from the same pool and settings, so the distributions match
    return {"n_nodes": len(idx), "mean_degree": edges.shape[1] / len(idx), "q_degree": int((edges[1] == q).sum()),
        "span_m": dist.max(), "q_nn_m": d_q.min(), "q_median_m": np.median(d_q),
        "edge_median_m": np.median(dist[edges[0], edges[1]]),
        "query_vel": float(v[q]), "query_abs_vel": abs(float(v[q])), "vel_std": float(v[vis].std()),
        "vel_range": float(np.ptp(v[vis])), "q_dev_from_mean": abs(float(v[q] - v[vis].mean()))}


def train_graph_stats(settings, seed, cache_dir, n=20000):
    """
    Graph statistics of the train set for the comparison with dev: n randomly sampled train pools, each drawn once
    like in the first training epoch. Cached like the baselines and recomputed only if the train data or the settings
    changed.

    Arguments:
        settings -- dict, the draw settings (DRAW_KEYS) of the evaluated runs
        seed -- int, seed of the pool sample and of the draws (the seed of the first evaluated run)
        cache_dir -- Path, folder of the cache (the train data folder)
        n -- int, number of sampled train pools

    Returns:
        stats -- polars.DataFrame, one row per sampled pool with the columns of graph_stats
    """
    # the train GRAPH files in the order s06 uses them, and the fingerprint of everything the statistics depend on
    train_files = train_dev_graph_files(0)
    parquet_path, meta_path = cache_dir / "train_graph_stats.parquet", cache_dir / "train_graph_stats.json"
    fingerprint = dataset_fingerprint(train_files, {**settings, "seed": seed, "n": n})

    # same fingerprint = same train data and settings, the cached statistics are still valid
    if parquet_path.exists() and meta_path.exists():
            meta = json.load(open(meta_path))
            if meta["fingerprint"] == fingerprint:
                print("train graph stats loaded from cache", flush=True)
                return pl.read_parquet(parquet_path)

    print("train graph stats computed (no cache, or train data or settings changed)", flush=True)
    # n sampled pools, drawn like in s06's training (fixed_draws=False, epoch 0); no GSE and no normalization needed
    index = s06.sample_pools(s06.build_pool_index(train_files), n, seed)
    ds = s06.PoolDataset(s06.load_pools(index, False, settings["min_graph_size"]), None, settings["k_graph"],
                         settings["subset_min"], settings["subset_max"], settings["min_graph_size"], False, seed=seed)

    # one row per pool, and the cache with its fingerprint next to it
    stats = pl.DataFrame([graph_stats(*ds.draw_subset_method(i)) for i in range(len(ds))])
    stats.write_parquet(parquet_path)
    with open(meta_path, "w") as f:
        json.dump({"fingerprint": fingerprint, "created": datetime.now().isoformat()}, f, indent=2)
    return stats


def idw(pool, idx, q, power):
    """
    Predicts the query's velocity by inverse distance weighting of all visible points of the same dev graph
    (baseline). Distances below 1 m are clipped, so very close points don't get an unbounded weight.

    Arguments:
        pool -- dict, the pool (see load_pools)
        idx -- numpy int array (m,), positions of the subset's points in the pool (see PoolDataset.draw_subset_method)
        q -- int, position of the query within idx
        power -- int or float, power of the inverse distance (1 or 2)

    Returns:
        prediction -- float, the predicted velocity in mm/yr
    """

    # coordinates and raw velocities (mm/yr) of the subset, in subset order (positions 0 ... m-1)
    lat, lon, vel = pool["lat"][idx], pool["lon"][idx], pool["vel"][idx]
    # pairwise distances between all points of the subset; the bearings are not needed
    dist, _ = s06.pairwise_geodesic(lat, lon)
    # we only need the distances from the query to the other points, which are in row q of the matrix
    query_dist_row = dist[q]
    # the query itself is deleted (its distance to itself is 0)
    query_dist_row = np.delete(query_dist_row, q)
    # its velocity is deleted as well, so distances and velocities stay in the same order
    vel = np.delete(vel, q)
    # very close points (and distance 0) would get an unbounded weight: distances below 1 m are set to 1 m
    query_dist_row = np.maximum(query_dist_row, 1.0)
    # inverse distance weights
    weights = 1 / query_dist_row ** power
    # weighted mean by vector operation
    result = np.sum(vel * weights) / np.sum(weights)

    return float(result)


def neighbour_mean(pool, idx, q, k_graph):
    """
    Predicts the query's velocity as the mean of its k_graph nearest visible points of the same dev graph (baseline).

    Arguments:
        pool -- dict, the pool (see load_pools)
        idx -- numpy int array (m,), positions of the subset's points in the pool (see PoolDataset.draw_subset_method)
        q -- int, position of the query within idx
        k_graph -- int, number of nearest visible points that are averaged

    Returns:
        mean -- float, the predicted velocity in mm/yr
    """

    # coordinates and raw velocities (mm/yr) of the subset, in subset order (positions 0 ... m-1)
    lat, lon, vel = pool["lat"][idx], pool["lon"][idx], pool["vel"][idx]

    # pairwise distances of the subset; the bearings are not needed
    dist, _ = s06.pairwise_geodesic(lat, lon)

    # distances from the query to all subset points; the query itself gets inf so it is never its own neighbour
    d_q = dist[q].copy()
    d_q[q] = np.inf

    # positions (within the subset) of the k nearest visible points
    nearest = np.argsort(d_q)[:k_graph]

    # mean of their velocities, as a plain python float
    return float(vel[nearest].mean())


def compute_variograms(dev_ds):
    """
    Fits one variogram per dev region x orbit for the ordinary kriging baseline (project.md §7.5). The points are all
    points of the group's dev pools minus the dev queries (fixed draws), so the query values never enter the fit.
    Empirical semivariance per bin with the robust Cressie-Hawkins estimator (Cressie & Hawkins 1980), weighted least
    squares fit with the Cressie (1985) weights N(h)/gamma(h)^2 of an Exponential, Spherical and Gaussian model with
    nugget, the best by r2 is kept. Distances (bins, len_scale) are in metres.

    Arguments:
        dev_ds -- PoolDataset, the dev dataset (fixed_draws=True)

    Returns:
        variograms -- dict, (region, orbit) -> fitted gstools model (see o_kriging)
        vario_log -- list of dicts, one per group: points, kept bins, largest lag and the fitted parameters (report)
    """

    collected = defaultdict(lambda: {"lat":[], "lon": [], "vel": [], "dist": []})

    for i in range(len(dev_ds)):
        # extract necessary information from the graph
        pool, idx, q = dev_ds.draw_subset_method(i)
        dist, _ = s06.pairwise_geodesic(pool["lat"][idx], pool["lon"][idx])

        # mask over the whole pool (not only the subset): everything except the query enters the variogram.
        # q is the position within idx, so the query's position in the pool is idx[q]
        query_pos = idx[q] 
        keep_mask = np.ones(len(pool["vel"]),dtype=bool)
        keep_mask[query_pos] = False
        key = (pool["region"], pool["orbit"])

        # distances from the query to the visible points (row q, without the 0 to itself): the distances the kriging
        # actually has to bridge, they set the largest lag of the group's variogram
        dist_to_q = np.delete(dist[q], q)

        # the defaultdict creates the empty lists the first time a key is accessed; every pool adds one array per list
        for x in ["lat", "lon", "vel"]:
            collected[key][x].append(pool[x][keep_mask])
        collected[key]["dist"].append(dist_to_q)

    variograms = {}
    vario_log = []

    # one fit per group: key = (region, orbit), value = dict of lat, lon, vel, dist (lists with one array per pool)
    for (region, orbit), data in collected.items():
        # one flat array per variable
        for x in ["lat", "lon", "vel", "dist"]:
            data[x] = np.concatenate(data[x])

        n = len(data["vel"])

        # bins up to the largest lag the kriging needs in this group, fine near 0 and coarse further out
        max_dist_to_q = np.percentile(data["dist"], VARIO_PERCENTILE) * VARIO_LAG_FACTOR
        bins = np.concatenate([[0.0], np.geomspace(VARIO_FIRST_EDGE_M, max_dist_to_q, VARIO_N_BINS)])

        # empirical semivariance per bin (Cressie-Hawkins estimator), from at most VARIO_SAMPLING_SIZE random points
        # (pairs grow with n^2); geo_scale in metres, so the bins are in metres
        bin_center, gamma, counts = gs.vario_estimate(pos=(data["lat"], data["lon"]), field=data["vel"], bin_edges=bins, latlon=True,
                                  geo_scale=gs.KM_SCALE * 1000, estimator="cressie", return_counts=True, sampling_size=min(n, VARIO_SAMPLING_SIZE),
                                  sampling_seed=0)

        # drop bins with too few pairs (noisy) and empty bins (gamma = 0 would divide by zero in the weights)
        valid = (counts >= VARIO_MIN_PAIRS) & (gamma > 0)
        bin_center, gamma, counts = bin_center[valid], gamma[valid], counts[valid]

        # Cressie (1985) weights N(h)/gamma(h)^2, with the empirical gamma. gstools squares the weights internally
        # (sigma = 1/weights in scipy's curve_fit), hence the root
        weights = np.sqrt(counts) / gamma

        # fit every candidate model with nugget and keep (r2, model); r2 is computed on the bins
        fits = []
        for model in [gs.Exponential, gs.Spherical, gs.Gaussian]:
            x = model(dim=2, latlon=True, geo_scale=gs.KM_SCALE * 1000)
            # upper bound of the range, otherwise variograms that keep rising (trend) end at absurd ranges
            x.len_scale_bounds = [0, VARIO_LEN_FACTOR * max_dist_to_q]
            _, _, r2 = x.fit_variogram(bin_center, gamma, nugget=True, weights=weights, max_eval=100000, return_r2=True)
            fits.append((r2, x))

        # the best model by r2 is used for the kriging, its parameters go to the report
        best_r2, best_model = max(fits, key=lambda t: t[0])

        variograms[(region, orbit)] = best_model
        vario_log.append({"region": region, "orbit": orbit, "n": n, "n_bins_kept": len(gamma),
                          "max_dist_to_q": max_dist_to_q, "model name": type(best_model).__name__,
                          "best model nugget": best_model.nugget, "best model var": best_model.var,
                          "best model len_scale": best_model.len_scale, "best R2": best_r2})

    return variograms, vario_log


def o_kriging(pool, idx, q, variograms):
    """
    Predicts the query's velocity by ordinary kriging from all visible points of the same dev graph (baseline), with
    the fitted variogram of the pool's region x orbit. exact=False treats the nugget as measurement noise, so the
    kriging variance includes it and sigma is comparable to the GNN's.

    Arguments:
        pool -- dict, the pool (see load_pools)
        idx -- numpy int array (m,), positions of the subset's points in the pool (see PoolDataset.draw_subset_method)
        q -- int, position of the query within idx
        variograms -- dict, (region, orbit) -> fitted gstools model (see compute_variograms)

    Returns:
        mu -- float, the predicted velocity (mm/yr)
        sigma -- float, the kriging standard deviation (mm/yr)
    """

    # the variogram of the pool's region and orbit
    model = variograms[(pool["region"], pool["orbit"])]

    # the visible points: the subset without the query
    lat = np.delete(pool["lat"][idx], q)
    lon = np.delete(pool["lon"][idx], q)
    vel = np.delete(pool["vel"][idx], q)

    # the query's position
    lat_q, lon_q = pool["lat"][idx][q], pool["lon"][idx][q]

    # kriging system of the visible points; positions as (lat, lon) because the model has latlon=True
    krig = gs.krige.Ordinary(model, cond_pos=(lat, lon), cond_val=vel, exact=False)
    # mean and variance at the query (positions as arrays, also for a single point; store=False keeps no field)
    mu, var = krig((np.array([lat_q]), np.array([lon_q])), return_var=True, store=False)

    # the variance can be slightly negative from rounding, hence the max
    return float(mu[0]), float(np.sqrt(max(var[0], 0.0)))


def evaluate_baselines(dev_ds, k_graph):
    """
    Predicts every dev query with the baselines, each from exactly the visible points of the GNN's dev graph (same
    subset and query, see PoolDataset.draw_subset_method). Runs once per run, the baselines don't depend on the model.

    Arguments:
        dev_ds -- PoolDataset, the dev dataset (fixed_draws=True)
        k_graph -- int, number of nearest visible points of the neighbour mean

    Returns:
        baselines -- polars.DataFrame, one row per dev graph in pool order: region, orbit, the query key graph_file and
            seed_pid (for the join with the GNN's predictions), y_mm and one column per baseline (mm/yr): neighbour_mean, idw_1, idw_2, ok_mu and the
            kriging standard deviation ok_sigma, plus the graph statistics of the dev graph (see graph_stats)
        vario_log -- list of dicts, the fitted variograms per region x orbit (see compute_variograms)
    """

    # list to capture the comparisons
    results = []

    # one variogram per dev region x orbit, fitted once before the loop (never per query)
    variograms, vario_log = compute_variograms(dev_ds)

    # len applied to dev_ds calls the method __len__ of that object; we iterate over every graph of the dev set in pool
    # order 0 ... n-1, the order in which the dev loader (shuffle=False) gives them to the GNN. The raw values of the
    # draw are used here, not the Data object (that only holds normalized features and no coordinates)
    for i in range(len(dev_ds)):
        pool, idx, q = dev_ds.draw_subset_method(i)
        # kriging gives a mean and a standard deviation
        ok_mu, ok_sigma = o_kriging(pool, idx, q, variograms)
        # one row per graph: the true value of the query and the prediction of every baseline
        results_per_graph = {"region": pool["region"], "orbit": pool["orbit"],
                             "graph_file": pool["graph_file"], "seed_pid": pool["seed_pid"],
                             "y_mm": pool["vel"][idx][q], "neighbour_mean": neighbour_mean(pool, idx, q, k_graph),
                             "ok_mu": ok_mu, "ok_sigma": ok_sigma,
                             "idw_1": idw(pool, idx, q, 1), "idw_2": idw(pool, idx, q, 2),
                             **graph_stats(pool, idx, q)}

        results.append(results_per_graph)
    # a dataframe built from the list of dicts, one row per graph
    return pl.DataFrame(results), vario_log


################################################
#======= Section 2: Run and Cache Helpers =====#
################################################
# Used by main, in the order they're first called there.

def load_run(run_dir):
    """
    Loads the settings and the files s06 wrote for one run. Runs with a missing file (failed or unfinished) are skipped.

    Arguments:
        run_dir -- Path, the run folder <project_dir>/checkpoints/<run_name>

    Returns:
        run -- dict with name, params (the settings saved in the checkpoint), predictions (polars.DataFrame, the
            GNN's dev predictions), history (list of dicts, loss per epoch) and metrics (dict, train/dev metrics),
            or None if a file is missing
    """
    # the four files s06 writes into the run folder
    names = ["best_model.pt", "dev_predictions.parquet", "history.json", "metrics.json"]
    if not all((run_dir / n).exists() for n in names):
        print(f"skipping {run_dir.name}: not all s06 files exist (failed or unfinished run)", flush=True)
        return None

    # the settings of the run are saved with the checkpoint (weights_only=False: params is a plain dict, not tensors)
    params = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)["params"]
    return {"name": run_dir.name, "params": params,
            "predictions": pl.read_parquet(run_dir / "dev_predictions.parquet"),
            "history": json.load(open(run_dir / "history.json")),
            "metrics": json.load(open(run_dir / "metrics.json"))}


def train_dev_graph_files(set):
    """
    Lists the train or dev GRAPH files from the region table, in exactly the same order as s06.build_datasets: the
    draws depend on the order of the pools and that comes from the order of the files, so this list must not be sorted.

    Arguments:
        set -- int, the set column of the region table: 0 = train, 1 = dev

    Returns:
        graph_files -- list of str, the GRAPH files of that set
    """
    gdf = s01.study_areas_io("read", base_path=Path(__file__).parent / "../../assets/study_areas_gdf")
    return [f for files in gdf.loc[gdf["set"] == set, "graph_file_list"] if files for f in files]


def dataset_fingerprint(dev_files, settings):
    """
    A hash of everything a cache depends on: the GRAPH files (path, size, modification time, so a rerun of s05
    changes it), the draw settings, the graph construction, the variogram settings and the version of the graph
    statistics (raise "graph_stats" whenever graph_stats changes, so both caches are recomputed). Used for the
    baselines (dev files) and the train graph statistics (train files).

    Arguments:
        dev_files -- list of str, the GRAPH files (see train_dev_graph_files)
        settings -- dict, the draw settings (DRAW_KEYS) of the evaluated runs (for the train statistics plus seed and n)

    Returns:
        fingerprint -- str, SHA-256 hex digest
    """
    content = {"files": [(str(f), os.path.getsize(f), os.path.getmtime(f)) for f in dev_files],
               "settings": settings,
               "graph": "delaunay",
               "vario": [VARIO_PERCENTILE, VARIO_LAG_FACTOR, VARIO_FIRST_EDGE_M, VARIO_N_BINS, VARIO_MIN_PAIRS,
                         VARIO_SAMPLING_SIZE, VARIO_LEN_FACTOR],
                "graph_stats": 1}
    
    # sort_keys: the same content always gives the same text and so the same hash
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def load_or_compute_baselines(dev_files, settings, cache_dir):
    """
    Returns the baselines of the dev graphs: from the cache if its fingerprint still matches, otherwise computed
    (evaluate_baselines) and written to the cache.

    Arguments:
        dev_files -- list of str, the dev GRAPH files (see dev_graph_files)
        settings -- dict, the draw settings (DRAW_KEYS) of the evaluated runs
        cache_dir -- Path, folder of the cache (the dev data folder)

    Returns:
        baselines -- polars.DataFrame, see evaluate_baselines
        vario_log -- list of dicts, the fitted variograms (see compute_variograms)
    """
    parquet_path, meta_path = cache_dir / "baselines.parquet", cache_dir / "baselines_meta.json"
    fingerprint = dataset_fingerprint(dev_files, settings)

    # same fingerprint = same dev data and settings, the cached baselines are still valid
    if parquet_path.exists() and meta_path.exists():
        meta = json.load(open(meta_path))
        if meta["fingerprint"] == fingerprint:
            print("baselines loaded from cache", flush=True)
            return pl.read_parquet(parquet_path), meta["vario_log"]

    print("baselines computed (no cache, or dev data or settings changed)", flush=True)
    # the same dev draws as in s06 (fixed draws, same settings); the baselines need neither GSE nor normalization stats
    pools = s06.load_pools(s06.build_pool_index(dev_files), False, settings["min_graph_size"])
    dev_ds = s06.PoolDataset(pools, None, settings["k_graph"], settings["subset_min"], settings["subset_max"],
                             settings["min_graph_size"], False, fixed_draws=True)
    baselines, vario_log = evaluate_baselines(dev_ds, settings["k_graph"])

    # the cache: the baselines and, next to them, the fingerprint they belong to
    baselines.write_parquet(parquet_path)
    with open(meta_path, "w") as f:
        json.dump({"fingerprint": fingerprint, "settings": settings, "created": datetime.now().isoformat(),
                   "vario_log": vario_log}, f, indent=2, default=float)
    return baselines, vario_log


################################################
#========= Section 3: Metric Helpers ==========#
################################################
# Used by compare_run.

def rmse_diff_ci(err_a, err_b, n_boot=1000, seed=0):
    """
    95 % paired bootstrap interval of RMSE(a) - RMSE(b): n_boot times the queries are drawn with replacement, with the
    same indices for both methods (paired: a hard query stays hard for both, so only the difference varies), and the
    2.5th and 97.5th percentile of the n_boot differences are returned. The queries are treated as independent; with
    spatially correlated queries the interval is somewhat too narrow (a block bootstrap over graph files would not be).

    Arguments:
        err_a, err_b -- numpy arrays (n,), the errors (prediction - truth) of two methods on the same queries
        n_boot -- int, number of resamples
        seed -- int, seed of the resampling, so the report is reproducible

    Returns:
        ci -- numpy array (2,), lower and upper limit in mm/yr; 0 outside the interval = significant difference
    """
    rng = np.random.default_rng(seed)
    # one difference per resample: n indices drawn with replacement, applied to both error arrays
    d = [np.sqrt(np.mean(err_a[i] ** 2)) - np.sqrt(np.mean(err_b[i] ** 2))
         for i in (rng.integers(0, len(err_a), len(err_a)) for _ in range(n_boot))]
    return np.percentile(d, [2.5, 97.5])


def per_group_metrics(dev_predictions, pred_col, group_col=None, sigma_col=None):
    """
    Computes the accuracy metrics of one prediction column against y_mm, overall or per group: n, RMSE, MAE, median
    absolute error and R² (1 - MSE / variance of y_mm), and with a sigma column the uncertainty metrics: the share
    inside pred ± z sigma for 50/90/95 %, the median width of the 90 % interval and the Spearman rank correlation of
    sigma and |error| (does a larger sigma go with a larger error?). Without sigma these keys are None, so every
    method has the same keys (s01.table_to_md takes its columns from the first row).

    Arguments:
        dev_predictions -- polars.DataFrame, one row per dev query with y_mm and the prediction columns
            (see compare_run)
        pred_col -- str, the prediction column (mu_mm, idw_1, ...)
        group_col -- str, list of str or None, column(s) to group by (region, ["region", "orbit"], abs_vel_bin, ...);
            None = one row over all queries
        sigma_col -- str or None, the predicted standard deviation (GNN: sigma_mm, kriging: ok_sigma)

    Returns:
        metrics -- list of dicts: one per group (sorted by the group), or a list with one dict without grouping
    """

    # the error and the truth as polars expressions, computed per row once they are used in select or agg
    err = pl.col(pred_col) - pl.col("y_mm")
    y = pl.col("y_mm")

    # a list of expressions for the accuracy metrics
    recipes = [pl.len().alias("n"), (err ** 2).mean().sqrt().alias("rmse"), err.abs().mean().alias("mae"),
        err.abs().median().alias("median_ae"), (1 - (err ** 2).mean() / ((y - y.mean()) ** 2).mean()).alias("r2")]
    # the uncertainty metrics; z = the two-sided normal quantile of the level (norm.ppf(0.5 + level / 200))
    s = pl.col(sigma_col) if sigma_col else None
    for level, z in [(50, 0.674), (90, 1.645), (95, 1.960)]:
        recipes.append(((err.abs() <= z * s).mean() if s is not None else pl.lit(None)).alias(f"inside_{level}"))
    recipes += [((2 * 1.645 * s).median() if s is not None else pl.lit(None)).alias("width_90"),
            (pl.corr(s, err.abs(), method="spearman") if s is not None else pl.lit(None)).alias("spearman_sigma_err")]

    # agg applies the expressions per group (sorted, group_by has no fixed order); select applies them to the whole
    # table and gives one row
    if group_col is not None:
        metrics = dev_predictions.group_by(group_col).agg(recipes).sort(group_col)
    else:
        metrics = dev_predictions.select(recipes)

    return metrics.to_dicts()


def per_method_table(joined, group_col, values, methods=tuple(RESULT_RECIPE)):
    """
    One row per group, one column per method (and metric): e.g. the RMSE of every method per region.

    Arguments:
        joined -- polars.DataFrame, one row per dev query with all prediction columns (see compare_run)
        group_col -- str, the column to group by (region, abs_vel_bin, q_nn_bin)
        values -- str or list of str, the metric(s) from per_group_metrics; one -> columns named by method,
            several -> columns named <metric>_<method>
        methods -- tuple of str, the names from RESULT_RECIPE to include (default all)

    Returns:
        table -- list of dicts, one per group (sorted), with n and the metric columns
    """
    # long: one row per method and group
    long = pl.DataFrame([{"method": m, **row} for m in methods
                         for row in per_group_metrics(joined, RESULT_RECIPE[m][0], group_col, RESULT_RECIPE[m][1])])
    # "2-5, 5-10, <2, >10" comes in the report in this weird order because sorting later is alphabetically
    long = long.with_columns(pl.col(group_col).cast(joined.schema[group_col]))
    # n is the same for every method, so it's taken once from the first one
    counts = long.filter(pl.col("method") == methods[0]).select(group_col, "n")
    # wide: one row per group; sorted at the end because a join doesn't keep the row order
    return counts.join(long.pivot(on="method", index=group_col, values=values), on=group_col).sort(group_col).to_dicts()


################################################
#======= Section 4: Markdown Log Helper =======#
################################################
# Used by main.

def report_md(runs, comparisons, vario_table, s05_params, train_stats, dev_stats):
    """
    Builds the s06b markdown report: the settings of s06 (one column per run) and s05, the dataset sizes, the graph
    statistics of train vs dev, one overview of the GNN of every run and the baselines, per run the training info,
    train vs dev, the tables per region, velocity and distance range and the figures (each in its own section with a
    short description, placed via s01.figure_marker), and the variograms in the appendix.

    Arguments:
        runs -- list of dicts, the evaluated runs (see load_run)
        comparisons -- list of dicts, one per run in the same order (see compare_run)
        vario_table -- list of dicts, the fitted variograms with the kriging errors per region x orbit (see main)
        s05_params -- dict, the arguments of the last s05 run (data/s05_params.json)
        train_stats -- polars.DataFrame, the train graph statistics (see train_graph_stats)
        dev_stats -- polars.DataFrame, the dev graph statistics, one row per dev query (the baselines, see
            evaluate_baselines)

    Returns:
        body -- str, markdown
        figures -- dict, {name: matplotlib.figure.Figure}: graph_stats once, and per run loss_curves_<run>,
            pred_vs_true_<run>, calibration_<run> and win_rate_<run> (see xx_plotting)
    """
    # rounds the floats of a metrics row, so the tables stay readable (names, counts and texts stay as they are)
    def rounded(row):
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}
    parts, figures = [], {}

    # a markdown table of rounded rows (used for every table of the report)
    def table(rows):
        return s01.table_to_md([rounded(r) for r in rows])

    # --- 1. settings: all s06 params, one column per run; the ones that differ between runs (in a grid: the
    # hyperparameters of the grid) first and in bold. stats = the normalization statistics, not a setting; str() so
    # that lists and None can be compared in a set
    keys = [k for k in runs[0]["params"] if k != "stats"]
    varying = [k for k in keys if len({str(r["params"].get(k)) for r in runs}) > 1]
    settings_rows = [{"parameter": f"**{k}**" if k in varying else k, **{r["name"]: r["params"].get(k) for r in runs}}
                     for k in varying + [k for k in keys if k not in varying]]
    # the number of pools and node features per run (from metrics.json)
    sizes = [{"run": r["name"], **{k: r["metrics"].get(k) for k in ("n_train_pools", "n_dev_pools", "n_node_features")}}
             for r in runs]
    parts += ["## Settings",
              "### s06 (training)",
              "The arguments of every evaluated run, one column per run. Parameters that differ between the runs "
              "(in a grid: the varied hyperparameters) come first and are bold; with a single run none is bold.",
              s01.table_to_md(settings_rows),
              "### s05 (graph pools)",
              "The arguments of the last s05 run (data/s05_params.json): how the seeds and their pools were built.",
              s01.params_to_md(s05_params),
              "### Dataset sizes",
              "Number of train and dev pools (one graph is drawn per pool and epoch) and node features per point "
              "(3 without GSE, 67 with).",
              table(sizes)]

    # --- 2. graph statistics, train vs dev
    parts += ["## Graph statistics",
              f"What the graphs look like: size and connectivity, distances from the query to the visible points and "
              f"the velocities around the query. Train = one draw (as in epoch 0) of {len(train_stats)} randomly "
              f"sampled train pools, dev = the {len(dev_stats)} fixed dev graphs of this evaluation. Both are shown "
              f"as densities, so the different sample sizes compare directly; median, std and n per set are next to "
              f"each panel. Where the two differ, the dev graphs are unlike what the model was trained on.",
              s01.figure_marker("graph_stats")]
    figures["graph_stats"] = xx.plot_graph_stat_distributions(train_stats, dev_stats)

    # --- 3. overview: the GNN of every run, then the baselines once (the same for all runs)
    # the GNN rows come first, so table_to_md takes all columns (incl. the sigma metrics) from them
    overview = [{"run": r["name"], **c["overall"][0]} for r, c in zip(runs, comparisons)]
    overview += [{"run": "-", **row} for row in comparisons[0]["overall"][1:]]
    parts += ["## Results on dev",
              "All methods predict the same dev queries from the same visible points: one GNN row per run, the "
              "baselines once (they don't depend on the run).",
              "- **Methods:** IDW1/IDW2 = inverse distance weighting with power 1/2 over all visible points (distances "
              "below 1 m clipped); Neighbour Mean = mean of the k_graph nearest visible points; Ordinary Kriging = "
              "kriging of all visible points with the variogram of the query's region × orbit (see appendix).\n"
              "- **Accuracy:** rmse, mae, median_ae in mm/yr; r2 = 1 - MSE / variance of the true velocities.\n"
              "- **Uncertainty** (only GNN and kriging give a sigma): inside_50/90/95 = share of queries inside "
              "mu ± z·sigma (ideal 0.50/0.90/0.95; above = intervals too wide, below = too narrow); width_90 = median "
              "width of the 90 % interval (mm/yr); spearman_sigma_err = rank correlation of sigma and |error| "
              "(> 0: a larger sigma goes with a larger error).\n"
              "- **Against kriging:** skill_vs_ok = 1 - RMSE / RMSE_OK (> 0 = lower RMSE than kriging); "
              "win_rate_vs_ok = share of queries with a smaller |error| than kriging; rmse_diff_ci95 = 95 % paired "
              "bootstrap interval of RMSE - RMSE_OK (0 outside the interval = significant difference).\n"
              "- **Note on rmse_diff_ci95:** the bootstrap resamples single queries and so treats them as independent. "
              "Queries of the same region and GRAPH file are spatially correlated, so the interval is likely somewhat "
              "too narrow. A block bootstrap (resampling whole GRAPH files instead of queries) would account for this "
              "and is not implemented yet; treat intervals that barely exclude 0 with care.",
              table(overview)]

    # --- 4. per run: training, train vs dev, the tables per group and the figures, each with a short description
    for run, comp in zip(runs, comparisons):
        h, name, q = run["history"], run["name"], comp["joined"]
        # .get / the minutes check: older runs without these fields still work
        info = {"epochs": len(h), "best epoch": run["metrics"]["best_epoch"],
                "stopped": run["metrics"].get("stopped", "?"),
                "min/epoch": float(np.mean([e["minutes"] for e in h])) if "minutes" in h[0] else None}
        parts += [f"## Run {name}",
                  "### Training",
                  "Epochs trained, the epoch of the saved checkpoint (lowest dev NLL), why the training stopped "
                  "(early_stopping or max_epochs) and the mean duration of an epoch incl. the dev evaluation.",
                  table([info]),
                  s01.figure_marker(f"loss_curves_{name}"),
                  "Plain NLL (normalized units) per epoch on train and dev, on the same scale; the dashed line marks "
                  "the best epoch. The train NLL is averaged during the epoch (dropout on, weights still changing), so "
                  "it lies slightly above a clean evaluation. Runs older than the train_nll logging show the β-NLL "
                  "instead, which is not on the same scale.",
                  "### Train vs dev",
                  "The saved checkpoint evaluated on the train and the dev graphs (NLL in normalized units, RMSE and "
                  "MAE in mm/yr). A much better train than dev value means overfitting.",
                  s01.table_to_md([{"set": s, **rounded(run["metrics"][s])} for s in ("train", "dev")]),
                  "### Per region",
                  "RMSE and MAE (mm/yr) of every method per dev region; n = dev queries of the region. RMSE weights "
                  "large errors more, so a method can win on MAE and lose on RMSE when it misses a few outliers.",
                  "**RMSE**", table(comp["rmse_per_region"]),
                  "**MAE**", table(comp["mae_per_region"]),
                  "### By true velocity and by distance",
                  "GNN vs kriging per range of the true |velocity| (mm/yr) and per range of the distance from the query "
                  "to its nearest visible point (m); n = dev queries in the range.",
                  "**By |true velocity| (mm/yr)**", table(comp["per_vel_bin"]),
                  "**By distance query → nearest visible point (m)**", table(comp["per_dist_bin"]),
                  "### Predicted vs true",
                  "Predicted vs true velocity of every dev query, GNN and kriging (colour = number of queries, log "
                  "scale; grey line = perfect prediction). A cloud flatter than the line at large |v| means the method "
                  "smooths extreme velocities towards the neighbourhood.",
                  s01.figure_marker(f"pred_vs_true_{name}"),
                  "### Calibration",
                  "Share of dev queries inside the predicted interval mu ± z·sigma (observed) against the level the "
                  "interval is meant to cover (nominal, 5 to 95 %). On the diagonal = honest uncertainty; above = "
                  "intervals too wide (too cautious), below = too narrow (overconfident).",
                  s01.figure_marker(f"calibration_{name}"),
                  "### Win rate by graph statistic",
                  "Share of dev queries where the GNN's |error| is smaller than kriging's, along each graph statistic "
                  "(10 bins with equal numbers of queries; shaded = 95 % interval; solid line = 0.5, dashed = overall "
                  "win rate). Above 0.5 = the GNN wins more often in that range, so weak ranges can be traced back to "
                  "graph properties.",
                  s01.figure_marker(f"win_rate_{name}")]
        figures |= {f"loss_curves_{name}": xx.plot_loss_curves(h),
                    f"pred_vs_true_{name}": xx.plot_pred_vs_true(q),
                    f"calibration_{name}": xx.plot_calibration(q),
                    f"win_rate_{name}": xx.plot_win_rate(q)}

    # --- 5. appendix: the variograms with the kriging accuracy per region x orbit
    parts += ["## Appendix: variograms of the ordinary kriging, per dev region × orbit",
              f"The variogram model behind every kriging prediction, with the kriging errors of the dev queries of "
              f"that group. Fitted once per region × orbit on all dev pool points minus the dev queries (at most "
              f"{VARIO_SAMPLING_SIZE} sampled points). Empirical semivariance with the Cressie-Hawkins estimator in "
              f"{VARIO_N_BINS} bins from 0 to the largest lag max_dist_to_q (the {VARIO_PERCENTILE}th percentile of the "
              f"query distances × {VARIO_LAG_FACTOR}, geometric spacing from {VARIO_FIRST_EDGE_M} m), bins with fewer "
              f"than {VARIO_MIN_PAIRS} pairs dropped; weighted least squares fit with the Cressie (1985) weights, best "
              f"of Exponential, Spherical and Gaussian (with nugget, range at most {VARIO_LEN_FACTOR} × max_dist_to_q) "
              f"by r². Nugget and var (partial sill) in mm²/yr², len_scale and max_dist_to_q in m. ok rmse / ok mae "
              f"in mm/yr; check = best R² < 0.3 or var ≈ 0 (weak fit: kriging is then close to a plain weighted mean).",
              table(vario_table)]

    return "\n\n".join(parts), figures


##############################################################################
#============================================================================#
# 3) Processing
#============================================================================#
##############################################################################

def compare_run(run, baselines):
    """
    Joins the GNN's dev predictions of one run with the baselines query by query and computes the metrics per method.

    Arguments:
        run -- dict, one evaluated run (see load_run)
        baselines -- polars.DataFrame, see evaluate_baselines

    Returns:
        comparison -- dict with
            overall -- list of dicts, one per method: method, the metrics of per_group_metrics and skill_vs_ok,
                win_rate_vs_ok, rmse_diff_ci95 (None for kriging itself)
            rmse_per_region, mae_per_region -- lists of dicts, one per region: n and one column per method
            per_vel_bin, per_dist_bin -- lists of dicts, one per |velocity| / query-nearest-point distance range:
                n and rmse, mae, inside_90 of GNN and kriging
            joined -- polars.DataFrame, one row per dev query with all predictions and graph statistics (figures)
    """
    # one row per dev query: the GNN's prediction next to the baselines (region/orbit are in both, kept once)
    joined = run["predictions"].join(baselines.drop("region", "orbit"), on=["graph_file", "seed_pid"],
                                     how="inner", suffix="_baseline")

    # every query must be found on both sides, and both sides must have the same true value (same graph and query)
    if joined.height != run["predictions"].height or joined.height != baselines.height:
        raise ValueError(f"{run['name']}: GNN and baselines don't cover the same dev queries")
    # the tolerance covers the float32 round trip (normalize -> de-normalize) of the GNN's y_mm
    if not np.allclose(joined["y_mm"], joined["y_mm_baseline"], atol=1e-2):
        raise ValueError(f"{run['name']}: y_mm of GNN and baselines differ, the dev draws are not the same")

    # bins for the tables by velocity and by distance (q_nn_m comes with the baselines, see graph_stats); cut gives an
    # Enum that sorts in label order
    joined = joined.with_columns(
        pl.col("y_mm").abs().cut([2, 5, 10], labels=["<2", "2-5", "5-10", ">10"]).alias("abs_vel_bin"),
        pl.col("q_nn_m").cut([50, 150, 300], labels=["<50", "50-150", "150-300", ">300"]).alias("q_nn_bin"))

    # overall: one row per method
    overall = [{"method": name, **per_group_metrics(joined, pred, None, sigma)[0]}
               for name, (pred, sigma) in RESULT_RECIPE.items()]

    # the error per query of every method as a numpy array (same row order for all, needed for the paired bootstrap)
    err = {name: (joined[pred] - joined["y_mm"]).to_numpy() for name, (pred, _) in RESULT_RECIPE.items()}

    # every method against kriging: skill (1 - RMSE ratio), win rate per query, bootstrap interval of the
    # RMSE difference; every row gets all three keys so table_to_md doesn't drop them (kriging itself: None)
    ok = err["Ordinary Kriging"]
    rmse_ok = np.sqrt(np.mean(ok ** 2))
    for row in overall:
        if row["method"] == "Ordinary Kriging":
            row |= {"skill_vs_ok": None, "win_rate_vs_ok": None, "rmse_diff_ci95": None}
            continue
        e = err[row["method"]]
        lo, hi = rmse_diff_ci(e, ok)
        row |= {"skill_vs_ok": 1 - np.sqrt(np.mean(e ** 2)) / rmse_ok,
                "win_rate_vs_ok": float(np.mean(np.abs(e) < np.abs(ok))),
                "rmse_diff_ci95": f"[{lo:.3f}, {hi:.3f}]"}

    # the tables per group; the bin tables only for GNN vs kriging, all 5 methods x 3 metrics would be 15 columns
    vs_ok = ("GNN", "Ordinary Kriging")
    return {"overall": overall,
            "rmse_per_region": per_method_table(joined, "region", "rmse"),
            "mae_per_region": per_method_table(joined, "region", "mae"),
            "per_vel_bin": per_method_table(joined, "abs_vel_bin", ["rmse", "mae", "inside_90"], vs_ok),
            "per_dist_bin": per_method_table(joined, "q_nn_bin", ["rmse", "mae", "inside_90"], vs_ok),
            "joined": joined}


##############################################################################
#============================================================================#
# 4) Main
#============================================================================#
##############################################################################

def main(project_dir, runs):
    """
    Evaluates one run or several runs of a grid on dev: loads their s06 files, the (cached) baselines, compares them
    query by query, adds what is the same for all runs (kriging errors per variogram, s05 settings, train graph
    statistics) and builds one report.

    Arguments:
        project_dir -- str, project root (same as in s01/s06), the runs are in <project_dir>/checkpoints/
        runs -- list of str, run names or glob patterns, e.g. ["single_123"] or ["grid_456_*"]

    Returns:
        body -- str, markdown of the report
        figures -- dict, the report's figures (see report_md)
    """
    # every run folder matching one of the names or patterns, each only once
    checkpoint_dir = Path(project_dir) / "checkpoints"
    run_dirs = sorted({d for pattern in runs for d in checkpoint_dir.glob(pattern) if d.is_dir()})
    loaded = [r for r in (load_run(d) for d in run_dirs) if r is not None]
    if not loaded:
        raise RuntimeError(f"no complete run found for {runs} in {checkpoint_dir}")

    # all runs must have been evaluated on the same dev graphs, otherwise one set of baselines doesn't fit all
    settings_per_run = {r["name"]: {k: r["params"][k] for k in DRAW_KEYS} for r in loaded}
    settings = next(iter(settings_per_run.values()))
    if any(s != settings for s in settings_per_run.values()):
        raise ValueError(f"the runs use different draw settings, evaluate them separately: {settings_per_run}")

    # the baselines once for all runs, from the cache in the dev data folder if still valid
    baselines, vario_log = load_or_compute_baselines(train_dev_graph_files(1), settings, Path(project_dir) / "data" / "dev")
    comparisons = [compare_run(r, baselines) for r in loaded]

    # kriging accuracy per region x orbit next to its fitted variogram (appendix table); check flags weak fits
    ok = pl.DataFrame(per_group_metrics(baselines, "ok_mu", ["region", "orbit"])).select(
        "region", "orbit", pl.col("rmse").alias("ok rmse"), pl.col("mae").alias("ok mae"))
    vario_table = (pl.DataFrame(vario_log).join(ok, on=["region", "orbit"], how="left")
                   .with_columns(((pl.col("best R2") < 0.3) | (pl.col("best model var") < 1e-6)).alias("check"))
                   .sort("region", "orbit").to_dicts())

    # the s05 settings (written by s05 next to the data) and the train graph statistics (cached in data/train)
    s05_params = json.load(open(Path(project_dir) / "data" / "s05_params.json"))
    train_stats = train_graph_stats(settings, loaded[0]["params"]["seed"], Path(project_dir) / "data" / "train")

    # baselines = the dev graph statistics too (one row per dev query, see evaluate_baselines)
    return report_md(loaded, comparisons, vario_table, s05_params, train_stats, baselines)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--project-dir", dest="project_dir", type=str, required=True,
                        help="project root (same as in s01/s06), the runs are in <project-dir>/checkpoints/")
    parser.add_argument("--runs", dest="runs", nargs="+", required=True,
                        help="run names or glob patterns in <project-dir>/checkpoints, e.g. single_123 or 'grid_456_*'")

    args = parser.parse_args()

    start = datetime.now()
    body, figures = main(args.project_dir, args.runs)

    # the markdown log of the evaluation, written only if it got this far
    s01.write_md_log("s06b_baseline_comparison", start, vars(args), body=body, figures=figures)
