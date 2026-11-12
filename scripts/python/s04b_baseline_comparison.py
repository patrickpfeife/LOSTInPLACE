#######################################
# s04b_baseline_comparison

"""
Evaluation of trained s04 runs on dev, on the CPU after the GPU training: the baselines (IDW, neighbour mean, ordinary
kriging) on exactly the same dev graphs as the GNN, cached once per dataset and recomputed only if the dev data or the
settings changed, GNN vs baselines per region and orbit, and one markdown report for one run or all runs of a grid.
Run via scripts/slurm/submit_train_eval.sh (after s04) or scripts/slurm/04b_baseline_comparison.slurm.
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
import s04_train_model as s04
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

# the draw settings that decide which dev graphs are built (saved in every s04 checkpoint); all evaluated runs must share them
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
# Used by evaluate_baselines (moved from s04): the baselines on the dev graphs (project.md §7.5).

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
    dist, _ = s04.pairwise_geodesic(lat, lon)
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
    dist, _ = s04.pairwise_geodesic(lat, lon)

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
        dist, _ = s04.pairwise_geodesic(pool["lat"][idx], pool["lon"][idx])

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
            kriging standard deviation ok_sigma
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
                             "idw_1": idw(pool, idx, q, 1), "idw_2": idw(pool, idx, q, 2)}

        results.append(results_per_graph)
    # a dataframe built from the list of dicts, one row per graph
    return pl.DataFrame(results), vario_log


################################################
#======= Section 2: Run and Cache Helpers =====#
################################################
# Used by main, in the order they're first called there.

def load_run(run_dir):
    """
    Loads the settings and the files s04 wrote for one run. Runs with a missing file (failed or unfinished) are skipped.

    Arguments:
        run_dir -- Path, the run folder <project_dir>/checkpoints/<run_name>

    Returns:
        run -- dict with name, params (the settings saved in the checkpoint), predictions (polars.DataFrame, the
            GNN's dev predictions), history (list of dicts, loss per epoch) and metrics (dict, train/dev metrics),
            or None if a file is missing
    """
    # the four files s04 writes into the run folder
    names = ["best_model.pt", "dev_predictions.parquet", "history.json", "metrics.json"]
    if not all((run_dir / n).exists() for n in names):
        print(f"skipping {run_dir.name}: not all s04 files exist (failed or unfinished run)", flush=True)
        return None

    # the settings of the run are saved with the checkpoint (weights_only=False: params is a plain dict, not tensors)
    params = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)["params"]
    return {"name": run_dir.name, "params": params,
            "predictions": pl.read_parquet(run_dir / "dev_predictions.parquet"),
            "history": json.load(open(run_dir / "history.json")),
            "metrics": json.load(open(run_dir / "metrics.json"))}


def dev_graph_files():
    """
    Lists the dev GRAPH files from the region table, in exactly the same order as s04.build_datasets: the dev draws
    depend on the order of the pools and that comes from the order of the files, so this list must not be sorted.

    Arguments:
        None -- reads the region table (s01.study_areas_io)

    Returns:
        dev_files -- list of str, the dev GRAPH files
    """
    gdf = s01.study_areas_io("read", base_path=Path(__file__).parent / "../../assets/study_areas_gdf")
    return [f for files in gdf.loc[gdf["set"] == 1, "graph_file_list"] if files for f in files]


def dataset_fingerprint(dev_files, settings):
    """
    A hash of everything the baselines depend on: the dev files (path, size, modification time, so a rerun of s03
    changes it), the draw settings, the graph construction and the variogram settings.

    Arguments:
        dev_files -- list of str, the dev GRAPH files (see dev_graph_files)
        settings -- dict, the draw settings (DRAW_KEYS) of the evaluated runs

    Returns:
        fingerprint -- str, SHA-256 hex digest
    """
    content = {"files": [(str(f), os.path.getsize(f), os.path.getmtime(f)) for f in dev_files],
               "settings": settings,
               "graph": "delaunay",
               "vario": [VARIO_PERCENTILE, VARIO_LAG_FACTOR, VARIO_FIRST_EDGE_M, VARIO_N_BINS, VARIO_MIN_PAIRS,
                         VARIO_SAMPLING_SIZE, VARIO_LEN_FACTOR]}
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
    # the same dev draws as in s04 (fixed draws, same settings); the baselines need neither GSE nor normalization stats
    pools = s04.load_pools(s04.build_pool_index(dev_files), False, settings["min_graph_size"])
    dev_ds = s04.PoolDataset(pools, None, settings["k_graph"], settings["subset_min"], settings["subset_max"],
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

def per_group_metrics(dev_predictions, pred_col, group_col=None, sigma_col=None):
    """
    Computes n, RMSE and MAE of one prediction column against y_mm, overall or per group, and the share inside
    pred ± 1.645 sigma if a sigma column is given.

    Arguments:
        dev_predictions -- polars.DataFrame, one row per dev query with y_mm and the prediction columns
            (see evaluate_on_dev)
        pred_col -- str, the prediction column (mu_mm, idw_1, ...)
        group_col -- str or None, column to group by (region, orbit); None = one row over all queries
        sigma_col -- str or None, the predicted standard deviation (so far only the GNN has one)

    Returns:
        metrics -- list of dicts: one per group (sorted by the group), or a list with one dict without grouping
    """

    # the error as a polars expression, computed per row once it is used in select or agg
    err = pl.col(pred_col) - pl.col("y_mm")
    # a list of expressions for accuracy metrics
    metrics_recipes = [pl.len().alias("n"), (err ** 2).mean().sqrt().alias("rmse"), err.abs().mean().alias("mae")]

    if sigma_col is not None:
        # <= comparison yields True or False. The mean of that yields the share of True, so how many are in interval
        metrics_recipes.append((err.abs() <= (1.645 * pl.col(sigma_col))).mean().alias("inside_90"))

    # agg applies the expressions per group (sorted, group_by has no fixed order); select applies them to the whole
    # table and gives one row
    if group_col is not None:
        metrics = dev_predictions.group_by(group_col).agg(metrics_recipes).sort(group_col)
    else:
        metrics = dev_predictions.select(metrics_recipes)

    return metrics.to_dicts()


################################################
#======= Section 4: Markdown Log Helper =======#
################################################
# Used by main.

def report_md(runs, comparisons, vario_log):
    """
    Builds the s04b markdown log: one summary table of all runs, the baselines once (they are the same for all runs),
    the fitted variograms, and per run the train vs dev metrics, the errors per region and orbit and the loss curves.

    Arguments:
        runs -- list of dicts, the evaluated runs (see load_run)
        comparisons -- list of dicts, one per run in the same order (see compare_run)
        vario_log -- list of dicts, the fitted variograms (see compute_variograms)

    Returns:
        body -- str, markdown
        figures -- dict, {"loss_curves_<run>": matplotlib.figure.Figure} (see xx.plot_loss_curves)
    """
    # rounds the floats of a metrics row, so the tables stay readable (names and counts stay as they are)
    def rounded(row):
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}

    # one row per run: the main hyperparameters and the dev metrics
    summary = [{"run": r["name"], "GSE": r["params"]["use_gse"], "layers": r["params"]["n_layers"],
                "hidden": r["params"]["hidden_dim"], "lr": f'{r["params"]["lr"]:g}',   # as text, rounding to 3 decimals would make 3e-4 a 0.0
                "beta": r["params"].get("beta", 0.0),
                "epochs": len(r["history"]), "best epoch": r["metrics"]["best_epoch"],
                **{f"dev {k}": v for k, v in r["metrics"]["dev"].items()}} for r in runs]
    # the baselines are the same for every run, so they are taken from the first one
    baselines_overall = [row for row in comparisons[0]["overall"] if row["method"] != "GNN"]

    parts = [
        "## Runs, dev metrics",
        "NLL in normalized units, RMSE and MAE in mm/yr, inside_90 = share of queries inside mu ± 1.645 sigma.",
        s01.table_to_md([rounded(r) for r in summary]),
        "## Baselines on dev",
        "All methods on the same dev graphs (same visible points and query). IDW1/IDW2 = inverse distance weighting "
        "with power 1/2 over all visible points (distances below 1 m clipped), Neighbour Mean = mean of the k_graph "
        "nearest visible points, Ordinary Kriging = ordinary kriging of all visible points with the variogram of the "
        "query's region x orbit (see below). RMSE and MAE in mm/yr; inside_90 only for Ordinary Kriging "
        "(mu ± 1.645 sigma). The GNN of every run is in the tables per run.",
        s01.table_to_md([rounded(r) for r in baselines_overall]),
        # the fitted variograms, so every kriging result can be traced back to its model
        "### Variograms of the ordinary kriging, per dev region × orbit",
        f"Fitted once per region × orbit on all dev pool points minus the dev queries (at most {VARIO_SAMPLING_SIZE} sampled points). "
        f"Empirical semivariance with the Cressie-Hawkins estimator in {VARIO_N_BINS} bins from 0 to the largest lag "
        f"max_dist_to_q (the {VARIO_PERCENTILE}th percentile of the query distances × {VARIO_LAG_FACTOR}, geometric "
        f"spacing from {VARIO_FIRST_EDGE_M} m), bins with fewer than {VARIO_MIN_PAIRS} pairs dropped; weighted least "
        f"squares fit with the Cressie (1985) weights, best of Exponential, Spherical and Gaussian (with nugget, range at most {VARIO_LEN_FACTOR} × max_dist_to_q) by r². "
        f"Nugget and var (partial sill) in mm²/yr², len_scale and max_dist_to_q in m.",
        s01.table_to_md([rounded(r) for r in vario_log]),
    ]

    # per run: train vs dev, GNN and baselines per region, GNN per orbit, loss curves
    figures = {}
    for run, comp in zip(runs, comparisons):
        parts += [f"## Run {run['name']}",
                  s01.table_to_md([{"set": s, **rounded(run["metrics"][s])} for s in ("train", "dev")]),
                  "### RMSE per region", s01.table_to_md([rounded(r) for r in comp["rmse_per_region"]]),
                  "### MAE per region", s01.table_to_md([rounded(r) for r in comp["mae_per_region"]]),
                  "### GNN per orbit", s01.table_to_md([rounded(r) for r in comp["gnn_per_orbit"]])]
        figures[f"loss_curves_{run['name']}"] = xx.plot_loss_curves(run["history"])

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
            overall -- list of dicts, one per method: method, n, rmse, mae (inside_90 with a sigma)
            rmse_per_region, mae_per_region -- lists of dicts, one per region, one column per method
            gnn_per_orbit -- list of dicts, the GNN's n, rmse, mae, inside_90 per orbit
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

    # overall: one row per method
    overall = [{"method": name, **per_group_metrics(joined, pred, None, sigma)[0]}
               for name, (pred, sigma) in RESULT_RECIPE.items()]

    # per region: first one row per method and region (long), then one row per region with one column per method (wide)
    per_region_long = pl.DataFrame([{"method": name, **row} for name, (pred, sigma) in RESULT_RECIPE.items()
                                    for row in per_group_metrics(joined, pred, "region", sigma)])

    return {"overall": overall,
            "rmse_per_region": per_region_long.pivot(on="method", index="region", values="rmse").to_dicts(),
            "mae_per_region": per_region_long.pivot(on="method", index="region", values="mae").to_dicts(),
            "gnn_per_orbit": per_group_metrics(joined, "mu_mm", "orbit", "sigma_mm")}


##############################################################################
#============================================================================#
# 4) Main
#============================================================================#
##############################################################################

def main(project_dir, runs):
    """
    Evaluates one run or several runs of a grid on dev: loads their s04 files, the (cached) baselines, compares them
    query by query and builds one report.

    Arguments:
        project_dir -- str, project root (same as in s01/s04), the runs are in <project_dir>/checkpoints/
        runs -- list of str, run names or glob patterns, e.g. ["single_123"] or ["grid_456_*"]

    Returns:
        body -- str, markdown of the report
        figures -- dict, the loss curves per run
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
    baselines, vario_log = load_or_compute_baselines(dev_graph_files(), settings, Path(project_dir) / "data" / "dev")
    comparisons = [compare_run(r, baselines) for r in loaded]

    return report_md(loaded, comparisons, vario_log)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--project-dir", dest="project_dir", type=str, required=True,
                        help="project root (same as in s01/s04), the runs are in <project-dir>/checkpoints/")
    parser.add_argument("--runs", dest="runs", nargs="+", required=True,
                        help="run names or glob patterns in <project-dir>/checkpoints, e.g. single_123 or 'grid_456_*'")

    args = parser.parse_args()

    start = datetime.now()
    body, figures = main(args.project_dir, args.runs)

    # the markdown log of the evaluation, written only if it got this far
    s01.write_md_log("s04b_baseline_comparison", start, vars(args), body=body, figures=figures)
