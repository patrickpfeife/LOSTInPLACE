#######################################
# xx_plotting

#============================================================#
# Import Statements
#============================================================#

import math
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import norm
from importlib.util import spec_from_file_location, module_from_spec
from shapely.geometry import Polygon
from pyproj import Transformer

#============================================================#
# Define Global Variables
#============================================================#

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR.parent.parent / "assets"
PLOTS_DIR = ASSETS_DIR / "plots"

COVERAGE_PATH = ASSETS_DIR / "egms_coverage_countries.geojson"
REAL_SUB_PATH = ASSETS_DIR / "real_subsidence_pointbased.gpkg"
STUDY_AREAS_PATH = ASSETS_DIR / "study_areas_gdf.gpkg"
GNSS_LIST_PATH = ASSETS_DIR / "gnss_station_list.txt"

# Categorical palette (dataviz skill's validated default, fixed order, never cycled)
COLOR_REAL = "#2a78d6"          # slot 1 blue   -- real subsidence regions
COLOR_RANDOM = "#eb6834"        # slot 2 orange -- random background points
COLOR_TRAIN = "#2a78d6"         # slot 1 blue
COLOR_DEV = "#eb6834"           # slot 2 orange
COLOR_VAL = "#1baf7a"           # slot 3 aqua
COLOR_GNSS = "#4a3aa7"          # slot 7 violet -- GNSS stations
COLOR_COVERAGE_FILL = "#e8e7e2"  # neutral surface, context polygons
COLOR_COVERAGE_EDGE = "#c3c2b7"

SET_LABELS = {0: "train", 1: "dev", 2: "val"}
SET_COLORS = {0: COLOR_TRAIN, 1: COLOR_DEV, 2: COLOR_VAL}

# the s04b report figures: GNN vs the strongest baseline (slots 1 and 2, the pair validated for all chart forms)
COLOR_GNN = "#2a78d6"           # slot 1 blue
COLOR_KRIGING = "#eb6834"       # slot 2 orange
INK_SECONDARY = "#52514e"       # text next to the plots (never in a series color)
INK_MUTED = "#898781"           # reference lines (1:1, 50 %, diagonal)

# the graph statistics of the s04b report (columns of the per-graph tables of s04b): column -> axis label.
# The order is the order of the panels in plot_graph_stat_distributions and plot_win_rate
GRAPH_STAT_LABELS = {
    "n_nodes": "nodes per graph",
    "mean_degree": "edges per node",
    "q_degree": "edges at the query",
    "span_m": "span: largest node distance (m)",
    "q_nn_m": "query → nearest visible point (m)",
    "q_median_m": "median query → visible distance (m)",
    "edge_median_m": "median edge length (m)",
    "query_vel": "query velocity (mm/yr)",
    "query_abs_vel": "|query velocity| (mm/yr)",
    "vel_std": "velocity std of the visible points (mm/yr)",
    "vel_range": "velocity range of the visible points (mm/yr)",
    "q_dev_from_mean": "|query − mean of the visible points| (mm/yr)",
}

#============================================================#
# Helper Functions
#============================================================#

def create_plot_dir():
    """
    Creates the assets/plots directory (for the SVG figures this script produces) if it doesn't exist yet.

    Arguments:
        None -- uses the global PLOTS_DIR path

    Returns:
        None -- creates the folder on disk as a side effect
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_download_data_module():
    """
    Dynamically imports 01_download_data.py, so this script can reuse its functions (e.g. point_to_bbox)
    without duplicating them. A plain `import 01_download_data` isn't valid syntax since the filename
    starts with a digit, hence the importlib workaround. Importing it this way does not run its main(),
    since __name__ inside the loaded module is not "__main__".

    Arguments:
        None -- loads the fixed path SCRIPT_DIR / "01_download_data.py"

    Returns:
        module -- the loaded 01_download_data module, with its functions accessible as attributes
    """
    module_path = SCRIPT_DIR / "01_download_data.py"
    spec = spec_from_file_location("download_data", module_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_figure(fig, name):
    """
    Saves a matplotlib figure as an SVG (vector format, easy to include in LaTeX) into PLOTS_DIR.

    Arguments:
        fig -- matplotlib.figure.Figure, the figure to save
        name -- string, output filename without extension

    Returns:
        None -- writes PLOTS_DIR/name.svg to disk and closes the figure
    """
    output_path = PLOTS_DIR / f"{name}.svg"
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def load_gnss_stations(gnss_list_path):
    """
    Reads the Nevada Geodetic Laboratory station list into a GeoDataFrame of station points.

    Arguments:
        gnss_list_path -- path (str or Path) to the raw DataHoldings.txt-style station list

    Returns:
        stations -- geopandas.GeoDataFrame, one row per station, geometry in EPSG:4326
    """
    # usecols=range(11) drops the optional trailing StaOrigName field, whose values sometimes
    # contain spaces and would otherwise break the whitespace-separated parsing
    station_list = pd.read_csv(gnss_list_path, sep=r"\s+", usecols=range(11))
    geometry = gpd.points_from_xy(station_list["Long(deg)"], station_list["Lat(deg)"])
    return gpd.GeoDataFrame(station_list, geometry=geometry, crs="EPSG:4326")


#============================================================#
# Processing
#============================================================#

def plot_egms_coverage(coverage_path):
    """
    Plots the EGMS coverage countries as a standalone context map.

    Arguments:
        coverage_path -- path (str or Path) to the egms_coverage_countries.geojson file

    Returns:
        None -- saves "egms_coverage.svg" to PLOTS_DIR
    """
    coverage = gpd.read_file(coverage_path)

    fig, ax = plt.subplots(figsize=(8, 8))
    coverage.plot(ax=ax, facecolor=COLOR_COVERAGE_FILL, edgecolor=COLOR_COVERAGE_EDGE, linewidth=0.4)
    ax.set_title(f"EGMS coverage countries (n={len(coverage)})")
    ax.set_axis_off()
    save_figure(fig, "egms_coverage")


def plot_real_subsidence_bboxes(real_sub_path, coverage_path, point_to_bbox):
    """
    Plots the real subsidence points together with their query bounding boxes, reusing point_to_bbox
    from 01_download_data.py so the illustrated boxes are exactly what the download step queries.

    Arguments:
        real_sub_path -- path (str or Path) to the real_subsidence_pointbased.gpkg file
        coverage_path -- path (str or Path) to the egms_coverage_countries.geojson file
        point_to_bbox -- callable, the point_to_bbox function reused from 01_download_data.py

    Returns:
        None -- saves "real_subsidence_bboxes.svg" to PLOTS_DIR
    """
    real_subsidence = gpd.read_file(real_sub_path)
    coverage = gpd.read_file(coverage_path).to_crs(real_subsidence.crs)

    t2projected = Transformer.from_crs(real_subsidence.crs, "EPSG:3035", always_xy=True)
    t2wgs84 = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    bboxes = real_subsidence.apply(point_to_bbox, axis=1, result_type="reduce",
                                    t2projected=t2projected, t2wgs84=t2wgs84, source_crs=real_subsidence.crs)
    bbox_polygons = gpd.GeoSeries([Polygon(corners) for corners in bboxes], crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(10, 10))
    coverage.plot(ax=ax, facecolor=COLOR_COVERAGE_FILL, edgecolor=COLOR_COVERAGE_EDGE, linewidth=0.4, zorder=1)
    bbox_polygons.plot(ax=ax, facecolor="none", edgecolor=COLOR_REAL, linewidth=1.2, zorder=2)
    real_subsidence.plot(ax=ax, color=COLOR_REAL, markersize=15, zorder=3, label="Real subsidence region")
    ax.set_title("Real subsidence regions and their query bounding boxes")
    ax.set_axis_off()
    ax.legend(loc="lower left", frameon=False)
    save_figure(fig, "real_subsidence_bboxes")


def plot_gnss_stations(gnss_list_path, coverage_path, real_sub_path):
    """
    Plots GNSS station locations alongside the EGMS coverage area and the real subsidence regions,
    to show where ground-truth validation data is available relative to the study areas.

    Arguments:
        gnss_list_path -- path (str or Path) to the GNSS station list (gnss_station_list.txt)
        coverage_path -- path (str or Path) to the egms_coverage_countries.geojson file
        real_sub_path -- path (str or Path) to the real_subsidence_pointbased.gpkg file

    Returns:
        None -- saves "gnss_stations.svg" to PLOTS_DIR
    """
    stations = load_gnss_stations(gnss_list_path)
    coverage = gpd.read_file(coverage_path).to_crs(stations.crs)
    real_subsidence = gpd.read_file(real_sub_path).to_crs(stations.crs)

    fig, ax = plt.subplots(figsize=(10, 10))
    coverage.plot(ax=ax, facecolor=COLOR_COVERAGE_FILL, edgecolor=COLOR_COVERAGE_EDGE, linewidth=0.4, zorder=1)
    stations.plot(ax=ax, color=COLOR_GNSS, markersize=2, zorder=2, label=f"GNSS station (n={len(stations)})")
    real_subsidence.plot(ax=ax, color=COLOR_REAL, markersize=20, zorder=3, label="Real subsidence region")
    ax.set_title("GNSS stations and real subsidence regions")
    ax.set_axis_off()
    ax.legend(loc="lower left", frameon=False)
    save_figure(fig, "gnss_stations")


def plot_train_dev_val_split(study_areas_path):
    """
    Plots the train/dev/val split as both a bar chart of counts and a map colored by split.

    Arguments:
        study_areas_path -- path (str or Path) to the study_areas_gdf.gpkg file

    Returns:
        None -- saves "train_dev_val_counts.svg" and "train_dev_val_map.svg" to PLOTS_DIR
    """
    study_areas = gpd.read_file(study_areas_path)

    counts = study_areas["set"].map(SET_LABELS).value_counts().reindex(SET_LABELS.values())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values, color=[SET_COLORS[k] for k in SET_LABELS])
    ax.set_ylabel("Number of regions")
    ax.set_title("Train / dev / val split")
    save_figure(fig, "train_dev_val_counts")

    fig, ax = plt.subplots(figsize=(10, 10))
    for set_value, label in SET_LABELS.items():
        subset = study_areas[study_areas["set"] == set_value]
        subset.plot(ax=ax, color=SET_COLORS[set_value], markersize=15, zorder=2, label=label)
    ax.set_title("Study areas by train / dev / val split")
    ax.set_axis_off()
    ax.legend(loc="lower left", frameon=False)
    save_figure(fig, "train_dev_val_map")


def plot_real_vs_random(study_areas_path, real_sub_path):
    """
    Plots a map distinguishing the real subsidence points from the randomly sampled background points
    within the combined study_areas_gdf, by matching each row's location against the known real
    subsidence locations (study_areas_gdf itself carries no explicit real/random flag column).

    Arguments:
        study_areas_path -- path (str or Path) to the study_areas_gdf.gpkg file
        real_sub_path -- path (str or Path) to the real_subsidence_pointbased.gpkg file

    Returns:
        None -- saves "real_vs_random_points.svg" to PLOTS_DIR
    """
    study_areas = gpd.read_file(study_areas_path)
    real_subsidence = gpd.read_file(real_sub_path)

    real_locations = set(real_subsidence["location"])
    is_real = study_areas["location"].isin(real_locations)

    fig, ax = plt.subplots(figsize=(10, 10))
    study_areas[~is_real].plot(ax=ax, color=COLOR_RANDOM, markersize=15, zorder=2, label="Random background")
    study_areas[is_real].plot(ax=ax, color=COLOR_REAL, markersize=15, zorder=3, label="Real subsidence")
    ax.set_title("Real subsidence vs. random background points")
    ax.set_axis_off()
    ax.legend(loc="lower left", frameon=False)
    save_figure(fig, "real_vs_random_points")


def plot_loss_curves(history):
    """
    Plots the train and dev NLL per epoch of a training run on the same scale and marks the epoch with the lowest dev
    loss. Train uses the plain NLL of the train batches (key "train_nll"), not the β-NLL that is optimized, so both
    curves are the same quantity. Older runs without "train_nll" fall back to the β-NLL (labelled as such).

    Arguments:
        history -- list of dicts, one per epoch, with the keys "epoch", "dev_loss" and "train_nll" (or only
            "train_loss", older runs) (see train_model in s04)

    Returns:
        fig -- matplotlib.figure.Figure, not saved here since s01.write_md_log saves the figures it gets
    """
    epochs = [h["epoch"] for h in history]
    # the epoch of the lowest dev loss is the one saved as the best checkpoint
    best = min(history, key=lambda h: h["dev_loss"])

    # plain NLL on train if the run logged it, otherwise the optimized β-NLL (a different scale, see the label)
    if "train_nll" in history[0]:
        train, train_label = [h["train_nll"] for h in history], "train (NLL)"
    else:
        train, train_label = [h["train_loss"] for h in history], "train (β-NLL, not the same scale as dev)"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train, color=COLOR_TRAIN, linewidth=2, label=train_label)
    ax.plot(epochs, [h["dev_loss"] for h in history], color=COLOR_DEV, linewidth=2, label="dev (NLL)")
    ax.axvline(best["epoch"], color=INK_MUTED, linestyle="--", linewidth=1, label=f"best epoch ({best['epoch']})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL (normalized units)")
    ax.set_title("Loss per epoch")
    ax.legend(frameon=False)

    return fig


def finite_values(table, col):
    """
    The finite values of one column of a table (polars or pandas), as a float numpy array.

    Arguments:
        table -- polars.DataFrame or pandas.DataFrame
        col -- str, the column

    Returns:
        values -- numpy float array without NaN, inf and missing values
    """
    values = np.asarray(table[col].to_numpy(), dtype=float)
    return values[np.isfinite(values)]


def plot_graph_stat_distributions(train_stats, dev_stats, n_cols=3, n_bins=40):
    """
    One panel per graph statistic (GRAPH_STAT_LABELS): the distributions of train and dev overlaid (density, so the
    different numbers of graphs don't matter) and next to the panel the median and std of both sets. The bins span
    the 0.5 ... 99.5 percentile of both sets together; the values outside are counted in the first / last bin, so a
    few extreme graphs don't squeeze the rest into one bin. Integer statistics with a small range get one bin per value.

    Arguments:
        train_stats, dev_stats -- polars.DataFrame, one row per graph, one column per statistic (see s04b)
        n_cols -- int, panels per row
        n_bins -- int, number of bins of the continuous statistics

    Returns:
        fig -- matplotlib.figure.Figure
    """
    # only the statistics both tables have, in the order of GRAPH_STAT_LABELS
    cols = [c for c in GRAPH_STAT_LABELS if c in train_stats.columns and c in dev_stats.columns]
    n_rows = math.ceil(len(cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 3.2 * n_rows), squeeze=False)

    for ax, col in zip(axes.flat, cols):
        sets = {"train": finite_values(train_stats, col), "dev": finite_values(dev_stats, col)}
        both = np.concatenate(list(sets.values()))
        lo, hi = np.percentile(both, [0.5, 99.5])

        # integers with a small range (node counts, degrees): one bin per value, centred on it
        if np.all(both == np.round(both)) and hi - lo <= 60:
            bins = np.arange(lo - 0.5, hi + 1.5)
        else:
            bins = np.linspace(lo, hi, n_bins + 1)

        side_text = []
        for name, color in [("train", COLOR_TRAIN), ("dev", COLOR_DEV)]:
            v = sets[name]
            # values outside the shown range go into the edge bins
            ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, density=True, histtype="stepfilled", color=color, alpha=0.15)
            ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, density=True, histtype="step", color=color, linewidth=2, label=name)
            side_text.append(f"{name} (n={len(v)})\nmedian {np.median(v):.3g}\nstd {np.std(v):.3g}")

        # median and std of both sets right of the panel, in text ink
        ax.text(1.03, 0.5, "\n\n".join(side_text), transform=ax.transAxes, va="center", ha="left", fontsize=8, color=INK_SECONDARY)
        ax.set_title(GRAPH_STAT_LABELS[col], fontsize=10)
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=8)

    # the legend once, the unused panels of the last row hidden
    axes.flat[0].legend(frameon=False, fontsize=8)
    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Graph statistics, train vs dev (0.5–99.5 percentile range shown, outliers in the edge bins)", fontsize=11)
    fig.tight_layout()

    return fig


def plot_win_rate(queries, pred_col="mu_mm", ref_col="ok_mu", ref_name="Ordinary Kriging", n_bins=10, n_cols=3):
    """
    Where does the GNN win? One panel per graph statistic (GRAPH_STAT_LABELS): the dev queries are split into n_bins
    bins with the same number of queries (quantiles of the statistic), per bin the share of queries where the GNN's
    absolute error is smaller than the reference's, at the median of the statistic in the bin, with the 95 % binomial
    interval as band (1.96 sqrt(p (1 - p) / n)). 50 % = as good as the reference; the dashed line is the overall share.

    Arguments:
        queries -- polars.DataFrame, one row per dev query: y_mm, the two prediction columns and the graph statistics
            (the baselines table of s04b joined with the GNN's predictions)
        pred_col, ref_col -- str, the GNN's and the reference's prediction column
        ref_name -- str, the reference's name for the labels
        n_bins -- int, quantile bins per statistic (fewer if the statistic has many equal values)
        n_cols -- int, panels per row

    Returns:
        fig -- matplotlib.figure.Figure
    """
    y = queries["y_mm"].to_numpy()
    # a win: the GNN is strictly closer to the true velocity than the reference
    win = np.abs(queries[pred_col].to_numpy() - y) < np.abs(queries[ref_col].to_numpy() - y)
    overall = win.mean()

    cols = [c for c in GRAPH_STAT_LABELS if c in queries.columns]
    n_rows = math.ceil(len(cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.2 * n_rows), squeeze=False)

    for ax, col in zip(axes.flat, cols):
        x = np.asarray(queries[col].to_numpy(), dtype=float)
        ok = np.isfinite(x)
        x, w = x[ok], win[ok]

        # quantile bin edges; np.unique drops repeated edges of statistics with many equal values (integers)
        edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
        # bin of every query (the maximum belongs to the last bin)
        b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)

        x_mid, share, half_width = [], [], []
        for k in range(len(edges) - 1):
            in_bin = b == k
            if not in_bin.any():
                continue
            p, n = w[in_bin].mean(), in_bin.sum()
            x_mid.append(np.median(x[in_bin]))
            share.append(p)
            half_width.append(1.96 * np.sqrt(p * (1 - p) / n))
        x_mid, share, half_width = map(np.array, (x_mid, share, half_width))

        ax.fill_between(x_mid, share - half_width, share + half_width, color=COLOR_GNN, alpha=0.2, linewidth=0)
        ax.plot(x_mid, share, color=COLOR_GNN, linewidth=2, marker="o", markersize=5)
        ax.axhline(0.5, color=INK_MUTED, linewidth=1)
        ax.axhline(overall, color=INK_MUTED, linewidth=1, linestyle="--")

        # distances and magnitudes over several orders of magnitude: log axis
        # (ticks at 1, 2, 5 x 10^k as plain numbers, not 4×10^-1)
        if x.min() > 0 and x.max() / x.min() >= 100:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        # 50 % always visible, otherwise the axis zooms to the data
        ax.set_ylim(min(0.4, (share - half_width).min() - 0.02), max(0.6, (share + half_width).max() + 0.02))
        ax.set_title(GRAPH_STAT_LABELS[col], fontsize=10)
        ax.set_ylabel("share of GNN wins", fontsize=8)
        ax.tick_params(labelsize=8)

    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)

    fig.suptitle(f"Share of dev queries where |error GNN| < |error {ref_name}| per quantile bin "
                 f"(band: 95 % interval, solid: 50 %, dashed: overall {overall:.1%})", fontsize=11)
    fig.tight_layout()

    return fig


def plot_pred_vs_true(queries, methods=None):
    """
    Predicted vs true velocity of the dev queries, one panel per method, as a hexbin with log colour scale (thousands
    of points would hide each other in a scatter) and the 1:1 line. Both axes span the 0.5 ... 99.5 percentile of
    the true velocities, the same for all panels.

    Arguments:
        queries -- polars.DataFrame, one row per dev query with y_mm and the prediction columns
        methods -- dict, display name -> prediction column; None = GNN and Ordinary Kriging

    Returns:
        fig -- matplotlib.figure.Figure
    """
    methods = methods or {"GNN": "mu_mm", "Ordinary Kriging": "ok_mu"}
    y = queries["y_mm"].to_numpy()
    lo, hi = np.percentile(y, [0.5, 99.5])

    fig, axes = plt.subplots(1, len(methods), figsize=(4.8 * len(methods), 4.4), squeeze=False)
    for ax, (name, col) in zip(axes.flat, methods.items()):
        pred = queries[col].to_numpy()
        rmse = np.sqrt(np.mean((pred - y) ** 2))
        hb = ax.hexbin(y, pred, gridsize=60, extent=(lo, hi, lo, hi), bins="log", mincnt=1, cmap="Blues")
        ax.plot([lo, hi], [lo, hi], color=INK_MUTED, linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(f"{name} (RMSE {rmse:.2f} mm/yr)", fontsize=10)
        ax.set_xlabel("true velocity (mm/yr)")
        ax.set_ylabel("predicted velocity (mm/yr)")
        fig.colorbar(hb, ax=ax, label="queries")
    fig.tight_layout()

    return fig


def plot_calibration(queries, methods=None):
    """
    Calibration of the predicted uncertainty: for nominal levels p = 5 ... 95 % the observed share of dev queries
    inside the central interval mu ± z sigma, with z = norm.ppf(0.5 + p / 2) (p = 90 % -> z = 1.645). On the diagonal
    the intervals are honest, below it too narrow (overconfident), above it too wide.

    Arguments:
        queries -- polars.DataFrame, one row per dev query with y_mm and the mean and sigma columns
        methods -- dict, display name -> (mean column, sigma column); None = GNN and Ordinary Kriging

    Returns:
        fig -- matplotlib.figure.Figure
    """
    methods = methods or {"GNN": ("mu_mm", "sigma_mm"), "Ordinary Kriging": ("ok_mu", "ok_sigma")}
    colors = [COLOR_GNN, COLOR_KRIGING]
    nominal = np.linspace(0.05, 0.95, 19)
    z = norm.ppf(0.5 + nominal / 2)
    y = queries["y_mm"].to_numpy()

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1)
    for (name, (mu_col, sigma_col)), color in zip(methods.items(), colors):
        abs_err = np.abs(queries[mu_col].to_numpy() - y)
        sigma = queries[sigma_col].to_numpy()
        observed = [(abs_err <= zi * sigma).mean() for zi in z]
        ax.plot(nominal, observed, color=color, linewidth=2, marker="o", markersize=4, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("nominal coverage of mu ± z sigma")
    ax.set_ylabel("observed share of dev queries inside")
    ax.set_title("Calibration (diagonal = honest)", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()

    return fig


#============================================================#
# Main
#============================================================#

def main():
    """
    Checks the required input files exist, sets up the plots directory, and generates every plot.

    Arguments:
        None

    Returns:
        None -- writes SVG figures to assets/plots
    """
    if not COVERAGE_PATH.exists() or not REAL_SUB_PATH.exists():
        raise FileNotFoundError("The EGMS coverage or real subsidence input file is missing!")

    create_plot_dir()

    download_data = load_download_data_module()

    plot_egms_coverage(COVERAGE_PATH)
    plot_real_subsidence_bboxes(REAL_SUB_PATH, COVERAGE_PATH, download_data.point_to_bbox)

    if GNSS_LIST_PATH.exists():
        plot_gnss_stations(GNSS_LIST_PATH, COVERAGE_PATH, REAL_SUB_PATH)
    else:
        print(f"Skipping GNSS plot: {GNSS_LIST_PATH} not found.")

    if STUDY_AREAS_PATH.exists():
        plot_train_dev_val_split(STUDY_AREAS_PATH)
        plot_real_vs_random(STUDY_AREAS_PATH, REAL_SUB_PATH)
    else:
        print(f"Skipping study-area plots: {STUDY_AREAS_PATH} not found (run 01_download_data.py first).")


if __name__ == '__main__':
    main()
