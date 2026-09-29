#######################################
# xx_plotting

#============================================================#
# Import Statements
#============================================================#

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
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
