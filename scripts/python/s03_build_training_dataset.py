#######################################
# 03_build_training_dataset

##############################################################################
#============================================================================#
# 1) Import Statements
#============================================================================#
##############################################################################

import argparse
import polars as pl
import numpy as np
import shapely
import rasterio

import s01_download_data as s01
import s02_preprocessing as s02

from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import transform
from pathlib import Path
from rasterio.windows import from_bounds

################################################
# 1.1) Constants
################################################

LC_CLASSES = {"Built-up": 50, "Grassland": 30, "Tree cover": 10, "Shrubland": 20,
              "Bare / sparse vegetation": 60, "Cropland": 40}

# Transformer from wgs84 lon/lat to the metric EGMS crs, built once and reused for every region
TO_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
MIN_AREA, MIN_PTS = 4_000_000, 10_000

##############################################################################
#============================================================================#
# 2) Helper Functions
#============================================================================#
##############################################################################

def preproc_csvs_for_region(region_path):
    # Returns the paths to all preprocessed burst csvs of a region (ascending and descending)
    return [s02.preprocessed_path(c) for c in s02.burst_csvs_for_region(region_path)]

def count_points(csvs):
    # Counts the rows of all csvs lazily, without loading them into memory
    return sum(pl.scan_csv(c).select(pl.len()).collect().item() for c in csvs)

def overlap_with_aoi(burst, aoi):
    pts = shapely.points(burst["easting"].to_numpy(), burst["northing"].to_numpy())
    area_intersected = shapely.convex_hull(shapely.multipoints(pts)).intersection(aoi)
    in_area = shapely.contains(area_intersected, pts)
    return pts, area_intersected, in_area

def count_points_in_aoi(csvs, aoi):
    n_total = 0
    for c in csvs:
        b = pl.read_csv(c, columns=["easting", "northing"])   # only 2 columns, cheap
        _, area_intersected, in_area = overlap_with_aoi(b, aoi)
        n = int(in_area.sum())
        if area_intersected.area >= MIN_AREA and n >= MIN_PTS:
            n_total += n
    return n_total

def sample_landcover(wc_path, lon, lat):
    """Samples the WorldCover class (uint8, 0 = no data / outside) at each lon/lat via one windowed read."""
    with rasterio.open(wc_path) as ds:
        win = from_bounds(lon.min(), lat.min(), lon.max(), lat.max(), ds.transform).round_offsets().round_lengths()
        arr = ds.read(1, window=win, boundless=True, fill_value=0)
        tr = ds.window_transform(win)
    col = np.floor((lon - tr.c) / tr.a).astype(np.int64)
    row = np.floor((lat - tr.f) / tr.e).astype(np.int64)
    ok = (row >= 0) & (row < arr.shape[0]) & (col >= 0) & (col < arr.shape[1])
    lc = np.zeros(len(lon), dtype=np.uint8)
    lc[ok] = arr[row[ok], col[ok]]
    return lc

def select_area_and_buffer_burst_edges(burst, aoi, buffer_m):

    # Turn the cs columns into spatial objects and build hull around them
    pts = shapely.points(burst["easting"].to_numpy(), burst["northing"].to_numpy())
    burst_poly = shapely.convex_hull(shapely.multipoints(pts))

    # Compute the intersection of the aoi polygon (already in EPSG:3035) and the burst
    area_intersected = burst_poly.intersection(aoi)

    #########################################
    # Before processing any more, we check whether the burst overlaps significantly with the aoi
    # minimal overlaps can be ignored, they do not contribute much to the training

    # points inside the overlap, computed once and reused for the filter below
    in_area = shapely.contains(area_intersected, pts)
    n_pts = int(in_area.sum())

    # Reject the burst since buffer will shrink it even more and no points might survive
    if area_intersected.area < 4000000 or n_pts < 10000:
        return None, None, None

    ##########################################
    # Find the points that are in the inner zone or buffer zone and flag them

    inner = area_intersected.buffer(-buffer_m)
    in_center = shapely.contains(inner, pts)

    burst = burst.with_columns(pl.Series("in_center", in_center.astype(np.int8)))
    burst = burst.filter(pl.Series(in_area))     # keep all points in the overlap

    return burst, area_intersected, n_pts

    
def find_n_seed_points(n_pts, k_nearest, density):
    # Finds the appropriate number of points per burst
    # Whole burst has typically 1.5 Mio points. So capacity is 15000 per burst with k_nearest=25.
    # with density ~ 0.66, this gives 10000 seed points per full burst
    # With k_nearest = 25, this gives 250000 points totally used for the graphs
    # That should be sufficient data and enough separation of points at the same time 
    capacity = n_pts // (4 * k_nearest)
    return int(capacity * density)

def select_center_nodes(n_seed_points, wc_merged_path, burst, area_intersected, burst_name, log_path,
                        min_dist_m=100.0, cell_size_m=2000.0, grid_share=2/3,
                        strat_exponent=0.5, rng_seed=42):
    """
    Flags seed points: 2/3 spread equally over a 2 x 2 km grid (random within each cell), 1/3
    stratified over land cover with extra weight on small classes. Only points with in_center == 1
    (outside the edge margin) can become seeds. In both phases a seed must be at least min_dist_m
    from every seed chosen so far. Shortfalls are not redistributed; intended vs real counts go to the log.

    Arguments:
        n_seed_points -- int, intended number of seeds for this burst
        wc_merged_path -- str or Path, WorldCover file of the region
        burst -- polars.DataFrame with easting, northing, longitude, latitude, in_center
        area_intersected -- shapely geometry (EPSG:3035), defines the grid origin
        burst_name -- str, used in the log
        log_path -- Path, log file the counts are appended to
        min_dist_m -- float, minimum distance between any two seeds (0 = off)
        cell_size_m -- float, grid cell size
        grid_share -- float, share of seeds placed on the grid
        strat_exponent -- float, class weight = class size ** strat_exponent
        rng_seed -- int, seed for reproducibility

    Returns:
        burst -- the input with an int8 column "seed" (1 = seed, 0 = other, margin points always 0)
    """
    rng = np.random.default_rng(rng_seed)

    # only points outside the edge margin are eligible, elig holds their row indices in burst
    elig = np.flatnonzero(burst["in_center"].to_numpy() == 1)
    E = burst["easting"].to_numpy()[elig]
    N = burst["northing"].to_numpy()[elig]
    # python lists for fast scalar access inside the loops below
    xs, ys = E.tolist(), N.tolist()
    # chosen[i] marks whether eligible point i became a seed
    chosen = np.zeros(len(elig), dtype=bool)

    # split the intended seeds into the grid share and the stratified share
    n_grid = int(round(n_seed_points * grid_share))
    n_strat = n_seed_points - n_grid

    # spatial hash of the accepted seeds for the minimum distance
    # hash cell size = min_dist_m, so a seed that is too close must lie in the same or one of the 8 neighbouring cells
    g = min_dist_m
    seed_hash = {}

    def far_enough(x, y):
        # checks whether a point keeps min_dist_m to every seed accepted so far
        if g <= 0:
            return True
        kx, ky = int(x // g), int(y // g)
        # only the 3 x 3 neighbourhood of hash cells has to be searched
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for sx, sy in seed_hash.get((kx + dx, ky + dy), ()):
                    # squared distances avoid the sqrt
                    if (sx - x) ** 2 + (sy - y) ** 2 < g * g:
                        return False
        return True

    def accept(i):
        # flags point i as seed and registers it in the spatial hash
        chosen[i] = True
        if g > 0:
            seed_hash.setdefault((int(xs[i] // g), int(ys[i] // g)), []).append((xs[i], ys[i]))

    def take(idx, n):
        # visits the candidates idx in random order and accepts up to n that are far enough from all seeds
        # returns the number actually accepted (can be < n if too many candidates are too close)
        n_acc = 0
        for i in rng.permutation(idx):
            if n_acc == n:
                break
            if far_enough(xs[i], ys[i]):
                accept(i)
                n_acc += 1
        return n_acc

    # 1) grid phase: group the points by cell and give every occupied cell the same quota
    # counters for the log
    n_cells, n_grid_real, strat_real = 0, 0, {}
    if len(elig) > 0 and n_seed_points > 0:
        # cell id per point: column index * 1e6 + row index of a grid anchored at the overlap's lower left corner
        xmin, ymin, _, _ = area_intersected.bounds
        cell = (np.floor((E - xmin) / cell_size_m).astype(np.int64) * 1_000_000
                + np.floor((N - ymin) / cell_size_m).astype(np.int64))
        # sort by cell id and split into one index array per occupied cell
        order = np.argsort(cell, kind="stable")
        _, starts = np.unique(cell[order], return_index=True)
        groups = np.split(order, starts[1:])
        n_cells = len(groups)

        # equal quota per cell, the remainder is spread as +1 over randomly picked cells
        base, extra = divmod(n_grid, n_cells)
        quotas = np.full(n_cells, base)
        quotas[rng.choice(n_cells, size=extra, replace=False)] += 1

        for group, quota in zip(groups, quotas):
            take(group, quota)
        n_grid_real = int(chosen.sum())

        # 2) stratified phase over land cover, weights ~ class size ** strat_exponent
        if n_strat > 0:
            # land cover class of every eligible point
            lc = sample_landcover(wc_merged_path, burst["longitude"].to_numpy()[elig], burst["latitude"].to_numpy()[elig])
            # still unchosen points per land cover class, empty classes are dropped
            avail = {name: np.flatnonzero((lc == code) & ~chosen) for name, code in LC_CLASSES.items()}
            avail = {name: idx for name, idx in avail.items() if len(idx) > 0}
            if avail:
                # exponent < 1 gives small classes more seeds than their size share would
                w = np.array([len(idx) ** strat_exponent for idx in avail.values()])
                raw = n_strat * w / w.sum()
                # round down, then hand the leftover seeds to the classes with the largest remainders
                n_c = np.floor(raw).astype(int)
                n_c[np.argsort(-(raw - n_c))[:n_strat - n_c.sum()]] += 1
                for (name, idx), n in zip(avail.items(), n_c):
                    strat_real[name] = (int(n), take(idx, n))

    # write intended vs real counts to the log
    n_real = int(chosen.sum())
    frac = n_real / n_seed_points if n_seed_points else float("nan")
    with open(log_path, "a") as f:
        f.write(f"\n[select_center_nodes] burst: {burst_name}\n")
        f.write(f"  seeds intended: {n_seed_points}   real: {n_real}   fraction: {frac:.3f}\n")
        f.write(f"  grid phase: intended {n_grid}, real {n_grid_real} ({n_cells} occupied cells)\n")
        for name, (n_int, n_r) in strat_real.items():
            f.write(f"  stratified {name}: intended {n_int}, real {n_r}\n")
        f.write(f"  min distance between seeds: {min_dist_m} m\n")

    # map the chosen eligible points back to the full burst rows
    seed = np.zeros(burst.height, dtype=np.int8)
    seed[elig[chosen]] = 1
    return burst.with_columns(pl.Series("seed", seed))


def assign_n_seed_points_to_regions(study_areas_gdf):
    # This function assigns the total number of seed points that a region should have 
    # This is to ensure the real subsidence regions get sufficient weight in the training dataset
    # THE GDF MUST MUST BE FILTERED TO TRAINING ONLY!!!
    # Uses the n_points_region column that is computed in main

    n_per_region = int(study_areas_gdf["n_points_region"].sum() // len(study_areas_gdf))
    return n_per_region



def select_candidate_pool():
    # selects the other points around the center nodes 
    pass

def retrieve_feature_information():
    # retrieves the edge and node feature information for a candidate pool
    pass

def build_tensor_bundle():
    # builds the minigraphs in a particular way
    pass

def combine_burst_train_sets():
    # takes many csv based train sets and puts them into one.
    pass







##############################################################################
#============================================================================#
# 3) Processing
#============================================================================#
##############################################################################

def build_train_dataset_per_burst(csv_path, aoi, k_nearest, density, wc_merged_path,
                                  log_path, min_dist_m, buffer_m, n_seed_points_per_region, n_points_region):

    #################################################
    # 1) Load, check point availability and buffer
    #################################################

    # Load the csv 
    burst = pl.read_csv(csv_path, columns=["pid","latitude","longitude","easting","northing","incidence_angle","mean_velocity"],
                        schema_overrides={**s02.EGMS_FLOAT_COLS, "pid": pl.String})

    # Burst is now only the points intersecting the study area with edge flag column
    burst, area_intersected, n_pts = select_area_and_buffer_burst_edges(burst, aoi, buffer_m=buffer_m)

    if burst is None:
        return

    #################################################
    # 2) Find the appropriate number of seed points in the burst
    #################################################

    # This computes the share of points that an individual burst has in their region
    # Is used to spread out the points equally and ensure real subsidence gets enough weight
    share = n_pts / n_points_region

    # only use the number of points in the center (not the edge margin) to determine the number of seed points
    n_center = int(burst["in_center"].sum())
    n_seed_points = min(int(n_seed_points_per_region * share),                 # regional budget share
                    find_n_seed_points(n_center, k_nearest, density))          # burst capacity
    #################################################
    # 3) Select the actual seed nodes
    #################################################
    
    p = Path(csv_path)
    burst = select_center_nodes(n_seed_points, wc_merged_path, burst, area_intersected,
                            burst_name=f"{p.parent.parent.name}/{p.parent.name}/{p.stem}",
                            log_path=log_path, min_dist_m=min_dist_m)


    #################################################
    # 4) Select the candidate pool around every point
    #################################################
    
    

    # 4) Selects the other points for the candidate pool
    # Loops over all the center nodes?
    # INPUT: Takes the dataset from before and the center nodes 
    # OUTPUT: Returns per center node, the candidate pool in form of a dataframe. 
    select_candidate_pool()

    # 5) Node and edge features need to be determined 
    # not sure whether that is completely possible at this stage
    # INPUT: A candidate pool dataframe
    # OUTPUT: The same dataframe with the node and edge features attached
    retrieve_feature_information()

    # 6) Builds the minigraphs from the pools
    # INPUT: A candidate pool as df including features
    # OUTPUT: Returns tensor bundle with all the information needed to build a new mini graph per epoch
    build_tensor_bundle()


def build_train_dataset_per_region(row, k_nearest, density, log_path, min_dist_m, buffer_m):
    # This collects the sets per burst and puts them in one file and returns
    # the path to that file

    # This returns a list with all paths to the csvs per region (ascending and descending)
    csvs_for_region = preproc_csvs_for_region(row["region_path"])

    ############################################
    # extract columns that other functions need
    ############################################
    wc_merged_path = row["wc_merged_path"]
    n_seed_points_per_region = row["n_seed_points_per_region"]
    # The total number of seeds this entire region should have is already computed
    # Now this has to be spread out over the individual bursts according to their share of points in the region
    n_points_region = row["n_points_region"]


    # The aoi polygon is the same for every burst of the region, so it is projected only once
    aoi = transform(TO_3035.transform, Polygon(row["egms_query_bbox"]))

    # Per csv, the training build is done and collected in a list
    train_sets_burst = [build_train_dataset_per_burst(csv, aoi, k_nearest, density, wc_merged_path,
                        log_path, min_dist_m, buffer_m, n_seed_points_per_region, n_points_region) for csv in csvs_for_region]

    # The per burst sets are combined to one per region
    # returns the file path to that file
    train_file = combine_burst_train_sets()

    return train_file



##############################################################################
#============================================================================#
# 4) Main
#============================================================================#
##############################################################################


def main(k_nearest, density, log_path, min_dist_m, buffer_m):
    
    # Import the gdf again to have access to all the file paths
    study_areas_gdf = s01.study_areas_io("read", base_path = Path(__file__).parent / "../../assets/study_areas_gdf")
    # Select the training regions only 
    train_mask = study_areas_gdf["set"] == 0

    # Counting the points of all bursts per training region once, reused for the seed assignment and the per burst share
    study_areas_gdf.loc[train_mask, "n_points_region"] = study_areas_gdf[train_mask].apply(
                    lambda row: count_points_in_aoi(preproc_csvs_for_region(row["region_path"]), project_aoi(row["egms_query_bbox"])), axis=1)

    # Assigning the number of seed points per region, so the real subsidence regions are not underrepresented
    study_areas_gdf.loc[train_mask, "n_seed_points_per_region"] = assign_n_seed_points_to_regions(study_areas_gdf[train_mask])
        
    # writing the file paths to the saved training datasets per region
    study_areas_gdf.loc[train_mask, "train_file"] = study_areas_gdf.loc[train_mask].apply(build_train_dataset_per_region, axis=1, k_nearest=k_nearest, density=density,
               log_path=log_path, min_dist_m=min_dist_m, buffer_m=buffer_m)
    
    # !!!! Determine the paths to every csv in ascending and descending in the region folder
    # So loop over the region_paths with apply in gdf, build a list to every egms csv and then loop over every csv using the function below
    # BUILDING the tensor bundle per burst. Takes in one region row from gdf and builds paths to csv from that 
    # !!!! Do not apply to validation regions, since they only need the gnss stations.
    # INPUT: path to a egms csv
    # OUTPUT: Saved tensor bundle
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--project-dir", dest="project_dir", type=str, required=True,
                        help="project root (same as in s01), the parameter log goes to <project-dir>/param_log/")
    parser.add_argument("--k-nearest", dest="k_nearest", type=int, required=True,
                        help="number of pool points per seed")
    parser.add_argument("--density", dest="density", type=float, required=True,
                        help="share (0, 1] of the burst capacity that is used for seeds")
    parser.add_argument("--min-dist", dest="min_dist", type=float, default=100.0,
                        help="minimum distance (m) between any two seeds")
    parser.add_argument("--buffer-m", dest="buffer_m", type=float, default=2000.0,
                        help="ring (m) around the core that pool points may come from, also the edge margin without seeds; "
                             "must not exceed s01.CLIP_BUFFER_M")

    args = parser.parse_args()

    # the bursts are clipped to core + s01.CLIP_BUFFER_M, a larger buffer would reach beyond the downloaded data
    if args.buffer_m > s01.CLIP_BUFFER_M:
        parser.error(f"--buffer-m must not be larger than s01.CLIP_BUFFER_M ({s01.CLIP_BUFFER_M} m)")
    # Make sure density has the right value range
    if not 0 < args.density <= 1:
        parser.error("--density must be in (0, 1]")

    # Saving the important parameters in a log file
    log_path = s02.write_param_log("s03_build_training_dataset", Path(args.project_dir) / "param_log", vars(args))
    
    main(args.k_nearest, args.density, log_path, args.min_dist, args.buffer_m)
 
