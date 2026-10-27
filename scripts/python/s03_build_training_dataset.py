#######################################
# s03_build_training_dataset

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
import geopandas as gpd

import s01_download_data as s01
import s02_preprocessing as s02

from pathlib import Path
from datetime import datetime
from pyproj import Transformer
from rasterio.transform import rowcol
from rasterio.windows import Window, from_bounds
from scipy.spatial import cKDTree

################################################
# 1.1) Constants
################################################

LC_CLASSES = {"Built-up": 50, "Grassland": 30, "Tree cover": 10, "Shrubland": 20,
              "Bare / sparse vegetation": 60, "Cropland": 40}

# minimum overlap of a burst with the core (m2) and minimum number of points in it
MIN_AREA, MIN_PTS = 4_000_000, 1_000

# AlphaEarth satellite embedding (GSE), read from the open bucket (anonymous HTTPS, no account needed)
GSE_YEAR = 2025
GSE_BUCKET_URL = "https://storage.googleapis.com/"
GSE_INDEX_URL = GSE_BUCKET_URL + "alphaearth_foundations/satellite_embedding/v1/annual/aef_index.csv"
# the 0.8 GB index csv is reduced once to this small parquet (gitignored)
GSE_INDEX_PATH = Path(__file__).parent / "../../assets/gse_index.parquet"
GSE_NODATA, GSE_BANDS = -128, 64

##############################################################################
#============================================================================#
# 2) Helper Functions
#============================================================================#
##############################################################################

##############################################
#====== Section 1: Seed Budget Helpers ======#
##############################################
# Used by main to count the seed-eligible points and set the budget, in the order they're first called.

def preproc_csvs_for_region(region_path):
    """
    Lists the preprocessed burst csvs (s02) of one region, both orbit directions.

    Arguments:
        region_path -- str or Path, the region's folder

    Returns:
        csvs -- list of Path, the "*_PREPROCESSED.csv" files
    """
    return [s02.preprocessed_path(c) for c in s02.burst_csvs_for_region(region_path)]


def count_points_in_aoi(csvs, aoi, buffer_m):
    """
    Counts the seed-eligible points of a region: per burst the points inside the overlap with the core shrunk by
    buffer_m (in_center), over the bursts that pass MIN_AREA / MIN_PTS -- the same rules as
    select_area_and_buffer_burst_edges, so the shares of the bursts add up to 1.

    Arguments:
        csvs -- list of Path, the region's preprocessed burst csvs
        aoi -- shapely Polygon (EPSG:3035), the region's core
        buffer_m -- float, edge margin (m) inside the overlap

    Returns:
        n_total -- int, the region's in_center points (n_center_region)
    """
    n_total = 0
    for c in csvs:
        # only 2 columns, cheap; the overrides keep them Float64 even in a burst clipped to 0 rows
        b = pl.read_csv(c, columns=["easting", "northing"], schema_overrides=s02.EGMS_FLOAT_COLS)
        pts, area_intersected, in_area = overlap_with_aoi(b, aoi)
        if area_intersected.area >= MIN_AREA and int(in_area.sum()) >= MIN_PTS:
            n_total += int(shapely.contains(area_intersected.buffer(-buffer_m), pts).sum())
    return n_total


def overlap_with_aoi(burst, aoi):
    """
    Intersects the convex hull of a burst's points with an AOI and flags the points inside the overlap.

    Arguments:
        burst -- polars.DataFrame with easting and northing (EPSG:3035)
        aoi -- shapely Polygon (EPSG:3035), the region's core

    Returns:
        pts -- numpy array of shapely Points, one per row of burst
        area_intersected -- shapely geometry, burst hull ∩ aoi
        in_area -- numpy bool array, True for the points inside area_intersected
    """
    pts = shapely.points(burst["easting"].to_numpy(), burst["northing"].to_numpy())
    area_intersected = shapely.convex_hull(shapely.multipoints(pts)).intersection(aoi)
    in_area = shapely.contains(area_intersected, pts)
    return pts, area_intersected, in_area


def assign_n_seed_points_to_regions(study_areas_gdf, k_nearest, density):
    """
    Seed budget that every region of one set gets, so all regions contribute equally (small regions give what they
    have): the seed capacity of an average region of the set. THE GDF MUST BE FILTERED TO ONE SET (train, dev or cal).

    Arguments:
        study_areas_gdf -- geopandas.GeoDataFrame, the regions of one set, with n_center_region (see main)
        k_nearest, density -- see find_n_seed_points

    Returns:
        budget -- int, seeds per region
    """
    n_per_region = int(study_areas_gdf["n_center_region"].sum() // len(study_areas_gdf))
    return find_n_seed_points(n_per_region, k_nearest, density)


def find_n_seed_points(n_pts, k_nearest, density):
    """
    Seed capacity of a set of points: every seed reserves 4 * k_nearest points for its candidate pool, and only a
    share density of that maximum is used, so the pools do not exhaust the area. E.g. 1,000,000 in_center points
    with k_nearest = 25 and density = 0.66 give 6,600 seeds.

    Arguments:
        n_pts -- int, number of seed-eligible (in_center) points
        k_nearest -- int, number of pool points per seed
        density -- float in (0, 1]

    Returns:
        n_seeds -- int, the seed capacity
    """
    capacity = n_pts // (4 * k_nearest)
    return int(capacity * density)


################################################
#====== Section 2: Seed and Pool Helpers ======#
################################################
# Used by build_train_dataset_per_burst, in the order they're first called (a helper's own helpers right after it).

def select_area_and_buffer_burst_edges(burst, aoi, buffer_m):
    """
    Cuts a burst down to its overlap with the region's core and flags the seed-eligible points. Bursts whose overlap
    is smaller than MIN_AREA or holds fewer than MIN_PTS points are rejected. All points of the overlap are kept
    (pool candidates); in_center = 1 marks the points further than buffer_m from the overlap's edge (possible seeds).

    Arguments:
        burst -- polars.DataFrame with easting and northing (EPSG:3035)
        aoi -- shapely Polygon (EPSG:3035), the region's core
        buffer_m -- float, edge margin (m) inside the overlap without seeds

    Returns:
        burst -- polars.DataFrame, the points of the overlap with an int8 column in_center (None if rejected)
        area_intersected -- shapely geometry, burst hull ∩ core (None if rejected)
        n_pts -- int, number of points in the overlap (None if rejected)
    """
    # Turn the csv columns into spatial objects and build the hull around them
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

    # Reject the burst, the edge margin would shrink the overlap even more and hardly any seeds could survive
    if area_intersected.area < MIN_AREA or n_pts < MIN_PTS:
        return None, None, None

    ##########################################
    # Find the points that are in the inner zone or buffer zone and flag them

    inner = area_intersected.buffer(-buffer_m)
    in_center = shapely.contains(inner, pts)

    burst = burst.with_columns(pl.Series("in_center", in_center.astype(np.int8)))
    burst = burst.filter(pl.Series(in_area))     # keep all points in the overlap

    return burst, area_intersected, n_pts


def select_seed_nodes(n_seed_points, wc_merged_path, burst, area_intersected, burst_name, seed_log,
                        min_dist_m=100.0, cell_size_m=2000.0, grid_share=2/3,
                        strat_exponent=0.5, rng_seed=42):
    """
    Flags seed points: 2/3 spread equally over a 2 x 2 km grid (random within each cell), 1/3
    stratified over land cover with extra weight on small classes. Only points with in_center == 1
    (outside the edge margin) can become seeds. In both phases a seed must be at least min_dist_m
    from every seed chosen so far. Shortfalls are not redistributed; intended vs real counts go to seed_log.

    Arguments:
        n_seed_points -- int, intended number of seeds for this burst
        wc_merged_path -- str or Path, WorldCover file of the region
        burst -- polars.DataFrame with easting, northing, longitude, latitude, in_center
        area_intersected -- shapely geometry (EPSG:3035), defines the grid origin
        burst_name -- str, used in the log
        seed_log -- list, one dict with the intended vs real counts of this burst is appended (markdown log of the run)
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

    # intended vs real counts, collected for the markdown log that is written at the end of the run
    n_real = int(chosen.sum())
    seed_log.append({"burst": burst_name, "intended": n_seed_points, "real": n_real,
                     "grid intended": n_grid, "grid real": n_grid_real, "occupied cells": n_cells,
                     "stratified": strat_real})

    # map the chosen eligible points back to the full burst rows
    seed = np.zeros(burst.height, dtype=np.int8)
    seed[elig[chosen]] = 1
    return burst.with_columns(pl.Series("seed", seed))


def sample_landcover(wc_path, lon, lat):
    """
    Samples the WorldCover class at every point via one windowed read over the points' bounding box.

    Arguments:
        wc_path -- str or Path, the region's merged WorldCover file (EPSG:4326)
        lon, lat -- numpy arrays, the point coordinates

    Returns:
        lc -- numpy uint8 array, the class per point (0 = no data / outside the raster)
    """
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


def select_candidate_pool(burst, k_nearest, max_dist, rng_seed=42):
    """
    Assigns a candidate pool to every seed of a burst: up to k_nearest points within max_dist, chosen
    angular-weighted by find_pool_points. A point belongs to at most one pool (no reuse), and seeds never end up
    in another seed's pool. The minimum pool size is not applied here but at training time, when the graphs
    are redrawn from the pools every epoch.

    Arguments:
        burst -- polars.DataFrame with pid, easting, northing and seed (all points of the overlap, see select_seed_nodes)
        k_nearest -- int, maximum number of pool points per seed
        max_dist -- float, maximum distance (m) of a pool point from its seed
        rng_seed -- int, seed for the random order in which the seeds pick their points

    Returns:
        burst -- the input with a String column "seed_pid": the pid of the seed whose pool the point belongs to
            (the seed itself included), null for points in no pool
    """
    # the coordinates of all points of the overlap (EPSG:3035, metres)
    xy = np.column_stack([burst["easting"].to_numpy(), burst["northing"].to_numpy()])
    # a KD-tree finds all points within max_dist of a seed fast, built once per burst
    tree = cKDTree(xy)

    # instead of a copy of the burst where taken points are deleted, a mask of the points that are still available
    # (deleting rows would change the row indices the tree refers to)
    # the seeds are taken out from the start, so no seed can become a pool point of another seed
    seed_idx = np.flatnonzero(burst["seed"].to_numpy() == 1)
    available = np.ones(burst.height, dtype=bool)
    available[seed_idx] = False

    # the new column, one entry per point: the pid of the seed it is assigned to, None = in no pool
    pids = burst["pid"].to_numpy()
    seed_pid = np.full(burst.height, None, dtype=object)

    # Loop over all seed points in random order, so the seeds that pick first are not always in the same corner
    rng = np.random.default_rng(rng_seed)
    for i in rng.permutation(seed_idx):
        # all points within max_dist of the seed, sorted by their distance to it
        cand = np.asarray(tree.query_ball_point(xy[i], r=max_dist), dtype=np.int64)
        cand = cand[np.argsort(np.hypot(xy[cand, 0] - xy[i, 0], xy[cand, 1] - xy[i, 1]))]
        # pools can be smaller than k_nearest in sparse areas, the minimum size is applied at training time
        pool = find_pool_points(cand, available, k_nearest, xy, xy[i])

        # the seed and its pool points get the seed's pid
        seed_pid[i] = pids[i]
        seed_pid[pool] = pids[i]
        # the pool points are not available anymore, so in the next run they cannot be picked anymore
        available[pool] = False

    # return the burst with the new column
    return burst.with_columns(pl.Series("seed_pid", seed_pid.tolist(), dtype=pl.String))


def find_pool_points(cand_idx, available, k_nearest, xy, seed_xy, n_sectors=8):
    """
    Angular-weighted k nearest: the circle around the seed is split into n_sectors equal sectors and the points are
    taken round by round, the nearest free point of every sector first, then the second nearest of every sector, ...
    until k_nearest are picked. So the pool surrounds the seed instead of lying on one side.

    Arguments:
        cand_idx -- numpy int array, row indices of all points within max_dist of the seed, sorted by distance
        available -- numpy bool array over all rows of the burst, True = point can still be picked
        k_nearest -- int, number of points to pick
        xy -- numpy array (n, 2), coordinates of all points of the burst (EPSG:3035)
        seed_xy -- numpy array (2,), coordinates of the seed
        n_sectors -- int, number of angular sectors

    Returns:
        pool -- numpy int array, row indices of the picked points (fewer than k_nearest if not enough are free)
    """
    # only the points that are still available
    free = cand_idx[available[cand_idx]]
    if len(free) <= k_nearest:
        return free

    # the sector of every free point, from its direction as seen from the seed
    angle = np.arctan2(xy[free, 1] - seed_xy[1], xy[free, 0] - seed_xy[0])
    sector = (np.floor((angle + np.pi) / (2 * np.pi) * n_sectors).astype(np.int64)) % n_sectors

    # the rank of every point within its sector (0 = nearest), free is already sorted by distance
    rank = np.zeros(len(free), dtype=np.int64)
    for s in range(n_sectors):
        in_s = sector == s
        rank[in_s] = np.arange(in_s.sum())

    # sort by rank first and by distance second (lexsort uses the last key first), then take the first k_nearest
    # empty sectors are simply skipped, their share goes to the other sectors
    order = np.lexsort((np.arange(len(free)), rank))
    return free[order[:k_nearest]]


######################################
#====== Section 3: GSE Helpers ======#
######################################
# Used by main and build_train_dataset_per_region, in the order they're first called.

def load_gse_index(year):
    """
    Loads the GSE tile index of one year: one row per tile with its UTM crs, bucket path and rectangle
    (utm_west/east/south/north in metres). On the first run the 0.8 GB csv is downloaded and reduced to a parquet
    without the large WKT column (GSE_INDEX_PATH), later runs only read that parquet.

    Arguments:
        year -- int, the embedding year (GSE_YEAR)

    Returns:
        gse_index -- polars.DataFrame, the tiles of that year
    """
    if not GSE_INDEX_PATH.exists():
        csv_path = Path(s01.pull_data_by_url(GSE_INDEX_URL, GSE_INDEX_PATH.parent, "aef_index.csv"))
        pl.scan_csv(csv_path).select(pl.exclude("WKT")).collect().write_parquet(GSE_INDEX_PATH)
        csv_path.unlink()
    return pl.read_parquet(GSE_INDEX_PATH).filter(pl.col("year") == year)


def retrieve_feature_information(graph_paths, core, gse_index, gse_log):
    """
    Samples the GSE embedding (GSE_YEAR) at every point of a region's GRAPH tables and adds it as int8 columns
    gse_0 ... gse_63 (raw quantised, de-quantised at training time). Every tile touching the core is opened once,
    and per tile only the internal blocks that contain points are read, all 64 bands at once. A value only counts
    if it is not GSE_NODATA, so a point outside a tile's footprint (tile rectangles of neighbouring UTM zones
    overlap) or over nodata can still be filled by another tile.

    Arguments:
        graph_paths -- list of str, the region's GRAPH parquet files (see build_train_dataset_per_burst)
        core -- shapely Polygon, the region's core (EPSG:3035)
        gse_index -- polars.DataFrame, the tile index of GSE_YEAR (see load_gse_index)
        gse_log -- list, one dict per GRAPH file with the number of points without GSE is appended (markdown log)

    Returns:
        None -- overwrites every GRAPH file with the gse columns added
    """
    # all GRAPH tables of the region fit into memory, each gets an int8 array for the 64 bands and a filled mask
    graphs = [pl.read_parquet(p) for p in graph_paths]
    gse = [np.full((g.height, GSE_BANDS), GSE_NODATA, dtype=np.int8) for g in graphs]
    filled = [np.zeros(g.height, dtype=bool) for g in graphs]

    # GDAL shall not list the bucket directory when a tile is opened, and retry failed range requests.
    # HTTP/2 multiplexing, merged range requests and threaded decompression read a block (64 bands) in ~3 s instead of ~10 s
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2",
                      GDAL_HTTP_MULTIPLEX="YES", GDAL_HTTP_VERSION="2", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                      GDAL_NUM_THREADS="ALL_CPUS"):
        for tile in tiles_for_core(core, gse_index):
            # the points are in EPSG:3035, the tile in its own UTM crs
            to_tile = Transformer.from_crs("EPSG:3035", tile["crs"], always_xy=True)
            # every tile is opened only once, directly from the bucket
            with rasterio.open("/vsicurl/" + GSE_BUCKET_URL + tile["path"].removeprefix("gs://")) as ds:
                # internal blocks of 1024 x 1024 px; the tiles are 8192 px wide, so there are no partial blocks
                bh, bw = ds.block_shapes[0]

                # collect the points of ALL GRAPH tables of the region that lie in this tile and have no valid value yet,
                # together with the table they belong to. The bursts of a region overlap (ascending and descending
                # cover the same core), so grouping all of them together lets every block be read only once per tile
                tab, idx, rows, cols = [], [], [], []
                for t, (g, done) in enumerate(zip(graphs, filled)):
                    todo = np.flatnonzero(~done)
                    if len(todo) == 0:
                        continue
                    x, y = to_tile.transform(g["easting"].to_numpy()[todo], g["northing"].to_numpy()[todo])
                    # only the points inside the tile rectangle
                    inside = (x >= tile["utm_west"]) & (x < tile["utm_east"]) & (y > tile["utm_south"]) & (y <= tile["utm_north"])
                    if not inside.any():
                        continue
                    # pixel row and column of every point in the tile
                    r, c = rowcol(ds.transform, x[inside], y[inside])
                    tab.append(np.full(inside.sum(), t)); idx.append(todo[inside])
                    rows.append(np.asarray(r)); cols.append(np.asarray(c))
                if not tab:
                    continue
                tab, idx, rows, cols = (np.concatenate(a) for a in (tab, idx, rows, cols))

                # group the points by internal block, every needed block is read once with all 64 bands
                block_id = (rows // bh) * 10_000 + cols // bw
                for b in np.unique(block_id):
                    in_b = np.flatnonzero(block_id == b)
                    r0, c0 = (b // 10_000) * bh, (b % 10_000) * bw
                    block = ds.read(window=Window(c0, r0, bw, bh))                # shape (64, bh, bw)
                    vals = block[:, rows[in_b] - r0, cols[in_b] - c0].T            # shape (points, 64)
                    # a value only counts if it is not nodata, otherwise another tile may still fill the point
                    ok = (vals != GSE_NODATA).any(axis=1)
                    # write the values back into the arrays of the tables the points came from
                    for t in np.unique(tab[in_b][ok]):
                        m = tab[in_b][ok] == t
                        gse[t][idx[in_b][ok][m]] = vals[ok][m]
                        filled[t][idx[in_b][ok][m]] = True

    # add the 64 columns and overwrite every GRAPH file; points without GSE keep GSE_NODATA and go to the log
    for p, g, arr, done in zip(graph_paths, graphs, gse, filled):
        g = g.with_columns([pl.Series(f"gse_{i}", arr[:, i]) for i in range(GSE_BANDS)])
        g.write_parquet(p)
        gse_log.append({"region": Path(p).parent.parent.name, "graph file": Path(p).name,
                        "points": len(done), "without GSE": int((~done).sum())})


def tiles_for_core(core, gse_index):
    """
    Finds the GSE tiles whose UTM rectangle intersects a region's core.

    Arguments:
        core -- shapely Polygon, the region's core (EPSG:3035)
        gse_index -- polars.DataFrame, the tile index of one year (see load_gse_index)

    Returns:
        tiles -- list of dicts, one index row per tile
    """
    # cheap pre-selection on the lon/lat bounds of the tiles
    w, s, e, n = gpd.GeoSeries([core], crs="EPSG:3035").to_crs("EPSG:4326").total_bounds
    cand = gse_index.filter((pl.col("wgs84_west") < e) & (pl.col("wgs84_east") > w) &
                            (pl.col("wgs84_south") < n) & (pl.col("wgs84_north") > s))
    # exact test: the core in the tile's own UTM crs against the tile rectangle
    tiles = []
    for t in cand.iter_rows(named=True):
        core_utm = gpd.GeoSeries([core], crs="EPSG:3035").to_crs(t["crs"]).iloc[0]
        if core_utm.intersects(shapely.box(t["utm_west"], t["utm_south"], t["utm_east"], t["utm_north"])):
            tiles.append(t)
    return tiles


##################################################
#======= Section 4: Markdown Log Helper =========#
##################################################
# Used at the end of a run, for the step's part of the markdown log (see s01.write_md_log).

def training_summary_md(study_areas_gdf, log, k_nearest, density):
    """
    Builds the s03 part of the markdown log: seeds (= graphs) and pool points per set, the budget calculation
    per set and its effect per region, the seed placement (grid vs land cover) and the points without GSE.

    Arguments:
        study_areas_gdf -- geopandas.GeoDataFrame, the region table after main
        log -- dict, the lists "bursts", "seeds" and "gse" collected during the run (see main)
        k_nearest, density -- the arguments of the run, to show the budget formula

    Returns:
        body -- str, markdown
    """
    set_names = {0: "train", 1: "dev", 2: "cal"}
    regions = study_areas_gdf.loc[study_areas_gdf["set"].isin(set_names)]
    region_set = dict(zip(regions["location"], regions["set"]))

    # the per burst and per GRAPH file entries summed up per region
    per_region = {}
    for e in log["bursts"]:
        r = per_region.setdefault(e["region"], {"bursts": 0, "accepted": 0, "capacity": 0, "seeds": 0, "pool points": 0, "without GSE": 0})
        r["bursts"] += 1; r["accepted"] += int(e["accepted"])
        r["capacity"] += e["capacity"]; r["seeds"] += e["seeds"]; r["pool points"] += e["pool points"]
    for e in log["gse"]:
        per_region[e["region"]]["without GSE"] += e["without GSE"]

    region_rows, set_rows, budget_rows = [], [], []
    for row in regions.itertuples():
        r = per_region.get(row.location, {"bursts": 0, "accepted": 0, "capacity": 0, "seeds": 0, "pool points": 0, "without GSE": 0})
        region_rows.append({"region": row.location, "set": set_names[row.set], "bursts accepted": f"{r['accepted']} of {r['bursts']}",
                            "in_center points": f"{int(row.n_center_region):,}", "capacity": f"{r['capacity']:,}",
                            "budget": f"{int(row.n_seed_points_per_region):,}", "seeds": f"{r['seeds']:,}",
                            "capped by the budget": "yes" if r["capacity"] > row.n_seed_points_per_region else "",
                            "mean pool size": round(r["pool points"] / r["seeds"], 1) if r["seeds"] else "",
                            "points without GSE": r["without GSE"]})

    # sorted by set, then by number of seeds
    region_rows.sort(key=lambda r: (list(set_names.values()).index(r["set"]), -int(r["seeds"].replace(",", ""))))

    for s, name in set_names.items():
        q = regions.loc[regions["set"] == s]
        rs = [per_region.get(l, {"accepted": 0, "capacity": 0, "seeds": 0, "pool points": 0, "without GSE": 0}) for l in q["location"]]
        budget = int(q["n_seed_points_per_region"].iloc[0]) if len(q) else 0
        cap, seeds = sum(r["capacity"] for r in rs), sum(r["seeds"] for r in rs)
        pool = sum(r["pool points"] for r in rs)
        set_rows.append({"set": name, "regions": len(q), "bursts accepted": sum(r["accepted"] for r in rs),
                         "GRAPH files": int(sum(len(g or []) for g in q["graph_file_list"])),
                         "seeds (= graphs)": f"{seeds:,}", "pool points": f"{pool:,}",
                         "mean pool size": round(pool / seeds, 1) if seeds else "",
                         "points without GSE": sum(r["without GSE"] for r in rs)})
        budget_rows.append({"set": name, "regions": len(q), "mean in_center points": f"{int(q['n_center_region'].mean()) if len(q) else 0:,}",
                            "budget per region": f"{budget:,}",
                            "regions below the budget (give all)": sum(r["capacity"] <= budget for r in rs),
                            "regions capped": sum(r["capacity"] > budget for r in rs),
                            "capacity": f"{cap:,}", "seeds": f"{seeds:,}",
                            "share of the capacity used": f"{100 * seeds / cap:.0f}%" if cap else ""})

    # seed placement: grid phase and land cover stratified phase, summed per set
    placement = {s: {"grid intended": 0, "grid real": 0, "stratified intended": 0, "stratified real": 0} for s in set_names}
    classes = {s: {} for s in set_names}
    for e in log["seeds"]:
        s = region_set[e["burst"].split("/")[0]]
        placement[s]["grid intended"] += e["grid intended"]; placement[s]["grid real"] += e["grid real"]
        for c, (n_int, n_real) in e["stratified"].items():
            placement[s]["stratified intended"] += n_int; placement[s]["stratified real"] += n_real
            classes[s][c] = classes[s].get(c, 0) + n_real
    placement_rows = [{"set": set_names[s], **{k: f"{v:,}" for k, v in d.items()}} for s, d in placement.items()]
    class_rows = [{"set": set_names[s], **{c: f"{classes[s].get(c, 0):,}" for c in LC_CLASSES}} for s in set_names]

    return "\n\n".join([
        "## Summary", s01.table_to_md(set_rows),
        "## Budget per set",
        f"Every region of a set gets the same seed budget, the seed capacity of an average region of the set: "
        f"budget = (mean in_center points // (4 * k_nearest)) * density, with k_nearest = {k_nearest}, density = {density}. "
        f"A burst gets min(budget * its share of the region's in_center points, its own capacity), so large regions are "
        f"capped at the budget and small ones give everything they have; nothing is redistributed.",
        s01.table_to_md(budget_rows),
        "## Seeds per region", s01.table_to_md(region_rows),
        "## Seed placement",
        "2/3 of the seeds on a 2 x 2 km grid, 1/3 stratified over the WorldCover classes (real seeds below); "
        "shortfalls come from the minimum distance between seeds.",
        s01.table_to_md(placement_rows), "Stratified seeds per land cover class:", s01.table_to_md(class_rows)])

##############################################################################
#============================================================================#
# 3) Processing
#============================================================================#
##############################################################################

def build_train_dataset_per_burst(csv_path, aoi, k_nearest, density, wc_merged_path,
                                  log, min_dist_m, buffer_m, n_seed_points_per_region, n_center_region,
                                  max_dist):
    """
    Builds the GRAPH table of one burst: overlap with the core and edge margin, number of seeds, seed selection and
    candidate pools. Only the points of the pools (seeds included) are saved, as <burst>_GRAPH.parquet next to the
    burst csv; the GSE columns are added later on region level (retrieve_feature_information).

    Arguments:
        csv_path -- Path, the burst's preprocessed csv
        aoi -- shapely Polygon (EPSG:3035), the region's core
        k_nearest, density -- see find_n_seed_points
        wc_merged_path -- str or Path, the region's merged WorldCover file
        log -- dict, the lists of the markdown log (see main); one entry is appended to log["bursts"]
        min_dist_m, buffer_m, max_dist -- see main
        n_seed_points_per_region -- int, the region's seed budget
        n_center_region -- int, the region's in_center points, for the burst's share of the budget

    Returns:
        graph_path -- str, path of the GRAPH file (None if the burst is rejected or gets no seeds)
    """

    #################################################
    # 1) Load, check point availability and buffer
    #################################################

    # one entry per burst csv for the markdown log, filled in step by step (region = folder name)
    p = Path(csv_path)
    entry = {"region": p.parent.parent.name, "accepted": False, "n_center": 0, "capacity": 0,
             "intended": 0, "seeds": 0, "pool points": 0}
    log["bursts"].append(entry)

    # Load the csv
    burst = pl.read_csv(csv_path, columns=["pid","latitude","longitude","easting","northing","incidence_angle","mean_velocity"],
                        schema_overrides={**s02.EGMS_FLOAT_COLS, "pid": pl.String})

    # Burst is now only the points intersecting the core, with the edge flag column
    burst, area_intersected, _ = select_area_and_buffer_burst_edges(burst, aoi, buffer_m=buffer_m)

    if burst is None:
        return None
    entry["accepted"] = True

    # only the points outside the edge margin can become seeds, they determine the number of seeds
    n_center = int(burst["in_center"].sum())
    entry["n_center"] = n_center
    if n_center == 0:
        return None

    #################################################
    # 2) Find the appropriate number of seed points in the burst
    #################################################

    # The burst's share of the region's seed-eligible points, to spread the region's budget over its bursts
    share = n_center / n_center_region

    entry["capacity"] = find_n_seed_points(n_center, k_nearest, density)
    n_seed_points = min(int(n_seed_points_per_region * share),                 # regional budget share
                    entry["capacity"])                                         # burst capacity
    entry["intended"] = n_seed_points

    #################################################
    # 3) Select the actual seed nodes
    #################################################

    burst = select_seed_nodes(n_seed_points, wc_merged_path, burst, area_intersected,
                            burst_name=f"{p.parent.parent.name}/{p.parent.name}/{p.stem}",
                            seed_log=log["seeds"], min_dist_m=min_dist_m)

    #################################################
    # 4) Select the candidate pool around every point
    #################################################

    burst = select_candidate_pool(burst, k_nearest, max_dist)

    #################################################
    # 5) Save the points of the pools as the burst's GRAPH table
    #################################################

    # only the points that are in a pool (the seed included) are needed further, the edge flag not anymore
    graph = burst.filter(pl.col("seed_pid").is_not_null()).drop("in_center")
    entry["seeds"] = int(graph["seed"].sum())
    entry["pool points"] = graph.height - entry["seeds"]
    if graph.height == 0:
        return None

    # same folder and base name as the raw burst csv, parquet so no *.csv glob of s01/s02 picks it up
    graph_path = p.with_name(p.stem.removesuffix("_PREPROCESSED") + "_GRAPH.parquet")
    graph.write_parquet(graph_path)

    return str(graph_path)


def build_train_dataset_per_region(row, k_nearest, density, log, min_dist_m, buffer_m, max_dist, gse_index):
    """
    Builds the GRAPH tables of all bursts of one region and samples the GSE embedding at their points. Every burst
    keeps its own file, so the pids (unique only within a burst) never have to be made unique.

    Arguments:
        row -- pandas.Series, a region row with region_path, geometry (core), wc_merged_path,
            n_seed_points_per_region and n_center_region
        k_nearest, density, log, min_dist_m, buffer_m, max_dist -- see build_train_dataset_per_burst
        gse_index -- polars.DataFrame, the GSE tile index (see load_gse_index)

    Returns:
        graph_file_list -- list of str, the region's GRAPH files
    """
    # all preprocessed burst csvs of the region (ascending and descending)
    csvs_for_region = preproc_csvs_for_region(row["region_path"])

    ############################################
    # extract columns that other functions need
    ############################################
    wc_merged_path = row["wc_merged_path"]
    n_seed_points_per_region = row["n_seed_points_per_region"]
    # The total number of seeds this entire region should have is already computed
    # Now this has to be spread out over the individual bursts according to their share of the in_center points
    n_center_region = row["n_center_region"]

    # Per csv, the training build is done and collected in a list
    train_sets_burst = [build_train_dataset_per_burst(csv, row.geometry, k_nearest, density, wc_merged_path,
                        log, min_dist_m, buffer_m, n_seed_points_per_region, n_center_region, max_dist
                        ) for csv in csvs_for_region]


    # rejected bursts and bursts without seeds return None, they have no GRAPH file
    graph_file_list = [g for g in train_sets_burst if g is not None]

    # the GSE is sampled on region level, so every tile touching the core is opened only once
    if graph_file_list:
        retrieve_feature_information(graph_file_list, row.geometry, gse_index, log["gse"])

    return graph_file_list


##############################################################################
#============================================================================#
# 4) Main
#============================================================================#
##############################################################################


def main(k_nearest, density, min_dist_m, buffer_m, max_dist):
    """
    Builds the training dataset for all train, dev and cal regions (val regions only need their GNSS stations):
    seeds, candidate pools and the GSE embedding per pool point, saved as one GRAPH parquet per burst.
    Everything for the markdown log is collected during the run and returned; the log is written at the end (__main__).

    Seeding, step by step, done separately for each set (train, dev, cal):
        1) Per region: count the seed-eligible points -> n_center_region (count_points_in_aoi). Per burst this is
           the overlap of the burst's convex hull with the region's core polygon, only for bursts whose overlap
           has at least MIN_AREA (4 km2) and MIN_PTS (1,000) points, shrunk by buffer_m (the edge margin).
        2) Per region: the seed budget n_seed_points_per_region, the same for every region of the set, so all
           regions contribute equally: the seed capacity of an average region of the set
           (find_n_seed_points of the mean n_center_region, assign_n_seed_points_to_regions).
        3) Per burst of a region (build_train_dataset_per_burst):
           a) Overlap: convex hull of the burst points intersected with the core. Bursts below MIN_AREA or
              MIN_PTS are skipped. All points of the overlap are kept, the candidate pools can draw from all of them.
           b) Edge margin: points closer than buffer_m to the overlap's edge get in_center = 0. They stay in the
              table and can be pool points, but never seeds, so every seed has points in all directions.
           c) Seed count: min(region budget * the burst's share of the region's in_center points,
              burst capacity = n_center // (4 * k_nearest) * density), with n_center = points with in_center = 1.
              The factor 4 reserves enough points for every seed's candidate pool. Large regions are capped at
              the budget, small ones give everything they have; nothing is redistributed.
           d) Seed selection among the in_center points (select_seed_nodes):
              - 2/3 of the seeds on a 2 x 2 km grid: every occupied cell gets the same quota, random within the cell;
              - 1/3 stratified over the WorldCover classes, weight = class size ** 0.5, so small classes get more
                than their size share;
              - in both phases every seed keeps min_dist_m to all seeds chosen so far. Shortfalls are not
                redistributed; intended vs real counts per phase and class go to the log.
              Result: int8 column seed (1 = seed).

    Arguments:
        k_nearest -- int, number of pool points per seed
        density -- float in (0, 1], share of the burst capacity used for seeds
        min_dist_m -- float, minimum distance (m) between any two seeds
        buffer_m -- float, edge margin (m) inside the overlap of burst and core without seeds
        max_dist -- float, maximum distance (m) of a pool point from its seed

    Returns:
        study_areas_gdf -- geopandas.GeoDataFrame, the region table with the counts, budgets and graph_file_list
            (also saved to disk)
        log -- dict, the lists "bursts", "seeds" and "gse" collected for the markdown log (see training_summary_md)
    """

    # Import the gdf again to have access to all the file paths
    study_areas_gdf = s01.study_areas_io("read", base_path = Path(__file__).parent / "../../assets/study_areas_gdf")

    # the GSE tile index is loaded once for all regions
    gse_index = load_gse_index(GSE_YEAR)
    # list column, it has to exist before the set-wise assignment below
    study_areas_gdf["graph_file_list"] = None
    # everything for the markdown log is collected here and written once at the end of the run
    log = {"bursts": [], "seeds": [], "gse": []}

    # minigraphs need to be created for the calibration and dev sets as well
    for i in [0, 1, 2]:
        # one set at a time (0 = train, 1 = dev, 2 = cal), so the budget only comes from that set's regions
        train_mask = study_areas_gdf["set"] == i

        # Counting the seed-eligible points of every region of the set once, reused for the budget and the per burst share
        study_areas_gdf.loc[train_mask, "n_center_region"] = study_areas_gdf[train_mask].apply(
                        lambda row: count_points_in_aoi(preproc_csvs_for_region(row["region_path"]), row.geometry, buffer_m), axis=1)

        # Assigning the same seed budget to every region of the set, so all regions contribute equally
        study_areas_gdf.loc[train_mask, "n_seed_points_per_region"] = assign_n_seed_points_to_regions(study_areas_gdf[train_mask], k_nearest, density)

        # building the GRAPH files of every region of the set, their paths go into the region table
        study_areas_gdf.loc[train_mask, "graph_file_list"] = study_areas_gdf.loc[train_mask].apply(build_train_dataset_per_region, axis=1, k_nearest=k_nearest, density=density,
                    log=log, min_dist_m=min_dist_m, buffer_m=buffer_m, max_dist=max_dist, gse_index=gse_index)

    s01.study_areas_io("write", study_areas_gdf, Path(__file__).parent / "../../assets/study_areas_gdf")

    return study_areas_gdf, log


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--k-nearest", dest="k_nearest", type=int, required=True,
                        help="number of pool points per seed")
    parser.add_argument("--density", dest="density", type=float, required=True,
                        help="share (0, 1] of the burst capacity that is used for seeds")
    parser.add_argument("--min-dist", dest="min_dist", type=float, default=50.0,
                        help="minimum distance (m) between any two seeds")
    parser.add_argument("--buffer-m", dest="buffer_m", type=float, default=2000.0,
                        help="edge margin (m) inside the overlap of burst and core: no seeds there, but its points can be pool points")
    parser.add_argument("--max-dist", dest="max_dist", type=float, default=2000.0,
                            help="maximum distance (m) around a seed point from which pool candidates may come")

    args = parser.parse_args()

    # Make sure density has the right value range
    if not 0 < args.density <= 1:
        parser.error("--density must be in (0, 1]")

    start = datetime.now()
    study_areas_gdf, log = main(args.k_nearest, args.density, args.min_dist, args.buffer_m, args.max_dist)

    # the markdown log of the run, written only if the run got this far
    s01.write_md_log("s03_build_training_dataset", start, vars(args),
                     body=training_summary_md(study_areas_gdf, log, args.k_nearest, args.density))

