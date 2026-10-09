#######################################
# 02_preprocessing

##############################################################################
#============================================================================#
# 1) Import Statements
#============================================================================#
##############################################################################

import argparse 
import rasterio
import s01_download_data as s01
import pandas as pd
import numpy as np

from pathlib import Path
from rasterio.merge import merge
from scipy.stats import mannwhitneyu


##############################################################################
#============================================================================#
# 2) Helper Functions
#============================================================================#
##############################################################################

################################################
#===== Section 1: Shared Helper Functions =====#
################################################
"""
2. What preprocessing needs to handle
Grounding this in what project.md actually specifies for this stage, roughly in the order it'd naturally happen:

Quality filtering first. Every EGMS point carries rmse, temporal_coherence, amplitude_dispersion, mp_type (PS vs DS). Decide thresholds and drop/flag low-quality points before anything downstream uses them — doing this after basis-fitting means you've wasted compute fitting curves to points you'll throw away anyway.

Basis-coefficient fitting, per §3.5 — the centerpiece of this stage. Fit each point's LOS series to the small basis (linear trend + annual/semiannual sinusoids + ~3 cubic-spline knots) via lstsq, once, cached. Critically, fit ascending and descending separately, on each orbit's own native epochs — §3.5 is explicit that forcing both onto one shared epoch grid means guessing at dates one orbit never observed. Only after both are in coefficient space do they reconstruct onto a common grid via series = Φ @ coeffs.

Global normalization, §4.6 — train split only. Mean/std for LOS coefficients, every predictor channel, and distances, computed by pooling all train-labeled regions together. Needs to happen after coefficient-fitting (you're normalizing coefficients, not raw series) and needs the split already assigned (which it is, via set).

Predictor alignment — one real engineering task per source. AlphaEarth tiles are per-UTM-zone; ESA WorldCover and whatever DEM/hydro source you pick will each have their own native CRS and resolution. Every predictor needs to be sampled at each EGMS point's actual location, not just downloaded and left as-is. §4.1 says DEM should land on an "EPSG:3035 unified grid" — I'd apply that same discipline to all of them for consistency, even though §3.3 only forbids absolute coordinates in the graph itself, not a shared working CRS for sampling.

AlphaEarth dequantization, before it's usable as a feature at all: x = ((v / 127.5) ** 2) * np.sign(v), int8 → unit-length float64. Also where you'd fix the full-tile-download issue above, if that gets addressed at this stage instead of in the download script — either point works, just needs to happen once, not per training step.

§10.9's position-leakage tests, flagged explicitly as "cheap, do before scaling." Now that AlphaEarth is actually being downloaded, this is the moment to run the position probe and geographic ablation — before you've built a whole training pipeline around embeddings that might be smuggling absolute location back in.

A sanity check on the split's geography, not just its labels. §4.5 wants train blocks spatially far from test/cal blocks so a test point's neighbours are never training points. The set column plus the 150km exclusion buffer should already guarantee this, but it's cheap to verify directly — compute the actual minimum distance between any train-labeled region and any cal/val-labeled region, and confirm it comfortably exceeds whatever k-NN neighbourhood radius you end up using.

Decide the output shape. Everything above collapses into one decision: what does a finished, per-point record actually look like (pid, projected coordinates, basis coefficients, predictor values, quality flags, split label), and in what file format — given hundreds of thousands of points per region, this is the first point where Parquet (you've already got polars) is probably worth it over CSV.

# remember to crop the egms coverage to the actual bounding box defined. otherwise, too much surrounding 
# points with barely any real displacement will be introduced through the regions that are actually just 
# supposed to cover the real subsidence

"""

################################################
#====== Section 2: EGMS Helper Functions ======#
################################################  

################################################
#====== Section 3: GNSS Helper Functions ======#
################################################

def remove_outlier(values, n=10, alpha=0.05, k=5):
    """
    Removes isolated spikes from a time series while leaving real, persistent steps (e.g.
    earthquakes, equipment changes) untouched. For each interior point, compares its deviation
    from a local median/MAD baseline (the n points immediately before/after it, excluding itself)
    against a robust threshold; candidates are then tested with a Mann-Whitney U test on the
    before/after windows, and only removed if the before/after difference is both statistically
    significant and large relative to the local noise level.

    Arguments:
        values -- array-like, the raw time series values (e.g. GNSS east/north/up displacement, mm)
        n -- int, number of points taken immediately before and after each candidate as its local context
        alpha -- float, significance threshold for the Mann-Whitney U test
        k -- float, number of local MADs a point must deviate by to become a candidate, and the
            same multiplier applied to the before/after shift when deciding if a candidate is a real step

    Returns:
        values_cp -- numpy.ndarray, same length as values, isolated spikes replaced by NaN, real
            steps left unchanged
    """
    # This function removes outliers from the GNSS time series without touching real steps
    # caused by earthquakes for example
    # Work on a copy of the real values
    values_cp = np.asarray(values, dtype = float).copy()

    for i in range(n, len(values_cp)-n):
        before = values_cp[i-n:i]
        after = values_cp[i+1:i+1+n]
        # bring both together in one array and compute the median for reference
        local = np.concatenate([before, after])
        local_context_median = np.median(local)
        # mad = median absolute deviation
        local_mad =  np.median(abs(local - local_context_median))

        if local_mad==0:
            continue

        # Finding candidates for removal
        if abs(values_cp[i] - local_context_median) > k * (1.4826 * local_mad):
            stats, p = mannwhitneyu(before, after, alternative="two-sided")
            # Compute the absolute magnitude of the difference between the before and after time series
            magnitude = abs(np.median(after)-np.median(before))

            # the step is only real if test significant and the magnitude is large enough
            # Second criteria ensures that in very low noise level environments, spikes which are 
            # large relative to local noise level are caught.
            if p < alpha and magnitude > k * (1.4826 * local_mad):
                values_cp[i] = values_cp[i]
            else:
                values_cp[i] = np.nan

    return values_cp

def gnss_quality_stats(df_gnss, t0, t1):
    """
    Computes temporal coverage and the largest data gap for one GNSS station's time series
    within a given study period, used to decide whether that station's record is complete enough to keep.

    Arguments:
        df_gnss -- pandas.DataFrame, one station's loaded tenv3 file, with a "yyyy.yyyy" decimal-year column
        t0 -- float, start of the study period (decimal year, inclusive)
        t1 -- float, end of the study period (decimal year, exclusive)

    Returns:
        result -- dict with "n_points" (int, epochs inside the study period), "coverage_frac"
            (float, fraction of the study period actually spanned by data), and "max_gap_days"
            (float, the largest gap between consecutive epochs, in days)
    """
    # this function computes the statistics needed to decide whether a station is kept or not

    # Defining the two stats
    coverage_frac = 0.0
    max_gap_days = float("Inf")

    # Pull out the date column as a numpy array
    t = df_gnss["yyyy.yyyy"].to_numpy()
    # Creating a boolean mask for the time within the study period
    # Since yyyy.yyyy is decimal representation, the computations work directly on this
    mask = (t >= t0) & (t < t1)
    # Counting how many entries are within that period. If nothing is inside: return immediately
    n = mask.sum()
    if n < 2:
        return {"n_points":n, "coverage_frac":coverage_frac, "max_gap_days": max_gap_days}

    # Sorting the entries to be sure
    t_win = np.sort(t[mask])

    # compute the coverage fraction in the window
    coverage_frac = (t_win.max() - t_win.min()) / (t1 - t0)

    # np.diff computes the distance between consecutive entries, here in decimal years
    gaps_years = np.diff(t_win)
    # Turning the decimal years into days
    gaps_days = gaps_years * 365.25 
    # Find the maximum gap
    max_gap_days = gaps_days.max()

    return {"n_points":n, "coverage_frac":coverage_frac, "max_gap_days": max_gap_days}


def find_quality_filtered_stations(row, coverag_frac_lim=0.8, max_gap_days_lim=120):
    """
    Filters a region's GNSS stations down to the ones with sufficient temporal coverage and no
    gap larger than the given limit, using gnss_quality_stats on each station's already-downloaded
    tenv3 file.

    Arguments:
        row -- pandas.Series, a study_areas_gdf row with station_codes and station_file_paths
        coverag_frac_lim -- float, minimum fraction of the study period a station's data must span
        max_gap_days_lim -- float, largest allowed gap between consecutive epochs, in days

    Returns:
        good_stations, good_station_paths -- two parallel lists, the codes and file paths of the
            stations that passed both thresholds (same order, same length)
    """
    # This function filters out the stations that have incomplete GNSS time series...
    # Collect the stations that are ok in a set
    good_stations = []
    good_station_paths = []

    for station_code, station_path in zip(row["station_codes"], row["station_file_paths"]):

        # load the time series data
        gnss_df = pd.read_csv(station_path, sep=r"\s+")
        # compute the ts quality measures on every station df
        q_measures = gnss_quality_stats(gnss_df, 2019.0, 2024.0)
        # Check the quality measures
        if q_measures["coverage_frac"] >= coverag_frac_lim and q_measures["max_gap_days"] <= max_gap_days_lim:
            good_stations.append(station_code)
            good_station_paths.append(station_path)

    return good_stations, good_station_paths

def compute_station_velovities(row, t0, t1):
    """
    Computes the east/north/up trend (mm/yr) for every (already quality-filtered) GNSS station
    in a region, over the given study period: despikes each component with remove_outlier, then
    fits a straight line to what's left.

    Arguments:
        row -- pandas.Series, a study_areas_gdf row with station_codes and station_file_paths
        t0 -- float, start of the study period (decimal year, inclusive)
        t1 -- float, end of the study period (decimal year, exclusive)

    Returns:
        velocities -- list of dicts, one per station, each {"station_code":, "ve":, "nv":, "vu":}
            with the three trends in mm/yr
    """
    # This function computes the velocities for the stations that passed the quality filter.
    velocities = []

    for station_code, station_path in zip(row["station_codes"], row["station_file_paths"]):
        # load the time series data and filter for the time period of interest again
        gnss_df = pd.read_csv(station_path, sep=r"\s+")
        t = gnss_df["yyyy.yyyy"].to_numpy()

        # Defining the mask for temporal filtering
        mask = (t >= t0) & (t < t1)

        # Filtering the dataframe and the time stamps
        gnss_df = gnss_df[mask]
        t = t[mask]

        # Creating an initial dict for every station
        result = {"station_code":station_code}

        # Looping over the 3 components of the 3D displacement vector
        for short_name, col_name in zip(["ve", "nv", "vu"], ["__east(m)", "_north(m)", "____up(m)"]):
            # form meter to mm unit
            ts = gnss_df[col_name] * 1000
            # cleaning up the time series 
            ts = remove_outlier(ts, n=10, alpha=0.05, k=5)
            # filtering the time series and the date array to ensure dimension match
            valid = ~np.isnan(ts)
            ts, t_valid = ts[valid], t[valid]

            # fit the linear trend
            trend = np.polyfit(t_valid, ts, 1)[0]

            result[short_name] = trend
                    
        velocities.append(result)

    return velocities


#################################################
#===== Section 4: ESA10wc Helper Functions =====#
#################################################

def merge_wc_tiles(row):
    """
    Merges a region's individual ESA WorldCover tiles into one GeoTIFF and deletes the originals.
    Safe to call again on an already-merged region (returns the existing merged file without
    redoing anything) or a region with zero or one tile downloaded (nothing to merge).

    Arguments:
        row -- pandas.Series, a study_areas_gdf row with path (the region's folder)

    Returns:
        wc_path -- Path or None, the merged GeoTIFF's path, the single tile's path if there was
            only one, or None if no WorldCover tiles exist for this region at all
    """
    # This function merges the world cover tiles per location directory.

    folder = Path(row["path"])
    tifs = list(folder.glob("ESA_WorldCover*.tif"))

    wc_path = folder / "worldcover_merged.tif"
    if wc_path.exists():
        return wc_path          # already merged in an earlier run, reuse it, don't touch it
    if len(tifs) == 0:
        return None              # genuinely nothing downloaded for this region
    if len(tifs) == 1:
        return tifs[0]           # nothing to merge, the one file already is "the" worldcover file
        

    datasets = [rasterio.open(f) for f in tifs]
    mosaic, transform = merge(datasets)
    # .profile is a dict of everything needed to write a GeoTIFF: driver, dtype, CRS, nodata value, compression, band count, height, width, transform
    profile = datasets[0].profile
    # only the shape of the new tif changes, the rest stays the same as in the input
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform)
    # New worldcover tif path
    wc_path = folder / "worldcover_merged.tif"
    # uses the new profile to open a new tif in write mode
    with rasterio.open(wc_path, "w", **profile) as dst:
        dst.write(mosaic)
    # closes the opened datasets and deletes the individual tif files. 
    for ds in datasets:
        ds.close()
    for f in tifs:
        f.unlink()

    return wc_path

##############################################################################
#============================================================================#
# 3) Processing
#============================================================================#
##############################################################################

def preprocess_egms():
    """
    Placeholder for EGMS preprocessing -- not needed at this stage of the pipeline yet.

    Arguments:
        None

    Returns:
        None
    """
    # actually doesnt need preprocessing at this point.
    pass

def preprocess_gnss(study_areas_gdf):
    """
    Runs the full GNSS preprocessing step for the val regions: filters out low-quality stations,
    updates station_codes/station_file_paths to only the survivors, and computes each survivor's
    east/north/up trend.

    Arguments:
        study_areas_gdf -- geopandas.GeoDataFrame, with station_codes/station_file_paths already
            populated for val (set==3) rows by 01_download_data.py

    Returns:
        study_areas_gdf -- the same GeoDataFrame, with station_codes/station_file_paths filtered
            down to the quality-passing stations and a new station_velocities column added
    """
    # Filter the dataframe for the validation set because only there, gnss data is even available on disk
    val_mask = study_areas_gdf["set"] == 3
    # Filter out the stations with insuffiecient temporal coverage or too large gaps
    result = study_areas_gdf.loc[val_mask].apply(find_quality_filtered_stations, axis=1,
                                              coverag_frac_lim=0.8, max_gap_days_lim=120)
    
    # Putting the right result in the right column
    study_areas_gdf.loc[val_mask, "station_codes"] = result.apply(lambda r: r[0])
    study_areas_gdf.loc[val_mask, "station_file_paths"] = result.apply(lambda r: r[1])

    # Computing the station velocities
    study_areas_gdf["station_velocities"] = study_areas_gdf.loc[val_mask].apply(compute_station_velovities, axis=1, t0=2019.0, t1=2024.0)

    return study_areas_gdf

def preprocess_esa10mwc(study_areas_gdf):
    """
    Merges every region's individual ESA WorldCover tiles into one GeoTIFF.

    Arguments:
        study_areas_gdf -- geopandas.GeoDataFrame, with path set for every region

    Returns:
        study_areas_gdf -- the same GeoDataFrame, with a new wc_path column (the merged GeoTIFF's
            path per region)
    """
    # merging the individual esa worldcover tiles
    study_areas_gdf["wc_path"] = study_areas_gdf.apply(merge_wc_tiles, axis = 1)
    
    return study_areas_gdf

##############################################################################
#============================================================================#
# 4) Main
#============================================================================#
##############################################################################

def main():
    """
    Loads study_areas_gdf, runs the EGMS/GNSS/ESA WorldCover preprocessing steps in sequence,
    and saves the updated GeoDataFrame back to disk.

    Arguments:
        None

    Returns:
        None
    """
    # Import the gdf again to have access to all the file paths
    study_areas_gdf = s01.study_areas_io("read", base_path = Path(__file__).parent / "../../assets/study_areas_gdf")

    preprocess_egms()

    study_areas_gdf = preprocess_gnss(study_areas_gdf)

    study_areas_gdf = preprocess_esa10mwc(study_areas_gdf)

    s01.study_areas_io("write", study_areas_gdf, Path(__file__).parent / "../../assets/study_areas_gdf")
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--projectdir", required=True)

    args = parser.parse_args()

    main() 

