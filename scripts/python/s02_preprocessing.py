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
import polars as pl

from pathlib import Path
from rasterio.merge import merge
from scipy.stats import mannwhitneyu
from datetime import datetime

##############################################################################
#============================================================================#
# 2) Helper Functions
#============================================================================#
##############################################################################

################################################
#===== Section 1: Shared Helper Functions =====#
################################################

def write_param_log(step_name, params):
    """
    Writes a short log file recording the runtime parameters a processing step was called with,
    for provenance -- so a later rerun or a cached output can be checked against what parameters
    actually produced it.

    Arguments:
        step_name -- string, name of the processing step producing this log (e.g. "s02_preprocessing")
        params -- dict, the runtime parameters to record (e.g. vars(args) from argparse)

    Returns:
        log_path -- Path, the log file that was written
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = PARAM_LOG_DIR / f"{step_name}_log_{timestamp}.log"

    if log_path.exists():
        log_path.unlink()

    with open(log_path, "w") as f:
        f.write(f"step: {step_name}\n")
        f.write(f"time: {datetime.now().isoformat()}\n")
        f.write("parameters:\n")
        for key, value in params.items():
            f.write(f"  {key}: {value}\n")

    return log_path

def burst_csvs_for_region(region_path):
    """
    Lists every burst csv downloaded for one region, both orbit directions.

    Arguments:
        region_path -- path (str or Path) to the region's folder (study_areas_gdf's region_path column)

    Returns:
        csv_paths -- list of Path, every "*.csv" under region_path/ascending and region_path/descending
    """
    region_path = Path(region_path)
    return list(region_path.glob("ascending/*.csv")) + list(region_path.glob("descending/*.csv"))


def all_burst_csvs(study_areas_gdf):
    """
    Lists every burst csv downloaded for every region in study_areas_gdf, pooled into one flat list.

    Arguments:
        study_areas_gdf -- geopandas.GeoDataFrame, with region_path set for every region

    Returns:
        csv_paths -- list of Path, every burst csv across every region (both orbit directions)
    """
    csvs = []
    for region_path in study_areas_gdf["region_path"]:
        csvs.extend(burst_csvs_for_region(region_path))

    return csvs


################################################
#====== Section 2: EGMS Helper Functions ======#
################################################

def compute_qa_stats(csvs, coherence_q=0.05, ampdisp_q=0.95):
    """
    Computes the given quantiles of temporal_coherence and amplitude_dispersion, pooled across
    the given burst csvs. Used by preprocess_egms to derive thresholds when none are supplied.

    Arguments:
        csvs -- list of Path, burst csvs to pool (e.g. from all_burst_csvs)
        coherence_q -- float, quantile to compute for temporal_coherence (the "bad" tail is low)
        ampdisp_q -- float, quantile to compute for amplitude_dispersion (the "bad" tail is high)

    Returns:
        result -- dict, {"coherence_q": ..., "ampdisp_q": ...}, the two pooled quantile values
    """
    lf = pl.concat([
        pl.scan_csv(csv_path).select(["temporal_coherence", "amplitude_dispersion"])
        for csv_path in csvs
    ])

    result = lf.select([
        pl.col("temporal_coherence").quantile(coherence_q).alias("coherence_q"),
        pl.col("amplitude_dispersion").quantile(ampdisp_q).alias("ampdisp_q"),
    ]).collect().row(0, named=True)

    return result


def filter_egms(csvs, egms_coherence, egms_ampdisp_max):
    """
    Filters each given EGMS burst csv down to points passing both quality thresholds, writing
    each result next to the original with "_PREPROCESSED" appended to the filename. Originals
    are never touched, so a rerun with different thresholds is always possible. Any stale
    filtered output from a previous run is removed first.

    Arguments:
        csvs -- list of Path, burst csvs to filter (e.g. from all_burst_csvs)
        egms_coherence -- float, minimum temporal_coherence a point must have to be kept
        egms_ampdisp_max -- float, maximum amplitude_dispersion a point may have to be kept

    Returns:
        None -- writes a "*_PREPROCESSED.csv" next to each input csv as a side effect
    """
    for csv in csvs:
        csv = Path(csv)
        out_path = csv.with_name(f"{csv.stem}_PREPROCESSED{csv.suffix}")

        if out_path.exists():
            out_path.unlink()

        lf = pl.scan_csv(csv).filter(
            (pl.col("temporal_coherence") >= egms_coherence) &
            (pl.col("amplitude_dispersion") <= egms_ampdisp_max)
        )
        lf.sink_csv(out_path)


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
        row -- pandas.Series, a study_areas_gdf row with gnss_station_codes and gnss_station_paths
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

    for station_code, station_path in zip(row["gnss_station_codes"], row["gnss_station_paths"]):

        # load the time series data
        gnss_df = pd.read_csv(station_path, sep=r"\s+")
        # compute the ts quality measures on every station df
        q_measures = gnss_quality_stats(gnss_df, 2019.0, 2024.0)
        # Check the quality measures
        if q_measures["coverage_frac"] >= coverag_frac_lim and q_measures["max_gap_days"] <= max_gap_days_lim:
            good_stations.append(station_code)
            good_station_paths.append(station_path)

    return good_stations, good_station_paths

def compute_station_velocities(row, t0, t1):
    """
    Computes the east/north/up trend (mm/yr) for every (already quality-filtered) GNSS station
    in a region, over the given study period: despikes each component with remove_outlier, then
    fits a straight line to what's left.

    Arguments:
        row -- pandas.Series, a study_areas_gdf row with gnss_station_codes_filtered and
            gnss_station_paths_filtered (the quality-passing subset, see find_quality_filtered_stations)
        t0 -- float, start of the study period (decimal year, inclusive)
        t1 -- float, end of the study period (decimal year, exclusive)

    Returns:
        velocities -- list of dicts, one per station, each {"station_code":, "ve":, "nv":, "vu":}
            with the three trends in mm/yr
    """
    # This function computes the velocities for the stations that passed the quality filter.
    velocities = []

    for station_code, station_path in zip(row["gnss_station_codes_filtered"], row["gnss_station_paths_filtered"]):
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
    Merges a region's individual ESA WorldCover tiles into one GeoTIFF. The originals are kept
    (not deleted) specifically so this step can be redone later -- deleting them would make a
    rerun impossible, since there'd be nothing left to merge from.
    Safe to call again on an already-merged region (returns the existing merged file without
    redoing anything) or a region with zero or one tile downloaded (nothing to merge).

    Arguments:
        row -- pandas.Series, a study_areas_gdf row with region_path (the region's folder)

    Returns:
        wc_merged_path -- Path or None, the merged GeoTIFF's path, the single tile's path if
            there was only one, or None if no WorldCover tiles exist for this region at all
    """
    # This function merges the world cover tiles per location directory.

    folder = Path(row["region_path"])
    tifs = list(folder.glob("ESA_WorldCover*.tif"))

    wc_merged_path = folder / "worldcover_merged.tif"
    if wc_merged_path.exists():
        return wc_merged_path    # already merged in an earlier run, reuse it, don't touch it
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
    # uses the new profile to open a new tif in write mode
    with rasterio.open(wc_merged_path, "w", **profile) as dst:
        dst.write(mosaic)
    # closes the opened datasets -- the individual tiles are deliberately NOT deleted, see docstring
    for ds in datasets:
        ds.close()

    return wc_merged_path

##############################################################################
#============================================================================#
# 3) Processing
#============================================================================#
##############################################################################

def preprocess_egms(study_areas_gdf, egms_coherence=None, egms_ampdisp_max=None, coherence_q=0.05, ampdisp_q=0.95):
    """
    Filters every burst csv's points down to the ones passing quality thresholds on
    temporal_coherence and amplitude_dispersion, writing each result as a sibling
    "*_PREPROCESSED.csv" (see filter_egms). If egms_coherence/egms_ampdisp_max aren't given,
    both are derived from the data itself as the given quantiles, pooled across every burst csv
    in study_areas_gdf (see compute_qa_stats).

    Arguments:
        study_areas_gdf -- geopandas.GeoDataFrame, with region_path set for every region
        egms_coherence -- float or None, minimum temporal_coherence to keep a point; if None
            (together with egms_ampdisp_max), computed from coherence_q
        egms_ampdisp_max -- float or None, maximum amplitude_dispersion to keep a point; if None
            (together with egms_coherence), computed from ampdisp_q
        coherence_q -- float, quantile used to derive egms_coherence when it isn't given
        ampdisp_q -- float, quantile used to derive egms_ampdisp_max when it isn't given

    Returns:
        study_areas_gdf -- the same GeoDataFrame, unchanged (the filtered csvs are written to
            disk as a side effect, not tracked as a new column yet)
    """
    csvs = all_burst_csvs(study_areas_gdf)

    # if either threshold is missing, derive both from the data's own quantiles
    if egms_coherence is None or egms_ampdisp_max is None:
        qa_stats = compute_qa_stats(csvs, coherence_q=coherence_q, ampdisp_q=ampdisp_q)
        egms_coherence = qa_stats["coherence_q"]
        egms_ampdisp_max = qa_stats["ampdisp_q"]

    filter_egms(csvs, egms_coherence, egms_ampdisp_max)

    return study_areas_gdf

def preprocess_gnss(study_areas_gdf):
    """
    Runs the full GNSS preprocessing step for the val regions: filters out low-quality stations
    into new gnss_station_codes_filtered/gnss_station_paths_filtered columns (the originals from
    01_download_data.py are left untouched, so this can be rerun later with different thresholds),
    computes each survivor's east/north/up trend, and records the thresholds this run used.

    Arguments:
        study_areas_gdf -- geopandas.GeoDataFrame, with gnss_station_codes/gnss_station_paths
            already populated for val (set==3) rows by 01_download_data.py

    Returns:
        study_areas_gdf -- the same GeoDataFrame, with gnss_station_codes_filtered,
            gnss_station_paths_filtered, gnss_velocities, and gnss_filter_params added
    """
    # Filter the dataframe for the validation set because only there, gnss data is even available on disk
    val_mask = study_areas_gdf["set"] == 3
    coverag_frac_lim, max_gap_days_lim = 0.8, 120
    t0, t1 = 2019.0, 2024.0

    # Filter out the stations with insuffiecient temporal coverage or too large gaps
    result = study_areas_gdf.loc[val_mask].apply(find_quality_filtered_stations, axis=1,
                                              coverag_frac_lim=coverag_frac_lim, max_gap_days_lim=max_gap_days_lim)

    # New columns, not overwriting gnss_station_codes/gnss_station_paths -- keeps the original
    # candidate list around so a future rerun with different thresholds has something to work from
    study_areas_gdf["gnss_station_codes_filtered"] = None
    study_areas_gdf["gnss_station_paths_filtered"] = None
    study_areas_gdf.loc[val_mask, "gnss_station_codes_filtered"] = result.apply(lambda r: r[0])
    study_areas_gdf.loc[val_mask, "gnss_station_paths_filtered"] = result.apply(lambda r: r[1])

    # Computing the station velocities, on the filtered (quality-passing) stations only
    study_areas_gdf["gnss_velocities"] = None
    study_areas_gdf.loc[val_mask, "gnss_velocities"] = study_areas_gdf.loc[val_mask].apply(
        compute_station_velocities, axis=1, t0=t0, t1=t1)

    # Records the parameters this run used. Plain overwrite every run, no history kept on purpose.
    study_areas_gdf["gnss_filter_params"] = None
    study_areas_gdf.loc[val_mask, "gnss_filter_params"] = [
        {"coverag_frac_lim": coverag_frac_lim, "max_gap_days_lim": max_gap_days_lim, "t0": t0, "t1": t1}
        for _ in range(val_mask.sum())
    ]

    return study_areas_gdf

def preprocess_esa10mwc(study_areas_gdf):
    """
    Merges every region's individual ESA WorldCover tiles into one GeoTIFF. Deletes any
    previously-merged output first (never an original tile -- see the name check below), so
    this step can genuinely be rerun instead of merge_wc_tiles silently skipping already-merged
    regions.

    Arguments:
        study_areas_gdf -- geopandas.GeoDataFrame, with region_path set for every region

    Returns:
        study_areas_gdf -- the same GeoDataFrame, with a new wc_merged_path column (the merged
            GeoTIFF's path per region)
    """
    # clear stale merged output so merge_wc_tiles' own "already merged" check doesn't skip the redo.
    # only ever deletes files actually named worldcover_merged.tif -- never an original tile, which
    # is what wc_merged_path holds instead when a region only ever had exactly one tile to begin with
    if "wc_merged_path" in study_areas_gdf.columns:
        for p in study_areas_gdf["wc_merged_path"].dropna():
            p = Path(p)
            if p.name == "worldcover_merged.tif" and p.exists():
                p.unlink()

    # merging the individual esa worldcover tiles
    study_areas_gdf["wc_merged_path"] = study_areas_gdf.apply(merge_wc_tiles, axis = 1)

    return study_areas_gdf

##############################################################################
#============================================================================#
# 4) Main
#============================================================================#
##############################################################################

def main(run_egms=True, run_gnss=True, run_worldcover=True, egms_coherence=None, egms_ampdisp_max=None,
         coherence_q=0.05, ampdisp_q=0.95):
    """
    Loads study_areas_gdf, runs the EGMS/GNSS/ESA WorldCover preprocessing steps in sequence
    (each individually skippable), and saves the updated GeoDataFrame back to disk.

    Arguments:
        run_egms -- bool, whether to run preprocess_egms
        run_gnss -- bool, whether to run preprocess_gnss
        run_worldcover -- bool, whether to run preprocess_esa10mwc
        egms_coherence -- float or None, passed through to preprocess_egms
        egms_ampdisp_max -- float or None, passed through to preprocess_egms
        coherence_q -- float, passed through to preprocess_egms
        ampdisp_q -- float, passed through to preprocess_egms

    Returns:
        None
    """
    # Import the gdf again to have access to all the file paths
    study_areas_gdf = s01.study_areas_io("read", base_path = Path(__file__).parent / "../../assets/study_areas_gdf")

    if run_egms:
        study_areas_gdf = preprocess_egms(study_areas_gdf, egms_coherence, egms_ampdisp_max, coherence_q, ampdisp_q)

    if run_gnss:
        study_areas_gdf = preprocess_gnss(study_areas_gdf)

    if run_worldcover:
        study_areas_gdf = preprocess_esa10mwc(study_areas_gdf)

    s01.study_areas_io("write", study_areas_gdf, Path(__file__).parent / "../../assets/study_areas_gdf")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # every step runs by default -- pass the matching --no-... flag to skip just that one
    parser.add_argument("--no-egms", dest="run_egms", action="store_false", default=True,
                         help="skip the EGMS preprocessing step")
    parser.add_argument("--no-gnss", dest="run_gnss", action="store_false", default=True,
                         help="skip the GNSS preprocessing step")
    parser.add_argument("--no-worldcover", dest="run_worldcover", action="store_false", default=True,
                         help="skip the WorldCover merge step")
    parser.add_argument("--egms-coherence", dest="egms_coherence", type=float, default=None,
                         help="minimum temporal_coherence to keep a point; omit to derive it from the data")
    parser.add_argument("--egms-ampdisp-max", dest="egms_ampdisp_max", type=float, default=None,
                         help="maximum amplitude_dispersion to keep a point; omit to derive it from the data")
    parser.add_argument("--egms-coherence-q", dest="coherence_q", type=float, default=0.05,
                         help="quantile used to derive --egms-coherence when it's omitted")
    parser.add_argument("--egms-ampdisp-q", dest="ampdisp_q", type=float, default=0.95,
                         help="quantile used to derive --egms-ampdisp-max when it's omitted")

    args = parser.parse_args()

    # Saving the important parameters in a log file
    write_param_log("s02_preprocessing", vars(args))

    main(run_egms=args.run_egms, run_gnss=args.run_gnss, run_worldcover=args.run_worldcover, 
         egms_coherence=args.egms_coherence, egms_ampdisp_max=args.egms_ampdisp_max,
         coherence_q=args.coherence_q, ampdisp_q=args.ampdisp_q)

