#######################################
# s02_preprocessing

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
import json

from pathlib import Path
from rasterio.merge import merge
from scipy.stats import mannwhitneyu

################################################
# 1.1) Constants
################################################

EGMS_KEEP_COLS = ["pid", "latitude", "longitude", "easting", "northing", "incidence_angle",
                  "track_angle", "los_east", "los_north", "los_up", "mean_velocity"]
EGMS_NORM_COLS = ["mean_velocity", "incidence_angle"]
EGMS_NORM_STATS_PATH = Path(__file__).parent / "../../assets/egms_norm_stats.json"
EGMS_FLOAT_COLS = {c: pl.Float64 for c in ["latitude", "longitude", "easting", "northing", "incidence_angle", "track_angle",
                                            "los_east", "los_north", "los_up", "mean_velocity",
                                            "temporal_coherence", "amplitude_dispersion"]}

##############################################################################
#============================================================================#
# 2) Helper Functions
#============================================================================#
##############################################################################

################################################
#===== Section 1: Shared Helper Functions =====#
################################################

def burst_csvs_for_region(region_path):
    """
    Lists the original burst csvs of one region (both orbits), skipping "*_PREPROCESSED.csv".

    Arguments:
        region_path -- str or Path, the region's folder

    Returns:
        csv_paths -- list of Path
    """
    region_path = Path(region_path)
    csvs = sorted(region_path.glob("ascending/*.csv")) + sorted(region_path.glob("descending/*.csv"))

    return [c for c in csvs if not c.stem.endswith("_PREPROCESSED")]


def all_burst_csvs(study_areas_gdf):
    """
    Lists the original burst csvs of every region in study_areas_gdf.

    Arguments:
        study_areas_gdf -- GeoDataFrame with a region_path column

    Returns:
        csv_paths -- list of Path
    """
    csvs = []
    for region_path in study_areas_gdf["region_path"]:
        csvs.extend(burst_csvs_for_region(region_path))

    return csvs

def preprocessed_path(csv_path):
    """
    Returns the path of a burst csv's preprocessed version (same folder, "_PREPROCESSED" appended).

    Arguments:
        csv_path -- str or Path, an original burst csv

    Returns:
        out_path -- Path
    """
    csv_path = Path(csv_path)
    return csv_path.with_name(f"{csv_path.stem}_PREPROCESSED{csv_path.suffix}")

################################################
#====== Section 2: EGMS Helper Functions ======#
################################################

def compute_qa_stats(csvs, coherence_q=0.05, ampdisp_q=0.95, sample_every=20):
    """
    Computes pooled quantiles of temporal_coherence and amplitude_dispersion over the given csvs,
    using only every sample_every-th row of each csv to keep memory low.

    Arguments:
        csvs -- list of Path, burst csvs to pool
        coherence_q -- float, quantile for temporal_coherence
        ampdisp_q -- float, quantile for amplitude_dispersion
        sample_every -- int, keep every n-th row of each csv

    Returns:
        result -- dict, {"coherence_q": ..., "ampdisp_q": ...}
    """
    lf = pl.concat([
        pl.scan_csv(csv_path, schema_overrides=EGMS_FLOAT_COLS).select(["temporal_coherence", "amplitude_dispersion"]).gather_every(sample_every)
        for csv_path in csvs
    ], how="vertical_relaxed")

    result = lf.select([
        pl.col("temporal_coherence").quantile(coherence_q).alias("coherence_q"),
        pl.col("amplitude_dispersion").quantile(ampdisp_q).alias("ampdisp_q"),
    ]).collect().row(0, named=True)

    return result


def filter_egms(csvs, egms_coherence, egms_ampdisp_max):
    """
    Keeps only points passing both quality thresholds and only EGMS_KEEP_COLS, written as
    "*_PREPROCESSED.csv" next to each original. Originals stay untouched.

    Arguments:
        csvs -- list of Path, original burst csvs
        egms_coherence -- float, minimum temporal_coherence
        egms_ampdisp_max -- float, maximum amplitude_dispersion

    Returns:
        None
    """
    for csv in csvs:
        out_path = preprocessed_path(csv)

        # Deleting existing files if there
        if out_path.exists():
            out_path.unlink()

        # Filter rows based on thresholds, then select only the needed cols
        lf = pl.scan_csv(csv, schema_overrides=EGMS_FLOAT_COLS).filter(
            (pl.col("temporal_coherence") >= egms_coherence) &
            (pl.col("amplitude_dispersion") <= egms_ampdisp_max)
        ).select(EGMS_KEEP_COLS)
        lf.sink_csv(out_path)

def compute_norm_stats(study_areas_gdf, egms_coherence, egms_ampdisp_max, stats_path=EGMS_NORM_STATS_PATH):
    """
    Computes mean/std of EGMS_NORM_COLS over the preprocessed csvs of train regions only and
    writes them to a json file.

    Arguments:
        study_areas_gdf -- GeoDataFrame with region_path and set columns
        egms_coherence -- float, filter threshold used (recorded only)
        egms_ampdisp_max -- float, filter threshold used (recorded only)
        stats_path -- Path, output json file

    Returns:
        stats -- dict, mean/std per column plus counts and thresholds
    """
    # Statistics for normalization are only computed from the training set
    train_gdf = study_areas_gdf.loc[study_areas_gdf["set"] == 0]
    csvs = [preprocessed_path(c) for c in all_burst_csvs(train_gdf)]

    lf = pl.concat([pl.scan_csv(c, schema_overrides=EGMS_FLOAT_COLS).select(EGMS_NORM_COLS) for c in csvs], how="vertical_relaxed")

    # compute the mean and std for the columns that need normalization
    row = lf.select([
        pl.len().alias("n_points"),
        *[pl.col(c).mean().alias(f"{c}_mean") for c in EGMS_NORM_COLS],
        *[pl.col(c).std().alias(f"{c}_std") for c in EGMS_NORM_COLS],
    ]).collect(engine="streaming").row(0, named=True)

    # save the stats in assets directory
    stats = {
        "source": "preprocessed csvs of train regions (set == 0)",
        "n_csvs": len(csvs),
        "n_points": row["n_points"],
        "filter": {"egms_coherence": egms_coherence, "egms_ampdisp_max": egms_ampdisp_max},
    }
    for c in EGMS_NORM_COLS:
        stats[c] = {"mean": row[f"{c}_mean"], "std": row[f"{c}_std"]}

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats

################################################
#====== Section 3: GNSS Helper Functions ======#
################################################

def remove_outlier(values, n=10, alpha=0.05, k=5):
    """
    Replaces isolated spikes with NaN but keeps real steps (e.g. earthquakes). A point deviating
    more than k MADs from its neighbourhood is kept only if a Mann-Whitney test shows a
    significant and large before/after shift.

    Arguments:
        values -- array-like, the time series
        n -- int, points before and after used as local context
        alpha -- float, significance level of the Mann-Whitney test
        k -- float, MAD multiplier for candidates and for the step size

    Returns:
        values_cp -- numpy.ndarray, same length, spikes set to NaN
    """
    # This function removes outliers from the GNSS time series without touching real steps
    # caused by earthquakes for example
    # Work on a copy of the real values
    values = np.asarray(values, dtype=float)
    values_cp = values.copy()

    # windowed approach with n values before and after the one that is being checked 
    for i in range(n, len(values)-n):
        before = values[i-n:i]
        after = values[i+1:i+1+n]
        local = np.concatenate([before, after])
        local_context_median = np.median(local)
        local_mad = np.median(abs(local - local_context_median))

        if local_mad == 0:
            continue

        if abs(values[i] - local_context_median) > k * (1.4826 * local_mad):
            _, p = mannwhitneyu(before, after, alternative="two-sided")
            magnitude = abs(np.median(after) - np.median(before))
            # a real step is significant and large; everything else is an isolated spike
            if not (p < alpha and magnitude > k * (1.4826 * local_mad)):
                values_cp[i] = np.nan

    return values_cp


def gnss_quality_stats(df_gnss, t0, t1):
    """
    Computes temporal coverage and largest gap of one GNSS series within [t0, t1).

    Arguments:
        df_gnss -- pandas.DataFrame, one tenv3 file with a "yyyy.yyyy" column
        t0 -- float, start (decimal year, inclusive)
        t1 -- float, end (decimal year, exclusive)

    Returns:
        result -- dict with n_points, coverage_frac and max_gap_days
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


def find_quality_filtered_stations(row, t0, t1, coverage_frac_lim=0.8, max_gap_days_lim=120):
    """
    Keeps the region's GNSS stations with enough coverage and no overlong gap in [t0, t1).

    Arguments:
        row -- pandas.Series, study_areas_gdf row with gnss_station_codes and gnss_station_paths
        t0 -- float, start (decimal year, inclusive)
        t1 -- float, end (decimal year, exclusive)
        coverage_frac_lim -- float, minimum covered fraction of the period
        max_gap_days_lim -- float, maximum allowed gap in days

    Returns:
        good_stations, good_station_paths -- two parallel lists of the passing stations
    """
    # This function filters out the stations that have incomplete GNSS time series...
    # Collect the stations that are ok in a set
    good_stations = []
    good_station_paths = []

    for station_code, station_path in zip(row["gnss_station_codes"], row["gnss_station_paths"]):

        # load the time series data
        gnss_df = pd.read_csv(station_path, sep=r"\s+")
        # compute the ts quality measures on every station df
        q_measures = gnss_quality_stats(gnss_df, t0, t1)
        # Check the quality measures
        if q_measures["coverage_frac"] >= coverage_frac_lim and q_measures["max_gap_days"] <= max_gap_days_lim:
            good_stations.append(station_code)
            good_station_paths.append(station_path)

    return good_stations, good_station_paths

def compute_station_velocities(row, t0, t1):
    """
    Fits an east/north/up linear trend (mm/yr) per quality-filtered station after despiking.

    Arguments:
        row -- pandas.Series, study_areas_gdf row with gnss_station_codes_filtered and gnss_station_paths_filtered
        t0 -- float, start (decimal year, inclusive)
        t1 -- float, end (decimal year, exclusive)

    Returns:
        velocities -- list of dicts {"station_code", "ve", "vn", "vu"}, one per station
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
        for short_name, col_name in zip(["ve", "vn", "vu"], ["__east(m)", "_north(m)", "____up(m)"]):
            # from metres to mm
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


################################################
#= Section 4: ESA WorldCover Helper Functions =#
################################################

def merge_wc_tiles(row):
    """
    Merges a region's WorldCover tiles into worldcover_merged.tif, keeping the original tiles.
    Reuses an existing merge; with zero or one tile nothing is merged.

    Arguments:
        row -- pandas.Series, study_areas_gdf row with region_path

    Returns:
        wc_merged_path -- Path or None, merged file, the single tile, or None if no tiles exist
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

def preprocess_egms(study_areas_gdf, egms_coherence=None, egms_ampdisp_max=None, compute_thresholds=False,
                    coherence_q=0.05, ampdisp_q=0.95, sample_every=20, file_every=3):
    """
    Quality-filters every burst csv (see filter_egms) and writes the train normalization stats
    (see compute_norm_stats). With compute_thresholds, both thresholds are derived as pooled
    quantiles from a sample of the csvs; otherwise the given values are used.

    Arguments:
        study_areas_gdf -- GeoDataFrame with region_path and set columns
        egms_coherence -- float or None, minimum temporal_coherence (required if not computed)
        egms_ampdisp_max -- float or None, maximum amplitude_dispersion (required if not computed)
        compute_thresholds -- bool, derive both thresholds from the data instead
        coherence_q -- float, quantile used to derive egms_coherence
        ampdisp_q -- float, quantile used to derive egms_ampdisp_max
        sample_every -- int, keep every n-th row of each sampled csv
        file_every -- int, use every n-th csv for the quantiles

    Returns:
        study_areas_gdf -- unchanged; outputs are written to disk
    """
    csvs = all_burst_csvs(study_areas_gdf)

    if compute_thresholds:
        qa_stats = compute_qa_stats(csvs[::file_every], coherence_q=coherence_q, ampdisp_q=ampdisp_q,
                                    sample_every=sample_every)
        egms_coherence = qa_stats["coherence_q"]
        egms_ampdisp_max = qa_stats["ampdisp_q"]
        print(f"Computed thresholds: coherence >= {egms_coherence}, amplitude_dispersion <= {egms_ampdisp_max}", flush=True)

    filter_egms(csvs, egms_coherence, egms_ampdisp_max)

    compute_norm_stats(study_areas_gdf, egms_coherence, egms_ampdisp_max)

    return study_areas_gdf


def preprocess_gnss(study_areas_gdf):
    """
    Filters the val regions' GNSS stations by coverage and gaps and fits their east/north/up
    trends. The unfiltered station columns are kept for reruns.

    Arguments:
        study_areas_gdf -- GeoDataFrame with gnss_station_codes/paths for val rows (set == 3)

    Returns:
        study_areas_gdf -- with gnss_station_codes_filtered, gnss_station_paths_filtered,
            gnss_velocities and gnss_filter_params added
    """
    # Filter the dataframe for the validation set because only there, gnss data is even available on disk
    val_mask = study_areas_gdf["set"] == 3
    coverage_frac_lim, max_gap_days_lim = 0.8, 120
    t0, t1 = 2019.0, 2024.0

    # Filter out the stations with insufficient temporal coverage or too large gaps
    result = study_areas_gdf.loc[val_mask].apply(find_quality_filtered_stations, axis=1, t0=t0, t1=t1,
                                            coverage_frac_lim=coverage_frac_lim, max_gap_days_lim=max_gap_days_lim)

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
        {"coverage_frac_lim": coverage_frac_lim, "max_gap_days_lim": max_gap_days_lim, "t0": t0, "t1": t1}
        for _ in range(val_mask.sum())
    ]

    return study_areas_gdf

def preprocess_esa10mwc(study_areas_gdf):
    """
    Merges every region's WorldCover tiles, deleting stale merged files first so the merge is redone.

    Arguments:
        study_areas_gdf -- GeoDataFrame with a region_path column

    Returns:
        study_areas_gdf -- with a wc_merged_path column added
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
         compute_thresholds=False, coherence_q=0.05, ampdisp_q=0.95, sample_every=20, file_every=3):
    """
    Loads study_areas_gdf, runs the selected preprocessing steps and saves the gdf.

    Arguments:
        run_egms, run_gnss, run_worldcover -- bool, which steps to run
        egms_coherence, egms_ampdisp_max, compute_thresholds, coherence_q, ampdisp_q,
            sample_every, file_every -- passed to preprocess_egms

    Returns:
        None
    """
    # Import the gdf again to have access to all the file paths
    study_areas_gdf = s01.study_areas_io("read", base_path = Path(__file__).parent / "../../assets/study_areas_gdf")

    if run_egms:
        study_areas_gdf = preprocess_egms(study_areas_gdf, egms_coherence=egms_coherence,
                                          egms_ampdisp_max=egms_ampdisp_max, compute_thresholds=compute_thresholds,
                                          coherence_q=coherence_q, ampdisp_q=ampdisp_q,
                                          sample_every=sample_every, file_every=file_every)

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
                         help="minimum temporal_coherence to keep a point; required unless --egms-compute-thresholds")
    parser.add_argument("--egms-ampdisp-max", dest="egms_ampdisp_max", type=float, default=None,
                         help="maximum amplitude_dispersion to keep a point; required unless --egms-compute-thresholds")
    parser.add_argument("--egms-compute-thresholds", dest="compute_thresholds", action="store_true", default=False,
                         help="derive both thresholds as quantiles from a sample of the csvs (off by default)")
    parser.add_argument("--egms-coherence-q", dest="coherence_q", type=float, default=0.05,
                         help="quantile used to derive the coherence threshold")
    parser.add_argument("--egms-ampdisp-q", dest="ampdisp_q", type=float, default=0.95,
                         help="quantile used to derive the amplitude_dispersion threshold")
    parser.add_argument("--egms-sample-every", dest="sample_every", type=int, default=20,
                         help="for the quantiles, keep every n-th row of each sampled csv")
    parser.add_argument("--egms-file-every", dest="file_every", type=int, default=3,
                         help="for the quantiles, use every n-th csv")

    args = parser.parse_args()

    # thresholds come either from the data or from the command line, never from both
    if args.run_egms:
        given = args.egms_coherence is not None or args.egms_ampdisp_max is not None
        if args.compute_thresholds and given:
            parser.error("pass either --egms-compute-thresholds or the threshold values, not both")
        if not args.compute_thresholds and (args.egms_coherence is None or args.egms_ampdisp_max is None):
            parser.error("--egms-coherence and --egms-ampdisp-max are required unless --egms-compute-thresholds is set")

    # Saving the important parameters in a log file
    s01.write_param_log("s02_preprocessing", s01.PARAM_LOG_DIR, vars(args))

    main(run_egms=args.run_egms, run_gnss=args.run_gnss, run_worldcover=args.run_worldcover,
         egms_coherence=args.egms_coherence, egms_ampdisp_max=args.egms_ampdisp_max,
         compute_thresholds=args.compute_thresholds, coherence_q=args.coherence_q, ampdisp_q=args.ampdisp_q,
         sample_every=args.sample_every, file_every=args.file_every)

