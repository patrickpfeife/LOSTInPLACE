#######################################
# 01_download_egms

#============================================================#
# Import Statements
#============================================================#

import geopandas as gpd
import json
import time
import requests
import jwt
import os
import numpy as np
import pandas as pd

from pathlib import Path
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely import union_all
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


#============================================================#
# Define Global Variables
#============================================================#

PROJECT_DIR = Path("/dss/dsstbyfs02/scratch/0C/di54haf/LOSTInPLACE")
DATA_DIR = PROJECT_DIR / "data"
# For the data split between training, dev and validation regions
TRAIN_DIR = DATA_DIR / "train"
DEV_DIR = DATA_DIR / "dev"
CAL_DIR = DATA_DIR / "cal"
VAL_DIR = DATA_DIR / "val"

#============================================================#
# Helper Functions
#============================================================#

def create_file_system():
    """
    Creates the project's directory tree (data, train, dev, val) under PROJECT_DIR if it doesn't exist yet.

    Arguments:
        None -- uses the global PROJECT_DIR / DATA_DIR / TRAIN_DIR / DEV_DIR / VAL_DIR paths

    Returns:
        None -- creates the folders on disk as a side effect
    """
    # setting up the file system for the project
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)


def get_access_token(token_path):
    """
    Requests a short-lived OAuth access token for the EGMS API from a service-account key file.

    Arguments:
        token_path -- path (str or Path) to the service-account JSON key file

    Returns:
        access_token -- string, the bearer access token used to authenticate EGMS API requests
    """
    # This is from the egms api repo https://github.com/copernicus-land/egms-api/blob/main/EGMS-API.ipynb
    service_key = json.load(open(token_path, 'rb'))
    private_key = service_key['private_key'].encode('utf-8')
    claim_set = {
        "iss": service_key['client_id'],
        "sub": service_key['user_id'],
        "aud": service_key['token_uri'],
        "iat": int(time.time()),
        "exp": int(time.time() + (60 * 60)),
    }
    grant = jwt.encode(claim_set, private_key, algorithm='RS256')
    result = requests.post(service_key["token_uri"], headers={ "Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded" },
            data={ "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": grant } )
    access_token_info_json = result.json()
    access_token = access_token_info_json.get('access_token')
    return access_token

def point_to_bbox(row, t2projected, t2wgs84, source_crs):
    """
    Builds a lon/lat bounding box around a point by offsetting it west/east/north/south by the row's margins.

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with a Point geometry and numeric west/east/north/south margins (km)
        t2projected -- pyproj.Transformer, transforms the point's CRS to a metric CRS
        t2wgs84 -- pyproj.Transformer, transforms that metric CRS to EPSG:4326

    Returns:
        bbox -- list of [lon, lat] pairs, the four corners in order top-left, bottom-left, bottom-right, top-right
    """
    
    if source_crs == "EPSG:3035":
        x, y = row.geometry.x, row.geometry.y      # already projected, no transform needed
    else:
        x, y = t2projected.transform(row.geometry.x, row.geometry.y)   # e.g. EPSG:4326 -> EPSG:3035

    # compute the min/max values
    top_left_x = x - (row["west"] * 1000)
    top_left_y = y + (row["north"] * 1000)

    top_right_x = x + (row["east"] * 1000)
    top_right_y = y + (row["north"] * 1000)

    bottom_left_x = x - (row["west"] * 1000)
    bottom_left_y = y - (row["south"] * 1000)

    bottom_right_x = x + (row["east"] * 1000)
    bottom_right_y = y - (row["south"] * 1000)

    tl_lon, tl_lat = t2wgs84.transform(top_left_x, top_left_y)
    tr_lon, tr_lat = t2wgs84.transform(top_right_x, top_right_y)
    bl_lon, bl_lat = t2wgs84.transform(bottom_left_x, bottom_left_y)
    br_lon, br_lat = t2wgs84.transform(bottom_right_x, bottom_right_y)

    return [[tl_lon, tl_lat], [bl_lon, bl_lat], [br_lon, br_lat], [tr_lon, tr_lat]]

def query_egms(row, headers, api_endpoint = "https://egms.land.copernicus.eu/insar-api/archive"):
    """
    Searches the EGMS archive for products overlapping a row's bbox and builds the download link for each hit.

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with a bbox field (list of [lon, lat] corners)
        headers -- dict, HTTP headers including the bearer Authorization token
        api_endpoint -- string, base URL of the EGMS API

    Returns:
        result -- dict, the raw JSON search result returned by the API
        links -- list of strings, one download link per hit in result["hits"]
    """
    query = {"id": None,
         "bbox": row["bbox"],
         "levels" : ["L2B"],
         "releases" : ["2019-2023"]
         }

    r = requests.post(f"{api_endpoint}/search", headers=headers, data=json.dumps(query))
    result = r.json()

    # Constructing a download link for all the products in the result
    # code from the egms api repo
    links = []
    for hit in result["hits"]:
        link = f"{api_endpoint}/download/{hit['filename']}?id={result['id']}"
        links.append(link)

    return result, links
    

def pull_egms(row, headers, orbit):
    """
    Picks the relative orbit (of a given direction) whose bursts overlap the study area most, then downloads its files.

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with bbox, query_results, download_links, location and set
        headers -- dict, HTTP headers including the bearer Authorization token
        orbit -- string, orbit direction to keep, e.g. "ascending" or "descending"

    Returns:
        None -- writes the downloaded files to disk under TRAIN_DIR, DEV_DIR or VAL_DIR
    """
    # build a Polygon to compare with to filter out the best relative orbit.
    study_area = Polygon(row["bbox"])

    # keep only hits/links from the requested orbit direction
    pairs = [(hit, link) for hit, link in zip(row["query_results"]["hits"], row["download_links"])
             if hit["direction"] == orbit]

    # group by relative orbit, aggregating all bursts of that relative orbit
    by_rel_orbit = {}
    for hit, link in pairs:
        by_rel_orbit.setdefault(hit["relativeOrbit"], []).append((hit, link))

    # keep only the relative orbit whose combined bursts overlap the study area most
    best_rel_orbit = max(
        by_rel_orbit,
        key=lambda ro: union_all([Polygon(hit["poly"]) for hit, _ in by_rel_orbit[ro]]).intersection(study_area).area
    )
    hits, links = zip(*by_rel_orbit[best_rel_orbit])

    # some of the real subsidence areas go into training but some also go into validation 
    # the set variable decides into which region the data goes
    # Generally:  0 = train    1 = dev   2 = cal   3 = val
    if row["set"] == 0:
        return_path = TRAIN_DIR / row["location"]
        save_path = return_path / str(orbit)
    elif row["set"] == 1:
        return_path = DEV_DIR / row["location"]
        save_path = return_path  / str(orbit)
    elif row["set"] == 2:
        return_path = CAL_DIR / row["location"]
        save_path = return_path / str(orbit)
    elif row["set"] == 3:
            return_path = VAL_DIR / row["location"]
            save_path = return_path / str(orbit)
    else:
        raise ValueError("Not all rows have regions assigned!")

    # creating the directory where the data goes
    save_path.mkdir(parents=True, exist_ok=True)

    for hit,link in zip(hits, links):
         filename = hit["filename"]
         with requests.get(link, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(save_path / filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

    return str(return_path)


def sample_random_points(polygon, n_points, min_dist, max_attempts, crs=None):
    """
    Samples n_points random points inside a polygon via rejection sampling, so accepted points stay at least min_dist apart.

    Arguments:
        polygon -- shapely geometry, area to sample within (already buffered/clipped, e.g. coverage minus real_subsidence buffers)
        n_points -- int, target number of points to return
        min_dist -- float, minimum distance required between any two sampled points, in the units of crs
        max_attempts -- int, upper bound on the number of sampling attempts
        crs -- CRS to assign to the returned GeoDataFrame, since polygon itself carries no CRS

    Returns:
        points -- geopandas.GeoDataFrame, one row per accepted point, with a geometry column
    """

    random_generator = np.random.default_rng(6576)

    attempts = 0
    n = 0
    poly = gpd.GeoDataFrame(geometry=[polygon])
    points = gpd.GeoSeries([polygon]).sample_points(1, rng=random_generator)

    while attempts < max_attempts-1 and n < n_points-1:
        too_close_area = points.buffer(min_dist).union_all()
        poly.geometry = poly.difference(too_close_area)
        new_point = poly.sample_points(1, rng=random_generator)
        points = pd.concat([points, new_point], ignore_index=True)

        attempts += 1
        n += 1

    return gpd.GeoDataFrame(geometry=points, crs=crs)


def closest_city_name(row, reverser):
    """
    Returns the name of the nearest city/town to a GeoDataFrame row's point, via reverse geocoding.

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with a Point geometry in EPSG:4326 (lon/lat)
        reverser -- callable, a geopy reverse-geocoding function (e.g. a RateLimiter-wrapped Nominatim.reverse)

    Returns:
        name -- string or None, the city/town/village name from OpenStreetMap, or None if nothing was found
    """
    location = reverser((row.geometry.y, row.geometry.x), exactly_one=True)
    address = location.raw.get("address", {}) if location else {}
    return address.get("city") or address.get("town") or address.get("village")

def pull_data_by_url(url, out_dir, file):
    # Downloading the station list or other resources from NGL
    with requests.get(url, stream=True) as r:
            r.raise_for_status()
            file_path = out_dir / file
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
    return str(file_path)

def stations_in_bbox(row, stations):
    # create a polygon from the bbox col
    bbox = Polygon(row["bbox"])
    # .within returns boolean, so selection on stations first
    # then return them as list to be added as col in the gdf
    station_codes_within = stations.loc[stations.within(bbox), "Sta"].tolist()

    return station_codes_within

def pull_stations_in_bbox_data(row):
    # base url for the Eurasia fixed time series
    base_url = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/EU/"    #<ssss>.<plate>.tenv3
    # empty list to catch the paths to the gnss files
    ts_file_paths = []
    # loop over all stations that were found within that region
    for station_code in row["station_codes"]:
       # create individual url for every station
       url = f"{base_url}{station_code}.EU.tenv3"
       # reuse the download function
       ts_file_path = pull_data_by_url(url, Path(row["path"]), f"{str(station_code)}_gnss_time_series.tenv3")
       # catch the file paths to fill a new dataframe column
       ts_file_paths.append(ts_file_path)

    return ts_file_paths

def study_areas_io(mode, gdf=None, base_path=None):
    """
    Splits a GeoDataFrame into a gpkg (geometry + scalar columns) and a JSON file (the list/dict
    columns gpkg can't store), or reconstructs it from both files. Rows are matched by a dedicated
    "uid" column generated fresh on every write, rather than by index or "location" (which isn't
    guaranteed unique), so the two files can never get mismatched against each other.

    Arguments:
        mode -- string, "write" to split and save gdf, "read" to load and recombine
        gdf -- geopandas.GeoDataFrame, required when mode is "write"; the frame to split and save
        base_path -- path (str or Path) without extension, the shared filename stem for both files

    Returns:
        result -- on "write": None, writes base_path.gpkg and base_path_complex.json to disk.
                  on "read": geopandas.GeoDataFrame, the recombined frame
    """
    base_path = Path(base_path)
    gpkg_path = base_path.with_suffix(".gpkg")
    json_path = base_path.parent / f"{base_path.stem}_complex.json"
    complex_cols = ["bbox", "query_results", "download_links", "station_codes", "station_file_paths", "gse_tiles_by_year"]

    if mode == "write":
        gdf = gdf.copy()
        gdf["uid"] = range(len(gdf))

        existing_complex_cols = [c for c in complex_cols if c in gdf.columns]
        gdf.drop(columns=existing_complex_cols).to_file(gpkg_path, driver="GPKG")
        gdf[["uid"] + existing_complex_cols].to_json(json_path, orient="records")

    elif mode == "read":
        simple_gdf = gpd.read_file(gpkg_path)
        complex_df = pd.read_json(json_path, orient="records")
        return simple_gdf.merge(complex_df, on="uid", how="left")

    else:
        raise ValueError(f"mode must be 'write' or 'read', got {mode!r}")

def load_gse_index(index_url):
    """
    Loads the AlphaEarth tile index into a GeoDataFrame, geometry from its WKT column.

    Arguments:
        index_url_or_path -- the local path (or, if you download it once, a URL) to aef_index.csv

    Returns:
        index_gdf -- geopandas.GeoDataFrame, one row per tile, with year/utm_zone/path/geometry
    """
    # download the index file
    index_file_path = pull_data_by_url(index_url, Path(__file__).parent / "../../assets/", "gse_index_file.csv"  )
    # read the csv file and build geodata from wkt column
    index_file_df = pd.read_csv(index_file_path, sep=",")
    index_gdf = gpd.GeoDataFrame(index_file_df, geometry=gpd.GeoSeries.from_wkt(index_file_df["WKT"]), crs="EPSG:4326")

    return index_gdf

def tiles_for_row(row, index_gdf, years):
    """
    Finds which AlphaEarth tiles intersect a study-area row's bbox, for a set of years.

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with a bbox field (list of [lon, lat] corners)
        index_gdf -- geopandas.GeoDataFrame, the loaded tile index (see load_gse_index)
        years -- iterable of ints, which years to keep (e.g. range(2019, 2024) for the 5-year EGMS window)

    Returns:
        matches -- geopandas.GeoDataFrame, the index rows (tiles) that intersect the bbox, for those years
    """

    # First we build a Polygon from the bbox column
    bbox = Polygon(row["bbox"])
    # Filtering the index gdf based on the wanted years first because that is cheap operation
    index_gdf = index_gdf[index_gdf["year"].isin(years)]
    # Now find all tiles that intersect with the bounding box
    tiles = index_gdf[index_gdf.intersects(bbox)]

    # Returning a dictionary to be able to add the years to the file names to make them distinguishable
    tiles_by_year = {}
    for year, group in tiles.drop(columns="geometry").groupby("year"):
        tiles_by_year[year] = group.to_dict("records")

    return tiles_by_year

def gs_uri_to_https(gs_uri):
    """
    Converts a gs://bucket/key URI to its public HTTPS equivalent for plain download.

    Arguments:
        gs_uri -- string, e.g. "gs://alphaearth_foundations/satellite_embedding/v1/annual/2019/31N/x....tiff"

    Returns:
        https_url -- string, e.g. "https://storage.googleapis.com/alphaearth_foundations/..."
    """

    without_prefix = gs_uri.removeprefix("gs://")
    bucket, key = without_prefix.split("/", 1)
    return f"https://storage.googleapis.com/{bucket}/{key}"

def pull_gse_data(row):
    """
    Downloads every AlphaEarth tile covering a study-area row, for the given years, into its folder.

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with bbox, path and location
        index_gdf -- geopandas.GeoDataFrame, the loaded tile index
        years -- iterable of ints, which years to download

    Returns:
        tile_paths -- list of strings, local file paths of everything downloaded for this row
    """

    out_dir = Path(row["path"])
    for year, tiles in row["gse_tiles_by_year"].items():
        for tile in tiles:
            url = gs_uri_to_https(tile["path"])
            filename = f"{year}_{Path(tile['path']).name}"   # prefixes the year, since the tile name alone doesn't carry it
            pull_data_by_url(url, out_dir, filename)


#============================================================#
# Processing
#============================================================#

def download_egms(real_sub_path, token_path, egms_coverage, n_rnd, mindist_rnd, maxit_rnd, real_sub_buffer_value = 400000):
    """
    Downloads EGMS L2b data for the real subsidence regions and for extra randomly sampled regions, for both orbit directions.

    Arguments:
        real_sub_path -- path (str or Path) to the real_subsidence_pointbased.gpkg file
        token_path -- path (str or Path) to the EGMS API service-account key file
        egms_coverage -- path (str or Path) to the geojson of the EGMS coverage area
        real_sub_buffer_value -- float, buffer radius (metres) around each real subsidence point excluded from random sampling
        n_rnd -- int, number of extra random points to sample across the EGMS coverage area
        mindist_rnd -- float, minimum distance (metres) required between the random points
        maxit_rnd -- int, upper bound on sampling attempts for the random points

    Returns:
        None -- downloads files under TRAIN_DIR/DEV_DIR/VAL_DIR
    """

    ################################################
    #=== Section 1: The Real Subsidence Regions ===#
    ################################################

    # This function handles the download of the real subsidence regions
    # Some of them go into training, some into validation (handled by the set column in the real_subsidence.gpkg)

    # Loading the real subsidence cases
    real_subsidence = gpd.read_file(real_sub_path)

    # Compute the bounding boxes for every real subsidence region
    t2projected = Transformer.from_crs(real_subsidence.crs, "EPSG:3035", always_xy=True)
    t2wgs84 = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    real_subsidence["bbox"] = real_subsidence.apply(point_to_bbox, axis=1, result_type="reduce", t2projected=t2projected, t2wgs84=t2wgs84, source_crs=real_subsidence.crs)

    # get the access token for the egms api
    access_token = get_access_token(token_path)
    headers = {"Authorization" : f"Bearer {access_token}", "Accept" : "application/json"}

    # query the archive for overlapping bursts
    real_subsidence[["query_results", "download_links"]] = real_subsidence.apply(query_egms, axis=1, headers=headers, result_type = "expand")

    # Download the products with filter before
    # the set variable decides into which region the data goes
    # Generally:  0 = train    1 = dev   2 = cal   3 = val
    real_subsidence["path"] = real_subsidence.apply(pull_egms, axis = 1, headers=headers, orbit = "ascending")
    real_subsidence["path"] = real_subsidence.apply(pull_egms, axis = 1, headers=headers, orbit = "descending")

    ################################################
    #=== Section 2: The Normal Training Regions ===#
    ################################################

    # This section implements the random sampling across Europe to enrich the training dataset.
    # Loading the geojson containing the entire egms coverage area. Polygon to constrain the sampling
    coverage_egms = gpd.read_file(egms_coverage).to_crs("EPSG:3035").union_all()
    # Buffer the real_subsidence points because in their vicinity shall not be sampled
    real_sub_buffer = real_subsidence.to_crs("EPSG:3035").buffer(real_sub_buffer_value).union_all()
    # egms coverage area without the already represented areas through real subsidence areas
    allowed_area = coverage_egms.difference(real_sub_buffer)

    egms_random = sample_random_points(allowed_area, n_rnd, mindist_rnd, maxit_rnd, crs = "EPSG:3035")

    # adding the same columns to the dataframe that the initial real_subsidence df has 
    egms_random["north"] = 25
    egms_random["east"] = 25
    egms_random["south"] = 25
    egms_random["west"] = 25
    egms_random["set"] = np.random.default_rng(42).choice([0, 1, 2], size=len(egms_random), p=[0.6, 0.2, 0.2]) 

    # setting up the geopy functionality to determine the closest city
    geolocator = Nominatim(user_agent="lostinplace_thesis")
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)  # Nominatim's usage policy caps at 1 req/s
    # filling the location col via this function 
    egms_random["location"] = egms_random.apply(closest_city_name, axis = 1, reverser = reverse)

    # Creating the bbox and querying the archive on the random point samples
    egms_random["bbox"] = egms_random.apply(point_to_bbox, axis=1, result_type="reduce", t2projected=t2projected, t2wgs84=t2wgs84, source_crs=egms_random.crs)
    egms_random[["query_results", "download_links"]] = egms_random.apply(query_egms, axis=1, headers=headers, result_type = "expand")
    
    # downloading the egms data for the new areas
    egms_random["path"] = egms_random.apply(pull_egms, axis = 1, headers=headers, orbit = "ascending")
    egms_random["path"] = egms_random.apply(pull_egms, axis = 1, headers=headers, orbit = "descending")

    # Now concattenating both gdfs. This gdf is representative for the entire study area.
    # First making sure they are both in the same crs
    real_subsidence = real_subsidence.to_crs("EPSG:3035")
    egms_random = egms_random.to_crs("EPSG:3035")

    study_areas_gdf = pd.concat([real_subsidence, egms_random])

    study_areas_io("write", study_areas_gdf, Path(__file__).parent / "../../assets/study_areas_gdf")
    
    return study_areas_gdf


def download_gnss(study_areas_gdf):
    """
    Downloads GNSS station time series for the validation areas. Not implemented yet.

    Arguments:
        None

    Returns:
        None
    """
    # Creating a col for the station codes. Only the val regions shall receive a valid entry
    # For the train and dev regions this stays None
    study_areas_gdf["station_codes"] = None

    # pull the list of all stations
    gnss_station_list_path = pull_data_by_url("https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt", 
                            out_dir=Path(__file__).parent / "../../assets/",
                            file="gnss_station_list.txt")

    # Reading the whole station list from the path returned by pull_data_by_url above
    station_list = pd.read_csv(gnss_station_list_path, sep=r"\s+", parse_dates=["Dtbeg", "Dtend", "Dtmod"], usecols=range(11))
    # Adjust the longitude from 360 to -180/+180 convention
    station_list["Long(deg)"] = station_list["Long(deg)"].mask(station_list["Long(deg)"] > 180, station_list["Long(deg)"] - 360)
    # creating a geometry column
    station_list["geometry"] = gpd.points_from_xy(station_list["Long(deg)"], station_list["Lat(deg)"])
    # create a gdf from those points for further analysis
    stations = gpd.GeoDataFrame(station_list, geometry= "geometry" , crs="EPSG:4326")

    # Creating a mask to apply station_codes_in_bbox only to the validation regions
    val_mask = study_areas_gdf["set"] == 3
    # apply the function and write the result back into the val row, leaving None for train/dev
    study_areas_gdf.loc[val_mask, "station_codes"] = study_areas_gdf.loc[val_mask].apply(
        stations_in_bbox, axis=1, stations=stations)

    ##########################################################
    # Now downloading the actual time series

    # Again only compute on the validation areas
    study_areas_gdf["station_file_paths"] = None
    study_areas_gdf.loc[val_mask, "station_file_paths"] = study_areas_gdf.loc[val_mask].apply(
            pull_stations_in_bbox_data, axis=1)

    # Saving the updated geodataframe
    study_areas_io("write", study_areas_gdf, Path(__file__).parent / "../../assets/study_areas_gdf")
        
    return study_areas_gdf

def download_gse(study_areas_gdf, years):
    """
    Downloads Google satellite embeddings covering the train/dev/val areas. Not implemented yet.

    Arguments:
        None

    Returns:
        None
    """

    # First download the index dataframe that contains information about the gse tiles and years
    # index_gdf is the raw index_gdf only filtered by the years that are relevant here
    index_gdf = load_gse_index("https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual/aef_index.csv")

    # This adds the tiles per year as a new column to the gdf
    study_areas_gdf["gse_tiles_by_year"] = study_areas_gdf.apply(tiles_for_row, axis=1, index_gdf=index_gdf, years=years)

    # Downloading the tiles into the location folder with year prefixes in filenames
    study_areas_gdf.apply(pull_gse_data, axis=1)

    study_areas_io("write", study_areas_gdf, Path(__file__).parent / "../../assets/study_areas_gdf")

    return study_areas_gdf

def download_esa_worldcover():
    """
    Downloads miscellaneous auxiliary data, e.g. country boundaries. Not implemented yet.

    Arguments:
        None

    Returns:
        None
    """
    pass


#============================================================#
# Main
#============================================================#

def main():
    """
    Checks the required input files exist, sets up the project's file system, and runs the EGMS download.

    Arguments:
        None

    Returns:
        None -- raises FileNotFoundError if the real subsidence file is missing
    """

    ##############################
    # Check necessary files are there

    if not os.path.exists(Path(__file__).parent / "../../assets/real_subsidence_pointbased.gpkg"):
        raise FileNotFoundError("The real subsidence regions are missing!")
    
    ##############################
    # Creating the file system for the project 
    create_file_system()

    ##############################
    # Downloading the data
    study_areas_gdf = download_egms(Path(__file__).parent / "../../assets/real_subsidence_pointbased.gpkg", 
                        Path(__file__).parent / "../../assets/token.jwt",
                        Path(__file__).parent / "../../assets/egms_coverage_countries.geojson",
                        real_sub_buffer_value = 150000,
                        n_rnd=35, mindist_rnd=150000, maxit_rnd=200)
    
    # Use the dataframe of the regions for which egms was downloaded to select and download
    # the GNSS stations and time series.
    study_areas_gdf = download_gnss(study_areas_gdf)

    # download google satellite embeddings for every region
    study_areas_gdf = download_gse(study_areas_gdf, [2019, 2020, 2021, 2022, 2023])


if __name__ == '__main__':
    main() 
