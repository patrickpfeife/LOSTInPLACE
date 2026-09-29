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
    # Generally:  0 = train    1 = dev   2 = val
    if row["set"] == 0:
        save_path = TRAIN_DIR / row["location"] / str(orbit)
    elif row["set"] == 1:
        save_path = DEV_DIR / row["location"] / str(orbit)
    elif row["set"] == 2:
        save_path = VAL_DIR / row["location"] / str(orbit)
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
    # Generally:  0 = train    1 = dev   2 = val
    real_subsidence.apply(pull_egms, axis = 1, headers=headers, orbit = "ascending")
    real_subsidence.apply(pull_egms, axis = 1, headers=headers, orbit = "descending")

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
    egms_random["set"] = np.random.default_rng(42).choice([0, 1], size=len(egms_random), p=[0.8, 0.2]) 

    # setting up the geopy functionality to determine the closest city
    geolocator = Nominatim(user_agent="lostinplace_thesis")
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)  # Nominatim's usage policy caps at 1 req/s
    # filling the location col via this function 
    egms_random["location"] = egms_random.apply(closest_city_name, axis = 1, reverser = reverse)

    # Creating the bbox and querying the archive on the random point samples
    egms_random["bbox"] = egms_random.apply(point_to_bbox, axis=1, result_type="reduce", t2projected=t2projected, t2wgs84=t2wgs84, source_crs=egms_random.crs)
    egms_random[["query_results", "download_links"]] = egms_random.apply(query_egms, axis=1, headers=headers, result_type = "expand")
    
    # downloading the egms data for the new areas
    egms_random.apply(pull_egms, axis = 1, headers=headers, orbit = "ascending")
    egms_random.apply(pull_egms, axis = 1, headers=headers, orbit = "descending")

    # Now concattenating both gdfs. This gdf is representative for the entire study area.
    # First making sure they are both in the same crs
    real_subsidence = real_subsidence.to_crs("EPSG:3035")
    egms_random = egms_random.to_crs("EPSG:3035")

    study_areas_gdf = pd.concat([real_subsidence, egms_random])
    study_areas_gdf.to_file(Path(__file__).parent / "../../assets/study_areas_gdf.gpkg", driver="gpkg")

    return study_areas_gdf


    # current state:
    # for all real subsidence areas, egms data was downloaded but not unpacked yet.
    # Some of them went into train and some into val as specified in the set column of the df.
    # 1) Generate random points with minimum distance to the center point of the real subsidence areas.
    # 2) Download egms for those regions as well. 

    # Next steps:
    # 3) run the unzip and and cover area functions on the complete egms downloads(in 02_preprocessing)
    # 4) Download GNSS for all validation areas (the point_to_bbox will be used herefor)
    # 5) ensure that enough GNSS stations are there and time series are of good quality. (checked)
    # 6) Download google satellite embeddings covering all the areas in train, dev and val

    # !!! for the GNSS sites: ascending and descending must overlap!!!
    # this should be ensured: all overlapping products are downloaded and in the preprocessing
    # cropped to the areas. So every GNSS site should be covered by two orbits.

    



def download_gnss():
    """
    Downloads GNSS station time series for the validation areas. Not implemented yet.

    Arguments:
        None

    Returns:
        None
    """
    pass

def download_gse():
    """
    Downloads Google satellite embeddings covering the train/dev/val areas. Not implemented yet.

    Arguments:
        None

    Returns:
        None
    """
    pass

def download_miscellaneous():
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
                        real_sub_buffer_value = 400000,
                        n_rnd=25, mindist_rnd=200000, maxit_rnd=200)
    ##############################
    # Pre-Processing the data


if __name__ == '__main__':
    main() 
