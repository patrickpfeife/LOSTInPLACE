#######################################
# 01_download_egms

##############################################################################
#============================================================================#
# 1) Import Statements
#============================================================================#
##############################################################################

import geopandas as gpd
import json
import time
import requests
import jwt
import os
import numpy as np
import pandas as pd
import polars as pl
import zipfile
import rasterio
import argparse

from pathlib import Path
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely import union_all
from shapely.geometry import box
from rasterio.mask import mask as rio_mask

##############################################################################
#============================================================================#
# 2) Helper Functions
#============================================================================#
##############################################################################

################################################
#===== Section 1: Shared Helper Functions =====#
################################################
# Used across more than one processing step below, not specific to a single one.

def create_file_system():
    """
    Creates the project's directory tree (data, train, dev, cal, val) under PROJECT_DIR if it doesn't exist yet.

    Arguments:
        None -- uses the global PROJECT_DIR / DATA_DIR / TRAIN_DIR / DEV_DIR / CAL_DIR / VAL_DIR paths

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
    # building the paths of the two files from the shared filename stem
    base_path = Path(base_path)
    gpkg_path = base_path.with_suffix(".gpkg")
    json_path = base_path.parent / f"{base_path.stem}_complex.json"

    # the columns holding lists or dicts. gpkg can't store those, so they go into the json
    complex_cols = ["bbox", "station_codes", "station_file_paths", 
                    "gse_tiles_by_year", "gse_crop_paths", "large_bbox", "world_cover_tile_urls"]

    if mode == "write":
        # work on a copy so the uid column doesn't end up in the gdf of the caller
        gdf = gdf.copy()
        # fresh uid per row, to match the rows of both files again when reading
        gdf["uid"] = range(len(gdf))

        # only the complex columns that exist already, since they get added step by step
        existing_complex_cols = [c for c in complex_cols if c in gdf.columns]
        # everything but the complex columns goes into the gpkg
        gdf.drop(columns=existing_complex_cols).to_file(gpkg_path, driver="GPKG")
        # the uid and the complex columns go into the json
        gdf[["uid"] + existing_complex_cols].to_json(json_path, orient="records")

    elif mode == "read":
        # read both files and join them again via the uid column
        simple_gdf = gpd.read_file(gpkg_path)
        complex_df = pd.read_json(json_path, orient="records")
        return simple_gdf.merge(complex_df, on="uid", how="left")

    else:
        raise ValueError(f"mode must be 'write' or 'read', got {mode!r}")

def requests_get(url, part_file_path, file_path, headers=None):
    """
    Streams a URL's response body to part_file_path, then renames it to file_path once the
    download finished without error (so a crash mid-download leaves only the .part file behind).
    If the server answers with an error status, a requests.exceptions.HTTPError is raised and
    nothing is written. For downloads with headers (EGMS) the server's own message is printed first.

    Arguments:
        url -- string, the URL to download
        part_file_path -- Path, temporary path written to while streaming
        file_path -- Path, final path part_file_path is renamed to on success
        headers -- dict or None, request headers (e.g. EGMS bearer token); None for plain downloads

    Returns:
        None -- writes to disk and renames part_file_path to file_path as a side effect
    """
    if headers == None:
        # Downloading the station list or other resources from NGL
            with requests.get(url, stream=True) as r:
                    # stop here if the server answered with an error status
                    r.raise_for_status()
                    # write the response to the temporary file in chunks of 1 MB
                    with open(part_file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            f.write(chunk)
                    
            # download finished without error, so the temporary file gets its final name
            part_file_path.rename(file_path)
    else:
        # Downloading from EGMS, the headers carry the access token
        with requests.get(url, headers=headers, stream=True) as r:
            # raise_for_status only reports the status code, so print the server's own message first
            if not r.ok:
                print(f"Download failed [{r.status_code}] {url}\n    server response: {r.text[:500]}", flush=True)
            r.raise_for_status()
            # write the response to the temporary file in chunks of 1 MB
            with open(part_file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

        # download finished without error, so the temporary file gets its final name
        part_file_path.rename(file_path)


def pull_data_by_url(url, out_dir, file):
    """
    Streams a URL's response body to a file, used for any plain-HTTP download (station list,
    GNSS time series, AlphaEarth tiles, ...).

    Arguments:
        url -- string, the URL to download
        out_dir -- Path, the directory to save into
        file -- string, the filename to save as

    Returns:
        file_path -- string, the full path the file was saved to (out_dir / file)
    """
    # a temporary filename and a final filename, to tell a stale download from a complete one
    file_path = out_dir / file
    part_file_path = out_dir / f"PART_{file}"

    # checking three scenarios: 1) only stale file exists
    if part_file_path.exists():
        part_file_path.unlink()
        requests_get(url, part_file_path, file_path)
    # 2) file does not exist at all (has never been downloaded before)
    elif not file_path.exists():
        requests_get(url, part_file_path, file_path)
    # 3) complete file exists
    else: 
        pass

    return str(file_path)

################################################
#====== Section 2: EGMS Download Helpers ======#
################################################
# Used by download_egms, in the order they're first called there.

def point_to_bbox(row, t2projected, t2wgs84, source_crs):
    """
    Builds a lon/lat bounding box around a point by offsetting it west/east/north/south by the row's margins.

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with a Point geometry and numeric west/east/north/south margins (km)
        t2projected -- pyproj.Transformer, transforms the point's CRS to a metric CRS
        t2wgs84 -- pyproj.Transformer, transforms that metric CRS to EPSG:4326
        source_crs -- the row's actual CRS; if "EPSG:3035" the point is used as-is, otherwise t2projected is applied

    Returns:
        bbox -- list of [lon, lat] pairs, the four corners in order top-left, bottom-left, bottom-right, top-right
    """

    # the point is needed in a metric crs, to add the margins in metres
    if source_crs == "EPSG:3035":
        x, y = row.geometry.x, row.geometry.y      # already projected, no transform needed
    else:
        x, y = t2projected.transform(row.geometry.x, row.geometry.y)   # e.g. EPSG:4326 -> EPSG:3035

    # the margins are given in km, so convert them to metres
    west = row["west"] * 1000
    north = row["north"] * 1000
    east = row["east"] * 1000
    south = row["south"] * 1000

    # compute the min/max values
    top_left_x = x - west
    top_left_y = y + north

    top_right_x = x + east
    top_right_y = y + north

    bottom_left_x = x - west
    bottom_left_y = y - south

    bottom_right_x = x + east
    bottom_right_y = y - south

    # transform the four corners back to lon/lat
    tl_lon, tl_lat = t2wgs84.transform(top_left_x, top_left_y)
    tr_lon, tr_lat = t2wgs84.transform(top_right_x, top_right_y)
    bl_lon, bl_lat = t2wgs84.transform(bottom_left_x, bottom_left_y)
    br_lon, br_lat = t2wgs84.transform(bottom_right_x, bottom_right_y)

    return [[tl_lon, tl_lat], [bl_lon, bl_lat], [br_lon, br_lat], [tr_lon, tr_lat]]

def get_access_token(token_path):
    """
    Requests a short-lived OAuth access token for the EGMS API from a service-account key file.
    Raises a RuntimeError with the server's answer if no token comes back.

    Arguments:
        token_path -- path (str or Path) to the service-account JSON key file

    Returns:
        access_token -- string, the bearer access token used to authenticate EGMS API requests
    """
    # This is from the egms api repo https://github.com/copernicus-land/egms-api/blob/main/EGMS-API.ipynb
    # reading the service-account key file
    service_key = json.load(open(token_path, 'rb'))
    private_key = service_key['private_key'].encode('utf-8')
    # the claim set says who asks for the token and that it shall be valid for one hour
    claim_set = {
        "iss": service_key['client_id'],
        "sub": service_key['user_id'],
        "aud": service_key['token_uri'],
        "iat": int(time.time()),
        "exp": int(time.time() + (60 * 60)),
    }
    # sign the claim set with the private key
    grant = jwt.encode(claim_set, private_key, algorithm='RS256')
    # exchange the signed grant for the actual access token
    result = requests.post(service_key["token_uri"], headers={ "Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded" },
            data={ "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": grant } )
    access_token_info_json = result.json()
    access_token = access_token_info_json.get('access_token')
    
    # In case the token generation fails, this will deliver a meaningful error message
    if not access_token:
        raise RuntimeError(f"EGMS token request failed: {access_token_info_json}")
    
    return access_token

def query_egms(row, headers, api_endpoint = "https://egms.land.copernicus.eu/insar-api/archive"):
    """
    Searches the EGMS archive for products overlapping a row's bbox and builds the download link for each hit.
    Raises a RuntimeError if the search fails or is rejected (e.g. because of an invalid token).

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with a bbox field (list of [lon, lat] corners)
        headers -- dict, request headers carrying the bearer access token (see get_access_token)
        api_endpoint -- string, base URL of the EGMS API

    Returns:
        result -- dict, the raw JSON search result returned by the API
        links -- list of strings, one download link per hit in result["hits"]
    """
    
    # the search query: all L2B products of the 2019-2023 release that overlap the bbox
    query = {"id": None,
         "bbox": row["bbox"],
         "levels" : ["L2B"],
         "releases" : ["2019-2023"]
         }

    # send the query to the archive
    r = requests.post(f"{api_endpoint}/search", headers=headers, data=json.dumps(query))
    # stop here if the server answered with an error status
    # otherwise this only shows up later as a confusing KeyError
    if not r.ok:
        raise RuntimeError(f"EGMS search failed [{r.status_code}] for {row['location']}: {r.text[:500]}")

    result = r.json()

    # a search without a valid token still answers with 200, but with status False and no id
    # stop here as well, because no download works without the id
    if not result.get("status") or result.get("id") is None:
        raise RuntimeError(f"EGMS search was rejected for {row['location']}: {str(result)[:500]}")

    # Constructing a download link for all the products in the result
    # code from the egms api repo
    links = []
    for hit in result["hits"]:
        link = f"{api_endpoint}/download/{hit['filename']}?id={result['id']}"
        links.append(link)

    return result, links


def pull_egms(row, token_path, orbit, failed_downloads):
    """
    Picks the relative orbit (of a given direction) whose bursts overlap the study area most, then downloads its files.
    Files that are already on disk are skipped. Files the server refuses are recorded in
    failed_downloads and skipped as well, so a single bad file doesn't stop the whole run.

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with bbox, location and set
        token_path -- path (str or Path) to the EGMS API service-account key file, used to fetch
            a fresh access token and search id once they are close to expiring (both expire after
            1 hour, and a single row's files can take longer than that to all download)
        orbit -- string, orbit direction to keep, e.g. "ascending" or "descending"
        failed_downloads -- list, every file that could not be downloaded is appended to it as a
            dict (location, orbit, filename, status_code, server_response)

    Returns:
        return_path -- string, the location-level folder the files were saved under (TRAIN_DIR, DEV_DIR, CAL_DIR or VAL_DIR / location)
            Also returned if there are no products for this orbit direction
    """

    # some of the real subsidence areas go into training but some also go into validation
    # the set variable decides into which region the data goes
    # Generally:  0 = train    1 = dev   2 = cal   3 = val
    # This comes first so return_path is known even if nothing is downloaded for this orbit
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
    
    # Create an access token and note when it was generated (expires after one hour)
    access_token = get_access_token(token_path)
    token_issued_at = time.time()
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

    # Call query egms once per line since the ids also expire
    # Calling it once upfront on the entire df leads to expired ids once the download is actually performed
    result, links = query_egms(row=row, headers=headers)

    # the study area as polygon, to measure the overlap with the bursts below
    study_area = Polygon(row["bbox"])
    # keep only the products of the wanted orbit direction, together with their links
    pairs = [(hit, link) for hit, link in zip(result["hits"], links)
                 if hit["direction"] == orbit]

    # group by relative orbit, aggregating all bursts of that relative orbit
    by_rel_orbit = {}
    for hit, link in pairs:
        by_rel_orbit.setdefault(hit["relativeOrbit"], []).append((hit, link))

    # Nothing to download for this orbit direction. Still return the path because download_egms
    # assigns the path column once per orbit direction
    if not by_rel_orbit:
        print(f"No {orbit} products found for {row['location']}, skipping", flush=True)
        return str(return_path)
    
    # keep only the relative orbit whose combined bursts overlap the study area most
    best_rel_orbit = max(
        by_rel_orbit,
        key=lambda ro: union_all([Polygon(hit["poly"]) for hit, _ in by_rel_orbit[ro]]).intersection(study_area).area
    )
    # split the (hit, link) pairs of that orbit into two separate tuples again
    hits, links = zip(*by_rel_orbit[best_rel_orbit])

    # creating the directory where the data goes
    save_path.mkdir(parents=True, exist_ok=True)

    for hit, link in zip(hits, links):
        # how many seconds old the current token is
        token_age = time.time() - token_issued_at
        # tokens are valid for 60 min; refresh once we're within 10 min of that to be safe
        if token_age > 50 * 60:
            access_token = get_access_token(token_path)
            token_issued_at = time.time()
            headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
            # the search id expires together with the token, so the search is repeated to get a new one
            result, _ = query_egms(row=row, headers=headers)

        # put the current search id into the link (it changes whenever the search is repeated)
        link = f"{link.split('?id=')[0]}?id={result['id']}"

        # Checking whether a file already exists or not 
        # after unzipping only the csv is left, so that one counts as already downloaded as well
        file_path = save_path / hit["filename"]
        part_file_path = save_path / f"PART_{hit["filename"]}"
        csv_path = save_path / f"{file_path.stem}.csv"

        # checking three scenarios: 1) only stale file exists, so remove it and download again
        if part_file_path.exists():
            part_file_path.unlink()
        # 2) complete file exists, so move on to the next file
        elif file_path.exists() or csv_path.exists():
            continue

        # 3) file does not exist at all (has never been downloaded before)
        # the download itself, reached in scenario 1) and 3)
        # the server sometimes refuses a link right after the search (401) or still counts finished
        # downloads as running (429), so every file gets up to five attempts
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                requests_get(link, part_file_path, file_path, headers=headers)
                # download worked, no further attempt needed
                break
            # an error status from the server ends up here
            except requests.exceptions.HTTPError as e:
                # last attempt failed as well: give up on this file but keep going with the rest
                # note which file failed and why, download_egms writes that list to disk at the end
                if attempt == max_attempts:
                    failed_downloads.append({"location": row["location"],
                                             "orbit": orbit,
                                             "filename": hit["filename"],
                                             "status_code": e.response.status_code,
                                             "server_response": e.response.text[:500]})
                    break

                # 401 means the server does not accept the search id of the link
                # the server asks to rerun the search in that case, so a new token and id are fetched
                if e.response.status_code == 401:
                    access_token = get_access_token(token_path)
                    token_issued_at = time.time()
                    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
                    result, _ = query_egms(row=row, headers=headers)
                    link = f"{link.split('?id=')[0]}?id={result['id']}"

                # give the server some time before the next attempt, a bit longer every time
                time.sleep(10 * attempt)

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

    # fixed seed, so the same points are sampled on every run
    random_generator = np.random.default_rng(6576)

    # counters for the loop below
    attempts = 0
    n = 0
    # the area that is still allowed for sampling, it shrinks with every new point
    poly = gpd.GeoDataFrame(geometry=[polygon])
    # the first point can be anywhere in the polygon
    points = gpd.GeoSeries([polygon]).sample_points(1, rng=random_generator)

    while attempts < max_attempts-1 and n < n_points-1:
        # buffer all points sampled so far and remove that area from the allowed area
        # this way the next point keeps the minimum distance to all of them
        too_close_area = points.buffer(min_dist).union_all()
        poly.geometry = poly.difference(too_close_area)
        # sample the next point from what is left and add it to the others
        new_point = poly.sample_points(1, rng=random_generator)
        points = pd.concat([points, new_point], ignore_index=True)

        attempts += 1
        n += 1

    # sample_points returns a MultiPoint per row, even if it only holds a single point
    # explode turns them into plain Points, the later steps need their x and y
    points = points.explode(ignore_index=True)

    return gpd.GeoDataFrame(geometry=points, crs=crs)


def unzip(data_dir):
    """
    Extracts every zip found under a directory (recursively) into its own folder, then deletes the zip.

    Arguments:
        data_dir -- path (str or Path), root directory to search for "*.zip" files in

    Returns:
        None -- extracts files to disk and removes the original zips
    """
    # finding all the zips upfront and putting into a list
    zips = list(Path(data_dir).rglob("*.zip"))
    # iterate over the zip paths that were found
    for zip_path in zips:
        # open the zips, extract them and remove the original files
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(zip_path.parent)
        zip_path.unlink()

################################################
#====== Section 3: GNSS Download Helpers ======#
################################################
# Used by download_gnss, in the order they're first called there.

def stations_in_bbox(row, stations):
    """
    Returns the 4-digit station codes ("Sta") of the GNSS stations within a row's bbox polygon.

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with a bbox field (list of [lon, lat] corners)
        stations -- geopandas.GeoDataFrame, station points with a "Sta" column, same CRS as the bbox

    Returns:
        station_codes_within -- list of strings, the "Sta" codes of stations within the bbox
    """
    # create a polygon from the bbox col
    bbox = Polygon(row["bbox"])
    # .within returns boolean, so selection on stations first
    # then return them as list to be added as col in the gdf
    station_codes_within = stations.loc[stations.within(bbox), "Sta"].tolist()

    return station_codes_within

def pull_stations_in_bbox_data(row):
    """
    Downloads the GNSS time series (plate-fixed Eurasia frame) for every station code in a row's
    station_codes, into the row's location folder.

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with station_codes and path

    Returns:
        ts_file_paths -- list of strings, local file paths of the downloaded time series
    """
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

#################################################
#======= Section 4: GSE Download Helpers =======#
#################################################
# Used by download_gse, in the order they're first called there (bbox_per_csv and gs_uri_to_https
# are placed right before the function that calls them: large_bbox and pull_gse_tiles).

def bbox_per_csv(csv_file):
    """
    Computes the axis-aligned bounding box of the easting/northing points in one EGMS csv.

    Arguments:
        csv_file -- path (str or Path) to the EGMS csv, with easting and northing columns (EPSG:3035, metres)

    Returns:
        bbox -- shapely.geometry.Polygon, rectangle covering the min/max easting and northing of the file
    """
    # lazy computation of the relevant columns min and max values
    # .collect() does the actual computation
    # returns a one row df which is turned into a dict using .row(0, named=True)
    crs_cols = pl.scan_csv(csv_file).select(pl.col("easting").min().alias("min_easting"),
                                            pl.col("easting").max().alias("max_easting"),
                                            pl.col("northing").min().alias("min_northing"),
                                            pl.col("northing").max().alias("max_northing")).collect().row(0, named=True)

    bbox = box(crs_cols["min_easting"], crs_cols["min_northing"],
               crs_cols["max_easting"], crs_cols["max_northing"])

    return bbox

def large_bbox(row):
    """
    Computes the smallest rectangle spanning the union of all downloaded burst footprints (both
    orbit directions) for a study area, so GSE/ESA WorldCover downloads cover more than the
    narrow query bbox -- enough to support points queried later, not just the original margins.

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with path (the location folder containing
            ascending/ and descending/ subfolders of downloaded, unzipped EGMS csvs)

    Returns:
        corners -- list of [lon, lat] pairs, the four corners, same format as the bbox column
    """
    # computes the footprint for every csv in the two orbit folders per location
    bboxes = []
    for folder in ["ascending", "descending"]:
        csvs = (Path(row["path"]) / folder).glob("*.csv")
        for csv_path in csvs:
            # A list with all the individual bbox is created
            bboxes.append(bbox_per_csv(csv_path))

    # The union over the bboxes and their envelope
    envelope = union_all(bboxes).envelope
    # Transform to lat/lon. Index df for the gse data is also in lat/lon
    poly_4326 = gpd.GeoSeries([envelope], crs="EPSG:3035").to_crs("EPSG:4326").iloc[0]
    # Return it in the same format as the point_to_bbox function
    return [[x, y] for x, y in list(poly_4326.exterior.coords)[:-1]]

def load_gse_index(index_url):
    """
    Loads the AlphaEarth tile index into a GeoDataFrame, geometry from its WKT column.

    Arguments:
        index_url -- URL to download the aef_index.csv tile index from

    Returns:
        index_gdf -- geopandas.GeoDataFrame, one row per tile, with year/utm_zone/path/geometry
    """
    # download the index file
    index_file_path = pull_data_by_url(index_url, Path(__file__).parent / "../../assets/", "gse_index_file.csv"  )
    # read the csv file and build geodata from wkt column
    index_file_df = pd.read_csv(index_file_path, sep=",")
    index_gdf = gpd.GeoDataFrame(index_file_df, geometry=gpd.GeoSeries.from_wkt(index_file_df["WKT"]), crs="EPSG:4326")

    return index_gdf

def tiles_for_row_gse(row, index_gdf, years):
    """
    Finds which AlphaEarth tiles intersect a study-area row's large_bbox, grouped by year.

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with a large_bbox field (list of [lon, lat] corners)
        index_gdf -- geopandas.GeoDataFrame, the loaded tile index (see load_gse_index)
        years -- iterable of ints, which years to keep (e.g. range(2019, 2024) for the 5-year EGMS window)

    Returns:
        tiles_by_year -- dict, {year: [tile_dict, ...]} -- only years with at least one matching tile appear
    """

    # First we build a Polygon from the bbox column
    bbox = Polygon(row["large_bbox"])
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

    # cut off the gs:// prefix and split the rest into bucket and key
    without_prefix = gs_uri.removeprefix("gs://")
    bucket, key = without_prefix.split("/", 1)
    # put both behind the public https address of google cloud storage
    return f"https://storage.googleapis.com/{bucket}/{key}"

def pull_gse_tiles(part_crop_path, crop_path, tile, row):
    """
    Crops a single AlphaEarth tile to row's large_bbox via a remote windowed read and writes it
    to part_crop_path, renaming to crop_path on success.

    Arguments:
        part_crop_path -- Path, temporary path written to while cropping
        crop_path -- Path, final path part_crop_path is renamed to on success
        tile -- dict, one tile record from the GSE index (path, crs, ...)
        row -- pandas.Series, a GeoDataFrame row with a large_bbox field (list of [lon, lat] corners)

    Returns:
        result -- string, crop_path if the tile overlapped large_bbox and was written; None if
            the tile didn't overlap (nothing is written in that case)
    """
    # the public https url of the tile, rasterio can read from it directly
    url = gs_uri_to_https(tile["path"])
    with rasterio.open(url) as src:
        # Reproject the study area's bbox into this tile's own CRS -- rio_mask needs
        # the geometry in the same CRS as the raster, not lon/lat.
        bbox_poly = gpd.GeoSeries([Polygon(row["large_bbox"])], crs="EPSG:4326").to_crs(tile["crs"]).iloc[0]
    
        try:
            # Does the intersection + windowed read + crop in one call: reads only
            # the pixels covering bbox_poly (not the whole tile), and returns them
            # already cropped, along with the correct transform for just this crop.
            data, out_transform = rio_mask(src, [bbox_poly], crop=True)
        except ValueError:
            # Raised when bbox_poly doesn't overlap this tile at all.
            return None
    
        # the crop is smaller than the tile, so size and transform in the profile are adjusted
        profile = src.profile.copy()
        profile.update(height=data.shape[1], width=data.shape[2], transform=out_transform)
    
    # write the crop to the temporary file first
    with rasterio.open(part_crop_path, "w", **profile) as dst:
        dst.write(data)

    # writing finished without error, so the temporary file gets its final name
    part_crop_path.rename(crop_path)

    return str(crop_path)


def crop_gse_data(row):
    """
    Crops every AlphaEarth tile covering a study-area row's large_bbox directly over HTTPS
    (COG range reads -- no full-tile download), saving only the needed pixels per tile/year.

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with path and gse_tiles_by_year

    Returns:
        crop_paths -- list of strings, local file paths of the cropped GeoTIFFs written for this row
    """
    # defining path for output directory and list that catches all the paths to the files
    out_dir = Path(row["path"])
    crop_paths = []

    for year, tiles in row["gse_tiles_by_year"].items():
        for tile in tiles:
            # to check whether a file is already successfully downloaded, 
            # a temporary filename and a final file name are defined
            crop_path = out_dir / f"{year}_{Path(tile['path']).stem}_crop.tif"
            part_crop_path = out_dir / f"PART_{year}_{Path(tile['path']).stem}_crop.tif"

            # checking three scenarios: 1) only stale file exists  
            if part_crop_path.exists():
                part_crop_path.unlink()
                r = pull_gse_tiles(part_crop_path, crop_path, tile, row)
                if r is not None:
                    crop_paths.append(r) 
            # 2) file does not exist at all (has never been downloaded before)
            elif not crop_path.exists():
                r = pull_gse_tiles(part_crop_path, crop_path, tile, row)
                if r is not None:
                    crop_paths.append(r) 
            # complete file exists
            elif crop_path.exists(): 
                crop_paths.append(str(crop_path))

    return crop_paths

##################################################
#======= Section 5: ESA Worldcover Helper =======#
##################################################

def pull_worldcover(row, index_gdf):
    """
    Finds the ESA WorldCover (v200, 2021) tiles intersecting a study-area row's large_bbox and
    downloads each one into the row's location folder.

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with large_bbox and path
        index_gdf -- geopandas.GeoDataFrame, the WorldCover tile grid, with an "ll_tile" column
            (see download_esa_worldcover)

    Returns:
        tile_urls -- list of strings, the download URL of every tile that intersects the row
    """
    # First we build a Polygon from the bbox column
    bbox = Polygon(row["large_bbox"])
    # Find all tiles that intersect with that large bbox per location
    tiles = index_gdf[index_gdf.intersects(bbox)]
    # extract only the tile codes in that column to build the download urls for that tile
    tile_codes = tiles["ll_tile"]
    # empty list to catch the urls
    tile_urls = []
    for tile_code in tile_codes:
        url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile_code}_Map.tif"
        tile_urls.append(url)
    # Pull the tiles reusing pull_data_by_url
    for tile_url in tile_urls:
        pull_data_by_url(tile_url, Path(row["path"]), Path(tile_url).name)

    return tile_urls

##############################################################################
#============================================================================#
# 3) Processing
#============================================================================#
##############################################################################

def download_egms(real_sub_path, token_path, egms_coverage, n_rnd, mindist_rnd, maxit_rnd, real_sub_buffer_value = 400000):
    """
    Downloads EGMS L2b data for the real subsidence regions and for extra randomly sampled regions, for both orbit directions.
    Files that could not be downloaded are listed in assets/failed_egms_downloads.csv.

    Arguments:
        real_sub_path -- path (str or Path) to the real_subsidence_pointbased.gpkg file
        token_path -- path (str or Path) to the EGMS API service-account key file
        egms_coverage -- path (str or Path) to the geojson of the EGMS coverage area
        real_sub_buffer_value -- float, buffer radius (metres) around each real subsidence point excluded from random sampling
        n_rnd -- int, number of extra random points to sample across the EGMS coverage area
        mindist_rnd -- float, minimum distance (metres) required between the random points
        maxit_rnd -- int, upper bound on sampling attempts for the random points

    Returns:
        study_areas_gdf -- geopandas.GeoDataFrame, the real subsidence regions plus the randomly
            sampled regions combined (EPSG:3035), with bbox and path set
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

    # Collects every file that could not be downloaded, filled by pull_egms and written to disk below
    failed_downloads = []

    # Download the products with filter before
    # the set variable decides into which region the data goes
    # Generally:  0 = train    1 = dev   2 = cal   3 = val
    real_subsidence["path"] = real_subsidence.apply(pull_egms, axis = 1, token_path=token_path, orbit = "ascending", failed_downloads=failed_downloads)
    real_subsidence["path"] = real_subsidence.apply(pull_egms, axis = 1, token_path=token_path, orbit = "descending", failed_downloads=failed_downloads)

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

    # sampling the random points within the allowed area
    egms_random = sample_random_points(allowed_area, n_rnd, mindist_rnd, maxit_rnd, crs = "EPSG:3035")

    # adding the same columns to the dataframe that the initial real_subsidence df has 
    egms_random["north"] = 25
    egms_random["east"] = 25
    egms_random["south"] = 25
    egms_random["west"] = 25
    # the random regions only go into train, dev and cal (0, 1, 2), with a fixed seed for the split
    egms_random["set"] = np.random.default_rng(42).choice([0, 1, 2], size=len(egms_random), p=[0.6, 0.2, 0.2]) 

    # naming the random regions after their set with a running number per set
    # e.g. train_1, train_2, ... dev_1, dev_2, ... cal_1, cal_2, ...
    set_names = {0: "train", 1: "dev", 2: "cal"}
    egms_random["location"] = egms_random["set"].map(set_names) + "_" + (egms_random.groupby("set").cumcount() + 1).astype(str)

    # Creating the bbox and querying the archive on the random point samples
    egms_random["bbox"] = egms_random.apply(point_to_bbox, axis=1, result_type="reduce", t2projected=t2projected, t2wgs84=t2wgs84, source_crs=egms_random.crs)
    
    # downloading the egms data for the new areas
    egms_random["path"] = egms_random.apply(pull_egms, axis = 1, token_path=token_path, orbit = "ascending", failed_downloads=failed_downloads)
    egms_random["path"] = egms_random.apply(pull_egms, axis = 1, token_path=token_path, orbit = "descending", failed_downloads=failed_downloads)

    # Writing the files that could not be downloaded to disk so they are not lost silently
    if failed_downloads:
        failed_path = Path(__file__).parent / "../../assets/failed_egms_downloads.csv"
        pd.DataFrame(failed_downloads).to_csv(failed_path, index=False)
        print(f"{len(failed_downloads)} EGMS files could not be downloaded, see {failed_path}", flush=True)
    
    # Unzipping the egms zip files for later use
    unzip(DATA_DIR)

    # Now concattenating both gdfs. This gdf is representative for the entire study area.
    # First making sure they are both in the same crs
    real_subsidence = real_subsidence.to_crs("EPSG:3035")
    egms_random = egms_random.to_crs("EPSG:3035")

    study_areas_gdf = pd.concat([real_subsidence, egms_random])

    # Saving the geodataframe
    study_areas_io("write", study_areas_gdf, Path(__file__).parent / "../../assets/study_areas_gdf")
    
    return study_areas_gdf


def download_gnss(study_areas_gdf):
    """
    Finds the GNSS stations within each validation region's bbox and downloads their time series,
    leaving station_codes/station_file_paths as None for train/dev/cal regions.

    Arguments:
        study_areas_gdf -- geopandas.GeoDataFrame, from download_egms, with bbox/location/set/path

    Returns:
        study_areas_gdf -- the same GeoDataFrame, with station_codes and station_file_paths added
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

    # Creating a mask to apply stations_in_bbox only to the validation regions
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
    Crops the AlphaEarth satellite embedding tiles covering every study area's large_bbox (the
    union of its downloaded burst footprints, not just the narrow query bbox), for every
    requested year -- read directly over HTTPS (COG range reads), without downloading whole tiles.

    Arguments:
        study_areas_gdf -- geopandas.GeoDataFrame, from download_egms/download_gnss, with path set
        years -- iterable of ints, which years to download (e.g. [2019, 2020, 2021, 2022, 2023])

    Returns:
        study_areas_gdf -- the same GeoDataFrame, with large_bbox, gse_tiles_by_year and
            gse_crop_paths added
    """

    # Computing the larger bbox for the covariates on the dataframe
    study_areas_gdf["large_bbox"] = study_areas_gdf.apply(large_bbox, axis=1)

    # First download the index dataframe that contains information about the gse tiles and years
    # index_gdf is the raw index_gdf only filtered by the years that are relevant here
    index_gdf = load_gse_index("https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual/aef_index.csv")

    # This adds the tiles per year as a new column to the gdf
    study_areas_gdf["gse_tiles_by_year"] = study_areas_gdf.apply(tiles_for_row_gse, axis=1, index_gdf=index_gdf, years=years)

    # Downloading the tiles into the location folder with year prefixes in filenames
    study_areas_gdf["gse_crop_paths"] = study_areas_gdf.apply(crop_gse_data, axis=1)

    # Saving the updated geodataframe
    study_areas_io("write", study_areas_gdf, Path(__file__).parent / "../../assets/study_areas_gdf")

    return study_areas_gdf

def download_esa_worldcover(study_areas_gdf, index_url):
    """
    Downloads the ESA WorldCover (v200, 2021) tiles covering every study area's large_bbox,
    reusing the same large_bbox column download_gse computes.

    Arguments:
        study_areas_gdf -- geopandas.GeoDataFrame, from download_gse, with large_bbox and path set
        index_url -- URL to download the WorldCover tile grid geojson from

    Returns:
        None -- study_areas_gdf is updated and saved in place, not returned
    """
    # download the index file with the tile grid
    index_path = pull_data_by_url(index_url, Path(__file__).parent / "../../assets/", "esa_world_cover_index.geojson")

    # read it as gdf to find the tiles per region
    index_gdf = gpd.read_file(index_path)

    # Downloading the tiles and adding their urls as a new column to the gdf
    study_areas_gdf["world_cover_tile_urls"] = study_areas_gdf.apply(pull_worldcover, axis=1, index_gdf=index_gdf)

    # Saving the updated geodataframe
    study_areas_io("write", study_areas_gdf, Path(__file__).parent / "../../assets/study_areas_gdf")
    

##############################################################################
#============================================================================#
# 4) Main
#============================================================================#
##############################################################################

def main(real_sub_buffer_value, n_rnd, mindist_rnd, maxit_rnd):
    """
    Checks the required input files exist, sets up the project's file system, and runs the
    EGMS, GNSS, GSE and ESA WorldCover downloads in sequence.

    Arguments:
        real_sub_buffer_value -- float, buffer radius (metres) around each real subsidence point
            excluded from random sampling (passed through to download_egms)
        n_rnd -- int, number of extra random points to sample across the EGMS coverage area
        mindist_rnd -- float, minimum distance (metres) required between the random points
        maxit_rnd -- int, upper bound on sampling attempts for the random points

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
                        real_sub_buffer_value = real_sub_buffer_value,
                        n_rnd=n_rnd, mindist_rnd=mindist_rnd, maxit_rnd=maxit_rnd)
    
    # Use the dataframe of the regions for which egms was downloaded to select and download
    # the GNSS stations and time series.
    study_areas_gdf = download_gnss(study_areas_gdf)

    # download google satellite embeddings for every region
    study_areas_gdf = download_gse(study_areas_gdf, [2019, 2020, 2021, 2022, 2023])

    # Download the ESA world cover for later land use stratification
    download_esa_worldcover(study_areas_gdf, index_url="https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/esa_worldcover_grid.geojson")


if __name__ == '__main__':

    # reading the command line arguments that the slurm script passes
    parser = argparse.ArgumentParser()

    parser.add_argument("--projectdir", required=True)
    parser.add_argument("--real_sub_buffer_value", required=True)
    parser.add_argument("--n_rnd", required=True)
    parser.add_argument("--mindist_rnd", required=True)
    parser.add_argument("--maxit_rnd", required=True)

    args = parser.parse_args()

    # the arguments arrive as strings, so the numeric ones are converted to int
    REAL_SUB_BUFFER_VALUE = int(args.real_sub_buffer_value)
    N_RND = int(args.n_rnd)
    MINDIST_RND = int(args.mindist_rnd)
    MAXIT_RND = int(args.maxit_rnd)

    # the paths of the project's file system, used as globals by the functions above
    PROJECT_DIR = Path(args.projectdir)
    DATA_DIR = PROJECT_DIR / "data"
    # For the data split between training, dev and validation regions
    TRAIN_DIR = DATA_DIR / "train"
    DEV_DIR = DATA_DIR / "dev"
    CAL_DIR = DATA_DIR / "cal"
    VAL_DIR = DATA_DIR / "val"

    # running the whole download
    main(real_sub_buffer_value = REAL_SUB_BUFFER_VALUE,
          n_rnd=N_RND, mindist_rnd=MINDIST_RND, maxit_rnd=MAXIT_RND) 
