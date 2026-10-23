#######################################
# s01_download_data

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
import argparse
import shapely

from pathlib import Path
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely import union_all
from shapely.geometry import box
from shapely.ops import transform

################################################
# Section 1.1: Constants
################################################

# Buffer (metres) around each region's core that is downloaded and kept when the bursts are clipped.
# The pool buffer of the later steps may vary, but must not be larger than this.
CLIP_BUFFER_M = 5000

##############################################################################
#============================================================================#
# 2) Helper Functions
#============================================================================#
##############################################################################


################################################
#===== Section 1: Shared Helper Functions =====#
################################################
# Used across more than one processing step below, in the order they're first called.

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
    PARAM_LOG_DIR.mkdir(parents=True, exist_ok=True)


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
    if headers is None:
        # Plain downloads (NGL station list and time series, ESA WorldCover tiles)
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


def study_areas_io(mode, gdf=None, base_path=None):
    """
    Splits a GeoDataFrame into a gpkg (geometry + scalar columns) and a JSON file (the list/dict
    columns gpkg can't store), or reconstructs it from both files. Rows are matched by a dedicated
    "uid" column generated fresh on every write, rather than by index or "location" (which isn't
    guaranteed unique), so the two files can never get mismatched against each other.
    Writes always go to a "_local" copy (base_path_local.gpkg / base_path_local_complex.json, gitignored),
    so the git-tracked original files are never overwritten. Reads take the "_local" copy if it exists,
    otherwise the original.

    Arguments:
        mode -- string, "write" to split and save gdf, "read" to load and recombine
        gdf -- geopandas.GeoDataFrame, required when mode is "write"; the frame to split and save
        base_path -- path (str or Path) without extension, the shared filename stem of the original files

    Returns:
        result -- on "write": None, writes base_path_local.gpkg and base_path_local_complex.json to disk.
                  on "read": geopandas.GeoDataFrame, the recombined frame
    """
    # building the paths of the two files from the shared filename stem
    base_path = Path(base_path)
    local_path = base_path.parent / f"{base_path.stem}_local"
    # reading falls back to the tracked original as long as no local copy has been written yet
    if mode == "read" and not local_path.with_suffix(".gpkg").exists():
        local_path = base_path
    gpkg_path = local_path.with_suffix(".gpkg")
    json_path = local_path.parent / f"{local_path.stem}_complex.json"

    # the columns holding lists or dicts. gpkg can't store those, so they go into the json
    complex_cols = ["egms_query_bbox", "egms_union_bbox", "gnss_station_codes", "gnss_station_paths",
                     "wc_tile_urls", "gnss_station_codes_filtered", "gnss_station_paths_filtered",
                     "gnss_velocities", "gnss_filter_params"]

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


def pull_data_by_url(url, out_dir, file):
    """
    Downloads a URL to out_dir / file unless the complete file is already there, used for every
    plain-HTTP download (GNSS station list and time series, ESA WorldCover tiles). A stale
    PART_ file from an interrupted earlier run is removed and downloaded again.

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
# Used by download_egms, in the order they're first called there (a helper's own helpers right after it).

def polygon_to_coords(geom, t2wgs84):
    """
    Converts a polygon's exterior ring from EPSG:3035 to a list of [lon, lat] pairs, the format of
    the EGMS search query (egms_query_bbox). The search only uses the min/max lon/lat of the ring,
    so the ring is not densified.

    Arguments:
        geom -- shapely Polygon in EPSG:3035 (single part; holes are ignored)
        t2wgs84 -- pyproj.Transformer from EPSG:3035 to EPSG:4326 (always_xy)

    Returns:
        coords -- list of [lon, lat] pairs, the ring's vertices without the closing vertex
    """
    ring = geom.exterior.coords
    return [list(t2wgs84.transform(x, y)) for x, y in list(ring)[:-1]]   # drop the closing vertex


def pull_egms(row, token_path, orbit, failed_downloads, t2proj):
    """
    Downloads the EGMS bursts of one orbit direction for one region. The search hits are checked against
    the actual query area (core + clip_buffer_m, EPSG:3035), since the API only searches its lon/lat bounding box.
    Of the intersecting bursts, only the relative orbit with the largest combined overlap is kept, and all
    of its bursts are downloaded, however small their overlap (the minimum-overlap check is done in s02).
    Files already on disk are skipped. Files the server refuses are recorded in failed_downloads and
    skipped as well, so a single bad file doesn't stop the whole run.

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with geometry (core, EPSG:3035), clip_buffer_m,
            egms_query_bbox, location and set
        token_path -- path (str or Path) to the EGMS API service-account key file, used to fetch
            a fresh access token and search id once they are close to expiring (both expire after
            1 hour, and a single row's files can take longer than that to all download)
        orbit -- string, orbit direction to keep, e.g. "ascending" or "descending"
        failed_downloads -- list, every file that could not be downloaded is appended to it as a
            dict (location, orbit, filename, status_code, server_response)
        t2proj -- pyproj.Transformer from EPSG:4326 to EPSG:3035 (always_xy), for the burst footprints

    Returns:
        return_path -- string, the region folder (TRAIN_DIR, DEV_DIR, CAL_DIR or VAL_DIR / location).
            It is always created and returned, also if there are no products for this orbit direction
    """

    # some of the real subsidence areas go into training but some also go into validation
    # the set variable decides into which region the data goes
    # Generally:  0 = train    1 = dev   2 = cal   3 = val
    # This comes first so return_path exists even if nothing is downloaded for this orbit
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

    return_path.mkdir(parents=True, exist_ok=True)

    # Create an access token and note when it was generated (expires after one hour)
    access_token = get_access_token(token_path)
    token_issued_at = time.time()
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

    # Call query egms once per line since the ids also expire
    # Calling it once upfront on the entire df leads to expired ids once the download is actually performed
    result, links = query_egms(row=row, headers=headers)

    # the query area (core + buffer) in EPSG:3035, to check and measure the overlap with the bursts below
    study_area = row.geometry.buffer(row["clip_buffer_m"])

    # keep only the products of the wanted orbit direction, together with their links,
    # and only those that really intersect the query area, not just its lon/lat bounding box
    pairs = [(hit, link) for hit, link in zip(result["hits"], links)
         if hit["direction"] == orbit and transform(t2proj.transform,
                                                    Polygon(hit["poly"])).intersects(study_area)]

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
        key=lambda ro: union_all([transform(t2proj.transform, Polygon(hit["poly"])) for hit, _ in by_rel_orbit[ro]]).intersection(study_area).area
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
    Searches the EGMS archive for L2b products of a row's query area and builds the download link for each hit.
    The API only searches the min/max lon/lat of the query polygon, so hits can lie outside the actual
    area; pull_egms checks them against it. Raises a RuntimeError if the search fails or is rejected
    (e.g. because of an invalid token).

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with egms_query_bbox (list of [lon, lat] pairs) and location
        headers -- dict, request headers carrying the bearer access token (see get_access_token)
        api_endpoint -- string, base URL of the EGMS API

    Returns:
        result -- dict, the raw JSON search result returned by the API
        links -- list of strings, one download link per hit in result["hits"]
    """

    # the search query: all L2B products of the 2019-2023 release within the query's lon/lat bounds
    query = {"id": None,
         "bbox": row["egms_query_bbox"],
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


def clip_to_query_area(csv_path, query_area):
    """
    Clips an EGMS burst csv in place to the points inside a region's query area (core + CLIP_BUFFER_M).
    All columns are read as text, so the kept rows are written back unchanged.

    Arguments:
        csv_path -- path (str or Path) to the unzipped EGMS csv, with easting and northing (EPSG:3035)
        query_area -- shapely Polygon in EPSG:3035, ideally prepared with shapely.prepare

    Returns:
        None -- overwrites csv_path
    """
    # cheap lazy pre-filter on the bounding box of the query area
    xmin, ymin, xmax, ymax = query_area.bounds
    df = (pl.scan_csv(csv_path, infer_schema_length=0)
            .filter(pl.col("easting").cast(pl.Float64).is_between(xmin, xmax)
                    & pl.col("northing").cast(pl.Float64).is_between(ymin, ymax))
            .collect())
    # exact point-in-polygon test on what is left
    inside = shapely.contains_xy(query_area, df["easting"].cast(pl.Float64).to_numpy(),
                                 df["northing"].cast(pl.Float64).to_numpy())
    df.filter(pl.Series(inside)).write_csv(csv_path)


################################################
#====== Section 3: GNSS Download Helpers ======#
################################################
# Used by download_gnss, in the order they're first called there.

def stations_in_bbox(row, stations):
    """
    Returns the 4-digit station codes ("Sta") of the GNSS stations within a row's core polygon
    (not the buffered query area).

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with geometry (core, EPSG:3035)
        stations -- geopandas.GeoDataFrame, station points with a "Sta" column, in EPSG:3035

    Returns:
        station_codes_within -- list of strings, the "Sta" codes of the stations within the core
    """
    # .within returns boolean, so selection on stations first
    # then return them as list to be added as col in the gdf
    station_codes_within = stations.loc[stations.within(row.geometry), "Sta"].tolist()

    return station_codes_within


def pull_stations_in_bbox_data(row):
    """
    Downloads the GNSS time series (plate-fixed Eurasia frame) for every station code in a row's
    gnss_station_codes, into the row's region folder.

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with gnss_station_codes and region_path

    Returns:
        ts_file_paths -- list of strings, local file paths of the downloaded time series
    """
    # base url for the Eurasia fixed time series
    base_url = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/EU/"    #<ssss>.<plate>.tenv3
    # empty list to catch the paths to the gnss files
    ts_file_paths = []
    # loop over all stations that were found within that region
    for station_code in row["gnss_station_codes"]:
       # create individual url for every station
       url = f"{base_url}{station_code}.EU.tenv3"
       # reuse the download function
       ts_file_path = pull_data_by_url(url, Path(row["region_path"]), f"{str(station_code)}_gnss_time_series.tenv3")
       # catch the file paths to fill a new dataframe column
       ts_file_paths.append(ts_file_path)

    return ts_file_paths


##################################################
#====== Section 4: ESA WorldCover Helpers =======#
##################################################
# Used by download_esa_worldcover, in the order they're first called there.

def large_bbox(row):
    """
    Computes the smallest rectangle spanning all downloaded burst footprints (both orbit
    directions) of a region, so the ESA WorldCover tiles cover every downloaded point, not
    only the query area.

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with region_path (the location folder containing
            ascending/ and descending/ subfolders of downloaded, unzipped EGMS csvs)

    Returns:
        corners -- list of [lon, lat] pairs, the four corners, same format as egms_query_bbox
    """
    # computes the footprint for every csv in the two orbit folders per location
    bboxes = []
    for folder in ["ascending", "descending"]:
        csvs = (Path(row["region_path"]) / folder).glob("*.csv")
        for csv_path in csvs:
            # A list with all the individual bbox is created
            bboxes.append(bbox_per_csv(csv_path))

    bboxes = [b for b in bboxes if b is not None]

    # The union over the bboxes and their envelope
    envelope = union_all(bboxes).envelope
    # Transform to lat/lon. Index df for the gse data is also in lat/lon
    poly_4326 = gpd.GeoSeries([envelope], crs="EPSG:3035").to_crs("EPSG:4326").iloc[0]
    # Return it in the same format as egms_query_bbox
    return [[x, y] for x, y in list(poly_4326.exterior.coords)[:-1]]


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

    # a burst clipped to zero points has no extent
    if crs_cols["min_easting"] is None:
        return None


    bbox = box(crs_cols["min_easting"], crs_cols["min_northing"],
               crs_cols["max_easting"], crs_cols["max_northing"])

    return bbox


def pull_worldcover(row, index_gdf):
    """
    Finds the ESA WorldCover (v200, 2021) tiles intersecting a row's egms_union_bbox and
    downloads each one into the row's region folder.

    Arguments:
        row -- pandas.Series, a GeoDataFrame row with egms_union_bbox and region_path
        index_gdf -- geopandas.GeoDataFrame, the WorldCover tile grid, with an "ll_tile" column
            (see download_esa_worldcover)

    Returns:
        tile_urls -- list of strings, the download URL of every tile that intersects the row
    """
    # First we build a Polygon from the bbox column
    bbox = Polygon(row["egms_union_bbox"])
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
        pull_data_by_url(tile_url, Path(row["region_path"]), Path(tile_url).name)

    return tile_urls


##############################################################################
#============================================================================#
# 3) Processing
#============================================================================#
##############################################################################


def download_egms(real_sub_path, token_path, egms_coverage, n_rnd, mindist_rnd, maxit_rnd, egms_rand_extension, real_sub_buffer_value = 400000):
    """
    Downloads EGMS L2b data for the real subsidence regions and for extra randomly sampled regions, for both orbit directions.
    Every region is a core polygon (geometry, EPSG:3035) plus clip_buffer_m = CLIP_BUFFER_M; EGMS is queried with
    core + buffer. The random regions are squares around random points, sampled outside a zone around the
    real subsidence regions so that no query areas overlap. Files that could not be downloaded are listed
    in assets/failed_egms_downloads.csv.

    Arguments:
        real_sub_path -- path (str or Path) to the real_subsidence_polygonbased.gpkg file (any CRS,
            converted to EPSG:3035; one single-part core polygon per region, columns location and set)
        token_path -- path (str or Path) to the EGMS API service-account key file
        egms_coverage -- path (str or Path) to the geojson of the EGMS coverage area
        n_rnd -- int, number of random regions to sample across the EGMS coverage area
        mindist_rnd -- float, minimum distance (metres) between the random region centres
        maxit_rnd -- int, upper bound on sampling attempts for the random points
        egms_rand_extension -- float, half side length (metres) of the square core of a random region
        real_sub_buffer_value -- float, minimum gap (metres) between the query areas of the real
            subsidence regions and those of the random regions

    Returns:
        study_areas_gdf -- geopandas.GeoDataFrame, the real subsidence regions plus the random regions
            (EPSG:3035), with clip_buffer_m, egms_query_bbox and region_path set
    """

    ################################################
    #=== Section 1: The Real Subsidence Regions ===#
    ################################################

    # The handpicked real subsidence regions. Some go into training, some into validation
    # (set column of the gpkg: 0 = train, 1 = dev, 2 = cal, 3 = val)
    real_subsidence = gpd.read_file(real_sub_path).to_crs("EPSG:3035")
    # stored per region, the later steps use it to tell the core from the buffer ring
    real_subsidence["clip_buffer_m"] = CLIP_BUFFER_M
    t2wgs84 = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    # the query area = core + buffer, only kept as [lon, lat] list, the geometry column stays the core
    query_geoms = real_subsidence.geometry.buffer(real_subsidence["clip_buffer_m"])
    real_subsidence["egms_query_bbox"] = [polygon_to_coords(g, t2wgs84) for g in query_geoms]

    # Collects every file that could not be downloaded, filled by pull_egms and written to disk below
    failed_downloads = []

    # Download both orbit directions, the burst footprints come in lon/lat and are projected for the overlap
    t2proj = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    real_subsidence["region_path"] = real_subsidence.apply(pull_egms, axis = 1, token_path=token_path, orbit = "ascending", failed_downloads=failed_downloads, t2proj=t2proj)
    real_subsidence["region_path"] = real_subsidence.apply(pull_egms, axis = 1, token_path=token_path, orbit = "descending", failed_downloads=failed_downloads, t2proj=t2proj)

    ################################################
    #======== Section 2: The Random Regions ========#
    ################################################

    # Random regions across Europe enrich the dataset with ordinary (mostly stable) ground.
    # The EGMS coverage area constrains the sampling
    coverage_egms = gpd.read_file(egms_coverage).to_crs("EPSG:3035").union_all()
    # No random centre within this zone around the real subsidence cores, so real_sub_buffer_value is the
    # minimum gap between the query areas: the random square reaches egms_rand_extension * sqrt(2) from its
    # centre (corner), and both query areas add CLIP_BUFFER_M. Overlapping query areas would put the same
    # points into two regions, possibly of different sets
    real_sub_buffer = real_subsidence.buffer(real_sub_buffer_value + egms_rand_extension * np.sqrt(2) + 2 * CLIP_BUFFER_M).union_all()
    # egms coverage area without the zones around the real subsidence regions
    allowed_area = coverage_egms.difference(real_sub_buffer)

    # sampling the random points within the allowed area
    egms_random = sample_random_points(allowed_area, n_rnd, mindist_rnd, maxit_rnd, crs = "EPSG:3035")

    # the random cores: squares with half side length egms_rand_extension (metres) around the sampled points
    egms_random["geometry"] = egms_random.geometry.buffer(egms_rand_extension, cap_style="square")
    egms_random["clip_buffer_m"] = CLIP_BUFFER_M

    # Query geometry = core + buffer, exactly as for the real subsidence regions
    query_geoms = egms_random.geometry.buffer(egms_random["clip_buffer_m"])
    egms_random["egms_query_bbox"] = [polygon_to_coords(g, t2wgs84) for g in query_geoms]

    # the random regions only go into train, dev and cal (0, 1, 2), with a fixed seed for the split
    egms_random["set"] = np.random.default_rng(42).choice([0, 1, 2], size=len(egms_random), p=[0.6, 0.2, 0.2])

    # naming the random regions after their set with a running number per set
    # e.g. train_1, train_2, ... dev_1, dev_2, ... cal_1, cal_2, ...
    set_names = {0: "train", 1: "dev", 2: "cal"}
    egms_random["location"] = egms_random["set"].map(set_names) + "_" + (egms_random.groupby("set").cumcount() + 1).astype(str)

    # downloading the egms data for the random regions
    egms_random["region_path"] = egms_random.apply(pull_egms, axis = 1, token_path=token_path, orbit = "ascending", failed_downloads=failed_downloads, t2proj=t2proj)
    egms_random["region_path"] = egms_random.apply(pull_egms, axis = 1, token_path=token_path, orbit = "descending", failed_downloads=failed_downloads, t2proj=t2proj)

    # Writing the files that could not be downloaded to disk so they are not lost silently
    if failed_downloads:
        failed_path = Path(__file__).parent / "../../assets/failed_egms_downloads.csv"
        pd.DataFrame(failed_downloads).to_csv(failed_path, index=False)
        print(f"{len(failed_downloads)} EGMS files could not be downloaded, see {failed_path}", flush=True)

    # Unzipping the egms zip files for later use
    unzip(DATA_DIR)

    # Concatenating both gdfs, this is the region table all later steps read
    # (both are in EPSG:3035 already, the to_crs only makes sure)
    real_subsidence = real_subsidence.to_crs("EPSG:3035")
    egms_random = egms_random.to_crs("EPSG:3035")

    study_areas_gdf = pd.concat([real_subsidence, egms_random], ignore_index=True)

    # clip every burst to its region's query area, only the points there are ever used
    for _, row in study_areas_gdf.iterrows():
        query_area = row.geometry.buffer(row["clip_buffer_m"])
        shapely.prepare(query_area)     # speeds up the many point-in-polygon tests
        for csv_path in Path(row["region_path"]).glob("*/*.csv"):
            if "_PREPROCESSED" not in csv_path.name:    # s02 outputs, in case s01 is rerun after s02
                clip_to_query_area(csv_path, query_area)


    # Saving the geodataframe
    study_areas_io("write", study_areas_gdf, Path(__file__).parent / "../../assets/study_areas_gdf")

    return study_areas_gdf


def download_gnss(study_areas_gdf):
    """
    Finds the GNSS stations within each validation region's core polygon and downloads their time series,
    leaving gnss_station_codes/gnss_station_paths as None for train/dev/cal regions.

    Arguments:
        study_areas_gdf -- geopandas.GeoDataFrame, from download_egms, with geometry (EPSG:3035)/set/region_path

    Returns:
        study_areas_gdf -- the same GeoDataFrame, with gnss_station_codes and gnss_station_paths added
    """
    # Creating a col for the station codes. Only the val regions shall receive a valid entry
    # For the train and dev regions this stays None
    study_areas_gdf["gnss_station_codes"] = None

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
    stations = gpd.GeoDataFrame(station_list, geometry= "geometry" , crs="EPSG:4326").to_crs("EPSG:3035")

    # Creating a mask to apply stations_in_bbox only to the validation regions
    val_mask = study_areas_gdf["set"] == 3
    # apply the function and write the result back into the val row, leaving None for train/dev
    study_areas_gdf.loc[val_mask, "gnss_station_codes"] = study_areas_gdf.loc[val_mask].apply(
        stations_in_bbox, axis=1, stations=stations)

    ##########################################################
    # Now downloading the actual time series

    # Again only compute on the validation areas
    study_areas_gdf["gnss_station_paths"] = None
    study_areas_gdf.loc[val_mask, "gnss_station_paths"] = study_areas_gdf.loc[val_mask].apply(
            pull_stations_in_bbox_data, axis=1)

    # Saving the updated geodataframe
    study_areas_io("write", study_areas_gdf, Path(__file__).parent / "../../assets/study_areas_gdf")

    return study_areas_gdf


def download_esa_worldcover(study_areas_gdf, index_url):
    """
    Downloads the ESA WorldCover (v200, 2021) tiles covering every region's egms_union_bbox,
    computing that column here (see large_bbox).

    Arguments:
        study_areas_gdf -- geopandas.GeoDataFrame, from download_gnss, with region_path set
        index_url -- URL to download the WorldCover tile grid geojson from

    Returns:
        None -- study_areas_gdf is updated and saved in place, not returned
    """
    # the smallest rectangle spanning every downloaded burst's footprint, per study area
    study_areas_gdf["egms_union_bbox"] = study_areas_gdf.apply(large_bbox, axis=1)

    # download the index file with the tile grid
    index_path = pull_data_by_url(index_url, Path(__file__).parent / "../../assets/", "esa_world_cover_index.geojson")

    # read it as gdf to find the tiles per region
    index_gdf = gpd.read_file(index_path)

    # Downloading the tiles and adding their urls as a new column to the gdf
    study_areas_gdf["wc_tile_urls"] = study_areas_gdf.apply(pull_worldcover, axis=1, index_gdf=index_gdf)

    # Saving the updated geodataframe
    study_areas_io("write", study_areas_gdf, Path(__file__).parent / "../../assets/study_areas_gdf")


##############################################################################
#============================================================================#
# 4) Main
#============================================================================#
##############################################################################


def main(real_sub_buffer_value, n_rnd, mindist_rnd, maxit_rnd, egms_rand_extension):
    """
    Checks the required input file exists, sets up the project's file system, and runs the
    EGMS, GNSS and ESA WorldCover downloads in sequence.

    Arguments:
        real_sub_buffer_value, n_rnd, mindist_rnd, maxit_rnd, egms_rand_extension -- passed to
            download_egms, see there

    Returns:
        None -- raises FileNotFoundError if the real subsidence file is missing
    """

    ##############################
    # Check necessary files are there

    if not os.path.exists(Path(__file__).parent / "../../assets/real_subsidence_polygonbased.gpkg"):
        raise FileNotFoundError("The real subsidence regions are missing!")

    ##############################
    # Creating the file system for the project
    create_file_system()

    ##############################
    # Downloading the data
    study_areas_gdf = download_egms(Path(__file__).parent / "../../assets/real_subsidence_polygonbased.gpkg",
                        Path(__file__).parent / "../../assets/token.jwt",
                        Path(__file__).parent / "../../assets/egms_coverage_countries.geojson",
                        real_sub_buffer_value = real_sub_buffer_value,
                        n_rnd=n_rnd, mindist_rnd=mindist_rnd, maxit_rnd=maxit_rnd, egms_rand_extension=egms_rand_extension)

    # Use the dataframe of the regions for which egms was downloaded to select and download
    # the GNSS stations and time series.
    study_areas_gdf = download_gnss(study_areas_gdf)

    # Download the ESA world cover for later land use stratification
    download_esa_worldcover(study_areas_gdf, index_url="https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/esa_worldcover_grid.geojson")


if __name__ == '__main__':

    # reading the command line arguments that the slurm script passes
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-dir", dest="project_dir", type=str, required=True,
                        help="root of the project's data tree, data/{train,dev,cal,val}/ is created under it")
    parser.add_argument("--real-sub-buffer-value", dest="real_sub_buffer_value", type=int, required=True,
                        help="minimum gap (m) between the query areas of the real subsidence and the random regions")
    parser.add_argument("--n-rnd", dest="n_rnd", type=int, required=True,
                        help="number of random regions sampled across the EGMS coverage area")
    parser.add_argument("--mindist-rnd", dest="mindist_rnd", type=int, required=True,
                        help="minimum distance (m) between the random region centres")
    parser.add_argument("--maxit-rnd", dest="maxit_rnd", type=int, required=True,
                        help="upper bound on sampling attempts for the random regions")
    parser.add_argument("--egms-rand-extension", dest="egms_rand_extension", type=int, required=True,
                        help="half side length (m) of the square core of a random region")

    args = parser.parse_args()

    # the paths of the project's file system, used as globals by the functions above
    PROJECT_DIR = Path(args.project_dir)
    DATA_DIR = PROJECT_DIR / "data"
    # For the data split between training, dev and validation regions
    TRAIN_DIR = DATA_DIR / "train"
    DEV_DIR = DATA_DIR / "dev"
    CAL_DIR = DATA_DIR / "cal"
    VAL_DIR = DATA_DIR / "val"
    # For logging the processing parameter
    PARAM_LOG_DIR = PROJECT_DIR / "param_log"

    # running the whole download
    main(real_sub_buffer_value=args.real_sub_buffer_value, n_rnd=args.n_rnd, mindist_rnd=args.mindist_rnd,
         maxit_rnd=args.maxit_rnd, egms_rand_extension=args.egms_rand_extension)
