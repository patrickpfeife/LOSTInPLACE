#######################################
# 01_download_egms

#============================================================#
# Import Statements
#============================================================#

import geopandas as gpd
import sys
import json
import time
import requests
import jwt
import cryptography
import os

from pathlib import Path
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import unary_union

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
    # setting up the file system for the project 
    PROJECT_DIR.mkdir(parents=True, exist_ok=True) 
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)


def get_access_token(token_path):
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

def point_to_bbox(row, t2projected, t2wgs84):
    # reproject to appropriate crs
    # assumes that the correct projection is already set
    x, y = t2projected.transform(row.geometry.x, row.geometry.y)

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
    

def pull_egms(row, headers, orbit, set):

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
        key=lambda ro: unary_union([Polygon(hit["poly"]) for hit, _ in by_rel_orbit[ro]]).intersection(study_area).area
    )
    hits, links = zip(*by_rel_orbit[best_rel_orbit])

    # some of the real subsidence areas go into training but some also go into validation 
    # the set variable decides into which region the data goes
    # Generally:  0 = train    1 = dev   2 = val
    if set == 0:
        save_path = TRAIN_DIR / row["location"] / str(orbit)
    elif set == 1:
        save_path = DEV_DIR / row["location"] / str(orbit)
    elif set == 2:
        save_path = VAL_DIR / row["location"] / str(orbit)

    # creating the directory where the data goes
    save_path.mkdir(parents=True, exist_ok=True)

    for hit,link in zip(hits, links):
         filename = hit["filename"]
         with requests.get(link, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(save_path / filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)


#============================================================#
# Processing
#============================================================#

def download_egms(real_sub_path, token_path):

    # Loading the real subsidence cases
    real_subsidence = gpd.read_file(real_sub_path)

    # Compute the bounding boxes for every real subsidence region
    t2projected = Transformer.from_crs(real_subsidence.crs, "EPSG:3035", always_xy=True)
    t2wgs84 = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    real_subsidence["bbox"] = real_subsidence.apply(point_to_bbox, axis=1, result_type="reduce", t2projected=t2projected, t2wgs84=t2wgs84)

    # get the access token for the egms api
    access_token = get_access_token(token_path)
    headers = {"Authorization" : f"Bearer {access_token}", "Accept" : "application/json"}

    # query the archive for overlapping bursts
    real_subsidence[["query_results", "download_links"]] = real_subsidence.apply(query_egms, axis=1, headers=headers, result_type = "expand")

    # Download the products with filter before
    # the set variable decides into which region the data goes
    # Generally:  0 = train    1 = dev   2 = val
    real_subsidence.apply(pull_egms, axis = 1, headers=headers, orbit = "ascending", set = 2)
    real_subsidence.apply(pull_egms, axis = 1, headers=headers, orbit = "descending", set = 2)



def download_gnss():
    pass

def download_gse():
    pass

def download_miscellaneous():
    pass


#============================================================#
# Main
#============================================================#

def main():

    ##############################
    # Check necessary files are there

    if not os.path.exists(Path(__file__).parent / "../../assets/real_subsidence_pointbased.gpkg"):
        raise FileNotFoundError("The real subsidence regions are missing!")
    
    ##############################
    # Creating the file system for the project 
    create_file_system()

    ##############################
    # Downloading the data
    download_egms(Path(__file__).parent / "../../assets/real_subsidence_pointbased.gpkg", 
                  Path(__file__).parent / "../../assets/token.jwt")
    ##############################
    # Pre-Processing the data


if __name__ == '__main__':
    main() 
