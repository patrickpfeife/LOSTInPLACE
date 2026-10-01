#######################################
# 02_preprocessing

#============================================================#
# Import Statements
#============================================================#

import polars as pl
import zipfile 


from shapely.geometry import box 

#============================================================#
# Define Global Variables
#============================================================#










#============================================================#
# Helper Functions
#============================================================#

def bbox_per_csv(csv_file):
    """Compute the axis-aligned bounding box of the easting/northing points in one EGMS csv.

    Args:
        csv_file (str or Path): path to the EGMS csv, with `easting` and `northing` columns (EPSG:3035, metres).
    Returns:
        shapely.geometry.Polygon: rectangle covering the min/max easting and northing of the file.
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

def bbox_envelope(individual_bboxes):
    """Merge a folder's per-csv bounding boxes into one geometry and write it to a gpkg in that folder.

    Args:
        individual_bboxes (list[shapely.geometry.Polygon]): bounding boxes from bbox_per_csv, in EPSG:3035.
        folder (Path): folder to write "covered_area.gpkg" into (the folder the csvs came from).
    Returns:
        None. Writes "covered_area.gpkg" to `folder`.
    """
    # takes a list of bboxes from individual egms csv downloads
    # and returns the union of them
    union = union_all(individual_bboxes)
    union = gpd.GeoDataFrame(geometry=[union], crs="EPSG:3035")
    




#============================================================#
# Processing
#============================================================#


# remember to crop the egms coverage to the actual bounding box defined. otherwise, too much surrounding 
# points with barely any real displacement will be introduced through the regions that are actually just 
# supposed to cover the real subsidence

unzip(data_dir)

    # {} creates a set which removes duplicates. So for a folder with 6 csvs, the path is collected only once
    csv_folder = {p.parent for p in Path(data_dir).rglob("*.csv")}

    for folder in csv_folder:
        csvs = folder.glob("*.csv")
        bboxes = []
        for csv_path in csvs:
            bboxes.append(bbox_per_csv(csv_path))
        bbox_union(bboxes, folder)


#============================================================#
# Main
#============================================================#

def main():



if __name__ == '__main__':
    main() 

