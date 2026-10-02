#######################################
# 02_preprocessing

##############################################################################
#============================================================================#
# 1) Import Statements
#============================================================================#
##############################################################################

import argparse 

##############################################################################
#============================================================================#
# 2) Define Global Variables
#============================================================================#
##############################################################################

parser = argparse.ArgumentParser()

parser.add_argument("--datadir", required=True)

args = parser.parse_args()





##############################################################################
#============================================================================#
# 2) Helper Functions
#============================================================================#
##############################################################################

################################################
#===== Section 1: Shared Helper Functions =====#
################################################

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

