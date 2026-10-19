#######################################
# 03_build_training_dataset

##############################################################################
#============================================================================#
# 1) Import Statements
#============================================================================#
##############################################################################

import s01_download_data as s01



##############################################################################
#============================================================================#
# 2) Helper Functions
#============================================================================#
##############################################################################

################################################
#===== Section 1: Shared Helper Functions =====#
################################################

def select_area_and_buffer_burst_edges():
    # This function buffers the burst edges so points too close to the edges are not included
    pass

def find_n_points():
    # Finds the appropriate number of points per burst
    pass

def select_center_nodes():
    # Finds the center nodes for the minigraphs
    pass

def select_candidate_pool():
    # selects the other points around the center nodes 
    pass

def retrieve_feature_information():
    # retrieves the edge and node feature information for a candidate pool

def build_tensor_bundle():
    # builds the minigraphs in a particular way
    pass




##############################################################################
#============================================================================#
# 3) Processing
#============================================================================#
##############################################################################

def build_train_dataset_per_burst():
    # per set per region there are multiple bursts. This function is applied to every
    # individual burst building a training dataset for it. 
    # saves this training dataset as cache to build the global dataset from many small ones later.
    # INPUT: takes a path to a csv
    # OUTPUT: saves a training dataset per csv as cached data 

    # 1) Determine actual area from which the points are drawn. many bursts only overlap little with the actual 
    # area of interest. Especially in the real subsidence areas, this could lead to many points being selected from outside the aoi.
    # Therefore the area that actually overlaps with the aoi must be used to extract points
    # Take out the points at the very edges because they might distort the dataset
    # INPUT: takes the loaded egms file (lazy using polars is best here)
    # OUTPUT: returns only that part of the dataset that meets criteria
    select_area_and_buffer_burst_edges()

    # 2) Select appropriate number of points for the remaining area (The area that remains from previous step can have
    # very different size so we cannot pass one fixed number of points to every burst)
    # INPUT: Takes the datset from previous function 
    # OUTPUT: Returns a scalar that says how many center nodes should be found in that burst
    find_n_points()

    # 3) Now the center nodes are selected using the land cover stratification (lc strate might not be sufficient to actually ensure a 
    # good spatial distribution. PS points tend to not exist in forests, fields etc per definition, so need to look into that. apply grid and 
    # number of points per grid cell maybe to ensure good distribution over the region. Also minimum distance)
    # INPUT: Takes the dataset from before and the scalar
    # OUTPUT: Returns the center nodes (maybe as row indices, not clear yet)
    select_center_nodes()

    # 4) Selects the other points for the candidate pool
    # Loops over all the center nodes?
    # INPUT: Takes the dataset from before and the center nodes 
    # OUTPUT: Returns per center node, the candidate pool in form of a dataframe. 
    select_candidate_pool()

    # 5) Node and edge features need to be determined 
    # not sure whether that is completely possible at this stage
    # INPUT: A candidate pool dataframe
    # OUTPUT: The same dataframe with the node and edge features attached
    retrieve_feature_information()

    # 6) Builds the minigraphs from the pools
    # INPUT: A candidate pool as df including features
    # OUTPUT: Returns tensor bundle with all the information needed to build a new mini graph per epoch
    build_tensor_bundle()
    pass

##############################################################################
#============================================================================#
# 4) Main
#============================================================================#
##############################################################################


def main():
    
    # Import the gdf again to have access to all the file paths
    study_areas_gdf = s01.study_areas_io("read", base_path = Path(__file__).parent / "../../assets/study_areas_gdf")



    # !!!! Determine the paths to every csv in ascending and descending in the region folder
    # So loop over the region_paths with apply in gdf, build a list to every egms csv and then loop over every csv using the function below
    # BUILDING the tensor bundle per burst. Takes in one region row from gdf and builds paths to csv from that 
    # !!!! Do not apply to validation regions, since they only need the gnss stations.
    # INPUT: path to a egms csv
    # OUTPUT: Saved tensor bundle
    build_train_dataset_per_burst()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--projectdir", required=True)

    args = parser.parse_args()

    main() 

