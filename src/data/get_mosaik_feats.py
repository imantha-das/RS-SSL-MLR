# ==============================================================================
# Note the following script requires python 3.10/3.11
# ==============================================================================

import os
import sys
from mosaiks import get_features
import numpy as np 
import pandas as pd
import dask.dataframe as dd
from dask import delayed
from pathlib import Path
from tqdm import tqdm
import argparse 


# Resolves a conflict in Geopandas. Improves speed.
os.environ["USE_PYGEOS"] = "0"

def apply_get_features(row:pd.Series):
    lat, lon = row['GPS_Y'], row['GPS_X']
    # We dont have year in this dataset is from 2014 - 2016 based on the image files 
    # Sentinel 2 images are ideal as they have larger resilution (10m) vs landsat (30m) resolution
    # Get the year from the image name
    #* note for each lat/lon there are multiple images that work, we have selected the earliest date.
    #* However this should be too much of a problem as we look at the year rather than date
    if row.image_name is np.nan:
        year:int = 2015
    else:
        year:int = int(row.image_name[:4]) 

    # # Original code uses a Centroid_id but we dont really have one
    # file_name = f"{lat}_{lon}_{year}.csv"
    # # Check if the file already exists
    # if os.path.exists(file_name):
    #     # Read the DataFrame from the file
    #     result = pd.read_csv(file_name)
    #     result['year'] = year  # Ensure the year column is correct
    #     return result

    result = get_features(
        [lat],
        [lon],
        datetime=str(year), # or ["2013-01-01", "2013-12-31"] or ...
        satellite_name = "sentinel-2-l2a", # or "sentinel-2-l2a",
        image_width=10000,
        image_resolution = 10,
        image_bands=['B02','B03','B04'],
        # image_bands=["SR_B2", "SR_B3", "SR_B4"], # for landsat
        model_device = "cpu",
        # parallelize = True,
        # dask_chunksize = 500
    )

    retry_year = year
    result['year'] = retry_year
    # Retry logic if the result contains all NaNs
    while result.loc[:, result.columns != "year"].isna().all().all() and retry_year <= 2020:
        print(f"Features for default year couldnt be found !, Atempting for year : {retry_year + 1} for lat/lon values: {lat,'/',lon} ...")
        retry_year += 1
        result = get_features([lat], [lon], str(retry_year))
        result['year'] = retry_year
    # result type is pad.core.frame.DataFrame which cannot be concatenated, so convert it to a pandas dataframe
    result = pd.DataFrame(result)
    # we want all the features (domain + mosaiks) so concatenate
    row_df = row.to_frame().transpose() # Convert to a dataframe and transpose it to get as a single row dataframe
    row_df = row_df.reset_index(drop = True) # having multiple indices causes concatenating row wise a problem
    result = result.reset_index(drop = True)
    feats = pd.concat([row_df, result], axis = 1)

    return feats

if __name__ == "__main__":

    assert sys.version_info >= (3,10) and sys.version_info < (3,12), "Incorrect python version, use 3.10 or 3.11"

    parser = argparse.ArgumentParser(description="Compute Mosaik features on a given dataset")
    parser.add_argument("-data_file", type = str, help = "Path to csv file containing lat/lon values to which you need mosaik feattures", default="data/processed/mlr_pts_no_missing.csv")
    parser.add_argument("-save_folder", type = str, help = "Folder where mosaik features csv file will be saved", default = "data/interim")
    args = parser.parse_args()
    
    # Get dataframe containing lat long values
    df = pd.read_csv(args.data_file)

    # Compute features and store them
    feats = []
    for i in tqdm(range(0, df.shape[0])):
        row:pd.Series = df.iloc[i]
        feat = apply_get_features(row)
        feats.append(feat)

    X = pd.concat(feats)
    X.reset_index(inplace = True)
    if "index" in X.columns.values:
        X.drop("index", axis = 1, inplace = True)
    if "Unnamed: 0" in X.columns.values:
        X.drop("Unnamed: 0", axis = 1 , inplace = True)

    file_name = os.path.basename(args.data_file).split(".")[0] + "_mosaiks_feats.csv"
    X.to_csv(os.path.join(args.save_folder, file_name))