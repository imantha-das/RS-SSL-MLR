# ==============================================================================
# Assigns path to corresponding images for each lat/lon points from malaria dataset
# This script reqiores all images to be already extracted to respective folders using
# "ext_or_msk_ROI.py" prior to using this script.
# "ext_or_msk" also performs this task while extracting however it just does for
# a particular folder. Here we aim to apply this 
# ==============================================================================

import os
from glob import glob

import numpy as np
import pandas as pd

from typing import List

from datetime import datetime
from termcolor import colored
from tqdm import tqdm
    
def get_imgp_for_latlon_pts(img_paths:List[str], df:pd.DataFrame, dataset:str)-> pd.DataFrame:
    """
    Looks for sample name and aligns corresponding image names with paths
    Inputs 
        - data_fold : Paths to Images (These must be extracted images containing sample name as the siffix)
        - df : Dataframe containing sample_name, lat, lon (This could be an intermediate dataframe where
        image paths for a certain folder has been completed. Note there are seperate  folders and we need to apply
        this function to each of them)
    """
    def rearrange_by_date(files:List[str], dataset:str):
        """
        This function will rearrange files based on date so the we can access the first image easily.
        Note this dataset contains multiple images been taken on the same day. We will not rearange
        by the suffix of the image files (2,4 etc.) as that doesnt mean much. 
        """
        match dataset:
            case "drn":
                ext_dates = list(map(lambda x : datetime.strptime(x.split("_")[1], "%Y%m%d"), files))
                ext_dates_sort_idx = np.argsort(ext_dates)
                return [files[i] for i in ext_dates_sort_idx]
            case "sen2a":
                #todo : You need re-extract sentinel images with date included in the file name.
                return files
            case _:
                raise(ValueError("Pass sen2a or drn for variable dataset"))
        
    assert set(["Sample", "GPS_X", "GPS_Y"]).issubset(df.columns.values), "Columns Sample/GPS_X/GPS_Y cannot be found in dataframe"
    # Get the column name which is folder name containg the images.
    colname = img_paths[0].split("/")[-2] # The second entry once splited is the parent folder of path

    # Construct a new column with parent dolder containing images as the name and add an empty list to each entry
    df[colname] = df.apply(lambda _ : [], axis = 1)
    
    # Make "Sample" column the index
    df.set_index("Sample", inplace = True)

    # Loop through the  dataframe index which are the Samples and check if any of the images have a corresponding sample extension.
    for sname_fdf in tqdm(df.index):
        for img_p in img_paths:
            file_name = img_p.split("/")[-1]
            sample_name_with_ext = file_name.split("_")[-1]
            sname_ff = sample_name_with_ext.split(".")[0]
            
            # If the images have the same sample extension as dataframe sample ...
            if sname_ff == sname_fdf:
                df.loc[sname_fdf][colname].append(file_name)
        # we need to rearrange here based on date 
        #! Note there are multiple images for the same date (i.e 02, 04) we will be not sorting based on this though    
        if len(df.loc[sname_fdf][colname]) > 0:
            rearranged_files = rearrange_by_date(df.loc[sname_fdf][colname], dataset)
            df.loc[sname_fdf][colname] = rearranged_files


    df.reset_index(inplace = True)
    return df


if __name__ == "__main__":
    # Get Paths for Drone Images
    pts_imgp_p = "data/interim/sshsph_mlr/geopts_and_imgp_v0.csv"
    data_fold_drn_p = "data/interim/sshsph_drn/drn_c3c4_256x_ext"
    df_pts_imgp = pd.read_csv(pts_imgp_p)
    img_paths = glob(os.path.join(data_fold_drn_p, "*"))
    
    df_pts_imgp = get_imgp_for_latlon_pts(img_paths, df_pts_imgp, "drn")

    # Get Paths for Sentinel Images
    data_fold_sat_p = "data/interim/gee_sat/sen2a_c3_256x_ext"
    img_paths_sat = glob(os.path.join(data_fold_sat_p, "*"))
    df_pts_imgp = get_imgp_for_latlon_pts(img_paths_sat, df_pts_imgp, "sen2a")

    # Print data the only has atleast one image from either dataset
    #print(df_pts_imgp[df_pts_imgp["drn_c3c4_256x_ext"].apply(len) > 0])
    print(df_pts_imgp[(df_pts_imgp["sen2a_c3_256x_ext"].apply(len) > 0) | (df_pts_imgp["drn_c3c4_256x_ext"].apply(len) > 0)] )

    # Save dataframe
    df_pts_imgp.to_csv("data/interim/sshsph_mlr/mlr_geopts_imgp_v1.csv", index=False)