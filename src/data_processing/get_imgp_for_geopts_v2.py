# ==============================================================================
# Performs the same process as get_imgp_for_geopts.py except for dataset 2.
# Here the extracted images hadnt saved the timestamp. Due to restrictions on
# time to procude anoter dataset (extraction takes a long time), we will use
# pick the first image in the event there is multiple images. The reason why we 
# need to run this script after extracting ROI using ext_or_mask_ROI.py is due
# to cleaning image chip whci has missing values.
# ==============================================================================
import os
import pandas as pd
from typing import List
from glob import glob
from tqdm import tqdm

def get_imgp_for_latlon_pts(img_paths:List[str], df:pd.DataFrame, store_colname):
    """
    Inputs
    - img_paths : A list of image paths
    - df : Dataframe containing GPS coordinames, samplesname etc.
    """
    # This will store the remaining cleaned images available in root
    df[store_colname] = df.apply(lambda _ : [], axis = 1)

    # Make "Sample" column the index
    df.set_index("Sample", inplace = True)

    # loop through the images and check if the sample
    for sname_fdf in tqdm(df.index):
        for img_p in img_paths:
            file_name = img_p.split("/")[-1]
            sample_name_with_ext = file_name.split("_")[-1]
            sname_ff = sample_name_with_ext.split(".")[0]
            # If the images have the same sample extension as dataframe sample ...
            if sname_ff == sname_fdf:
                df.loc[sname_fdf][store_colname].append(file_name)

    df.reset_index(inplace = True)
    return df
        

if __name__ == "__main__":
    df = pd.read_csv("data/interim/sshsph_mlr/lfmykmns_mlr_geopts_imgp_c13_256x_v0.csv")
    data_fold = "data/interim/gee_sat/sen2a_c13_256x_ext"
    img_paths = glob(os.path.join(data_fold, "*"))
    col_name = "sen2a_c13_ext_paths_v2"
    df = get_imgp_for_latlon_pts(img_paths, df, col_name)
    # Save dataframe 
    df.to_csv("data/processed/sshsph_mlr/lfmykmns_mlr_geopts_imgp_c13_256x_v1.csv", index = False)