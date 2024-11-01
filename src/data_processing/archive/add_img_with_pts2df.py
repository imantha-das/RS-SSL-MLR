# ==============================================================================
# Desc : This scripts provides functions to identify points that coincide with 
# images. 
# NOTE : There are sample points where there is many images that coincide. This
# is due to multiple images of the same location been taken during different
# time frames. Upon inspection (not thorougly) most images that are at different
# time scales display similar noise. (However there are cases which are not). Due
# to this reason the last image is selected
# ==============================================================================
import os
import pprint
from glob import glob
from typing import List, Dict

import pandas as pd
import numpy as np

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import cv2

from termcolor import colored
import argparse

parser = argparse.ArgumentParser(description = "Populate dataframe with corresponding images if present")
parser.add_argument("-pts_p" , type = str, help = "path to dataframe containing lat/lon values")
parser.add_argument("-img_p", type = str, help = "path to folder containing images")
parser.add_argument("-pts_dst_p", type = str , help = "path to where you want to store dataframe", default = "data/interim")


def get_point_coincide_images(df:pd.DataFrame,spl2imgs:Dict[str, List[str]], image_root:str)->pd.DataFrame:
    """Returns a dataframe with data for only rows that contain images
    (Not all images coincide with collected malaria data points)
    Inputs
        - df : Dataframe containing features and targets for malaria
        - image_paths : List of paths to extracted images (256x image patches centered based on lat/lon malaria values 
        - root : Path to root folder containing images
    NOTE : IMAGE PATHS CONTAIN THE SAME SAMPLE FROM DIFFERENT DATES. DUE TO THIS
    WE WILL ONLY CONISDER THE FIRST ONE FOUND IN THE IMAGE LIST. SPATIAL FEATURES
    MAY HAVE CHANGED OVER TIME THOUGH
    
    Outputs
        - df : returns the Dataframe containing features and targets with a new column containing path to image, if
        a particular sample has an image else this column will filled with a NA value.
    """

    # Create new column called image_paths to save images
    if "image_path" not in df.columns.values:
        df["image_name"] = pd.NA
        df["image_path"] = pd.NA

    if not df.index.name == "Sample":    
        df.set_index("Sample", inplace = True) 
    
    for sample_name,fnames in spl2imgs.items():
        # If there is more than one image per sample
        if len(fnames) > 1:
            # Select image name based on criteria
            selected_fname = select_image_based_on("last", fnames)
            # Store image path in dataframe
            df.loc[sample_name, "image_path"] = os.path.join(image_root, selected_fname)
            # Store selected filename in dataframe - for easy analysis
            df.loc[sample_name, "image_name"] = selected_fname
        # If there is only one imape per smaple
        else:
            if sample_name in df.index:
                # Store path in the dataframe
                df.loc[sample_name, "image_path"] = os.path.join(image_root, fnames[0])
                # Store filename in the dataframe - This is just for easy analysis
                df.loc[sample_name, "image_name"] = fnames[0]
            else:
                raise Exception(colored(f"Couldnt find {sample_name} in index", "red"))
            
    return df

def select_image_based_on(criteria:str, fnames:List[str]):
    """
    If there are multiple image for a sample select an image based on a criteria
    Inputs
        - fnames : Filenames of images coinciding with points (in a list)
        - criteria : criteria used to select an image (i.e first or last image)
            - todo : can form criterial such as select image based on mean/std of image
    """
    match criteria:
        case "first":
            return fnames[0]
        case "last":
            return fnames[-1]
        case _:
            raise Exception("Criteria not found!")

     
# ==============================================================================
# Helper Functions to identify recoccuring images
# ==============================================================================  
def get_all_images_per_sample(image_paths:List[str])->Dict[str, List[str]]:
    """
    Desc : There are at time several images (with different dates) for each sample
    This function returns a dictionary mapping sample name to all respective image names.
    i.e {
        '5YK3': ['20160818_01_5YK3.tif', '20160818_02_5YK3.tif']
        'ZEF3': ['20160323_02_ZEF3.tif'],
        ...
    }
    Inputs 
        - image_paths : Path to repository containing all images
    Outputs
        - spl2imgs : Dictionary containing, sample_name : fname 
    """
    spl2imgs = {}
    for img_p in image_paths:
        # Get basename from absolute path as "sample name" is stored here
        fname = os.path.basename(img_p)
        # Get "sample name" from fname : i.e "20141222_02_EREW.tif" where we want "EREW"
        sample_name = fname.split("_")[-1].split(".")[0]
        if sample_name in spl2imgs:
            spl2imgs[sample_name].append(fname)
        else:
            spl2imgs[sample_name] = [fname]

    return spl2imgs

def get_color_stats2(img_p):
    """
    Computes image color mean and standard deviation.
    The suffix 2 is due to another function existing in clean_noisy_images.py
    which does something slightly different.
    """
    # Read image from path
    img = cv2.imread(img_p)
    # Compute channel mean - across all channels
    cmu = img.mean()
    # Compute channel means - across seperate channels
    cmu3 = img.mean(axis = (0,1))
    # Compute standard deviation - across all channels
    cstd = img.std()
    # compute standard deviation - across seperate channels
    cstd3 = img.std(axis = (0,1))
    
    return img, cmu, cmu3, cstd, cstd3
    
def plot_image(img, mean, std, fig, row, col):
    """Plots an image with mean and standard deviation"""
    fig.add_trace(go.Image(z = img), row = row, col = col)

# -------------------------- End of Helper functions ------------------------- #

if __name__ == "__main__":

    args = parser.parse_args()
    
    # Load malaria dataset
    df = pd.read_csv(args.pts_p)

    # Path to folder containing images
    #images_root = "data/SSHSPH-RSMosaics-MY-v2.1/images/pts_centered_patches"
    image_paths = glob(os.path.join(args.img_p, "*"))

    # ----------------- Store all images that conicde with points ---------------- #
    # Get the corresponding image filename(s) for each sample
    spl2imgs = get_all_images_per_sample(image_paths)
    #pprint.pp(spl2imgs)

    # Get images that coincides with points
    df_con_pts = get_point_coincide_images(df, spl2imgs, args.img_p)
    print(df_con_pts)
    print(df_con_pts[df_con_pts["image_path"].notna()])

    df_con_pts.to_csv(args.pts_dst_p)

    # ------------------------------------- x ------------------------------------ #

    # THIS CODE CAN BE USED TO PLOT IMAGES WITH MORE THAN TWO SAMPLES TO COMPARE
    # ------- Filter samples where there is more than 2 images per samples ------- #
    # sample_images_gt2 = list(filter(lambda v: len(v) > 1, spl2imgs.values()))

    # for image_names in sample_images_gt2:
    #     nrow = 1; ncol = 1
    #     means = []
    #     stds = []
    #     p = make_subplots(rows = 1, cols = len(image_names))
    #     for image_name in image_names:
    #         # Join root folder path with image name
    #         img_p = os.path.join(args.img_p, image_name)
    #         # Get image, mean & std across all channels (for seperate channels use 4/5 postional aruments)
    #         img, cmu, _, cstd, _ = get_color_stats2(img_p)
    #         # Mean and Std
    #         means.append(np.round(cmu,2)) ; stds.append(np.round(cstd,2))
    #         # Plot 
    #         plot_image(img, 23,23, p, nrow, ncol)
    #         ncol += 1

    #     p.update_layout(title = f"means : {means}\nstds : {stds}")
    #     p.show()
    # ---------------------------------------------------------------------------- #