# ==============================================================================
# Following script removes any image that contains even a single pixel as nan
# This is to make processing pipeline easier without implementing any interpolation techniques
# However this can be added at later stage
# ==============================================================================
import os 
from glob import glob
import numpy as np
import rasterio
import shutil
from typing import List

from tqdm import tqdm
import argparse

def get_missing_percentage(img_paths:List[str],num_bands:int):
    """This is an external function explore if images contain high number of missing pixels"""
    # Construct an array to store missing values for each band, has a shape (num_imgs, num_bands)
    perc_miss_arr = np.zeros([len(img_paths), num_bands])
    # Loop over all images
    for idx in tqdm(range(len(img_paths))):
        img_p = img_paths[idx]
        with rasterio.open(img_p) as ds:
            img = ds.read()
        # get channels, Height and Width per image
        C,W,H = img.shape
        # Total nan count per band
        nan_counts_img = np.isnan(img).reshape(13,-1).sum(axis = 1) #(1, num_bands)
        #  Percentage of missing values in each band for each array
        perc_miss_arr[idx,:] = nan_counts_img / (W * H)

    return perc_miss_arr 

def remove_images_with_missing_gt_percentage(img_paths:List[str],archive_path:str,threshold:float = 0.05)->None:
    """Removes images if greater than a defined percentage"""
    if not os.path.exists(archive_path):
        os.mkdir(archive_path)

    for idx in tqdm(range(len(img_paths))):
        img_p = img_paths[idx]
        with rasterio.open(img_p) as ds:
            img = ds.read()
        # get channels, Height and Width per image
        C,W,H = img.shape
        # Total nan count per band
        nan_count_img = np.isnan(img).reshape(13,-1).sum(axis = 1) #(um_bands,) i.e (13,)
        # Per band missing value 
        nan_count_perc_b = nan_count_img / (W * H) #(num_bands,)
        # Get a single value to make life easier
        nan_count_perc = nan_count_perc_b.mean()
        # more than this percentage 
        if nan_count_perc > threshold: 
            print(f"moving file : {os.path.basename(img_p)} with avg missing percentage : {nan_count_perc} ...")
            shutil.move(src = img_p, dst = os.path.join(archive_path, os.path.basename(img_p)))

def remove_any_missing(img_paths:List[str], archive_path:str)->None:
    """
    Unlike the 'remoce_images_with_missing_gt_percentage' any image with missing values
    This isnt ideal but inpainting will not be required. If you want more images for downstream training
    use func above
    """
    if not os.path.exists(archive_path):
        os.mkdir(archive_path)

    clean_images = 0
    nan_images = 0
    for idx in tqdm(range(len(img_paths))):
        img_p = img_paths[idx]
        with rasterio.open(img_p) as ds:
            img = ds.read()
        # get channels, Height and Width per image
        C,W,H = img.shape
        if np.isnan(img).any():
            nan_images += 1
            print(f"moving file : {os.path.basename(img_p)} with atleast a single nan value")
            shutil.move(src = img_p, dst = os.path.join(archive_path, os.path.basename(img_p)))
        else:
            clean_images += 1


    print(f"Total clean images : {clean_images}, Nan images : {nan_images}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Argument Parser for handling missing values")
    parser.add_argument("-data_root", type = str, help = "path to root folder containing images that need misisng value sorting")
    parser.add_argument("-archive_root", type = str, help  = "path to where missing data will be stored, path doesnt exist will create folder")
    parser.add_argument("-miss_percentage", type = int, default = 5, help = "missing percentage, i.e 5 to remove images with more than 5% missing")
    args = parser.parse_args()

    # Get a list of images from the root folder
    img_paths = glob(os.path.join(args.data_root, "*"))
    # ------ Comment out if missing images removed to prevent looping agaain ----- #
    #remove_images_with_missing_gt_percentage(img_paths, args.archive_root, args.miss_percentage / 100)

    # --------------- To remove an image with even a single missing -------------- #
    remove_any_missing(img_paths,args.archive_root)

