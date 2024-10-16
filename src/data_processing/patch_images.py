# ==============================================================================
# Desc : Script to patch large RS images into small portions. This script can 
# handle images within the root folder or images within subdirectories which are 
# within root folder.

# Note : It might be a good idea to normalize Sentinel2A images to fall between 0-255
# Drone Images are already in this range
# ==============================================================================

import os
import argparse 
from glob import glob 
from tqdm import tqdm
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from patchify import patchify
from termcolor import colored
import cv2


def get_patched_images(path_to_img:str, save_img_at:str, patch_size:int = 256, normalize:bool = False, from_subdir:bool = False)->None:
    """
    Performs image patching and saves the resulting image chips in a specified folder.
    This function accounts for the Geo-spatial positioning of the image chips. i.e the
    image chips should align with the original image if loaded using a gis software
    Inputs
        - path_to_img : file path to image that needs processing
        - save_img_at : folder where image chips will be written to
        - patch_size : desired patch size required, default = 256
        - normalize : Use cv2 to normalize images. This is useful for sentinal images where
        RGB values are not in the range of 0-255
        - from_subdir : There are images that are containined in the root folder or within
        subdirectories inside the root folder. This argument is ONLY used for naming conventions.
        If image is inside root, the file name would be used as a prefix. If within a subdir
        the naming prefix will the subdirectory name.
    """
    # Use rasterio to open images
    with rasterio.open(path_to_img) as ds:
        img = ds.read() #read() function returns values (C,W,H) in numpy format
        bands = ds.count # no of dimensions in the image
        profile = ds.profile.copy()
        original_transform = ds.transform

    # Ensure we have only 3 bands else some additional steps need to be included
    assert bands == 3, colored(f"Num bands : {bands} not equal to 3", "red")
    # reshape image as rasterio shape comes in the form (C,W,H)
    img = np.moveaxis(img, source = [0,1,2], destination=[2,0,1]) #(C,W,H) -> (W,H,C)
    # Normalize image
    if normalize:
        print("Normalizing Image")
        # Normalize image values so that they will be in the range between 0-255
        img = cv2.normalize(img,None,alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
    # Get the number of channels, which chould be 3
    _,_,C = img.shape
    # Patchify object
    patches = patchify(img, patch_size=(patch_size, patch_size, C), step = patch_size) # example shape : (42, 9, 1, 256, 256, 3)

    # Loop through the rows and cols of the patchify object and store the patched images in a list
    if from_subdir:
        f_prefix = os.path.basename(os.path.dirname(path_to_img))
    else:
        f_prefix = os.path.basename(path_to_img).split(".")[0] 

    # Create folder to store images if directory doesnt exist
    if not os.path.exists(save_img_at):
        os.mkdir(save_img_at)
        
    for p_row in range(patches.shape[0]):
        for p_col in range(patches.shape[1]):
            patch = patches[p_row,p_col,0,:,:,:] #(256,256,3)
            # We cannot save in this format, channel needs to be in the first dimension
            patch = np.moveaxis(patch, source = [2,0,1], destination=[0,1,2]) #(3,256,256)
            # To get the geo referenced cooridnate correctly
            new_transform = original_transform * rasterio.Affine.translation(p_col * patch_size, p_row * patch_size)
            profile.update({
                "height" : patch_size, 
                "width" : patch_size, 
                "transform" : new_transform,
            })

            # Give a new file name 
            fname = f_prefix + "_" + f"{p_row}_{p_col}" + ".tif"
            fpath = os.path.join(save_img_at, fname)

            # Write image to folder
            with rasterio.open(fpath, 'w', **profile) as new_ds:
                new_ds.write(patch)

def plot_patches(patch_list):
    rows = int(np.sqrt(len(patch_list)))
    cols = rows

    fig, ax = plt.subplots(nrows = rows, ncols = cols, figsize = (10,10))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax[r,c].imshow(patch_list[idx])
            ax[r,c].axis("off")
            idx += 1

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type = str, help = "path to data directory")
    parser.add_argument("-save_dir", type = str, help = "path to directory where patched images are stored")
    parser.add_argument("-patch_size", type = int, default = 256, help = "size of patches")
    parser.add_argument("-normalize", action = argparse.BooleanOptionalAction)
    args = parser.parse_args()

    DATA_FOLDER = args.data_dir
    SAVE_FOLDER = args.save_dir
    PATCH_SIZE = args.patch_size
    NORMALIZE = args.normalize
    
    for f in tqdm(glob(os.path.join(DATA_FOLDER,"*"))):
        if os.path.isdir(f):
            # find for tif file within subdir
            img_f = list(filter(lambda x: x.endswith(".tif"), glob(os.path.join(f,"*"))))
            assert len(img_f) >0, "Tif file couldnt be found"
            print(f"Patching {os.path.basename(img_f[0])}...")
            get_patched_images(
                path_to_img = img_f[0],
                save_img_at = SAVE_FOLDER,
                patch_size = PATCH_SIZE,
                normalize = NORMALIZE,
                from_subdir = True
            )

        else:
            print(f"Patching {os.path.basename(f)}...")
            get_patched_images(
                path_to_img = f,
                save_img_at = SAVE_FOLDER,
                patch_size = PATCH_SIZE,
                normalize = NORMALIZE,
                from_subdir = False
            )

        

