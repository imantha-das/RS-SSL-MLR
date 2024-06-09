# ==============================================================================
# Desc : Script to patch large RS images into small portions
# ==============================================================================

import os
import argparse 
from os import path 
from glob import glob 
import numpy as np
import PIL
from PIL import Image
import skimage as ski
import matplotlib.pyplot as plt
from patchify import patchify
from termcolor import colored

PIL.Image.MAX_IMAGE_PIXELS = 1000000000

parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--folder_path", type = str, default = "channel3")
parser.add_argument("-sfp", "--save_folder_path", type = str, default = "test")
parser.add_argument("-ps", "--patch_size", type = int, default = 256)
parser.add_argument("-rs", "--resize", type = int, default = 0)

def patch_image(path_to_img:str, patch_size = 512,resize = None):
    img = Image.open(path_to_img) #For some reason has shape of (W,H) and no C, but when converted to numpy gets a C dim
    if resize != 0:
        resimg = img.resize(resize) 
        resimg_arr = np.asarray(resimg) #(W,C,H)
        assert resimg_arr.ndim == 3
        w,h,c = resimg_arr.shape
        print(f"Resizing image of shape : {np.asarray(img).shape} -> {resimg_arr.shape}")
        # patches.shape = (15, 15, 1, 512, 512, 3)
        patches = patchify(resimg_arr, patch_size = (patch_size, patch_size, c), step = patch_size)
    else:
        img_arr = np.asarray(img) #(W,C,H)
        assert img_arr.ndim ==3 # The image before converting to numpy has only 2 dims when usinging Image.open, so ensure it has 3 dims after numpy
        w,h,c = img_arr.shape
        patches = patchify(img_arr, patch_size = (patch_size, patch_size, c), step = patch_size)
    
    patch_list  = []
    for p_row in range(patches.shape[0]):
        for p_col in range(patches.shape[1]):
            patch = patches[p_row,p_col,0,:,:,:]
            patch_list.append(patch)
        
 
    return patch_list, img

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

def convert_and_save(arr, f_prefix, idx):
    img =Image.fromarray(arr)
    fname = f"{f_prefix}_{idx}.tif"
    print(f"saving image to : {fname}")
    img.save(fname)

if __name__ == "__main__":
    args = parser.parse_args()
    # Folder where the images are store
    data_root = "data/SSHSPH-RSMosaics-MY-v2.1/images"
    DATA_FOLDER = os.path.join(data_root, args.folder_path)
    SAVE_FOLDER = path.join(data_root, args.save_folder_path)
    PATCH_SIZE = args.patch_size
    RESIZE = args.resize
    
    # Creare a folder to store the patches
    if not path.exists(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)

    # Loop throught the folder containing all the images
    for folder in glob(path.join(DATA_FOLDER, "*")):
        # Loop through each of the files in the image folder ( there are many including the tif ...)
        for file in glob(path.join(folder, "*")):
            if file.endswith(".tif"):
                # Get the patches list and orginal image (if you want to have a look)
                patch_list, img = patch_image(file, patch_size = PATCH_SIZE, resize = RESIZE)
                # Save patches in the designated folder
                f_prefix = path.basename(path.dirname(file)) # get the foldername, i.w 20140429_03
                f_prefix = path.join(SAVE_FOLDER, f_prefix) # add path to begining, "data/SSHSPH.../test_ch3_p/20140429_03"
                print(f"file name prefix : {f_prefix}")
                for i in range(len(patch_list)):
                    convert_and_save(patch_list[i], f_prefix, i)

        

