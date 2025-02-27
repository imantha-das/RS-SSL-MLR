# ==============================================================================
# Desc : This script is used to identify and clean RS Images which conatins a 
# lot of noisy images. To do that the sciprt provides functions to compute image
# stats and move files to desired folders.

# Notes for future use
# There were 10118 extracted images from sentinel2A of which 1011 images contained
# NaN values however a large percentage had only 1% NaN's. We have removed all
# of them but can be used if they are required. The main reson to remove images is
# that there duplicate (more or less) extracted images from different times 


# ==============================================================================
import os 
from glob import glob 
import skimage as ski 
import numpy as np
from sklearn.preprocessing import StandardScaler
import rasterio
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from typing import List, Tuple
import plotly.io as pio 
from tqdm import tqdm
import shutil
from termcolor import colored

import argparse

parser = argparse.ArgumentParser(description = "Clean Noisy Images")
parser.add_argument("-src_fold", type = str, help = "Input folder containing images that require cleaning")
parser.add_argument("-dst_fold", type = str, help = "Input folder where to archive noisy images", default = "data/interim/test")

pio.templates.default = "plotly_white"

def get_image(path:str):
    """Desc : Returns an image from given path"""
    with rasterio.open(path) as ds:
        img = ds.read([1,2,3])
    return img

def get_color_stats(path):
    """
    Desc : Returns the color distribution and mean
    """
    # Load image
    img = get_image(path) #np.array #(c,w,h) 
    # Compute mean across all channels
    cmu = img.mean()
    # Compute mean seperately across channe;s
    cmu3 = img.mean(axis = (1,2)) #(3,)
    # Load image in Plotly
    img_rsp = np.moveaxis(img, source = (0,1,2), destination=(2,0,1))
    img_rsp = (img_rsp - (img_rsp.mean())) / img_rsp.std()

    p1 = go.Image(z = img_rsp, name = str(cmu)) # for plotting we need (w,h,c)
    # Save histograms as plotly traces as we need to add them to a subplot
    p2 = []
    for c in range(0,3):
        p2.append(go.Histogram(x = img[c, :, :].ravel()))

    return cmu, cmu3, (p1,p2)

def test_img_colorvals(image_paths, num_imgs):
    """Desc : Plots specified number of images randomly from dataset"""
    # Select on some image paths randomly
    img_paths = [np.random.choice(image_paths) for i in range(0, num_imgs)]
    # Make subplots : Give some names to argument "subplot_titles"
    p = make_subplots(rows = 2, cols = num_imgs, subplot_titles = [f"plot{i}" for i in range(num_imgs)])
    cmus = []
    for i in range(0, num_imgs):
        # (mean of image, mean of 3 channels seperate, (plotly figure, [plotly figures]))
        cmu, cmu3 , (p1,p2) = get_color_stats(img_paths[i])
        # Add the plotly figure as trace to the subplot
        p.add_trace(p1, row = 1, col = i+1)
        # Loop over list containing multiple figures and add them as traces to subplot
        for p_ in p2:
            p.add_trace(p_, row = 2, col = i+1)
        # Update the titles by indexing and updating the index
        p.layout.annotations[i].update(text=str(cmu))
    return p

def move_nan_to_dst(img_paths, dst_fold):
    """Gee Satelite data have images with NaN which may require to be removed if a large percentage contains NaN's"""
    def count_nans(img):
        """Counts the number of NaNs in an image"""
        return np.count_nonzero(np.isnan(img))
    
    # Create destination folder if not exists
    if not os.path.exists(dst_fold):
        os.mkdir(dst_fold)
    
    total_nan_counts = 0
    total_nonan_counts = 0
    for path in tqdm(img_paths):
        img = get_image(path)
        num_nans = count_nans(img)
        #* We are moving all images that have nan's. 
        if num_nans != 0:
            print(f"{path} : {num_nans/np.prod(img.shape):.3f}%")
            shutil.move(src = path, dst = dst_fold)
            total_nan_counts += 1
            
        else:
            total_nonan_counts += 1

    print(f"total images : {len(img_paths)}, total images with nan's : {total_nan_counts}, total clean images : {total_nonan_counts}")



def get_image_means(img_paths):
    """Desc : Test function to identify by visulaizing which images below cmu (mean across all channels) value should be removed"""
    cmus = []
    cmu0_paths = []
    cmu50_paths = []
    cmu75_paths = []
    cmu100_paths = []
    for path in tqdm(img_paths):
        cmu, _, (_,_) = get_color_stats(path)
        cmus.append(cmu)
        if cmu == 0:
            cmu0_paths.append(path)
        if cmu > 0 and cmu < 50:
            cmu50_paths.append(path)
        if cmu > 50 and cmu < 75:
            cmu75_paths.append(path)
        if cmu > 75 and cmu < 100:
            cmu100_paths.append(path)


    p = go.Figure()
    p.add_trace(go.Histogram(x = cmus))
    return cmus, cmu0_paths, cmu50_paths, cmu75_paths, cmu100_paths, p
    
def archive_images_less_than_cmu(img_paths: List[str],cmu_thresh:int, archive_folder_path:str):
    """
    Desc : Removes files beyond a cmu (channel mu/mean) threshold
    Inputs
        - root : root directory, consider this as dataset folder
        - img_path : list containing the path for each image
        - cmu_thresh : mean across all channels of image which decides which images get thrown away
    """
    if not os.path.exists(archive_folder_path):
        os.mkdir(archive_folder_path)

    for path in tqdm(img_paths):
        cmu, _, (_,_) = get_color_stats(path)
        if cmu < cmu_thresh:
            shutil.move(src = path, dst = archive_folder_path)
            


if __name__ == "__main__":
    #data_root = "data/SSHSPH-RSMosaics-MY-v2.1/images"
    #img_repo = "channel3_256x256p"
    #img_paths = glob(os.path.join(data_root,img_repo,"*"))
    args = parser.parse_args()
    src_fold = args.src_fold # folder containing datset to be preprocessed
    dst_fold = args.dst_fold # folder to store noisy images, if folder doesnt exist it will created when calling achive_images_less_than_cmu function
    img_paths = glob(os.path.join(args.src_fold, "*"))

    # Count Number of Nans in image / Move them to archive folder
    # move_nan_to_dst(img_paths, dst_fold)


    # Randomly pick n images from the defined repository
    pn = test_img_colorvals(img_paths, 4)    
    #pn.show()
    pn.write_image("tmp/test_img_colorvals.png")

    # Test function which computes the mean of images across all channels 
    # cmus,cmu0_paths, cmu50_paths,cmu75_paths, cmu100_paths, pn2 = get_image_means(img_paths)
    # #pn2.show()

    # Plot images to get an idea of below which cmu, the images need to be removed
    # pn3 = test_img_colorvals(cmu0_paths,4)
    # pn4 = test_img_colorvals(cmu50_paths,4)
    # pn5 = test_img_colorvals(cmu75_paths, 4)
    # pn6 = test_img_colorvals(cmu100_paths, 4)

    # pn3.show()
    # pn4.show()    
    # pn5.show()
    # pn6.show()

    # Number of images below a defined threshold
    # print(len(cmu0_paths),len(cmu50_paths),len(cmu75_paths), len(cmu100_paths), len(cmus))

    # Move images that have bad mean across channels, most of these are images that are completely black.
    #archive_images_less_than_cmu(img_paths, 75, dst_fold)

 