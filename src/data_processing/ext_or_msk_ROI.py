# ==============================================================================
# The following script is used to extract an image window centering around
# lat/lon values where malaria patients live. Once an image crop is extracted, the
# the original image CAN be masked at this region for later patching IF RQUIRED. 
# This way a validation set can be formed where the model hasnt seen sections of the 
# satelite/drone images where malaria patients reside.
# ==============================================================================

import os 
from glob import glob

import pandas as pd 
import numpy as np

import rasterio
import pyproj

from termcolor import colored
import matplotlib.pyplot as plt
import argparse

from typing import List, Union
from pprint import pprint
from tqdm import tqdm

# ---- Function to get a dict mapping of geo points that fall on an image ---- #

def get_samples_for_imgs(img_root:str, df:pd.DataFrame, img2samples:dict)->pd.DataFrame:
    """
    This function searches for images corresponding to geographic point. 
    """
    proj = pyproj.Proj(init = "epsg:32650")
    for f in glob(os.path.join(img_root, "*")):
        # Image within a subdirectory
        if os.path.isdir(f):
            # Search for tif file
            img_f = list(filter(lambda x: x.endswith(".tif"), glob(os.path.join(f, "*"))))
            assert len(img_f) == 1, colored(f"tif file not found or there are more than 1 tif file in subdir : {f}", "red")
            f = img_f[0] # There should be only one tif file in subdir
            # Open image file using rasterio
            ds = rasterio.open(f) 

        # Image in root folder
        else:
            ds = rasterio.open(f)
        
        # Keep track of all samples that relate to an image.
        img2samples[f] = []
        # Loop through the lat/lon files
        for i, row in df.iterrows():
            # convert EPSG 4326 to EPSG 32650
            proj_x, proj_y = proj(row.GPS_X, row.GPS_Y)

            if (ds.bounds.left < proj_x < ds.bounds.right) and (ds.bounds.bottom < proj_y < ds.bounds.top):
                img2samples[f].append(row.Sample)

    return img2samples

# ------- Class to Extract or Mask (a) Img(s) based on sample lat/lons ------- #

class ExtMskWin():
    """
    The following class will create a window centering around lat/lon points and extract image windows for a Sing Image.
    It can also mask sections of the original image where extracts were made. This is to can be useful if you dont want the
    SSL algorithms to see sections of the areas where malaria infected individual live (for generalizability if required)
    """
    def __init__(self, img_p:str, samples:List[str], mlr_pts:pd.DataFrame, ws:int, isdrone:bool, samp2extimgs):
        """
        Inputs
        - img_p : path to image
        - samples : a list of samples that correspond to an image
        - mlr_pts : Should contain GPS_X, GPS_Y coordinates along with samples 
        - ws : Window size, size of the patch / image chip that will be extracted
        - isdrone : boolean value stating if images are taken from a drone if false assume sentinel images
        - samp2extimgs : A dict that will holds which samples correspond to which images
        """ 
        self.mlr_pts = mlr_pts 
        self.ws = ws
        self.isdrone = isdrone
        self.img_p = img_p
        self.samples = samples
        
        # Make the sample the index of dataframe
        self.mlr_pts = self.mlr_pts.set_index("Sample")
        # Dictionary that keeps track of images and sample lat/lon that falls on the image - ...
        # ... this will be used to create a dataframe that has path to extracted image for corresponding samples
        self.samp2extimgs = samp2extimgs
        # Tracking dictionaries for masking
        self.samp2win = {}
        # We often use the file name for saving purposes
        self.fname = img_p.split("/")[-2]
            
    def ext_win(self,sample:Union[str,None], save_loc:str)-> None:
        """
        Inputs
            - sample : sample name for lat/lon coordinates
            - process : string value of "extract" or "mask"
                - "extract" : extracts an mxm image window with sample lat/lon being at the center
                - "mask" : masks an mxm image window with sample lat/lon being the center of mask
        """
        proj = pyproj.Transformer.from_crs(4326,32650)
        with rasterio.open(self.img_p) as dataset:
            self.img = dataset.read() # (C,W,H)
            self.C,_,_ = self.img.shape
            self.bands = [1,2,3] if self.C == 3 else [1,2,3,4]
            assert self.C == 3 or self.C == 4, colored(f"Got image channles {self.C}","red")
            profile = dataset.profile.copy() # copy profile of the original image
            profile.update(width = self.ws, height = self.ws, count = self.C) #count is the number of bands in rasterio

            # Get corresponding row for samples from dataframe
            row = self.mlr_pts.loc[sample]
            # get lat/lon values from dataframe
            x, y = row.GPS_X, row.GPS_Y
            # convert values from epsg:4326 to epsg:32650, Interchange x,y as epsg:4326 order is lat,lon and NOT lon,lat 
            proj_x, proj_y = proj.transform(y, x)
            # Get corresponding pixel positions
            px, py = rasterio.windows.rowcol(dataset.transform, proj_x, proj_y)
            # Make slice of image center around point
            window = rasterio.windows.Window(col_off = py - int(self.ws/2), row_off = px - int(self.ws/2), width = self.ws, height = self.ws)
            # Keep track of the window to be used for masking
            self.samp2win[sample] = window
            # Get the window transform
            window_transform = rasterio.windows.transform(window, dataset.transform)
            profile.update(transform = window_transform)
            # Get image in numpy format for saving
            img_win = dataset.read(self.bands, window = window, boundless = True) #(C,W,H)
            # Save extracted image
            self.save_ext_win(img_win,sample, profile, save_loc)

    def ext_samps_wins(self, save_loc:str)->None:
        """
        Extracts all samples in an corresponding to an image
        Inputs
            - save_loc : folder where extracted images will be saved
        """
        if len(self.samples) > 0:
            for sample in self.samples:
                self.ext_win(sample,save_loc) 

    def msk_samps_wins(self,save_loc:str)->None:
        """
        Masks areas of the original image where windows extracted (where samples fall on the image)
        Inputs
            - save_loc : location where maked images will be saved
        """
        with rasterio.open(self.img_p) as dataset:
            img = dataset.read() # (C,W,H)
            profile = dataset.profile.copy() # copy profile of the original image
            profile.update(nodata = 0) # default value is None for "nodata" which wont work, so set it to 0
            
            if len(self.samples) > 0:
                for sample in self.samples:
                    window = self.samp2win[sample]
                    img[:, window.row_off : window.row_off + self.ws, window.col_off : window.col_off + self.ws] = profile["nodata"]
                    profile.update(nodata = profile["nodata"])
                # at the end of masking save image
                self.save_msk_img(img,profile,save_loc)
                
            else:
                # If there is no samples falling on the original image, save it as it is as it can be patched without any issues
                self.C = img.shape[0]
                self.bands = [1,2,3] if self.C == 3 else [1,2,3,4]
                save_fname = self.fname + "_" + "C" + str(self.C) + "_" + "nomsk" + ".tif"
                # Write image to folder
                with rasterio.open(os.path.join(save_loc, save_fname), 'w', **profile) as new_dataset:
                    new_dataset.write_band(self.bands, img) #(C,W,H)

    
    def save_ext_win(self,img:np.ndarray,sample:str,profile:rasterio.profiles.Profile, save_loc:str):
        """
        Saves extracted image only (due to naming conventions)
        Inputs 
            - img : extracted image window
            - sample : sample no correponding to lat/lon points
            - profile : profile with georeference data
        """
        # File name for saving ...
        file_prefix = "drn" if self.isdrone else "sen2a"
        file_suffix = sample
        save_fname = file_prefix + "_" + self.fname + "_" + "C" +str(self.C) + "_" + file_suffix + ".tif"
        
        # Create save directory if it doesnt exist
        if not os.path.exists(save_loc):
            os.mkdir(save_loc)

        # store image relating to sample in dictionary
        self.samp2extimgs[sample].append(os.path.join(save_loc,save_fname))

        # Write image to folder
        with rasterio.open(os.path.join(save_loc, save_fname), 'w', **profile) as new_dataset:
            new_dataset.write_band(self.bands, img) #(C,W,H)
        
    def save_msk_img(self, mskimg:np.ndarray, profile:rasterio.profiles.Profile, save_loc:str)->None:
        """
        Saves ORGINAL image which has been maked at extracted windows
        Inputs 
            - mskimg : Original Image that has been masked where samples fall 
            - profile : Profile contianing geospatial info
            - save_loc : folder where image will be saved
        """
        # File name for saving
        save_fname = self.fname + "_" + "C" + str(self.C) + "_" + "msk" + ".tif"

        # Create saving directory if it doesnt exist
        if not os.path.exists(save_loc):
            os.mkdir(save_loc)       

        # Write image to folder
        with rasterio.open(os.path.join(save_loc, save_fname), 'w', **profile) as new_dataset:
            new_dataset.write_band(self.bands, mskimg) #(C,W,H)

    def store_path_to_df(self, save_path:str)->None:
        """
        Saves extracted image path to dataframe. This function should be called at the very end when
        all images relating to a sample stored after going throug all images.
        Inputs
            - save_path : where to save updated dataframe
        """
        col_name = f"drn_{'C' + str(self.C)}_ext_paths" if self.isdrone else f"sen2a_{'C' + str(self.C)}_ext_paths"
        # Create an empty column which contains empty list
        self.mlr_pts[col_name] = self.mlr_pts.apply(lambda _: [], axis = 1)
        
        for sample, extimgs in self.samp2extimgs.items():
            self.mlr_pts.at[sample,col_name] = extimgs #you have to use "at" to store a list of values, "loc" allows single values

        self.mlr_pts.to_csv(save_path)
        
    def get_plot(self,img_p:str):
        """
        Helper function to plot image
        Inputs 
            - img_p : path to image
        """
        fig = plt.Figure(figsize = (12,12))
        with rasterio.open(img_p) as ds:
            img = ds.read()
            img_rsp = np.moveaxis(img, source = [0,1,2], destination = [2,0,1])
        fig = plt.imshow(img_rsp)
        return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Extracts or Masks Image Windows")
    parser.add_argument("-img_root", type = str, help = "Path to folder containing images that needs preprocessing")
    parser.add_argument("-ext_save_p", type = str, help = "Path where extracted images will be saved")
    parser.add_argument("-df_p", type = str, default = "data/interim/sshsph_mlr/mlr_geopts_imgs.csv", help = "path to file contianing sample and corrsponding lat/lon values")
    parser.add_argument("-df_ext_save_p", type = str, help = "Where extracted image paths dataframe will be saved")
    
    parser.add_argument("-msk_save_p", type = str, default = None, help = "Path where masked images will be store")
    parser.add_argument("-ws", type = int, default = 256, help = "Window Size")
    parser.add_argument("-drone", action=argparse.BooleanOptionalAction, help = "are images from a drone")

    # -------------------------- Arguments for functions ------------------------- #
    args = parser.parse_args()
    IMG_ROOT = args.img_root
    DF_P = args.df_p
    EXT_SAVE_LOC = args.ext_save_p
    MSK_SAVE_LOC = args.msk_save_p
    DF_EXT_SAVE_P= args.df_ext_save_p
    WS = args.ws
    ISDRONE = args.drone

    # --------- Get dictionary containing geo points that fall on images --------- #
    # Empty Dict that will store images and corresponding geo points
    img2samples = {}
    # Pandas dictionaty that contains sample, lat, lon values 
    df= pd.read_csv(DF_P)
    # Images and Geo-Points that fall on image
    img2samples = get_samples_for_imgs(IMG_ROOT,df,img2samples)

    # ---------------------------- Extract and or Mask --------------------------- #
    # Empty dictionary to hold extracted images for each sample 
    samps2extimgs = {sample : [] for sample in df.Sample} 
    for img_p, samples in tqdm(img2samples.items()):
        # Cretae ExtMskWin object for extracting and or masking
        ext_msk_win = ExtMskWin(
            img_p=img_p,
            samples=samples,
            mlr_pts = df,
            ws = WS, 
            isdrone=ISDRONE,
            samp2extimgs=samps2extimgs
        )
        # TO extract images : Function loops through all samples corresponding to an image
        ext_msk_win.ext_samps_wins(save_loc = EXT_SAVE_LOC)
        # To mask original image : Loops through all samples (geo points) and mask these areas
        if MSK_SAVE_LOC:
            # If msk_save_loc specified then save masked images in declared path
            ext_msk_win.msk_samps_wins(save_loc = MSK_SAVE_LOC)
    
    # At the end of the loop save all 
    ext_msk_win.store_path_to_df(save_path = DF_EXT_SAVE_P)
