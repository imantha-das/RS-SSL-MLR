# ==============================================================================
# Desc : Script to extract portions (256x256 windows) of the image with geometric 
# points centering around the image. These geometric points contain data for 
# malaria. 
# The function get_geom_points_on_image filters images theat conicide
# with lat/lon values present in datframe.
# The function get_image_slices, slices a larger image tile with nxn window with
# corresponding lat/lon point being the center of this sliced image chip.
# ==============================================================================
import os
import pandas as pd
from glob import glob
import rasterio
import pyproj
from pyproj import Transformer 
from rasterio.windows import rowcol, Window
import argparse
from tqdm import tqdm
from termcolor import colored

parser = argparse.ArgumentParser(description = "Extract nxn window from RS Image centering lat/lon points from dataframe")
parser.add_argument("-geopts_p", type = str, help = "Path to dataframe containing lat/lon values for malaria dataset")
parser.add_argument("-rsimgs_p", type = str, help = "Path to Remote Sensing Images")
parser.add_argument("-svefld_p", type = str, help = "Path to where extracted Images will be saved")
args = parser.parse_args()

def get_geom_points_on_image(img_p:str, df:pd.DataFrame):
    """
    Desc : Returns a dataframe with points that fall/(coincide) on/(with) an image.
    This function assumes the images are Projected using EPSG:32650
    Inputs
        - img_p : path to image
        - df : Dataframe containing heometric point data (lat/lon)
    """ 
    proj = pyproj.Proj(init = "epsg:32650")
    dataset = rasterio.open(img_p)
    pts_on_img = []
    for i, row in df.iterrows():
        proj_x, proj_y = proj(row.GPS_X, row.GPS_Y)

        if (dataset.bounds.left < proj_x < dataset.bounds.right) and (dataset.bounds.bottom < proj_y < dataset.bounds.top):
            pts_on_img.append(row)

    return pd.DataFrame(pts_on_img)

def get_image_slice(img_p:str, pts:pd.DataFrame, ws:int , save_folder:str):
    """
    Desc : Makes a mxm window with the geometric point at its center and extract a window/patch
    from the larger image.
    Inputs
        - img_p : path to image
        - pts : dataframe containing only points that coincide with the image
        - ws : window size to extract portion from image
        - save_folder : folder where images will be saved.
    """
    # Projection to convert lat/lon values from EPSG:4326 to EPSG:32650 
    proj = Transformer.from_crs(4326,32650)

    with rasterio.open(img_p) as dataset:
        profile = dataset.profile.copy() # copy profile of the original image
        profile.update(width = ws, height = ws, count = 3) #count is the number of bands in rasterio
        
        for idx, row in pts.iterrows():
            # get lat/lon values from dataframe
            x, y = row.GPS_X, row.GPS_Y
            # convert values from epsg:4326 to epsg:32650
            # Interchange x,y as epsg:4326 order is lat,lon and NOT lon,lat 
            proj_x, proj_y = proj.transform(y, x)
            # Get corresponding pixel positions
            px, py = rowcol(dataset.transform, proj_x, proj_y)

            # Make slice of image center around point
            window = Window(col_off = py - int(ws/2), row_off = px - int(ws/2), width = ws, height = ws)
            # Get the window transform
            window_transform = rasterio.windows.transform(window, dataset.transform)
            profile.update(transform = window_transform)
            
            rgb_slice = dataset.read([1,2,3], window = window, boundless = True)
            
            # Save copy of .tiff file
            folder_name = dataset.name.split("/")[-2] #name of file
            suffix = row.Sample
            save_fname = folder_name + "_" + suffix + ".tif"

            if not os.path.exists(save_folder):
                os.mkdir(save_folder)

            with rasterio.open(os.path.join(save_folder, save_fname), 'w', **profile) as new_dataset:
                new_dataset.write_band([1,2,3], rgb_slice)

if __name__ == "__main__":
    # Malaria dataset
    malaria_df = pd.read_csv(args.geopts_p)
    # Paths to all the images - Looks through each folder and returns the tif path
    image_paths = glob(os.path.join(args.rsimgs_p,"*","*.tif")) # "*.tif" gets only the tif files within the images
    # get points that fall on image
    print(colored("Extracting images ...", "green"))
    for img_p in tqdm(image_paths):
        pts_on_img = get_geom_points_on_image(img_p, malaria_df)
        #print(pts_on_img[["GPS_X","GPS_Y","hadMalaria","pf","pv"]])
        get_image_slice(img_p, pts_on_img, 256, args.svefld_p)
        
    

