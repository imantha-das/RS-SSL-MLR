# ==============================================================================
# Lightly Dataset class cannot read georeferenced data by the looks of it. As
# such we will have to convert tiff images saved using rasterio in pil format. 
# we probably can still save it as tiff
# ==============================================================================
import os
import numpy as np
import rasterio
from  PIL import Image
from glob import glob
from tqdm import tqdm
import argparse
import plotly.express as px

parser = argparse.ArgumentParser(description = "Save rasterio image in PIL format")
parser.add_argument("-src_path", type = str, help = "path to data that needs conversion")
parser.add_argument("-dst_path", type = str, help = "Path to where data will be saved")
args = parser.parse_args()

if __name__ == "__main__":
    if not os.path.exists(args.dst_path):
        os.mkdir(args.dst_path)

    img_paths = glob(os.path.join(args.src_path, "*"))
    
    for img_path in tqdm(img_paths):
        # Read tiff image
        with rasterio.open(img_path) as ds:
            img = ds.read([1,2,3]) #(C,W,H)

        # PIL requires (W,H,C)
        img = np.moveaxis(img, source= (0,1,2), destination=(2,0,1))
        # Sentinel Images need to be scaled by 10000, Normalized value between 0-1
        img = img/10000
        # Multiply by 255 to bring to RGB range
        #img = np.clip(img * 255, 0, 255)
        img = np.clip(img, 0, 0.3) #reflectance values such vegetagtion lie between 0-0.3
        # Rescale back to 255 to save as uint8
        img = (img/0.3) * 255
        #Convert to uint8 to save as a pil image
        img = img.astype(np.uint8)
        # To save read image from array
        img = Image.fromarray(img)
        # Save image
        fname = os.path.basename(img_path)
        img.save(os.path.join(args.dst_path, fname))

        
    
