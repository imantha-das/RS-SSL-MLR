# ---------------------------------------------------------------------------- #
# Desc : Organises RSMosiac data raw data in to different folders   
#   - adds each corresponding images respective file into a folder (.tif,.prj etc.)
#   - seperates images with channel3 and channel4 to two seperate folders.                 
# Folder structure : root (RSMosiacs-MY) > Image Name > Each tif file belonging to image
# ---------------------------------------------------------------------------- #
import os 
import re
import numpy as np
import pandas as pd
import shutil 
from glob import glob 
from zipfile import ZipFile
from termcolor import colored
import skimage as ski
import argparse 

parser = argparse.ArgumentParser(description = "Organizes raw remote sensing data")
parser.add_argument("-data_path",type = str, help = "path to zip file")

def create_dataset(path_to_zip:str, dataset_name:str):
    """
    Unzips the .zip file and moves each of the corresponding files (.prj, .tfw, .tif, .xml, ...) 
    of an image to a respective folder.
    This function is not completely accurate and results in some tif files with similar names to be
    moved to the same folder which needs to be cleaned manually.
    """
    root_dir = os.path.dirname(path_to_zip)
    # Unzips zip file containing RS Mosiacs
    # with ZipFile(path_to_zip, "r") as f:
    #     print(f.extractall(os.path.join(root_dir, "tmp")))

    parent_folder = os.path.join(root_dir, dataset_name)
    if not os.path.exists(parent_folder): #i.e .../RSMosiacs
        os.mkdir(parent_folder)
        
    # make a folder called images within the dataset_folder where you would store the images
    dataset_folder = os.path.join(parent_folder, "images") #i.e .../RSMosiacs-MY/images
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)

    # Make a folder named tiles which contains tiles - larger images
    tile_folder = os.path.join(parent_folder, "tiles")
    if not os.path.exists(tile_folder):
        os.mkdir(tile_folder)

    # Note when using glob we get the full path
    # Note folder "tmp" is in the saame level as RSMosiacs-MY
    files = glob(os.path.join(root_dir, "tmp","*")) # i.e .../tmp
    
    # we need to move files such as .tif ... and others
    for file in sorted(files):
        if (os.path.isdir(file)) or (file.endswith(".pdf")):
            # A folder or a file that ends with .pdf
            if os.path.isdir(file): # tiles folder
                folder = file # File is actually a folder
                tile_files = glob(os.path.join(folder, "*"))
                for f in sorted(tile_files):
                    new_folder_name = os.path.basename(f.split(".")[:-1][0]) # Note this is the absolute path rather than just file name
                    to_folder = os.path.join(tile_folder, new_folder_name)
                    if not os.path.exists(to_folder):
                        os.mkdir(to_folder)
                    shutil.move(src = f, dst = to_folder)
                    print(f"{colored(f, 'green')} ---> {colored(to_folder, 'blue')}")
            else:
                # Move pdf file
                to_folder = os.path.join(parent_folder)
                shutil.move(src = file, dst = to_folder)
                
        else:
            search = re.findall(r"\d+", file)
            if len(search) == 3: # There is an extra digit that we dont want
                search = search[:-1]
            new_folder_name = "_".join(search)

            to_folder = os.path.join(dataset_folder, new_folder_name)
            if not os.path.exists(to_folder):
                os.mkdir(to_folder)

            shutil.move(src = file, dst = to_folder)
            print(f"{colored(file, 'green')} ---> {colored(to_folder, 'blue')}")

def get_num_imgs(parent_dir):
    """Function to identify if there is more than one image in a single folder"""
    folder_img_cnts = {}
    folders = glob(os.path.join(parent_dir, "*"))
    for folder in folders:
        tif_cnt = 0
        for file in os.listdir(folder):
            if file.endswith(".tif"):
                tif_cnt += 1
        folder_img_cnts[folder] = tif_cnt

    return folder_img_cnts

def split_dataset_based_on_channels(root_dir):
    """There are mixture of images with 3 and 4 channels, group them into different datasets"""
    parent_dir = os.path.dirname(root_dir)
    if not os.path.exists(os.path.join(parent_dir, "channel3")):
        os.mkdir(os.path.join(parent_dir, "channel3"))
    if not os.path.exists(os.path.join(parent_dir, "channel4")):
        os.mkdir(os.path.join(parent_dir, "channel4"))

    image_folders = glob(os.path.join(root_dir,"*"))
    to_ch3 = os.path.join(parent_dir, "channel3")
    to_ch4 = os.path.join(parent_dir, "channel4")

    for folder in image_folders:
        tif_count = 0
        for file in glob(os.path.join(folder, "*")):
            if file.endswith(".tif"):
                tif_count += 1
                img = ski.io.imread(file)
                if img.ndim == 3:
                    w,h,c = img.shape
                else:
                    c = None    

                if c == 3:
                    shutil.move(src = folder, dst = to_ch3)
                elif c== 4:
                    shutil.move(src = folder, dst = to_ch4)
                else:
                    print(f"{file} has more or less than 3 or 4 channels")
        
        assert tif_count == 1, "There is more or less than one tif image"
        
    
if __name__ ==  "__main__":

    path_to_zip = args.data_path

    # #create_dataset(path_to_zip, "RSMosiacs-MY")
    # folder_img_cnts = get_num_imgs("data/archive/RSMosiacs-MY/images")

    # # for k,v in folder_img_cnts.items():
    # #     print(k , v)

    # img_cnts_df = pd.DataFrame(folder_img_cnts, columns = ["folder_name", "tiff_cnt"])
    # img_cnts_df["folder_name"] = folder_img_cnts.keys()
    # img_cnts_df["tiff_cnt"] = folder_img_cnts.values()

    # print(img_cnts_df[img_cnts_df.tiff_cnt > 1])

    #! There are a couple folders in images that has more than 1 tif file, locate them and ...
    #! ... clean them manually.

    # Code to split dataset depending on the number of channels
    root_dir = "data/SSHSPH-RSMosaics-MY-v2.1/images"
    image_folders = glob(os.path.join(root_dir,"*"))

    split_dataset_based_on_channels(root_dir)
