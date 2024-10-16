# ==============================================================================
# Desc : Script to be used with QGIS python console. Helps with loading all images
# in partidular folder.
# ==============================================================================

from qgis.core import QgsProject, QgsRasterLayer
from glob import glob 
from os import path 
import os 

def load_img_subdir_to_qgis(root,layer_suffix = "_tile_ch3"):
    """Searches and loads images in subdirectories to QGIS. Each subdirectory
    contains multiple files such as .tif, .proj etc."""
    # Make an instance 
    project = QgsProject.instance()

    # Loop through all the folders 
    for folder in glob(path.join(root, "*")):
        # Loop through all the files in the folders
        for file in glob(path.join(folder, "*")):
            # Locate the tif file
            if file.endswith(".tif"):
                # give the layer a name which is the folder name
                layer_name = path.basename(path.dirname(file)) # get parent folder name where file is stored
                layer_name = layer_name + layer_suffix
                layer = QgsRasterLayer(file, layer_name, "gdal")
                if layer.isValid():
                    project.addMapLayer(layer)
                    
def load_img_dir_to_qgis(root):
    """Loads images in main root directory to qgis"""
    # Make an instance 
    project = QgsProject.instance()
    
    # Loop trough image files 
    for file in glob(path.join(root, "*")):
        if file.endswith(".tif"):
            layer_name = path.basename(file).split(".")[0]
            layer = QgsRasterLayer(file, layer_name, "gdal")
            if layer.isValid():
                project.addMapLayer(layer)
        else:
            raise Exception("non .tif file found")


root = "/home/imantha/workspace/RS-SSL-MLR/data/interim/channel4"
#root = "/home/imantha/workspace/RemSens_SSL/data/SSHSPH-RSMosaics-MY-v2.1/tiles"

load_img_subdir_to_qgis(root)
#load_img_dir_to_qgis(root)
    
            