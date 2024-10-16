import os
import shutil
from glob import glob

import numpy as np
import pandas as pd

import torch 
from torch.utils.data import Dataset 
from torchvision.transforms import ToTensor, Normalize, Compose 
from torch.utils.data import DataLoader

import rasterio

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

from typing import List, Tuple

from termcolor import colored
from tqdm import tqdm
import warnings

import malaria_config 
import config

def clean_image_dataset():
    """
    Construct an image dataset by moving all .tif files
    - Constructs data folder to store all mosiacs data
    - Within the data folder a folder named "train" to store all .tif images and "archive" to store all other data.
    """
    # Construct data folder
    if not os.path.exists("data"):
        os.mkdir("data")
    # Construct train folder
    if not os.path.exists("data/train"):
        os.mkdir("data/train")
    # construct archive folder
    if not os.path.exists("data/archive"):
        os.mkdir("data/archive")

    for f in glob("*"):
        if f.endswith(".tif"):
            shutil.move(src = f, dst = "data/train")
        if f.endswith(".enp") or f.endswith(".ovr") or f.endswith("pdf") or f.endswith(".prj") or f.endswith(".tfw") or f.endswith("tiles") or f.endswith(".xml") or f.endswith(".zip"):
            shutil.move(src = f, dst = "data/archive")

def remove_images(paths:list):
    """
    Removes Images that are ...
        - Not of type ndarray (incomplete info, in this case grayscale)
        - Images that donot have 3 channels
    Inputs
        - list of paths to images
    """
    nd_array_cnt = 0 # 
    other_cnt = 0
    img3_cnt = 0 # 3D image count
    img3not_cnt = 0 # anything that isnt 3D image count
    for p in paths:
        img = cv2.imread(p)
        # Check whether data is of type ndarray (as there are nontype data in the dataset)
        if isinstance(img, np.ndarray):
            nd_array_cnt += 1
            # Check if the data is 3 Dimensional
            if img.shape[2] == 3:
                img3_cnt += 1
            # Move any non 3D images to archive
            else:
                img3not_cnt += 1
                print(f"Not an 3D img : {p}")
                shutil.move(src = p , dst = "data/archive")
        # Check if the data is of type 'NoneType' 
        else:
            print(f"None type images : {p}")
            print("Moving to archive ...")
            shutil.move(src = p , dst = "data/archive")
            other_cnt += 1

    print(f"Clean image count : {nd_array_cnt}")
    print(f"Damaged images : {other_cnt}")

# ---------------------- Dataset Class for Drone Images ---------------------- #

class DroneDataset(Dataset):
    """
    Desc : Dataset class for loading Remote Sensing Images (captured in Malaysia)
    The respository contain .jpg images with no labels. Hence the dataset will
    only return values for X.
    """
    def __init__(self, paths:List[str], transform:Compose = None, scaling_factor = None)->Tuple[torch.Tensor, str]:
        """
        Desc : Initialization
        Parameters
            - paths : list of paths to each image
        """
        self.paths = paths
        self.transform = transform 
        self.scaling_factor = scaling_factor

    def __len__(self):
        """
        Desc : return the size of the dataset
        """
        return len(self.paths)
    
    def __getitem__(self, idx:int):
        """
        Desc : Returns a single item from dataset. Note only a single value 
        will be returned as this dataset doesnot have labels
        Parameters :
            - idx : index
        """
        with rasterio.open(self.paths[idx]) as ds:
            img = ds.read([1,2,3]) # (C,W,H)

        # convert to torch tensor : We havent used torchvision.ToTensor function as it automaticall normalize to range 0-1
        img = torch.tensor(img, dtype = torch.float32) #(C,W,H)
        # apply scaling factor if define, i.e 255 since drone images are within 0-255 range
        if self.scaling_factor:
            img = img/self.scaling_factor 

        if self.transform:
            img = self.transform(img)

        return img, self.paths[idx]
    
# --------------------- Dataset Class for Sentinel Images -------------------- #
class SentinelDataset(Dataset):
    """Desc : Dataset class for satelite data downloaded from google eath engine"""
    def __init__(self, paths:List[str], transform:Compose = None, scaling_factor = None)-> Tuple[torch.Tensor,str]:
        """
        Inputs 
             - paths : List of image paths
             - transform : Any Pytorch transformations
        NOTE : GEE states to divide dataset by 10,000 however values are difficult to be visualized, 
        so we will normalize the values btw 0 and 1
        """
        self.paths = paths
        self.transform = transform
        self.scaling_factor = scaling_factor

    def __len__(self):
        """Desc : Return dataset size"""
        return len(self.paths)
    
    def __getitem__(self, idx:int):
        """
        Desc : Return a single item Based on Index
        Inputs 
            - idx : index
        """
        with rasterio.open(self.paths[idx]) as ds:
            img = ds.read([1,2,3]) #(C,W,H)

        # convert to torch tensor : We havent used torchvision.ToTensor function as it automaticall normalize to range 0-1 
        img = torch.tensor(img, dtype = torch.float32) # (C,W,H)
        if self.scaling_factor:
            img = img / self.scaling_factor

        if self.transform:
            img = self.transform(img)

        return img, self.paths[idx]

 
# ------------ Dataset Class for Remote Sensing Images of Malaysia ----------- #

#todo : We need to incorporate rasterio
class SSHSPH_MALARIA_MY(Dataset):
    """
    Construct a Dataset for Malaria Data (i.e points on csv + images)
    Inputs
        - df : dataframe containg lat/lon points, malaria features & path to images
        - transform : various pytorch transforms to deal with images (i.e ToTensor)
    """
    def __init__(self, df:pd.DataFrame, target:str, transform:Compose = None, numeric_feat_names:List[str] = [], cat_feat_names:List[str] = [], train:bool = True, feat_transformer = None):
        self.df = df
        self.target = target
        self.transform = transform
        self.numeric_feat_names = numeric_feat_names
        self.cat_feat_names = cat_feat_names
        relevent_feat = self.numeric_feat_names + self.cat_feat_names
        # We fit and transform only during training
        if train:
            self.column_trans = make_column_transformer(
                (OneHotEncoder(), self.cat_feat_names), 
                remainder = "passthrough"
            )
            self.feat_enc = self.column_trans.fit_transform(self.df[relevent_feat]) #filter columns only with relevent features, i.e we dont want Sample column getting in there.
        # During validation or testing we use the existing transformer from training
        else:
            self.column_trans = feat_transformer #Feature transformer is passed in as an argument
            self.feat_enc = self.column_trans.transform(self.df[relevent_feat])
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx:int):
        # Get row from data frame
        df_row = self.df.iloc[idx]
        # Read image data from dataframe row
        img = cv2.imread(df_row.image_path) #(256,256,3)
        # Get label "pk" from dataframe row
        lab = df_row[self.target] #(,)
        # Get other features (domain specific) - Numeric Features + Cat Features
        # Note the encording has both cat + numerical features
        feat = self.feat_enc[idx, :].astype("float") #(55,)
        
        if self.transform:
            img_t = self.transform(img) #(3,256,256)
            feat_t = torch.tensor(feat, dtype = torch.float32) #(55,)
            lab_t = torch.tensor(lab, dtype = torch.long) #() Single tensor
            return img_t, lab_t, feat_t
        else:
            warnings.warn(colored("Using numpy instead of torch"))
            return img, lab, feat 
    
# --------------------- Computes Mean & Std for Datasets --------------------- #

class DATASET_MEAN_STD():
    """Gets the mean and standard deviation of a datastet"""
    def __init__(self, DATASET, batch_size:int = 512):
        super().__init__()
        self.DATASET = DATASET
        self.batch_size = batch_size

    def get_sshsph_my_stats(self,image_paths:List[str]):
        """
        Computes the mean and std for SSHSPH_MY dataset
        """
        c_sum, c_sq_sum, total_pixels = 0,0,0

        d = self.DATASET(image_paths)
        dl = DataLoader(d, batch_size = self.batch_size)
        for data,path in tqdm(dl): # note here we dont have traget else should be (data,labs)
            b,c,w,h = data.shape
            num_pixels = b * w * h
            assert c == 3, f"Num channels incorrect, got c = {c}"
            c_sum += torch.sum(data, dim = [0,2,3]) # Sum across channels, data.shape = (b,c,w,h)
            c_sq_sum += torch.sum(data**2 , dim = [0,2,3])

            total_pixels += num_pixels
        
        mean = c_sum / total_pixels
        std = ((c_sq_sum/total_pixels) - (mean**2))**0.5

        return mean,std

    def get_sshsph_my_malaria_stats(self,df:pd.DataFrame, target = "pk"):
        """
        Computes the mean and std for SSHSPH_MALARIA_MY dataset
        Note that the dataframe already contains the path to images
        Inputs
            - df : Dataframe containing points + image_paths + features
            - targets : target variable, not needed for this function except to pass to SSHSPH_MALARIA_MY dataset class
        """
        c_sum, c_sq_sum, total_pixels = 0,0,0

        d = self.DATASET(df, target)
        dl = DataLoader(d, batch_size = self.batch_size)
        for data,_,_ in  tqdm(dl): # we have both imgs and labs for this dataset class
            b,c,w,h = data.shape
            num_pixels = b * w * h # total pixels over batch per channel
            c_sum += torch.sum(data, dim = [0,2,3]) # Sum across channels, data.shape = (b,c,w,h)
            c_sq_sum += torch.sum(data**2 , dim = [0,2,3])
            
            total_pixels += num_pixels # aggregate num_pixels (batch) over entire dataset 

        mean = c_sum / total_pixels
        std = ((c_sq_sum/total_pixels) - (mean**2))**0.5

        return mean,std

# -------------------- To Load Model Weights ------------------- #

def load_model_weights(model, path_to_weights = "pretrain_weights/rsp-aid-resnet-50-e300-ckpt.pth", num_classes = 51):
    """
    Desc : Loads pretrained weights to a resnet 50/swin-vit model from the RSP repository.
    The weight file (i.e rsp-aid-resnet-50-e300-ckpt.pth) consists of a linear layer with an output of 51 hence we have to set num_classes to 51
    Inputs 
        - path_to_weights : path to the file containing weight (last layer is a Linear Layer with 51 neurons)
        - num_classes : number of classes, for the weight file (rsp-aid-resnet-50-e300-ckpt.pth) we need to set num classes to 51
    Outputs
        - res50 : i.e Resnet50 pretrained model
    """
    model_ = model(num_classes = num_classes)
    model_state = torch.load(path_to_weights) 
    model_.load_state_dict(model_state["model"]) # we can add argument .load_state_dict( ... , strict = False) if the weights dont load properly, random weights will be intialised for the weights that do not work
    return model_


if __name__ == "__main__":
    
    #* -------------------------- Cleaning image dataset -------------------------- #
    #clean_image_dataset()



    #* ------------------------ Using SSHPSH Dataset class ------------------------ #
    image_paths = glob("data/SSHSPH-RSMosaics-MY-v2.1/images/channel3_p/*")
    sshsph_my = SSHSPH_MY(
        image_paths , 
        transforms = Compose([ToTensor(), Normalize(mean = config.IMAGE_MEAN, std = config.IMAGE_STD)])
    )
    img = sshsph_my.__getitem__(1)

    #* ---------------------- Using SSHSPH_MALARIA_My2 Class ---------------------- #
    
    df = pd.read_csv("data/processed/mlr_pts_no_missing.csv")
    numerical_feat_names = malaria_config.numeric_feat 
    cat_feat_names = malaria_config.cat_feat
    target_name = malaria_config.target
    sshsph_mal_my = SSHSPH_MALARIA_MY(
        df,
        target_name,
        transform = Compose([ToTensor(), Normalize(mean = config.IMAGE_MEAN, std = config.IMAGE_STD)]),
        numeric_feat_names=numerical_feat_names,
        cat_feat_names=cat_feat_names
    )

    img,lab,feat = sshsph_mal_my.__getitem__(0)

    #* -------------------- To compute mean & std of dataset(s) ------------------- #
    image_paths = glob("data/SSHSPH-RSMosaics-MY-v2.1/images/channel3_256x256p/*")
    # df = pd.read_csv("data/SSHSPH-malaria-prevelance-v1.0/malaria_pts_with_images.csv")
    # train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 0)
    
    mean_std_dataset = DATASET_MEAN_STD(SSHSPH_MY, batch_size = 512)
    mean,std = mean_std_dataset.get_sshsph_my_stats(image_paths)
    #mean, std = mean_std_dataset.get_sshsph_my_malaria_stats(train_df)
    print(f"Mean : {mean}, Std : {std}")









