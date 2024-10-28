import malaria_config
import pandas as pd 
import cv2

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

from termcolor import colored
from typing import List
import warnings

# ------------ Dataset Class for RemoteSensing Images of Malaysia ----------- #

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
        
if __name__ == "__main__":
    #* ---------------------- Using SSHSPH_MALARIA_My2 Class ---------------------- #
    
    df = pd.read_csv("data/processed/mlr_pts_no_missing.csv")
    numerical_feat_names = malaria_config.numeric_feat 
    cat_feat_names = malaria_config.cat_feat
    target_name = malaria_config.target
    sshsph_mal_my = SSHSPH_MALARIA_MY(
        df,
        target_name,
        transform = Compose([ToTensor(), Normalize(mean = malaria_config.IMAGE_MEAN, std = malaria_config.IMAGE_STD)]),
        numeric_feat_names=numerical_feat_names,
        cat_feat_names=cat_feat_names
    )

    img,lab,feat = sshsph_mal_my.__getitem__(0)