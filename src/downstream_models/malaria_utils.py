import os
import sys
import pandas as pd 
import rasterio

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.models import resnet50
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

import pickle

from termcolor import colored

from typing import List, Union, Dict
from torchsummary import summary
import warnings
from tqdm import tqdm

# sys.path.append("src/ssl_models")

# from byol_train import BYOL
# from simsiam_train import SimSiam

# ------------------------ Load Finetuned SSL weights ------------------------ #
def load_ssl_weights(model, ssl_weights:str = "model_weights/ssl_weights/simsiam-is256-effbs256-ep1-bbRes-dsDrnSen2a-clClUcl-nmTTDrnSatNM/epoch:epoch=0.ckpt"):
    """Loads weights finetuned weights after SSL training to model"""
    model_state = torch.load(ssl_weights)
    model.load_state_dict(model_state["state_dict"])
    return model

# -------------- To verify if model weights are loaded properly ------------- #
def print_model_weights(model):
    """Print to check if the weights are loaded properly"""
    for name, param in model.named_parameters():
        print("-"*20)
        print(f"name : {name}")
        print(f"values : \n{param}")


# ------------ Dataset Class for RemoteSensing Images of Malaysia ----------- #

class SSHSPH_MALARIA_MY(Dataset):
    """
    Construct a Dataset for Malaria Data (i.e points on csv + images)
    Inputs
        - df : dataframe containg lat/lon points, malaria features & path to images
        - target_feat_names : dictionary containing names of numerical, categorical and target feature names.
        - fine_backbone : Fine tuned backbone, model weights preloaded.
        - feat_transformer : "make_colum_transfomer" function during training or trained ColumnTransformer during testing
    
    NOTES
        - categorical features are handled using One Hot Encoding
        - transforms handled within class
    """
    def __init__(self, 
                 df:pd.DataFrame, 
                 target_feat_names:Dict[str,list], fine_backbone:torch.nn.Sequential,img_transform:Union[dict, None] = None, feat_transformer:Union[make_column_transformer, None] = None):
        # Dataframe containing lat/lon point, image paths, feature and target
        self.df = df
        # Fine tuned backbone for extracting geospatial features from imagery
        self.backbone = fine_backbone
        # Other Malaria specific feature names
        relevent_feat = target_feat_names["numeric_feat_names"] + target_feat_names["cat_feat_names"]
        self.img_transform = img_transform
        # If training, we pass a feature transformer that must be trained for OHE
        if feat_transformer:
            self.column_trans = make_column_transformer(
                (OneHotEncoder(), target_feat_names["cat_feat_names"]), 
                remainder = "passthrough"
            )
            self.feat_enc = self.column_trans.fit_transform(self.df[relevent_feat]) #filter columns only with relevent features, i.e we dont want Sample column getting in there.
            # We need to save the transformer for later use - during testing
            with open("model_weights/mlr_weights/OHE_preprocessor.pkl", 'wb') as f:
                pickle.dump(self.feat_enc, f)
        # During testing we use the existing transformer from training
        else:
            self.column_trans = feat_transformer #Feature transformer is passed in as an argument
            self.feat_enc = self.column_trans.transform(self.df[relevent_feat])

        self.target_name = target_feat_names["target_name"]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx:int):
        # Get row from data frame
        df_row = self.df.iloc[idx]
        # Get path to identify which folder data is coming from, "sat or "drn"
        img_fold_name = os.path.dirname(df_row["selected_imgp"]).split("/")[-1]
        fold_prefix = img_fold_name.split("_")[0]
        # Read image data from dataframe row
        with rasterio.open(df_row["selected_imgp"]) as ds:
            img = ds.read([1,2,3])

        img = np.moveaxis(img, source=[0,1,2], destination = [2,0,1])
        
        #img = Image.open(df_row.selected_imgp) #(256,256,3)

        # Get label "pk" from dataframe row
        target = df_row[self.target_name] #(,)
        # Get other features (domain specific) - Numeric Features + Cat Features
        # Note the encording has both cat + numerical features
        feat = self.feat_enc[idx, :].astype("float") #(50,) , numpy array
        
        # if there are any image transforms
        if self.img_transform:
            transform = self.img_transform[fold_prefix] #img_transform is a dict containing transformation for "sen2a" / "drn" , folder_prefix contains these values
            img_t = transform(img).unsqueeze(0) #(1,3,256,256)
            B,C,H,W = img_t.shape
            if C == 4: # There is a mask in image, in that case just pick the first 3 channels
                img_t = img_t[:,0:3,:,:]
        # if there are no image transforms
        else:
            warnings.warn(colored("No Transformations applied !, output tensor will likely need further normalization", "red"))
            img_a = np.array(img)
            img_t = torch.from_numpy(img_a).float().unsqueeze(0) #(1,256,256,3)
            img_t = img_t.permute(0,3,1,2)

        with torch.no_grad():
            geo_feat_t = self.backbone(img_t).flatten(start_dim = 0) #must flatten to get dim in shape frim [1,1,2048,1] to shape [1,2048]
        
        feat_t = torch.tensor(feat).float() #(1,)
        feat_concat_t = torch.hstack([geo_feat_t, feat_t]) # for resnet 2048 + 50 (mlr features)
        target_t = torch.from_numpy(target.values.astype("int")).long()#(1,)

        return feat_concat_t, target_t

        
if __name__ == "__main__":
    #* ---------------------- Using SSHSPH_MALARIA_My2 Class ---------------------- #
    
    df = pd.read_csv("data/processed/sshsph_mlr/mlr_nomiss_vardrop_train_v2.csv")

    target_features_names = {
        "cat_feat_names" :  malaria_config.cat_feat,
        "numeric_feat_names" :  malaria_config.numeric_feat,
        "target_name" : malaria_config.target
    }
    # Define Resnet backbone as we need to pass it to SSL algorithms
    resnet = resnet50()
    resnet_backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
    ssl_weights = "model_weights/ssl_weights/simsiam-is256-effbs256-ep1-bbRes-dsDrnSen2a-clClUcl-nmTTDrnSatNM/epoch:epoch=0.ckpt"

    model_flag = ssl_weights.split("/")[-2].split("-")[0]
    
    match model_flag:
        case "simsiam":
            print(colored("Loading simsiam pretrained (finetuned) weights ...", "green"))
            model_params = ssl_config.simsiam_model_params
            model = SimSiam(model_params, resnet_backbone)
        case "byol":
            print(colored("Loading byol pretrained (finetuned) weights ...", "green"))
            model_params = ssl_config.byol_model_params
            model = BYOL(model_params, resnet_backbone)
        case _:
            raise(KeyError(f"Model flag incorrect, got {model_flag} but should get 'simsiam' or 'byol' !"))
        
    model = load_ssl_weights(model, ssl_weights)
    model_bb = model.backbone

    transforms = {
        "sen2a" : Compose([ToTensor(), Normalize(malaria_config.sat_img_mean, malaria_config.sat_img_std)]),
        "drn" : Compose([ToTensor(), Normalize(malaria_config.drn_img_mean, malaria_config.drn_img_std)])
    }
    
    sshsph_mal_my = SSHSPH_MALARIA_MY(
        df,
        target_features_names,
        model_bb,
        img_transform = transforms,
        feat_transformer=make_column_transformer
    )

    feat, target = sshsph_mal_my.__getitem__(0)
    print(feat.shape, target.shape)

    trainloader = DataLoader(sshsph_mal_my, batch_size=8)
    for X,y in tqdm(trainloader):
        print(colored(X.shape, "red"), colored(y.shape, "red"))
        break

