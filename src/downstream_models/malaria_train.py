# ==============================================================================
# Downstream Malaria Prediction.
# This script will be using SSHSPH_MALRIA_MY class for loading data. Note that this
# class returns the malaria specific features and geofeatures from a trained (ssl finetune)
# backbone. Hence a simple scikit-lear logisitc regressor can be applied to the dataset

# Please ammend the mlr_config files to indicate which features should be input into the model
# ==============================================================================

import os
import re
import sys
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models import resnet50
from sklearn.compose import make_column_transformer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from termcolor import colored
from malaria_utils import load_ssl_weights
import malaria_config

from malaria_utils import SSHSPH_MALARIA_MY

from typing import Tuple

sys.path.append("src/ssl_models")

import ssl_config
from byol_train import BYOL
from simsiam_train import SimSiam

def get_classifier_results(X,y, model):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, shuffle = True)

    clf = model()
    clf.fit(X_train, y_train)
    print(f"train acc : {clf.score(X_train, y_train)}, val acc : {clf.score(X_val, y_val)}")



def store_Xy(dataloader:DataLoader,ssl_weight_p:str)->None:
    """Saves X,y for futher use"""
    X, y = next(iter(dataloader))

    # save features in processed
    X = X.cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    print(colored(X.shape, "green"), colored(y.shape, "green"))

    fname_prefix = "-".join(ssl_weights_p.split("/")[-2].split("-")[0:3])#, os.path.basename(ssl_weights_p))
    fname_epoch =re.split(r"[:.]",os.path.basename(ssl_weights_p))[1]
    fname = fname_prefix + "-" + fname_epoch 
    save_fold = "data/processed/sshsph_mlr"
    save_p = os.path.join(save_fold, fname)
    with open(save_p + "-" + "geomlrX.npy", "wb") as f:
        np.save(f, X)

    with open(save_p + "-" + "y.npy", "wb") as f:
        np.save(f, y)

def get_Xy_npy(X_p:str,y_p:str)->Tuple[np.array,np.array]:
    "Loads X,y saved in npy format for further use"
    with open(X_p, "rb") as f:
        X = np.load(f)

    with open(y_p, "rb") as f:
        y = np.load(f)

    return X,y



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
    ssl_weights_p = "model_weights/ssl_weights/simsiam-is256-effbs256-ep1-bbRes-dsDrnSen2a-clClUcl-nmTTDrnSatNM/epoch:epoch=0.ckpt"

    model_flag = ssl_weights_p.split("/")[-2].split("-")[0]
    
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
        
    model = load_ssl_weights(model, ssl_weights_p)
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

    #feat, target = sshsph_mal_my.__getitem__(0)
    trainloader = DataLoader(sshsph_mal_my, batch_size=len(sshsph_mal_my))

    # --------------------- Store X, y results in npy format --------------------- #
    #store_Xy(trainloader, ssl_weight_p=ssl_weights_p)

    # ------------------ Load X, y results stored in npy format ------------------ #
    X_p = "data/processed/sshsph_mlr/simsiam-is256-effbs256-epoch=0-geomlrX.npy"
    y_p = "data/processed/sshsph_mlr/simsiam-is256-effbs256-epoch=0-y.npy"
    X,y = get_Xy_npy(X_p, y_p)

    # ------------------------ Fit model & print accuracy ------------------------ #
    get_classifier_results(X, y, LogisticRegression)
    
