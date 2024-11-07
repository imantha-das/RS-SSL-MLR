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
from glob import glob
import numpy as np
import pandas as pd

import rasterio.errors
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

from typing import Tuple, Union
import rasterio
import warnings
from tqdm import tqdm
import plotly.express as px
import argparse

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

sys.path.append("src/ssl_models")

import ssl_config
from byol_train import BYOL
from simsiam_train import SimSiam

def get_classifier_results(X,y, model, state =0):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state=state)

    clf = model()
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    print(f"train acc : {train_acc}, val acc : {val_acc}")

    return train_acc, val_acc



def store_Xy(dataloader:DataLoader,ssl_weights_p:str)->None:
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

def train_downstream_model(clf_model:Union[LogisticRegression],ssl_model:Union[SimSiam, BYOL],ssl_weights_p:str, feature_target_names:dict, state = 0):
    """Get representation for fine tuned model at each epoch and check train downstream classifier"""
    # Load the finetuned SSL weights and extract the backbone to get representaions 
    ssl_model = load_ssl_weights(ssl_model, ssl_weights_p)
    ssl_model_bb = ssl_model.backbone

    # Normalizations applied during SSL training, applied for downstream training
    ssl_transforms = {
        "sen2a" : Compose([ToTensor(), Normalize(malaria_config.sat_img_mean, malaria_config.sat_img_std)]),
        "drn" : Compose([ToTensor(), Normalize(malaria_config.drn_img_mean, malaria_config.drn_img_std)])
    }

    #* This Dataset class already HANDLES CONCATENATING THE X_GEO(ssl representation) WITH MALARIA DATASET FEATURES
    #* To select the features please ammend the malaria_config file
    sshsph_mal_my = SSHSPH_MALARIA_MY(
        df,
        feature_target_names,
        ssl_model_bb,
        img_transform = ssl_transforms,
        feat_transformer=make_column_transformer
    )

    # Define a trainloader where the batchsize is the same as the length (i.e a single batch in the trainloader)
    trainloader = DataLoader(sshsph_mal_my, batch_size=len(sshsph_mal_my), num_workers = 8)
    # Get the first and only batch
    X, y = next(iter(trainloader))

    # Save features in processed
    X = X.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    
    # Dwonstream classification
    train_acc, val_acc = get_classifier_results(X,y,clf_model, state = 0)

    return train_acc, val_acc




if __name__ == "__main__":
    # ------------------------------ Argument Parser ----------------------------- #

    parser = argparse.ArgumentParser(description = "Argument Parser for Downstream Malaria Training")
    parser.add_argument("-mlr_data_file", type = str, help = "path to malaria dataset file", 
                        default = "data/processed/sshsph_mlr/mlr_nomiss_vardrop_train_v2.csv")
    parser.add_argument("-ssl_weights_fold", type = str, help = "path to folder containing ssl weights")
    parser.add_argument("-train_last_epoch_weights_only", action = argparse.BooleanOptionalAction)

    args = parser.parse_args()
    
    #* ---------------------- Using SSHSPH_MALARIA_My2 Class ---------------------- #
    
    df = pd.read_csv(args.mlr_data_file)

    feature_target_names = {
        "cat_feat_names" :  malaria_config.cat_feat,
        "numeric_feat_names" :  malaria_config.numeric_feat,
        "target_name" : malaria_config.target
    }
    # Define Resnet backbone as we need to pass it to SSL algorithms
    resnet = resnet50()
    resnet_backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
    # The Weight folder contains the SSL training method (i.e simsiam) grab that from path name
    model_flag = args.ssl_weights_fold.split("/")[-1].split("-")[0]
    
    match model_flag:
        case "simsiam":
            print(colored("Loading simsiam pretrained (finetuned) weights ...", "green"))
            model_params = ssl_config.simsiam_model_params
            ssl_model = SimSiam(model_params, resnet_backbone)
        case "byol":
            print(colored("Loading byol pretrained (finetuned) weights ...", "green"))
            model_params = ssl_config.byol_model_params
            ssl_model = BYOL(model_params, resnet_backbone)
        case _:
            raise(KeyError(f"Model flag incorrect, got {model_flag} but should get 'simsiam' or 'byol' !"))

    # Get all the paths in the SSL weights folder, only a few are checkpoints ...
    ssl_fold_paths = glob(os.path.join(args.ssl_weights_fold, "*"))
    # Filter just the checkpoints
    ssl_weights_paths = list(filter(lambda x: x.endswith("ckpt"), ssl_fold_paths))
    # Checkpoint need to be sorted to ensure epoch 0 goes first
    ssl_weights_paths.sort()

    # If we want to just train only the wights from the last epoch only    
    if args.train_last_epoch_weights_only:
        print(colored("Training in last epoch checkpoint only ...", "green"))
        # get last ssl_weights_file
        ssl_weights_last_path = ssl_weights_paths[-1]
        # Downstream model training
        train_acc , val_acc = train_downstream_model(
            clf_model =LogisticRegression,
            ssl_model= ssl_model, #SimSiam or BYOL
            ssl_weights_p= ssl_weights_last_path,
            feature_target_names=feature_target_names,
        )
        #Write accuracy to text file
        with open(os.path.join(args.ssl_weights_fold,"last_epoch_acc.txt")) as f:
            f.write(f"train_acc : {train_acc}\n")
            f.write(f"val_acc : {val_acc}\n")
    # We are training on all model checkpoints and checking for learnt representaions
    else:
        print(colored("Training all model checkpoints ...", "green"))
        acc_tracker = {"weight_file" : [], "train_acc" : [], "val_acc" : []} 

        for ssl_w_p in ssl_weights_paths:
            # Downstream model training
            train_acc , val_acc = train_downstream_model(
                clf_model =LogisticRegression,
                ssl_model= ssl_model, #SimSiam or BYOL
                ssl_weights_p= ssl_w_p,
                feature_target_names=feature_target_names,
            )
            # Keep track of losses
            acc_tracker["train_acc"].append(train_acc)
            acc_tracker["val_acc"].append(val_acc)
            acc_tracker["weight_file"].append(ssl_w_p.split("/")[0])

        print(acc_tracker["train_acc"], acc_tracker["val_acc"])

        # Loss Plots
        p = px.line()
        p.add_scatter(x = np.arange(0, len(acc_tracker["train_acc"])), y = acc_tracker["train_acc"], name = "train accuracy")
        p.add_scatter(x = np.arange(0, len(acc_tracker["val_acc"])), y = acc_tracker["val_acc"], name = "validation accuracy")
        p.update_layout(xaxis_title = "epochs", yaxis_title = "accuracy", template = "plotly_white")
        p.write_image(os.path.join(args.ssl_weights_fold,"train_val_acc.png"))
        # Store losses in a CSV file
        df_acc = pd.DataFrame(acc_tracker)
        df_acc.to_csv(os.path.join(args.ssl_weights_fold,"train_val_acc.csv"), index = False)
        
        
    # model = load_ssl_weights(model, ssl_weights_p)
    # model_bb = model.backbone

    # transforms = {
    #     "sen2a" : Compose([ToTensor(), Normalize(malaria_config.sat_img_mean, malaria_config.sat_img_std)]),
    #     "drn" : Compose([ToTensor(), Normalize(malaria_config.drn_img_mean, malaria_config.drn_img_std)])
    # }
    
    # sshsph_mal_my = SSHSPH_MALARIA_MY(
    #     df,
    #     target_features_names,
    #     model_bb,
    #     img_transform = transforms,
    #     feat_transformer=make_column_transformer
    # )

    # #feat, target = sshsph_mal_my.__getitem__(0)
    # trainloader = DataLoader(sshsph_mal_my, batch_size=len(sshsph_mal_my))

    # X,y = next(iter(trainloader))
    # X = X.cpu().detach().numpy()
    # y = y.cpu().detach().numpy()
    # print(X)
    # print(y)
    # print(X.shape, y.shape)
    # # --------------------- Store X, y results in npy format --------------------- #
    # #store_Xy(trainloader, ssl_weights_p=ssl_weights_p)

    # # ------------------ Load X, y results stored in npy format ------------------ #
    # X_p = "data/processed/sshsph_mlr/simsiam-is256-effbs256-epoch=0-geomlrX.npy"
    # y_p = "data/processed/sshsph_mlr/simsiam-is256-effbs256-epoch=0-y.npy"
    # X,y = get_Xy_npy(X_p, y_p)

    # # ------------------------ Fit model & print accuracy ------------------------ #
    # get_classifier_results(X, y, LogisticRegression)
    
    # print(type(model))

