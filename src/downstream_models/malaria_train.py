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
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.models import resnet50
from sklearn.compose import make_column_transformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from termcolor import colored
from malaria_utils import load_ssl_weights
from malaria_utils import SSHSPH_MALARIA_MY

from typing import Tuple, Union
import rasterio
import yaml
import json
import warnings
from tqdm import tqdm
import plotly.express as px
import argparse

from timm.models.vision_transformer import vit_base_patch16_224, VisionTransformer

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

sys.path.append("src/ssl_models")

from simsiam import SimSiamBBResnet, SimSiamBBSwinViT
from byol import ByolBBResnet, ByolBBSwinViT
from dino import DinoBBResnet, DinoBBSwinViT
from mae import MaeBBViT
from ssl_utils import print_model_weights

sys.path.append("src/ssl_models/foundation_models/RSP/Scene Recognition/models")
from swin_transformer import SwinTransformer

def get_classifier_results(X,y, model, state =0):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state=state)

    clf = model(verbose =1)
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

def train_downstream_model(
    clf_model:Union[LogisticRegression],
    ssl_model:Union[SimSiamBBResnet,SimSiamBBSwinViT, ByolBBResnet, ByolBBSwinViT, DinoBBResnet, DinoBBSwinViT],
    ssl_weights_p:str,
    backbone_name:str, 
    ssl_strategy:str,
    feature_target_names:dict, 
    state = 0,
    ):
    """Get representation for fine tuned model at each epoch and check train downstream classifier"""

    # Load the finetuned SSL weights and extract the backbone to get representaions 
    print(colored(f"Loading ssl weights : {ssl_weights_p}", "green"))
    ssl_model = load_ssl_weights(ssl_model, ssl_weights_p)

    # Select backbone based on SSL model 
    if ssl_strategy == "simsiam" or ssl_strategy == "byol":
        if backbone_name == "Resnet":
            ssl_model_bb = ssl_model.backbone
        else:
            ssl_model_bb = ssl_model.backbone_model
    elif ssl_strategy == "dino":
        ssl_model_bb = ssl_model.student_backbone
    else:
        raise KeyError(colored(f"Incorrect strategy passed : {ssl_strategy}", "red"))

    # Normalizations applied during SSL training, applied for downstream training
    if backbone_name == "Resnet":
        ssl_transforms = {
            "sen2a" : Compose([ToTensor(), Normalize(ssl_config["sat_img_mean"], ssl_config["sat_img_std"])]),
            "drn" : Compose([ToTensor(), Normalize(ssl_config["drn_img_mean"], ssl_config["drn_img_std"])])
        }
    else:
        ssl_transforms = {
            "sen2a" : Compose([ToTensor(), Resize(224), Normalize(ssl_config["sat_img_mean"], ssl_config["sat_img_std"])]),
            "drn" : Compose([ToTensor(), Resize(224), Normalize(ssl_config["drn_img_mean"], ssl_config["drn_img_std"])])
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

def select_weights_on_nepochs(ssl_weights_paths:list, n_epochs:int):
    ssl_weights_paths_flt = ssl_weights_paths[0::n_epochs]
    if ssl_weights_paths_flt[-1] != ssl_weights_paths[-1]:
        ssl_weights_paths_flt.append(ssl_weights_paths[-1])
    return ssl_weights_paths_flt

if __name__ == "__main__":
    # ------------------------------ Argument Parser ----------------------------- #

    parser = argparse.ArgumentParser(description = "Argument Parser for Downstream Malaria Training")
    parser.add_argument("-mlr_data_file", type = str, help = "path to malaria dataset file", 
                        default = "data/processed/sshsph_mlr/mlr_nomiss_vardrop_train_v2.csv")
    parser.add_argument("-ssl_weights_root_fold", type = str, help = "path to root folder containing ssl weights")
    parser.add_argument("-version", type = int, default = 0, help = "enter ssl model version if any")
    parser.add_argument("-down_model", type = str, default = "lr", help = "Downstream training model, logistic regression, random forest, xgb")
    parser.add_argument("-every_n_epochs", type = int, default = 1, help = "execute downstream model on every n epochs")
    parser.add_argument("-train_last_epoch_weights_only", action = argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if not os.path.basename(args.ssl_weights_root_fold).startswith(("simsiam","byol","dino", "mae")):
        error_msg = colored("Please input ssl weights root folder (i.e dino-is256-effbs256-ep10-bbResnet-dsDrnSen2a-clClUcl-nmTTDrnSatNM)", "red")
        raise KeyError(error_msg)
    
    # ------------------------- Load Configuration Files ------------------------- #

    with open("src/ssl_models/ssl_config.yml") as f:
        ssl_config = yaml.safe_load(f)

    with open("src/downstream_models/malaria_config.yml") as f:
        mlr_config = yaml.safe_load(f)
    
    # ---------------------- Using SSHSPH_MALARIA_My2 Class ---------------------- #
    
    df = pd.read_csv(args.mlr_data_file)
   
    feature_target_names = {
        "cat_feat_names" :  mlr_config["cat_feat"],
        "numeric_feat_names" :  mlr_config["numeric_feat"],
        "target_name" : mlr_config["target"]
    }
    #todo : We might want to add the data selection (X,y) here ...

    # ------------------------------ Choose Backbone ----------------------------- #

    backbone_name = os.path.basename(args.ssl_weights_root_fold).split("-")[4]
    backbone_name = backbone_name[2:]
    ssl_name =os.path.basename(args.ssl_weights_root_fold).split("-")[0]

    print(colored(f"Loading {backbone_name} backbone", "green"))
    match backbone_name:
        case "Resnet" :
            backbone_model = resnet50()
            backbone_model = torch.nn.Sequential(*list(backbone_model.children())[:-1])
        case "Swin":
            backbone_model = SwinTransformer(num_classes = 51)
        case "ViT":
            backbone_model = vit_base_patch16_224(num_classes = 0)
        case _:
            raise ValueError(colored("Incorrect backbone name found", "red"))

    # ---------------------------- Get hyperparameters --------------------------- #

    version_name = f"version_{args.version}"
    with open(os.path.join(args.ssl_weights_root_fold, version_name, "hparams.yaml")) as f:
        hyperparams = yaml.safe_load(f)

    model_params = {"batch_size" : hyperparams["batch_size"]}
    
    # ----------------------------- Choose SSL Model ----------------------------- #

    print(colored(f"Loading SSL model : {ssl_name}", "green"))
    match ssl_name:
        case "simsiam":
            simsiam_params = ssl_config["simsiam_params"]
            model_params.update(simsiam_params)
            
            if backbone_name == "Resnet":
                ssl_model = SimSiamBBResnet(model_params, backbone_model) 
            if backbone_name == "Swin":
                ssl_model = SimSiamBBSwinViT(model_params_backbone_model)
        case "byol":
            byol_params = ssl_config["byol_params"]
            model_params.update(byol_params)

            if backbone_name == "Resnet":
                ssl_model = ByolBBResnet(model_params, backbone_model) 
            if backbone_name == "Swin":
                ssl_model = ByolBBSwinViT(model_params, backbone_model)
        case "dino":
            dino_params = ssl_config["dino_params"]
            model_params.update(dino_params)

            if backbone_name == "Resnet":
                ssl_model = DinoBBResnet(model_params, backbone_model) 
            if backbone_name == "Swin":
                ssl_model = DinoBBSwinViT(model_params_backbone_model)
        case _:
            raise(KeyError(f"Model flag incorrect, got {ssl_name} but should get 'simsiam','byol' or 'dino' !"))

    # -------------------------- Choose Downstream Model ------------------------- #
    match args.down_model:
        case "lr":
            clf_model = LogisticRegression
        case "rf":
            clf_model = RandomForestClassifier
        case "xgb":
            clf_model = XGBClassifier

    # ------------------------------- Load Weights ------------------------------- #

    # Get all the paths in the SSL weights folder, only a few are checkpoints ...
    ssl_fold_paths = glob(os.path.join(args.ssl_weights_root_fold, version_name, "checkpoints", "*"))
    assert len(ssl_fold_paths) > 0, colored(f"Check if version no contains checkpoints", "red")
    # Filter just the checkpoints
    ssl_weights_paths = list(filter(lambda x: x.endswith("ckpt"), ssl_fold_paths))
    assert len(ssl_weights_paths) > 0, colored(f"No .ckpt files found", "red")
    # Checkpoint need to be sorted to ensure epoch 0 goes first
    ssl_weights_paths = sorted(ssl_weights_paths, key=lambda x: int(x.split('epoch=')[1].split('.')[0]))

    # If we want to just train only the wights from the last epoch only    
    if args.train_last_epoch_weights_only:
        print(colored("Training in last epoch checkpoint only ...", "green"))
        # get last ssl_weights_file
        ssl_weights_last_path = ssl_weights_paths[-1]

        # Downstream model training
        train_acc , val_acc = train_downstream_model(
            clf_model =clf_model,
            ssl_model= ssl_model, #SimSiam or BYOL
            ssl_weights_p= ssl_weights_last_path,
            backbone_name = backbone_name,
            ssl_strategy = ssl_name,
            feature_target_names=feature_target_names,
        )
        # Remove accucracy yaml file if exists as we want clean file to store results
        fname = f"{args.down_model}_last_epoch_acc.yml"
        if os.path.exists(os.path.join(args.ssl_weights_root_fold, version_name, fname)):
            os.remove(os.path.join(args.ssl_weights_root_fold, version_name, fname))

        #Write accuracy to yaml file
        tracker = {
            ssl_weight_last_path.split("/")[-1].split(":")[-1].split(".")[0] :  [train_acc, val_acc]
        }

        with open(os.path.join(args.ssl_weights_root_fold, version_name, fname), "w") as f:
            yaml.dump(tracker, f)

    # We are training on all model checkpoints and checking for learnt representaions
    else:
        print(colored("Training all model checkpoints ...", "green"))

        ssl_weights_paths = select_weights_on_nepochs(ssl_weights_paths, args.every_n_epochs)
        
        # Remove accucracy yaml file if exists as we want clean file to store results
        fname = f"{args.down_model}_every{args.every_n_epochs}epochs_acc.yml"
        if os.path.exists(os.path.join(args.ssl_weights_root_fold, version_name, fname)):
            os.remove(os.path.join(args.ssl_weights_root_fold, version_name, fname))

        for ssl_w_p in ssl_weights_paths:
            # Downstream model training
            train_acc , val_acc = train_downstream_model(
                clf_model =LogisticRegression,
                ssl_model= ssl_model, #SimSiam or BYOL
                ssl_weights_p= ssl_w_p,
                backbone_name = backbone_name,
                ssl_strategy = ssl_name,
                feature_target_names=feature_target_names,
            )
            # Keep track of losses
            tracker = {
                ssl_w_p.split("/")[-1].split(":")[-1].split(".")[0] : [train_acc,val_acc]
            }
            
            # Write results after every epoch to yaml file
            with open(os.path.join(args.ssl_weights_root_fold, version_name, fname), "a+") as f:
                yaml.dump(tracker, f, default_flow_style = False)

        # Read stored data to plot results
        with open(os.path.join(args.ssl_weights_root_fold, version_name, fname), "r") as f:
            acc_tracker = yaml.safe_load(f)

        # Loss Plots
        p = px.scatter()
        epochs,train_acc,val_acc = [],[],[]

        # put results into a lists, so its easier plot values
        for e,m in acc_tracker.items():
            epochs.append(int(e.split("=")[1]))
            train_acc.append(m[0])
            val_acc.append(m[1])

        # Plotting accuracy
        p.add_scatter(x = epochs, y = train_acc, name = "train accuracy")
        p.add_scatter(x = epochs, y = val_acc, name = "validation accuracy")
        p.update_layout(xaxis_title = "epochs", yaxis_title = "accuracy", template = "plotly_white", title = f"{ssl_name}-{backbone_name}-{args.down_model}")
        p.write_image(os.path.join(args.ssl_weights_root_fold, version_name, f"{args.down_model}_train_val_acc.png"))

        
        


