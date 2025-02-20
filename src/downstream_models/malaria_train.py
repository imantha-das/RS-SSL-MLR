# ---------------------------------- Imports --------------------------------- #
import os
import re
import sys
from glob import glob
import numpy as np
import pandas as pd

import rasterio.errors
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.models import resnet50
from sklearn.compose import make_column_transformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from termcolor import colored
from malaria_utils import load_ssl_weights, setup_logger

from typing import Tuple, Union,List
import rasterio
import yaml
import json
import warnings
from tqdm import tqdm
import plotly.express as px
import pickle
import time
from datetime import datetime, timedelta
import argparse
import logging

from timm.models.vision_transformer import vit_base_patch16_224, VisionTransformer

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

sys.path.append("src/ssl_models")

from simsiam import SimSiamBBResnet, SimSiamBBSwinViT
from byol import ByolBBResnet, ByolBBSwinViT
from dino import DinoBBResnet, DinoBBSwinViT, DinoBBViT
from mae import MaeBBViT
from ssl_utils import print_model_weights

sys.path.append("src/ssl_models/foundation_models/RSP/Scene Recognition/models")
from swin_transformer import SwinTransformer

# ------------------------- Read Configuration Files ------------------------- #
with open("src/ssl_models/ssl_config.yml") as f:
    ssl_config = yaml.safe_load(f)

with open("src/downstream_models/malaria_config.yml") as f:
    mlr_config = yaml.safe_load(f)

def model_no_spatial_effects(mlr_data:tuple, clf_model:Union[LogisticRegression, RandomForestClassifier, XGBClassifier],feat_names:tuple):
    """This is a model for just malaria data without any spatial effects. We need to check
    if adding any spatial effects improve results. So we test a model"""
    # Get X_train, y_train ... values packaged as tuple
    X_train, X_val, y_train, y_val = mlr_data 
    num_cols, cat_cols = feat_names
    # Setup preprocessor to handle cat variables
    preprocessor = ColumnTransformer(
    transformers = [
        ("num", MinMaxScaler(), num_cols),
        ("cat", OneHotEncoder(), cat_cols)
        ]
    )
    # Instantiate downstream model and train
    clf = Pipeline(
    steps = [
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression())
        ]
    )
    clf.fit(X_train, y_train)
    # Trainset accuracy
    yhat_train = clf.predict(X_train)
    fpr,tpr,thresh = roc_curve(y_train, yhat_train)
    # AUC
    train_auc = auc(fpr,tpr)
    # F1-score
    train_f1 = f1_score(y_train, yhat_train)
    # Accuracy
    train_acc = accuracy_score(y_train, yhat_train)
    # Compute auc for val set
    yhat_val = clf.predict(X_val)
    fpr,tpr,thresh = roc_curve(y_val, yhat_val)
    # AUC
    val_auc = auc(fpr,tpr)
    # F1-score
    val_f1 = f1_score(y_val, yhat_val)
    # Accuracy
    val_acc = accuracy_score(y_val, yhat_val)
    print(colored(f"No spatial effects models train/val auc: {train_auc:.3f} , {val_auc:.3f}","blue"))
    return train_auc,  val_auc, train_f1, val_f1, train_acc, val_acc

# ------ Class that return Image as Tensor and Malaria features & target ----- #

class Get_Image_Tensors_Mlr_Feats(Dataset):
    """
    Returns Image as torch.Tensor and Malaria features seperately
    The Image tensors then need to be passed through trained encoder,
    which is done by a seperate class
    """
    def __init__(
        self,df:pd.DataFrame,
        img_transforms:Union[None, dict],
        num_feats:List[str], 
        cat_feats:List[str], 
        target:List[str],
        save_loc:str, 
        preprocessor:Union[None,ColumnTransformer],
        c:int = 3
        )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        super(Get_Image_Tensors_Mlr_Feats,self).__init__()
        self.df = df
        self.c = c

        # Image Transformations
        self.img_transforms = img_transforms

        self.target = target

        # Sero Prev feature preprocessing
        # We must filter the required columns only
        X = self.df[cat_feats + num_feats]
        
        # if there is a preprocessor : apply preprocessor, for validation data
        if preprocessor:
            self.preprocessor = preprocessor
            self.X_prev = self.preprocessor.transform(X)
        # if there is no preprocessor : declare and train a preprocessor, for train dataset  
        else:
            self.preprocessor = ColumnTransformer(
                transformers = [
                    ("num", MinMaxScaler(), num_cols),
                    ("cat", OneHotEncoder(handle_unknown = "ignore"), cat_cols),
                ], remainder = "passthrough"
            )
            self.X_prev = self.preprocessor.fit_transform(X)
        # We need to save transformer as .p0kl file to be used later for testing set
        with open(os.path.join(save_loc, "minmax_ohe_processor.pkl"),"wb") as f:
            pickle.dump(self.preprocessor, f)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx:int):
        # Get a row 
        df_row = self.df.iloc[idx]
        # Load image data
        with rasterio.open(df_row["selected_imgp"]) as ds:
            img = ds.read([i+1 for i in range(self.c)]) #i.e [1,2,3]
        
        img = np.moveaxis(img, source=[0,1,2], destination = [2,0,1])

        # Convert images to torch tensors and apply necessary transforms
        img_fold_name = os.path.dirname(df_row["selected_imgp"]).split("/")[-1]
        rs_name = img_fold_name.split("_")[0] # sen2a or drn
        
        if self.img_transforms:
            # Get transform based on rs_name ("sen2a","drn")
            transform = self.img_transforms[rs_name] # image transform : Compose([ToTensor(), Normalize(...)])
            # apply transform
            X_img = transform(img) #(3,256,256)
            
        # Get seroprevalence related features
        #* We will be keeping this in numpy as downstream model is sklearn model
        X_prev= self.X_prev[idx, :] #i.e (50,) 
        # Convert to torch tensors as the Dataloader will automatically convert otherwise
        X_prev = torch.tensor(X_prev)
 
        # Get target for each index
        y = self.df[self.target].iloc[idx].values
        y = torch.from_numpy(y)

        return X_img, X_prev, y #(Torch.Tensor, np.ndarray, np.ndarray)

# ------------------------- Get Geo Vector from Image ------------------------ #

def get_geo_vector(
    X_img:torch.Tensor,
    ssl_model:Union[SimSiamBBResnet,SimSiamBBSwinViT,ByolBBResnet, ByolBBSwinViT,DinoBBResnet, DinoBBSwinViT,MaeBBViT],
    ssl_weight_p:str,
    backbone_name:str,
    ssl_strategy:str,
    logger:logging.Logger,
    train_or_val:str
    ):
    """Function that applies self-supervised pretrain model to an Image tensor to get X_geo"""
    
    # Load SSL Weights to model : Finetuned
    try:
        print(colored(f"Loading ssl weights : {ssl_weight_p}", "green"))
        ssl_model = load_ssl_weights(ssl_model, ssl_weight_p)
    except:
        RuntimeError(colored("Failed to load weights", "red")) 
    
    # Extract backbone based on the following conditions
    match ssl_strategy:
        case "simsiam" | "byol":
            match backbone_name:
                case "resnet":
                    ssl_model_bb = ssl_model.backbone
                case "swin-vit":
                    ssl_model_bb = ssl_model.backbone_model
        case "dino":
            ssl_model_bb = ssl_model.student_backbone
        case "mae":
            ssl_model_bb = ssl_model.backbone
        case _:
            raise ValueError(f"Invalid ssl strategy : {ssl_stategy}")

    start = time.time()
    with torch.no_grad():
        X_geo = ssl_model_bb(X_img) #for resnet you get shape (*,2048,1,1)
        X_geo = X_geo.flatten(start_dim = 1) #(*,2048) or (*, 768)

    end = time.time()
    elapsed_time = (timedelta(seconds = end-start))
    print(f"Time taken for infering geo features : {elapsed_time} sec")
    epoch_num_for_weight = ssl_weight_p.split('epoch=')[1].split('.')[0]
    logger.info(f"{train_or_val}, epoch (weight) : {epoch_num_for_weight} , time taken for execution : {elapsed_time} mins")
    return X_geo

# ----------------------------- Downstream Model ----------------------------- #
def model_with_spatial_effects(
    mlr_data:tuple, 
    clf_model:Union[LogisticRegression, RandomForestClassifier, XGBClassifier], 
    met_save_name:str, 
    logger:logging.Logger,
    epoch_num_for_weights:str):
    """Downstream malaria prediction model
    Note that X has already been preprocessed by Get_Image_Tensors_Mlr_Feats class"""
    # Get X_train, y_train ... values packaged as tuple
    X_train, X_val, y_train, y_val = mlr_data 
    # Instantiate model
    clf = clf_model()
    # Train Model keep track of time 
    start = time.time()
    clf.fit(X_train, y_train)
    # Train Prediction
    yhat_train = clf.predict(X_train)
    # Validation prediction
    yhat_val = clf.predict(X_val)
    end = time.time()
    elapsed_time = timedelta(seconds = end - start)
    logger.info(f"train & val downstream training + preds, epoch (weight) : {epoch_num_for_weights}, time taken : {elapsed_time}")

    # Compute accuracy for train & validation
    # ROC
    fpr,tpr,thresh = roc_curve(y_train, yhat_train)
    # AUC
    train_auc = auc(fpr,tpr)
    # F1-score
    train_f1 = f1_score(y_train, yhat_train)
    # Accuracy
    train_acc = accuracy_score(y_train, yhat_train)
    # ROC
    fpr,tpr,thresh = roc_curve(y_val, yhat_val)
    # AUC
    val_auc = auc(fpr,tpr)
    # F1-score
    val_f1 = f1_score(y_val, yhat_val)
    # Accuracy
    val_acc = accuracy_score(y_val, yhat_val)

    # To Save
    acc_met = pd.DataFrame({
        "epoch_weight" : [epoch_num_for_weights],
        "train_auc" : [train_auc], 
        "train_f1" : [train_f1], 
        "train_acc" : [train_acc],
        "val_auc" : [val_auc],
        "val_f1" : [val_f1],
        "val_acc" : [val_acc]
    }).round(5)

    if not os.path.exists(met_save_name):
        acc_met.to_csv(met_save_name, index = False)
    else:
        acc_met.to_csv(met_save_name, mode = "a", header = False, index = False)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    # --------------------------------- Arguments -------------------------------- #
    parser = argparse.ArgumentParser(description="Argument Parser for Malaria Classifier")
    parser.add_argument("-mlr_csv", type = str, default = "data/processed/sshsph_mlr/mlr_nomiss_vardrop_train_v2.csv", help = "path to processed malaria dataset")
    parser.add_argument("-ssl_weights_root", type = str, help = "Path to root folder containing weights, hyperparam etc. This is NOT the version folder but one folder above")
    parser.add_argument("-version", type = int, default = 0, help = "ssl weights root folder will contain version numbers that containt the content")
    parser.add_argument("-down_clf", type = str, default = "lr", choices = ["lr", "rf", "xgb"], help = "Downstream classifier")
    parser.add_argument("-weight_every_n_epochs", type = int, default = 1, help = "Load ssl weights every n epochs to prevent long training times")
    #parser.add_argument("-train_last_epoch_weights_only", action = argparse.BooleanOptionalAction, help = "Train downstream model on ssl weights saved during last epoch")
    parser.add_argument("-weight_at_epoch", type = int, default = None, help = "Load a weight saved at a specific epoch")
    args = parser.parse_args()
    # --------------------------------- Save Root -------------------------------- #
    save_root = os.path.join(args.ssl_weights_root, f"version_{args.version}", "downstream")
    # Create a folder to store all the downstream information
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    # --------------------------- Read Malaria Dataset --------------------------- #

    # Read malaria dataset
    df = pd.read_csv(args.mlr_csv)

    # Filter only required colums
    cat_cols = mlr_config["cat_feat"]
    num_cols = mlr_config["numeric_feat"]
    tar_col = mlr_config["target"]
    
    # --------------------- Malaria Data : Train - Val Split --------------------- #
    df_train, df_val= train_test_split(df, test_size = mlr_config["test_size"], shuffle = False, random_state=0)
    X_train = df_train[num_cols + cat_cols]
    y_train = df_train[tar_col]
    X_val = df_val[num_cols + cat_cols]
    y_val = df_val[tar_col]
    
    # Package these to be used easily in functions
    mlr_data = (X_train, X_val, y_train, y_val)
    feat_names = (num_cols, cat_cols)
    
    # ----------------------------- Load hyperparams ----------------------------- #
    # Desc : Load hyperparameters file saved inside ssl_weights_root/"version_x"

    version_name = f"version_{args.version}"
    with open(os.path.join(args.ssl_weights_root, version_name, "hparams.yaml")) as f:
        hyper_config = yaml.safe_load(f)

    # Update model params with necessary hyperparameters : required to instantiate SimSiam model class for example
    model_params = {
        "batch_size" : hyper_config["batch_size"],
        "lr" : hyper_config["backbone"]
    }

    # ---------------------- Identify backbone + ssl method ---------------------- #
    # Get backbone from hyperparams 
    backbone_name = hyper_config['backbone']
    print(colored(f"Identifying backbone : {backbone_name}", "green"))
    match hyper_config["backbone"]:
        case "resnet" :
            backbone_model = resnet50()
            backbone_model = torch.nn.Sequential(*list(backbone_model.children())[:-1])
        case "swin-vit":
            backbone_model = SwinTransformer(num_classes = 51)
        case "vit":
            backbone_model = vit_base_patch16_224(num_classes = 0)
        case _:
            raise ValueError(colored("Incorrect backbone name found", "red"))

    # Get ssl method used from ssl_weights_root folder name
    ssl_name =os.path.basename(args.ssl_weights_root).split("-")[0] 
    print(colored(f"Instatiating ssl_model ({ssl_name}), with backbone ({backbone_name})", "green"))

    # --------------------------- Instantiate SSL Model -------------------------- #
    match ssl_name:
        case "simsiam":
            simsiam_params = ssl_config["simsiam_params"]
            model_params.update(simsiam_params)
            if backbone_name == "resnet":
                ssl_model = SimSiamBBResnet(model_params, backbone_model) 
            if backbone_name == "swin-vit":
                ssl_model = SimSiamBBSwinViT(model_params_backbone_model)
        case "byol":
            byol_params = ssl_config["byol_params"]
            model_params.update(byol_params)
            if backbone_name == "resnet":
                ssl_model = ByolBBResnet(model_params, backbone_model) 
            if backbone_name == "swin-vit":
                ssl_model = ByolBBSwinViT(model_params, backbone_model)
        case "dino":
            dino_params = ssl_config["dino_params"]
            model_params.update(dino_params)
            if backbone_name == "resnet":
                ssl_model = DinoBBResnet(model_params, backbone_model) 
            if backbone_name == "swin-vit":
                ssl_model = DinoBBSwinViT(model_params_backbone_model)
            if backbone_name == "vit":
                ssl_model = DinoBBViT(model_params, backbone_model)
        case "mae":
            mae_params = ssl_config["mae_params"]
            model_params.update(mae_params)
            if backbone_name == "vit":
                ssl_model = MaeBBViT(model_params, backbone_model)
        case _:
            raise(KeyError(f"Model flag incorrect, got {ssl_name} but should get 'simsiam','byol','dino','mae"))

    # -------------------------- Choose Downstream Model ------------------------- #
    match args.down_clf:
        case "lr":
            clf_model = LogisticRegression
        case "rf":
            clf_model = RandomForestClassifier
        case "xgb":
            clf_model = XGBClassifier
 
    # ---------------- Train and validate no spatial effects model --------------- #
    train_auc, val_auc, train_f1, val_f1, train_acc, val_acc = model_no_spatial_effects(mlr_data, clf_model, feat_names)
    acc_met_no_spatial_df = pd.DataFrame({
        "auc" : [train_auc, val_auc], 
        "f1" : [train_f1, val_f1], 
        "acc" : [train_acc, val_acc],
        "dataset" : ["train","val"]
    }).round(3)
    # Save Dataframe
    acc_met_no_spatial_df.set_index("dataset", inplace = True)
    acc_met_no_spatial_df.to_csv(os.path.join(save_root,f"metrics_mlr_{args.down_clf}_no_spatial.csv"))

    # -------------------------------- Load Images ------------------------------- #
    # As Vit require 224 paches we need to writ and if else clause. Note this isnt really necessary for Swin
    # as the model already has incorporated resizing.

    if backbone_name == "resnet":
        transforms = {
            "sen2a" : Compose([ToTensor(), Normalize(ssl_config["sat_img_mean"], ssl_config["sat_img_std"])]),
            "drn" : Compose([ToTensor(), Normalize(ssl_config["drn_img_mean"], ssl_config["drn_img_std"])])
        }
    else:
        # Resizing ... to work with ViT
        transforms = {
            "sen2a" : Compose([ToTensor(), Resize(224), Normalize(ssl_config["sat_img_mean"], ssl_config["sat_img_std"])]),
            "drn" : Compose([ToTensor(), Resize(224), Normalize(ssl_config["drn_img_mean"], ssl_config["drn_img_std"])])
        }


    #* Not this dataset HASN'T run ssl train encododer on image. We will be doing this to save time
    #* on loading images
    train_img_prev_dataset = Get_Image_Tensors_Mlr_Feats(
        df = df_train,
        img_transforms = transforms,
        num_feats = num_cols,
        cat_feats = cat_cols,
        target = tar_col,
        save_loc = save_root,
        preprocessor = None,
        c = 3
    )
    preprocessor = train_img_prev_dataset.preprocessor

    val_img_prev_dataset = Get_Image_Tensors_Mlr_Feats(
        df = df_val,
        img_transforms = transforms,
        num_feats = num_cols,
        cat_feats = cat_cols,
        target = tar_col,
        save_loc = save_root,
        preprocessor = preprocessor,
        c = 3
    )

    start = time.time()
    train_img_prev_loader = DataLoader(
        train_img_prev_dataset, 
        batch_size = len(train_img_prev_dataset), 
        shuffle = False,
    )
    val_img_prev_loader = DataLoader(val_img_prev_dataset, batch_size = len(val_img_prev_dataset), shuffle = False)
    
    X_train_img, X_train_prev, y_train = next(iter(train_img_prev_loader))
    X_val_img, X_val_prev, y_val = next(iter(val_img_prev_loader))
    end = time.time()

    print(colored(f"Time executed loading images : {(end-start):.3f}s", "green"))


    # ------------------- Get Geo Vector by applying SSL model ------------------- #
    # Locate ssl weights
    ssl_weights = glob(os.path.join(args.ssl_weights_root, f"version_{args.version}","checkpoints","*"))
    assert len(ssl_weights) > 0, colored(f"Check if version no contains checkpoints", "red")
    # Sort weights based on epoch file name as two digits (99) and single digit numbers (9) are confused
    ssl_weights = sorted(ssl_weights, key=lambda x: int(x.split('epoch=')[1].split('.')[0]))

    # Select weight at a single epoch specified
    if args.weight_at_epoch:
        ssl_weights = [os.path.join(args.ssl_weights_root,f"version_{args.version}","checkpoints",f"epoch:epoch={args.weight_at_epoch}")] # put last weight in a list so that it can be still be used inside a for loop
    # Select a muliple weights based on epoch frequency
    else:
        ssl_weights = ssl_weights[::args.weight_every_n_epochs]

    #Create log file to write
    logger = setup_logger(save_root)

    # Delete file that save metrics at start of the for loop
    met_save_name = os.path.join(save_root, f"metrics_mlr_{args.down_clf}_with_spatial.csv")
    if os.path.exists(met_save_name):
        os.remove(met_save_name)

    # We need to convert y outside the loop as its NOT returned by get_geo_vector()
    y_train = y_train.cpu().detach().numpy()
    y_val = y_val.cpu().detach().numpy()

    # Loop through each of the ssl weights ...
    for ssl_weight in ssl_weights:
        # Get Geospatial features by applying ssl finetuned models on images
        X_train_geo = get_geo_vector(
            X_img = X_train_img, 
            ssl_model = ssl_model, 
            ssl_weight_p = ssl_weight,
            backbone_name = backbone_name,
            ssl_strategy = ssl_name,
            logger = logger,
            train_or_val = "train"
        )
        X_val_geo = get_geo_vector(
            X_img = X_val_img, 
            ssl_model = ssl_model, 
            ssl_weight_p = ssl_weight,
            backbone_name = backbone_name,
            ssl_strategy = ssl_name,
            logger = logger,
            train_or_val = "val"
        )
        # Concatenate Geo & Sero-prevalence features
        #print(colored(X_train_geo.shape, "yellow"), colored(X_train_prev.shape, "yellow"))
        X_train_geo_prev = torch.hstack([X_train_geo, X_train_prev])
        X_val_geo_prev = torch.hstack([X_val_geo, X_val_prev])

        # Convert torch tensors to numpy arrays
        X_train_geo_prev = X_train_geo_prev.cpu().detach().numpy()
        X_val_geo_prev = X_val_geo_prev.cpu().detach().numpy()

        # Train downstream model
        epoch_num_for_weights = ssl_weight.split('epoch=')[1].split('.')[0]
        model_with_spatial_effects(
            mlr_data = (X_train_geo_prev, X_val_geo_prev, y_train, y_val), 
            clf_model = clf_model,
            met_save_name = met_save_name,
            logger = logger,
            epoch_num_for_weights = epoch_num_for_weights
        )

