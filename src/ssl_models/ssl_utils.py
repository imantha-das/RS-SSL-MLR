import os
import sys
import shutil
from glob import glob

import numpy as np
import pandas as pd

import torch 
from torch.utils.data import Dataset 
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torch.utils.data import DataLoader, ConcatDataset
from lightly.data import LightlyDataset

from lightly.transforms import SimSiamTransform, BYOLTransform, MAETransform
from lightly.transforms.dino_transform import DINOTransform

from lightly.data import LightlyDataset
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightly.transforms import MAETransform

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from timm.models.vision_transformer import vit_base_patch16_224, VisionTransformer

from mae import MaeBBViT

import cv2
import rasterio

from typing import List, Tuple, Union

from tqdm import tqdm
import yaml

import plotly.express as px

from termcolor import colored
from torchvision.models import resnet50

from PIL import Image
Image.MAX_IMAGE_PIXELS = 200_000_000

#! Note for some reason torchvision.models swin_t does load the weights properly
sys.path.append("foundation_models/RSP/Scene Recognition/models")
sys.path.append("src/ssl_models/foundation_models/RSP/Scene Recognition/models")
from swin_transformer import SwinTransformer


# -------------------- To Load Model Weights ------------------- #

def load_model_weights(model_name:str,  path_to_weights = "pretrain_weights/rsp-aid-resnet-50-e300-ckpt.pth", num_classes = 51):
    """
    Desc : Loads pretrained weights to a resnet 50/swin-vit model from the RSP repository.
    The weight file (i.e rsp-aid-resnet-50-e300-ckpt.pth) consists of a linear layer with an output of 51 hence we have to set num_classes to 51
    Inputs 
        - model :resnet50 or SwinTransformer instance
        - model_name : string to identfy model by
        - path_to_weights : path to the file containing weight (last layer is a Linear Layer with 51 neurons)
        - num_classes : number of classes, for the weight file (rsp-aid-resnet-50-e300-ckpt.pth) we need to set num classes to 51
    Outputs
        - res50 : i.e Resnet50 pretrained model
    """
    def load_weights(model:Union[resnet50, SwinTransformer], path_to_weights:str = path_to_weights, num_classes:int = num_classes):
        model_ = model(num_classes = num_classes)
        model_state = torch.load(path_to_weights) 
        model_.load_state_dict(model_state["model"]) # we can add argument .load_state_dict( ... , strict = False) if the weights dont load properly, random weights will be intialised for the weights that do not work
        return model_ 
    
    match model_name:
        case "resnet":
            print(colored("Loading Resnet weights ...", "green"))
            try:
                pretrain_model = load_weights(resnet50,path_to_weights,num_classes = 51)
            except RuntimeError:
                print(colored("Check if the correct weights are loaded for Resnet Backbone model", "red")) 
        case "swin-vit":
            print(colored("Loading Swin-vit weights ...", "green"))
            try:
                pretrain_model = load_weights(SwinTransformer,path_to_weights,num_classes=51)
            except RuntimeError:
                print(colored("Check if the correct weights are loaded for Swin-Vit Backbone model", "red")) 

    return pretrain_model

# -------------- To verify if model weights are loaded properly ------------- #

def print_model_weights(model):
    """Print to check if the weights are loaded properly"""
    for name, param in model.named_parameters():
        print("-"*20)
        print(f"name : {name}")
        print(f"values : \n{param}")

# ==============================================================================
# Helper funcs for train ssl funcs
# ==============================================================================

# ----------- Helper function to can load sinlgle or dual datasets ----------- #

# This projects contains Sentinel and Sen2a as well as Datasets such as Million-Aid
# At times we want a dataloader to contain 2 datasets (i.e sen2a + drn) or a single
# dataset. This function handles 

def get_dataloaders(model_params:dict, 
                    drn_fold:Union[None,str] = None,
                    sat_fold:Union[None,str] = None,
                    ssl_drn_transforms:Union[None,SimSiamTransform, BYOLTransform, DINOTransform, MAETransform] = None, 
                    ssl_sat_transforms:Union[None,SimSiamTransform, BYOLTransform, DINOTransform, MAETransform] = None
                    ):
    """
    Returns Trainloader which can load one or two datasets (drones &/ sat)
    This projects contains multiple datasets sunch as Sentinel and drone images as well as open sources datasets
    such as Million-Aid. At times we want a dataloader to contain 2 datasets (i.e sen2a + drn) while at other time
    just a single dataset. This function makes it easy to hadle single or multiple datasets.

    Note this function uses LightlDataset function as such it cannot handle multispectral image data.
    """
    #* Note we donot need to pass in ToTensor as Lightly SSL Transforms already incorporates this !
    if drn_fold is not None:
        drn_trainset = LightlyDataset(input_dir = drn_fold, transform=Compose([ssl_drn_transforms])) # .__getitem__() returns -> view1,view2,fname
    if sat_fold is not None:
        sat_trainset = LightlyDataset(input_dir= sat_fold, transform=Compose([ssl_sat_transforms]))
    
    assert not((drn_fold is None) & (sat_fold is None)), colored("Please input atleast one data folder", "red")
    
    if (drn_fold is not None) & (sat_fold is not None):
        drnsat_trainset = ConcatDataset([drn_trainset, sat_trainset])
        # DataLoader
        trainloader = DataLoader(
            drnsat_trainset, 
            batch_size= int(model_params["eff_batch_size"] / (model_params["nodes"] * model_params["devices"])),
            num_workers = model_params["dataloader_workers"]
        )
    elif (drn_fold is not None) & (sat_fold is None):
        trainloader = DataLoader(
            drn_trainset,
            batch_size= int(model_params["eff_batch_size"] / (model_params["nodes"] * model_params["devices"])),
            num_workers = model_params["dataloader_workers"]
        )
    else:
        trainloader = DataLoader(
            sat_trainset,
            batch_size= int(model_params["eff_batch_size"] / (model_params["nodes"] * model_params["devices"])),
            num_workers = model_params["dataloader_workers"]
        )
    return trainloader

# ------------ Helper Func : Used by all train ssl algo functions ------------ #

def get_trainer(model_params, data_params):
    # Save name
    match data_params["sat_data_name"]:
        case "million_aid":
            sat_name = "milaid"
        case "gee_sat":
            sat_name = "sen2a"
        case "be_net":
            sat_name = "benet"
        case _:
            sat_name = ""
    match data_params["drn_data_name"]:
        case "sshsph_drn":
            drn_name = "drn"
        case _:
            drn_name = ""

    save_name = "-".join([
        f"{model_params['ssl_model']}",
        f"effbs{model_params['eff_batch_size']}",
        f"ep{model_params['epochs']}",
        f"bb{model_params['backbone'].capitalize()}",
        f"ds{sat_name + drn_name}",
    ])

    # Checkpoint + Logging
    logger = CSVLogger(save_dir = data_params["save_weights_fold"], name = save_name)
    checkpoint_callback = ModelCheckpoint(
        #dirpath=os.path.join(args.save_weights_fold, save_name), 
        filename="epoch:{epoch}",
        save_on_train_epoch_end=True,
        save_weights_only = data_params["save_weights_only"],
        save_top_k = -1,
        every_n_epochs = data_params["save_freq"]
    )

    # Model Traiining
    trainer = pl.Trainer(
        default_root_dir = os.path.join(data_params["save_weights_fold"], save_name),
        devices = -1,
        num_nodes= model_params["nodes"],
        accelerator = "gpu",
        strategy = "ddp" if model_params["backbone"] == "resnet" else DDPStrategy(find_unused_parameters = True),
        max_epochs = model_params["epochs"],
        precision = model_params["precision"],
        logger = logger,
        callbacks = [checkpoint_callback],
        #auto_scale_batch_size = config.AUTO_SCALE_BATCH_SIZE # to find "max" batch_size that can be procesedwith resources (gpu)
    )
    return trainer 

# ==============================================================================
# Train SSL funcs
# ==============================================================================

# -------- Train MAE function for ssl_pretrain or ssl_finetune scripts ------- #

def train_mae(model_params:dict, data_params:dict,config:dict, backbone_name:str, pretrain_weights_file:Union[str,None]):

    # MAE Transforms
    # If there are is a satelitle folder path mentioned find mae transforms for each dataset
    if data_params["sat_fold_path"]:
        match data_params["sat_data_name"]:
            case "million_aid":
                mae_sat_trans = MAETransform(
                    normalize  ={"mean" : config["milaid_img_mean"], "std" : config["milaid_img_std"]}
                )
            case "gee_sat":
                mae_sat_trans = MAETransform(
                    normalize  ={"mean" : config["sen2a_img_mean"], "std" : config["sen2a_img_std"]}
                )
            case "be_net":
                mae_sat_trans = MAETransform(
                    normalize  ={"mean" : config["benet_img_mean"], "std" : config["benet_img_std"]}
                )
    # If there isnt just pass a "None" value for transforms
    else:
        mae_sat_trans = None

    if data_params["drn_fold_path"]:
        mae_drn_trans = MAETransform(
            normalize ={"mean" : config["drn_img_mean"], "std" : config["drn_img_std"]}
        )
    else:
        mae_drn_trans = None

    #DataLoader
    trainloader = get_dataloaders(
        model_params = model_params, #batchsixe etc passed as model_params
        drn_fold = data_params["drn_fold_path"],
        sat_fold = data_params["sat_fold_path"],
        ssl_drn_transforms = mae_drn_trans,
        ssl_sat_transforms = mae_sat_trans
    ) #* Note MAE transforms ouputs a shape of (*,3,224,224)

    assert backbone_name == "vit", colored("MAE requires a VIT backbone", "red")
    if pretrain_weights_file:
        #todo : we need to use get_pretrained_backbone func where we load weights
        pass
    else:
        backbone = vit_base_patch16_224(num_classes = 0, pretrained = True)

    mae = MaeBBViT(model_params, backbone)
    
    trainer = get_trainer(model_params, data_params)
    trainer.fit(mae, trainloader, ckpt_path = data_params["ckpt_path"])

if __name__ == "__main__":
    
    # -------------------------- Cleaning image dataset -------------------------- #
    #clean_image_dataset()
    with open("src/ssl_models/ssl_config.yml","r") as f:
        config = yaml.safe_load(f)


    # ------------------------ Using Sentinel Dataset class ------------------------ #
    image_paths = glob("data/interim/gee_sat/sen2a_c3_256x_pch/*")
    sentinel_dataset= SentinelDataset(
        image_paths, 
        scaling_factor= 10000,
        transform = Compose([Normalize(mean = config["sat_img_mean"], std = config["sat_img_std"])]),
    )
    img,path = sentinel_dataset.__getitem__(1)
    #print(f"img : {img} \nmin : {img.min()} , max : {img.max()}")

    # ------------------------- Using Drone Dataset Class ------------------------ #
    image_paths = glob("data/processed/sshsph_drn/drn_c3_256x_pch/*")
    drone_dataset= DroneDataset(
        image_paths, 
        scaling_factor= 255,
        transform = Compose([Normalize(mean = config["drn_img_mean"], std = config["drn_img_std"])]),
    )
    img,path = drone_dataset.__getitem__(50)
    #print(f"img : {img} \nmin : {img.min()} , max : {img.max()}")   


    # #* -------------------- To compute mean & std of dataset(s) ------------------- #
    # image_paths = glob("data/interim/gee_sat/sen2a_c3_256x_pch/*")
    # # df = pd.read_csv("data/SSHSPH-malaria-prevelance-v1.0/malaria_pts_with_images.csv")
    # # train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 0)
    
    # mean_std_dataset = DATASET_MEAN_STD(SentinelDataset, dataset_args= {},batch_size = 512)
    # mean,std = mean_std_dataset.get_mean_std_sentinel_or_drone(image_paths)
    # #mean, std = mean_std_dataset.get_sshsph_my_malaria_stats(train_df)
    # print(f"Mean : {mean}, Std : {std}")

    # ------------------------- Test Max Min computation ------------------------- #
    dataset_args = {
        "paths" : glob("data/processed/sshsph_drn/drn_c3_256x_pch/*"), 
        "scaling_factor" : None, 
        "transform" : Compose([
            Normalize(mean = config["drn_img_mean"], std = config["drn_raw_img_std"])
            ])
        }   
    mins_all_imgs, maxs_all_imgs = get_maxmin_stats(DroneDataset,dataset_args, save_suffix="norm")
    
    #todo : Explore Mins and Maxs and see why values go above the 1's. Note we havent scaled which should also be tested









