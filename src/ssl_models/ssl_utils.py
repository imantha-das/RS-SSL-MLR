import os
import sys
import shutil
from glob import glob

import numpy as np
import pandas as pd

import torch 
import torch.nn as nn
from torch.utils.data import Dataset 
from torchvision.transforms.v2 import ToImage, Normalize, Compose, Resize
from torch.utils.data import DataLoader, ConcatDataset
from lightly.data import LightlyDataset

from lightly.transforms import (SimSiamTransform, BYOLTransform, MAETransform, DINOTransform,
BYOLView1Transform, BYOLView2Transform)
from lightly.data import LightlyDataset
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from torchvision.models import resnet50

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from timm.models.vision_transformer import vit_base_patch16_224, VisionTransformer


from simsiam import SimSiamBBResnet, SimSiamBBSwinViT
from byol import ByolBBResnet, ByolBBSwinViT
from dino import DinoBBResnet, DinoBBSwinViT, DinoBBViT
from mae import MaeBBViT
from satmae import MaskedAutoencoderGroupChannelViT, SatMaeGroupViTBB

import cv2
import rasterio

from functools import partial
from typing import List, Tuple, Union

from tqdm import tqdm
import yaml

import plotly.express as px

from termcolor import colored


from PIL import Image
Image.MAX_IMAGE_PIXELS = 200_000_000

#! Note for some reason torchvision.models swin_t does load the weights properly
sys.path.append("foundation_models/RSP/Scene Recognition/models")
sys.path.append("src/ssl_models/foundation_models/RSP/Scene Recognition/models")
from swin_transformer import SwinTransformer

with open("src/ssl_models/ssl_config.yml", "r") as f:
    config = yaml.safe_load(f)

# ==============================================================================
# Helper funcs for train ssl funcs
# ==============================================================================

# ----------- Load Pretrained Model, Not backbone feature extractor ---------- #

def load_model_weights(model_name:str,  path_to_weights, num_classes):
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
        # Handle Resnet / Swin-Vit seperately as they come from RSP repo
        if model_name == "resnet" or model_name == "swin-vit":
            model_ = model(num_classes = num_classes)
            model_state = torch.load(path_to_weights) 
            model_.load_state_dict(model_state["model"]) # we can add argument .load_state_dict( ... , strict = False) if the weights dont load properly, random weights will be intialised for the weights that do not work
        # Handle ViT seperately as its trained in this repo
        else:
            model_ = MaeBBViT(
                model_params = {"lr" : None}, 
                backbone = model(num_classes = num_classes)
            )
            model_state = torch.load(path_to_weights)
            model_.load_state_dict(model_state["state_dict"])
        return model_ 
    
    match model_name:
        case "resnet":
            print(colored("Loading Resnet weights ...", "green"))
            try:
                pretrain_model = load_weights(resnet50,path_to_weights,num_classes = num_classes)
            except RuntimeError:
                print(colored("Unable to load weigts, Check if weights correspond to resnet model", "red")) 
        case "vit":
            print(colored("Loading ViT weights ...", "green")) 
            try:
                pretrain_model = load_weights(vit_base_patch16_224, path_to_weights, num_classes = num_classes)
            except RuntimeError:
                print(colored("Unable to load weights, check if weights correspond to vit model"))
        case "swin-vit":
            print(colored("Loading Swin-vit weights ...", "green"))
            try:
                pretrain_model = load_weights(SwinTransformer,path_to_weights,num_classes= num_classes)
            except RuntimeError:
                print(colored("Unable to load weigts, Check if weights correspond to swin-vit model", "red")) 

    return pretrain_model

# ---------------------- Load backbone feature extractor --------------------- #

def get_pretrained_backbone(backbone_name:str, pretrain_weights_file:str):
    """Loads model weights"""
    # Load model weights
    match backbone_name:
        case "resnet":
            # Load model weights for resnet
            resnet_bb_model = load_model_weights(model_name = backbone_name, path_to_weights=pretrain_weights_file, num_classes=51)
            # Get backbone 
            resnet_bb = torch.nn.Sequential(*list(resnet_bb_model.children())[:-1]) 
            return resnet_bb
        case "vit":
            # Load model for MAE with ViT 
            vit_bb_model = load_model_weights(model_name = backbone_name, path_to_weights = pretrain_weights_file, num_classes = 0)
            # Get ViT backbone
            vit_bb = vit_bb_model.backbone
            return vit_bb
        case "swin-vit":
            # Load model weights for Swin-vit
            swinvit_bb_model = load_model_weights(model_name = backbone_name, path_to_weights=pretrain_weights_file, num_classes=51)
            # we dont extract a backbone, rather we use .forward_features from SwinTranformer Class to get features
            return swinvit_bb_model
        case _:
            raise(KeyError("incorrect argument to 'backbone', please enter 'resnet', 'vit' or 'swin-vit'"))


# -------------- To verify if model weights are loaded properly ------------- #

def print_model_weights(model):
    """Print to check if the weights are loaded properly"""
    for name, param in model.named_parameters():
        print("-"*20)
        print(f"name : {name}")
        print(f"values : \n{param}")


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
        devices = model_params["devices"],
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

def train_mae(model_params:dict, data_params:dict, backbone_name:str, pretrain_weight_file:Union[str,None]):

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
                    normalize  ={"mean" : config["sat_img_mean"], "std" : config["sat_img_std"]}
                )
            case "be_net":
                mae_sat_trans = MAETransform(
                    normalize  ={"mean" : config["benet_img_mean"], "std" : config["benet_img_std"]}
                )
    # If there isnt a path for satellite, just pass a "None" value for transforms
    else:
        mae_sat_trans = None

    if data_params["drn_fold_path"]:
        mae_drn_trans = MAETransform(
            normalize ={"mean" : config["drn_img_mean"], "std" : config["drn_img_std"]}
        )
    # If there isnt a path for drone data, just pass a "None" value for transforms
    else:
        mae_drn_trans = None

    #DataLoader : The "get_dataloaders" function "combines" if two datasets are present
    trainloader = get_dataloaders(
        model_params = model_params, #batchsixe etc passed as model_params
        drn_fold = data_params["drn_fold_path"],
        sat_fold = data_params["sat_fold_path"],
        ssl_drn_transforms = mae_drn_trans,
        ssl_sat_transforms = mae_sat_trans
    ) #* Note MAE transforms ouputs a shape of (*,3,224,224)

    assert backbone_name == "vit", colored("MAE requires a ViT backbone", "red")
    # Finetuning
    if pretrain_weight_file:
        vit_bb = get_pretrained_backbone(backbone_name, pretrain_weight_file)
    # Pretraining
    else:
        vit_bb = vit_base_patch16_224(num_classes = 0, pretrained = True)

    mae = MaeBBViT(model_params, vit_bb)
    
    trainer = get_trainer(model_params, data_params)
    # Fine tuning
    if pretrain_weight_file:
        trainer.fit(mae, trainloader)
    # Pretraining  #todo : Not sure why ckpt_path required for pretraining
    else:
        trainer.fit(mae, trainloader, ckpt_path = data_params["ckpt_path"])

# ==============================================================================
# SimSiam Training function
# ==============================================================================

def train_simsiam(model_params:dict, data_params:dict, backbone_name:str, pretrained_weight_file:str)->None:
    
    # If we are using drone images, get transforms
    if data_params["drn_fold_path"]:
        simsiam_drn_transforms = SimSiamTransform(
            input_size = model_params["input_size"] if backbone_name == "resnet" else 224, 
            normalize = {"mean" : config["drn_img_mean"], "std" : config["drn_img_std"]}
        )
    else:
        simsiam_drn_transforms = None

    # If we are using satelite images, get transforms
    if data_params["sat_fold_path"]:
        simsiam_sat_transforms = SimSiamTransform(
            input_size = model_params["input_size"] if backbone_name == "resnet" else 224, 
            normalize = {"mean" : config["sat_img_mean"], "std" : config["sat_img_std"]}
        )
    else:
        simsiam_drn_transforms = None

    # DataLoader
    drnsen2a_trainloader = get_dataloaders(
        model_params,
        drn_fold = data_params["drn_fold_path"],
        sat_fold = data_params["sat_fold_path"],
        ssl_drn_transforms = simsiam_drn_transforms, 
        ssl_sat_transforms = simsiam_sat_transforms
    )
  
    # Select correct SimSiam class w.r.t the backbone
    if backbone_name == "resnet":
        resnet_bb = get_pretrained_backbone(backbone_name, pretrained_weight_file)
        simsiam = SimSiamBBResnet(model_params, resnet_bb)
        #print_model_weights(simsiam.backbone_model)
    elif backbone_name == "vit":
        #! We have saved ViT backbone itself so this is not needed use "get_pretrained_backbone"
        #vit_bb = get_pretrained_backbone(backbone_name, pretrained_weight_file)
        #todo : Need to implement SimSiam model for ViT backbone
        raise Exception("SimSiamBBViT not implemented yet !")
    elif backbone_name == "swin-vit":
        swinvit_bb_model = get_pretrained_backbone(backbone_name, pretrained_weight_file)
        simsiam = SimSiamBBSwinViT(model_params, swinvit_bb_model)
    else:
        raise KeyError("Incorrect value passed to 'backbone_name")

    # Train SimSiam model
    trainer = get_trainer(model_params, data_params)
    trainer.fit(simsiam, drnsen2a_trainloader)

# ==============================================================================
# Byol Training function
# ==============================================================================

def train_byol(model_params:dict, data_params:dict, backbone_name:str, pretrain_weight_file:str):

    # BYOL Transforms
    # Transforms for drones
    byol_drn_transforms = BYOLTransform(
        view_1_transform=BYOLView1Transform(
            input_size = model_params["input_size"] if backbone_name == "resnet" else 224, 
            normalize={"mean" : config["drn_img_mean"], "std" : config["drn_img_std"]}
        ),
        view_2_transform=BYOLView2Transform(
            input_size = model_params["input_size"] if backbone_name == "resnet" else 224,
            normalize={"mean" : config["drn_img_mean"], "std" : config["drn_img_std"]}
        )
    )

    # Transforms for satellite
    byol_sat_transforms = BYOLTransform(
        view_1_transform=BYOLView1Transform(
            input_size = model_params["input_size"] if backbone_name == "resnet" else 224, 
            normalize={"mean" : config["sat_img_mean"], "std" : config["sat_img_std"]}
        ),
        view_2_transform=BYOLView2Transform(
            input_size = model_params["input_size"] if backbone_name == "resnet" else 224,
            normalize={"mean" : config["sat_img_mean"], "std" : config["sat_img_std"]}
        )
    )

    # DataLoader
    drnsen2a_trainloader = get_dataloaders(
        model_params = model_params, 
        drn_fold = data_params["drn_fold_path"],
        sat_fold = data_params["sat_fold_path"],
        ssl_drn_transforms = byol_drn_transforms, 
        ssl_sat_transforms = byol_sat_transforms
    )

    # Select correct BYOL class depending on the backbone
    if backbone_name == "resnet":
        resnet_bb = get_pretrained_backbone(backbone_name, pretrain_weight_file)
        byol = ByolBBResnet(model_params, resnet_bb)
    elif backbone_name == "vit":
        vit_bb = get_pretrained_backbone(backbone_name, pretrained_weight_file)
        #todo : Need to implement SimSiam model for ViT backbone
        raise Exception("ByolBBViT not implemented yet !")
    elif backbone_name == "swin-vit":
        swinvit_bb_model = get_pretrained_backbone(backbone_name, pretrain_weight_file)
        byol = ByolBBSwinViT(model_params, swinvit_bb_model)
    else:
        raise KeyError("Incorrect value passed to 'backbone_name")

    # Train Byol model
    trainer = get_trainer(model_params, data_params)
    trainer.fit(byol, drnsen2a_trainloader)

# ==============================================================================
# Dino training function
# ==============================================================================

def train_dino(model_params:dict, data_params:dict, backbone_name:str, pretrain_weight_file:str):

    dino_drn_transforms = DINOTransform(
        # Input size really doesnt matter for SwinVit here as Dino Transforms give Global (224) and local Crops (96)
        normalize={"mean" : config["drn_img_mean"], "std" : config["drn_img_std"]}
    )
    dino_sat_transforms = DINOTransform(
        # Input size really doesnt matter for SwinVit here as Dino Transforms give Global (224) and local Crops (96)
        normalize={"mean" : config["sat_img_mean"], "std" : config["sat_img_std"]}
    )
    drnsen2a_trainloader = get_dataloaders(
        model_params = model_params,
        drn_fold = data_params["drn_fold_path"],
        sat_fold = data_params["sat_fold_path"],
        ssl_drn_transforms = dino_drn_transforms, 
        ssl_sat_transforms = dino_sat_transforms
    )

    # Select correct BYOL class depending on the backbone
    if backbone_name == "resnet":
        if pretrain_weight_file:
            # finetuning, load pretrain weight file
            resnet_bb = get_pretrained_backbone(backbone_name, pretrain_weight_file)
        else:
            # pretraining, no weight file
            resnet_bb = resnet50()
            resnet_bb = torch.nn.Sequential(*list(resnet_bb_model.children())[:-1]) 

        dino = DinoBBResnet(model_params, resnet_bb)

    elif backbone_name == "vit":
        if pretrain_weight_file:
            # finetuning, load pretrain weight file
            vit_bb = get_pretrained_backbone(backbone_name, pretrain_weight_file)
        else:
            # pretraining, no weights
            vit_bb = vit_base_patch16_224(num_classes = 0, pretrained = True)
        dino = DinoBBViT(model_params, vit_bb)
    elif backbone_name == "swin-vit":
        if pretrain_weight_file:
            # finetuning, load pretrain weights file
            swinvit_bb_model = get_pretrained_backbone(backbone_name, pretrain_weight_file)
        else:
            # pretraining, no weights
            raise NotImplementedError("Swin-vit pretraining not implemented !")
        dino = DinoBBSwinViT(model_params, swinvit_bb_model)
    else:
        raise KeyError("Incorrect value passed to 'backbone_name")
        # needs to be implemented

    # Train Dino model
    trainer = get_trainer(model_params, data_params)
    trainer.fit(dino, drnsen2a_trainloader)

# ==============================================================================
# SatMAE
# ==============================================================================

def train_satmae(model_params:dict, data_params:dict, backbone_name:str, pretrain_weight_file:str):

    # Get path to folder containing satellite images
    sat_fold = data_params["sat_fold_path"]
    assert sat_fold is not None, "SatMAE only runs on satelite images"
    sat_imgs = glob(os.path.join(sat_fold, "*"))
    # We dont have drone images 

    # We need datmae parameters
    satmae_params = config["satmae_params"]

    # Here norm is set to none as we dont want to divide by 10k
    sen2a_dataset = Sen2aMultiDataset(
        sat_imgs, 
        drop_bands = [1,9,10], 
        norm = None, 
        # We use GEE normalize mean/std normalize values from the satmae paper.
        # We will also need to resize the patchsize to 96 as this is the size the pretrian weights accomodate.
        transforms = Compose([
            Normalize(mean = config["sat_hyp_img_mean"], std = config["sat_hyp_img_std"]),
            Resize(satmae_params["pretrain_patch_hw"])
        ])
    )

    # Find the number of channels
    c,h,w = sen2a_dataset.__getitem__(0).shape
    model_params["resized_img_size"] = h

    # Dataloader
    sen2a_dataloader = DataLoader(
        sen2a_dataset,
        batch_size= int(model_params["eff_batch_size"] / (model_params["nodes"] * model_params["devices"])),
        num_workers = model_params["dataloader_workers"]
    )

    #todo : move this to an appropriate place
    grouped_bands = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9))

    # Instantiate masked auto encoder vit base model with patch_size = 8
    if backbone_name == "vit":
        try:
            mae_grp_vitb_p8 = MaskedAutoencoderGroupChannelViT(
                img_size = h,
                patch_size = satmae_params["patch_size"],
                #in_chans = len(grouped_bands), # usually 3, as we have 3 groups
                in_chans = c,
                spatial_mask = satmae_params["spatial_mask"],
                channel_groups = grouped_bands,
                channel_embed = satmae_params["channel_embed"],
                embed_dim = satmae_params["embed_dim"], # default : 1024
                depth = satmae_params["depth"],
                num_heads = satmae_params["num_heads"],
                decoder_channel_embed = satmae_params["decoder_channel_embed"],
                decoder_embed_dim = satmae_params["decoder_embed_dim"],
                decoder_depth = satmae_params["decoder_depth"],
                decoder_num_heads = satmae_params["decoder_num_heads"],
                mlp_ratio = satmae_params["mlp_ratio"],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                norm_pix_loss = satmae_params["norm_pix_loss"],
            )
        except RuntimeError:
            print(colored("Model weight mismatch !", "red"))

    else:
        raise KeyError("Only ViT implemented for satmae")

    # Load pretrained weights to the entire satmae model
    model_checkpoint = torch.load(pretrain_weight_file)
    mae_grp_vitb_p8.load_state_dict(model_checkpoint["model"])

    satmae = SatMaeGroupViTBB(model_params,mae_grp_vitb_p8)
    trainer = get_trainer(model_params, data_params)
    trainer.fit(satmae, sen2a_dataloader)

# ==============================================================================
# Sentinel2a Multi Channel Dataset
# ==============================================================================


class Sen2aMultiDataset(Dataset):
    def __init__(self, 
    img_paths:List[str], 
    drop_bands:List[int] = [1,9,10], #SatMae paper drops these bands
    clip:Union[float,None] = None, 
    norm:Union[float,None] = 10000,
    transforms:Union[Compose, None] = None
    ):
        self.img_paths = img_paths
        self.norm = norm
        self.clip = clip
        self.drop_bands = drop_bands
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,idx):
        img_path = self.img_paths[idx]
        with rasterio.open(img_path) as ds:
            img = ds.read()

        if self.norm:
            img /= self.norm

        if self.clip:
            img = np.clip(img, 0, self.clip)

        all_bands = np.arange(0,img.shape[0])
        selected_bands = [b for b in all_bands if b not in self.drop_bands]

        # Change axis from (C,H,W) -> (H,W,C) : Cause transforms.ToImage accepts it in this format it we rearrange c back to first dim
        #img = np.moveaxis(img, source = (0,1,2), destination = (2,0,1))
        img = torch.tensor(img)

        if self.transforms:
            img = self.transforms(img) #(C,H,W)

        img = img[selected_bands]
        assert img.shape[0] == len(selected_bands)
        return img

if __name__ == "__main__":


    #path_to_weights = path_to_weights = "model_weights/pretrain_weights/rsp-aid-resnet-50-e300-ckpt.pth"
    # path_to_weights = "model_weights/pretrain_weights/sshsph-aid-maevit-e299.ckpt"


    # backbone = get_pretrained_backbone("vit", path_to_weights)
    # print(backbone)

    root = "data/interim/gee_sat/sen2a_c13_512x_pch"
    img_paths = glob(os.path.join(root, "*")) 
    sen_mul_ds = Sen2aMultiDataset(img_paths, transforms = Compose([ToImage()]))
    sen_mul_ds.__getitem__(0)


    