import sys
import os
import re
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.models import resnet50
from torchvision.transforms import Compose, Resize, ToTensor
from torchsummary import summary

import argparse
import yaml
from termcolor import colored
from typing import List, Tuple,Dict, Union
from ssl_utils import load_model_weights, print_model_weights

from lightly.transforms import SimSiamTransform, BYOLView1Transform, BYOLView2Transform, BYOLTransform
from lightly.transforms.dino_transform import DINOTransform
from lightly.data import LightlyDataset
from simsiam import SimSiamBBResnet, SimSiamBBSwinVit
from byol import ByolBBResnet, ByolBBSwinVit
from dino import DinoBBResnet, DinoBBSwinVIT

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

#! Note for some reason torchvision.models swin_t does load the weights properly
sys.path.append("src/ssl_models/foundation_models/RSP/Scene Recognition/models")
from swin_transformer import SwinTransformer

# ------------------------------ Argument Parser ----------------------------- #

parser = argparse.ArgumentParser(description = "Train SSL algorithm") 
parser.add_argument("-ssl_model",type = str,help = "Enter SSL algorithm", choices=["byol","simsiam","dino"])
parser.add_argument("-backbone",type = str,help = "Enter model backbone", choices = ["resnet","swin-vit"])
parser.add_argument("-data_fold_drn", type = str, help = "Path to drone data folder", default = "data/processed/sshsph_drn/drn_c3_256x_pch")
parser.add_argument("-data_fold_sat", type = str, help = "Path to sentinel data folder",default = "data/interim/gee_sat/sen2a_c3_256x_clp0.3uint8_full_pch")
parser.add_argument("-pretrain_weights_fold", type = str, help = "Path to pretrained weights file", default="model_weights/pretrain_weights")
parser.add_argument("-save_weights_fold", type = str, help = "Path to where model weights + stats are saved", default = "model_weights/ssl_weights")
parser.add_argument("-epochs", type = int, help = "number of epochs", default = 20)
parser.add_argument("-eff_batch_size", type = int, help = "Effective batch size (batch_size * num_nodes * num_devices", default = 512)
parser.add_argument("-lr", type = float,help = "Enter learning rate, this will remove any schedulers that are being used", default = None)
parser.add_argument("-input_size",type = int,help = "Enter input image size",default = 256)
parser.add_argument("-devices", type = int, help = "No of GPU's", default = 4)
parser.add_argument("-nodes", type = int , help = "No of Nodes", default = 1)
parser.add_argument("-precision", type = int, help = "torch tensor precision", default = 32)
args = parser.parse_args()

# ==============================================================================
# Helper funcs to minimize SSL training code 
# ==============================================================================
# --------------------- Returns Pretrained Model Backbone -------------------- #

def get_pretrained_backbone(backbone_name:str, pretrain_weights_file:str):
    # Load model weights
    match backbone_name:
        case "resnet":
            # Load model weights for resnet
            resnet_bb_model = load_model_weights(model_name = backbone_name, path_to_weights=pretrain_weights_file, num_classes=51)
            # Get backbone 
            resnet_bb = torch.nn.Sequential(*list(resnet_bb_model.children())[:-1]) 
            return resnet_bb
        case "swin-vit":
            # Load model weights for Swin-vit
            swinvit_bb_model = load_model_weights(model_name = backbone_name, path_to_weights=pretrain_weights_file, num_classes=51)
            # we dont extract a backbone, rather we use .forward_features from SwinTranformer Class to get features
            return swinvit_bb_model
        case _:
            raise(KeyError("incorrect argument to 'backbone', please enter 'resnet' or 'swin-vit'"))

# -------------- Return dataloader based on SSL transformations -------------- #

def get_dataloaders(model_params:dict, 
                    ssl_drn_transforms:Union[SimSiamTransform, BYOLTransform], 
                    ssl_sat_transforms:Union[SimSiamTransform, BYOLTransform]
                    ):
    #* Note we donot need to pass in ToTensor as Lightly SSL Transforms already incorporates this !
    drn_trainset = LightlyDataset(input_dir = args.data_fold_drn, transform=Compose([ssl_drn_transforms])) # .__getitem__() returns -> view1,view2,fname
    sen2a_trainset = LightlyDataset(input_dir= args.data_fold_sat, transform=Compose([ssl_sat_transforms]))
    drnsen2a_trainset = ConcatDataset([drn_trainset, sen2a_trainset])
    # DataLoader
    drnsen2a_trainloader = DataLoader(
        drnsen2a_trainset, 
        batch_size= int(model_params["eff_batch_size"] / (model_params["nodes"] * model_params["devices"])),
        num_workers = config["dataloader_workers"]
    )
    return drnsen2a_trainloader

# ----------------------------- Trainer function ----------------------------- #

def get_trainer():
    # Save name
    save_name = "-".join([
        f"{args.ssl_model}",
        f"is{args.input_size}",
        f"effbs{args.eff_batch_size}",
        f"ep{args.epochs}",
        f"bb{args.backbone.capitalize()}",
        "dsDrnSen2a",
        "clClUcl",
        "nmTTDrnSatNM"
    ])
    # Checkpoint + Logging
    logger = CSVLogger(save_dir = args.save_weights_fold, name = save_name)
    checkpoint_callback = ModelCheckpoint(
        #dirpath=os.path.join(args.save_weights_fold, save_name), 
        filename="epoch:{epoch}",
        save_on_train_epoch_end=True,
        save_weights_only = True,
        save_top_k = -1
    )

    # Model Traiining
    trainer = pl.Trainer(
        default_root_dir = os.path.join(args.save_weights_fold, save_name),
        devices = -1,
        num_nodes= args.nodes,
        accelerator = "gpu",
        strategy = "ddp" if args.backbone == "resnet" else DDPStrategy(find_unused_parameters = True),
        max_epochs = args.epochs,
        precision = args.precision,
        logger = logger,
        callbacks = [checkpoint_callback],
        #auto_scale_batch_size = config.AUTO_SCALE_BATCH_SIZE # to find "max" batch_size that can be procesedwith resources (gpu)
    )
    return trainer

# ==============================================================================
# SimSiam Training function
# ==============================================================================

def train_simsiam(model_params:dict,backbone_name:str, pretrained_weight_file:str)->None:
    
    # SimSiam specific transforms
    simsiam_drn_transforms = SimSiamTransform(
        input_size = args.input_size if backbone_name == "resnet" else 224, 
        normalize = {"mean" : config["drn_img_mean"], "std" : config["drn_img_std"]}
    )
    simsiam_sat_transforms = SimSiamTransform(
        input_size = args.input_size if backbone_name == "resnet" else 224, 
        normalize = {"mean" : config["sat_img_mean"], "std" : config["sat_img_std"]}
    )

    # DataLoader
    drnsen2a_trainloader = get_dataloaders(model_params,simsiam_drn_transforms, simsiam_sat_transforms)
  
    # Select correct SimSiam class w.r.t the backbone
    if backbone_name == "resnet":
        resnet_bb = get_pretrained_backbone(backbone_name, pretrained_weight_file)
        simsiam = SimSiamBBResnet(model_params, resnet_bb)
        #print_model_weights(simsiam.backbone_model)
    else:
        swinvit_bb_model = get_pretrained_backbone(backbone_name, pretrained_weight_file)
        simsiam = SimSiamBBSwinVit(model_params, swinvit_bb_model)

    # Train SimSiam model
    trainer = get_trainer()
    trainer.fit(simsiam, drnsen2a_trainloader)

# ==============================================================================
# Byol Training function
# ==============================================================================

def train_byol(model_params:dict, backbone_name:str, pretrain_weight_file:str):

    # BYOL Transforms
    # Transforms for drones
    byol_drn_transforms = BYOLTransform(
        view_1_transform=BYOLView1Transform(
            input_size = args.input_size if args.backbone == "resnet" else 224, 
            normalize={"mean" : config["drn_img_mean"], "std" : config["drn_img_std"]}
        ),
        view_2_transform=BYOLView2Transform(
            input_size = args.input_size if args.backbone == "resnet" else 224,
            normalize={"mean" : config["drn_img_mean"], "std" : config["drn_img_std"]}
        )
    )

    # Transforms for satellite
    byol_sat_transforms = BYOLTransform(
        view_1_transform=BYOLView1Transform(
            input_size = args.input_size if args.backbone == "resnet" else 224, 
            normalize={"mean" : config["sat_img_mean"], "std" : config["sat_img_std"]}
        ),
        view_2_transform=BYOLView2Transform(
            input_size = args.input_size if args.backbone == "resnet" else 224,
            normalize={"mean" : config["sat_img_mean"], "std" : config["sat_img_std"]}
        )
    )

    # DataLoader
    drnsen2a_trainloader = get_dataloaders(model_params,byol_drn_transforms, byol_sat_transforms)

    # Select correct BYOL class depending on the backbone
    if backbone_name == "resnet":
        resnet_bb = get_pretrained_backbone(backbone_name, pretrain_weight_file)
        byol = ByolBBResnet(model_params, resnet_bb)
    else:
        swinvit_bb_model = get_pretrained_backbone(backbone_name, pretrain_weight_file)
        byol = ByolBBSwinVit(model_params, swinvit_bb_model)

    # Train Byol model
    trainer = get_trainer()
    trainer.fit(byol, drnsen2a_trainloader)

# ==============================================================================
# Dino training function
# ==============================================================================

def train_dino(model_params:dict, backbone_name:str, pretrain_weight_file:str):

    dino_drn_transforms = DINOTransform(
        # Input size really doesnt matter for SwinVit here as Dino Transforms give Global (224) and local Crops (96)
        normalize={"mean" : config["drn_img_mean"], "std" : config["drn_img_std"]}
    )
    dino_sat_transforms = DINOTransform(
        # Input size really doesnt matter for SwinVit here as Dino Transforms give Global (224) and local Crops (96)
        normalize={"mean" : config["sat_img_mean"], "std" : config["sat_img_std"]}
    )
    drnsen2a_trainloader = get_dataloaders(model_params,dino_drn_transforms, dino_sat_transforms)

    # Select correct BYOL class depending on the backbone
    if backbone_name == "resnet":
        resnet_bb = get_pretrained_backbone(backbone_name, pretrain_weight_file)
        dino = DinoBBResnet(model_params, resnet_bb)
    else:
        swinvit_bb_model = get_pretrained_backbone(backbone_name, pretrain_weight_file)
        dino = DinoBBSwinVIT(model_params, swinvit_bb_model)
        # needs to be implemented

    # Train Dino model
    trainer = get_trainer()
    trainer.fit(dino, drnsen2a_trainloader)


    
if __name__ == "__main__":

    # ------------------ Search for correct pretrain weight file ----------------- #
    pretrain_weights_files = glob(os.path.join(args.pretrain_weights_fold, "*"))
    for f in pretrain_weights_files:
        match args.backbone:
            case "resnet":
                if  bool(re.search(r"\bresnet\b",f)):
                    pretrain_weights_file = f
            case "swin-vit":
                if  bool(re.search(r"\bswin-vit\b",f)):
                    pretrain_weights_file = f
            case _:
                raise(ValueError(colored(f"No weights file found : {pretrain_weights_files}, Ensure the 'resnet' or 'swin-vit' is part of the file names", "red")))
                
    # ------------------- Errors for incorrect argparse inputs ------------------- #

    if args.ssl_model not in ["simsiam", "byol", "dino"]:
        raise(KeyError("Incorrect key passed to argument 'ssl_model', Please pick from the following options : simsiam / byol / dino"))
    if args.backbone not in ["resnet", "swin-vit"]:
        raise(KeyError("Incorrect key passed to argument 'backbone', Please pick from the following options : resnet / swin-vit"))

    # ----------------------- Load Configuration for Models ---------------------- #

    with open("src/ssl_models/ssl_config.yml", "r") as f:
        config = yaml.safe_load(f)

    # ----------------------- Get params & Initialize Model ---------------------- #


    #   Save name flag meanings - is : input size | effbs : effective batch size | ep : number of epochs | bb : backbone | ds : datasets used
    #                             cl : "Cl" for clean, "Ucl" for unclean and refer to the ds used earlier for dataset names respectively, i.e "ClUcl" means first dataset (drn) is cleaned & Second dataset is unclean| 
    #                             nm : normalization, TT refers ToTensor, Drn & Sat Nm refers normalization factors used above been applied to transformations

    model_params = {
        "batch_size" : int(args.eff_batch_size / (args.nodes * args.devices)),
        "eff_batch_size" : args.eff_batch_size,
        "epochs" : args.epochs,
        "backbone" : args.backbone, #add these to save as hyperparams
        "epochs" : args.epochs,
        "devices" : args.devices,
        "nodes" : args.nodes,
        "precision" : args.precision
    }
    if args.lr: # learning rate ususally set to none since we use cosine schedule. Its only passed when we want constant value
        model_params["lr"] = args.lr

    match args.ssl_model:
        case "simsiam":
            # Train Simsiam Model
            train_simsiam(model_params, args.backbone, pretrain_weights_file)

        case "byol":
            # Train Byol Model
            train_byol(model_params, args.backbone, pretrain_weights_file)

        case "dino":
            # Train Dino Model
            train_dino(model_params, args.backbone, pretrain_weights_file)

    


