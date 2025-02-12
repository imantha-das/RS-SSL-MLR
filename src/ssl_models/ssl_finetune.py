import sys
import os
import re
from glob import glob

import argparse

from termcolor import colored
from typing import List, Tuple,Dict, Union
from ssl_utils import (load_model_weights, print_model_weights,
get_dataloaders, get_trainer, get_pretrained_backbone, train_simsiam, train_byol, train_dino, train_mae
)

#! Note for some reason torchvision.models swin_t does load the weights properly
sys.path.append("src/ssl_models/foundation_models/RSP/Scene Recognition/models")
from swin_transformer import SwinTransformer

# ------------------------------ Argument Parser ----------------------------- #

parser = argparse.ArgumentParser(description = "Train SSL algorithm") 
parser.add_argument("-ssl_model",type = str,help = "Enter SSL algorithm", choices=["byol","simsiam","dino","mae"])
parser.add_argument("-backbone",type = str,help = "Enter model backbone", choices = ["resnet","swin-vit","vit"])
parser.add_argument("-epochs", type = int, default = 20, help = "number of epochs")
parser.add_argument("-eff_batch_size", type = int, default = 512, help = "Effective batch size (batch_size * num_nodes * num_devices")
parser.add_argument("-data_fold_drn", type = str, default = None, help = "Path to drone data folder")
parser.add_argument("-data_fold_sat", type = str,default = None, help = "Path to sentinel data folder")
parser.add_argument("-pretrain_weights_fold", type = str, default="model_weights/pretrain_weights", help = "Path to pretrained weights file")
parser.add_argument("-save_weights_fold", type = str, default = "model_weights/ssl_weights", help = "Path to where model weights + stats are saved")
parser.add_argument("-lr", type = float, default = None, help = "Enter learning rate, this will remove any schedulers that are being used")
parser.add_argument("-input_size",type = int,default = 256,help = "Enter input image size")
parser.add_argument("-devices", type = int, default = 2, help = "No of GPU's")
parser.add_argument("-nodes", type = int , default = 1, help = "No of Nodes")
parser.add_argument("-precision", type = int, default = 32, help = "torch tensor precision")
parser.add_argument("-dataloader_workers", type = int, default = 16, help = "number of workers for dataloader")
parser.add_argument("-save_freq", type = int, default = 1, help = "how frquent model weights are saved")
parser.add_argument("-save_all_weights", action = argparse.BooleanOptionalAction, help = "save model & optimizer weights only")
args = parser.parse_args()

if __name__ == "__main__":

    # ------------------ Search for correct pretrain weight file ----------------- #
    pretrain_weights_files = glob(os.path.join(args.pretrain_weights_fold, "*"))
    for f in pretrain_weights_files:
        match args.backbone:
            case "resnet":
                if  bool(re.search(r"\bresnet\b",f)):
                    pretrain_weights_file = f
            case "swin-vit":
                if  bool(re.search(r"\bswinvit\b",f)):
                    pretrain_weights_file = f
            case "vit":
                if  bool(re.search(r"\bvit\b",f)):
                    pretrain_weights_file = f                
            case _:
                raise(ValueError(colored(f"No weights file found : {pretrain_weights_file}, Ensure the 'resnet' or 'swin-vit' is part of the file names", "red")))

    assert "pretrain_weights_file" in locals(), colored("Pretrain Weight File Not Found !", "red")    
    print(colored(f"selected pretrain weights file : {pretrain_weights_file}","blue"))

    # ------------------- Errors for incorrect argparse inputs ------------------- #

    if args.ssl_model not in ["simsiam", "byol", "dino", "mae"]:
        raise(KeyError("Incorrect key passed to argument 'ssl_model', Please pick from the following options : simsiam / byol / dino / mae"))
    if args.backbone not in ["resnet", "swin-vit", "vit"]:
        raise(KeyError("Incorrect key passed to argument 'backbone', Please pick from the following options : resnet / swin-vit / vit"))

# ------------------------------- Define Params ------------------------------ #

    model_params = {
        "ssl_model" : args.ssl_model,
        "backbone" : args.backbone, #add these to save as hyperparams
        "input_size" : args.input_size,
        "batch_size" : int(args.eff_batch_size / (args.nodes * args.devices)),
        "eff_batch_size" : args.eff_batch_size,
        "epochs" : args.epochs,
        "devices" : args.devices,
        "nodes" : args.nodes,
        "precision" : args.precision,
        "dataloader_workers" : args.dataloader_workers,
        "lr" : args.lr
    }

    data_params = {
        "sat_data_name" : os.path.basename(os.path.dirname(args.data_fold_sat))if args.data_fold_sat else None,
        "drn_data_name" : os.path.basename(os.path.dirname(args.data_fold_drn)) if args.data_fold_drn else None,
        "sat_fold_path" : args.data_fold_sat,
        "drn_fold_path" : args.data_fold_drn,
        "save_weights_fold" : args.save_weights_fold,
        "save_freq" : args.save_freq,
        "save_weights_only" : False if args.save_all_weights else True
    }

    # -------------------------------- Train Funcs ------------------------------- #
    
    match args.ssl_model:
        case "simsiam":
            # Train Simsiam Model
            train_simsiam(model_params, data_params, args.backbone, pretrain_weights_file)

        case "byol":
            # Train Byol Model
            train_byol(model_params, data_params, args.backbone, pretrain_weights_file)

        case "dino":
            # Train Dino Model
            train_dino(model_params, data_params, args.backbone, pretrain_weights_file)
        
        case "mae":
            # Train MAE model
            train_mae(model_params, data_params, args.backbone, pretrain_weights_file)
    


