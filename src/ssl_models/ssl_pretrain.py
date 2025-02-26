import os
import sys
from glob import glob
import yaml
import argparse
from termcolor import colored


from ssl_utils import get_dataloaders, train_mae, train_dino

from typing import Union
# ---------------------------- Argeparse Arguments --------------------------- #
parser = argparse.ArgumentParser(description = "Pretrain SSL Models")
parser.add_argument("-ssl_model", type = str, help = "Enter SSL model for pretraining", choices = ["mae","dino"])
parser.add_argument("-backbone", type = str, help = "Enter backbone model", choices = ["resnet","vit"])
parser.add_argument("-epochs", type = int, default = 300, help = "Number of epochs")
parser.add_argument("-eff_batch_size", type = int,  default = 512, help = "Effective batch size (bs * nodes * devices)")
parser.add_argument("-data_fold_drn", type = str, default = None, help = "Path to drone data")
parser.add_argument("-data_fold_sat", type = str, default = None, help = "Path to satelite data")
parser.add_argument("-save_weights_fold", type = str, default = "model_weights/pretrain_weights",help = "Path to where model weights are saved")
parser.add_argument("-lr", type = float, default = None, help = "Enter learning rate, this will remove any schedulers that are being used")
parser.add_argument("-input_size",type = int, default = 256, help = "Enter input image size")
parser.add_argument("-devices", type = int, default = 4, help = "No of GPU's")
parser.add_argument("-nodes", type = int, default = 1, help = "Number of nodes")
parser.add_argument("-precision", type = int, default = 32, help = "torch tensor precisions")
parser.add_argument("-dataloader_workers", type = int, default = 16, help = "number of workers for dataloader")
parser.add_argument("-save_freq", type = int, default = 20, help = "save_frequency")
parser.add_argument("-ckpt_path", type =str, default = None, help = "path to checkpoint to resume training")
parser.add_argument("-save_all_weights", action = argparse.BooleanOptionalAction, help = "save model & optimizer weights only")
args = parser.parse_args()

if __name__ == "__main__":

    # ------------------------------- Model params ------------------------------- #
    model_params = {
        "ssl_model" : args.ssl_model,
        "backbone" : args.backbone, #add these to save as hyperparams
        "batch_size" : int(args.eff_batch_size / (args.nodes * args.devices)),
        "eff_batch_size" : args.eff_batch_size,
        "epochs" : args.epochs,
        "input_size" : args.input_size,
        "devices" : args.devices,
        "nodes" : args.nodes,
        "precision" : args.precision,
        "dataloader_workers" : args.dataloader_workers,
    }
    model_params["lr"] = args.lr


    data_params = {
        "sat_data_name" : os.path.basename(os.path.dirname(args.data_fold_sat))if args.data_fold_sat else None,
        "drn_data_name" : os.path.basename(os.path.dirname(args.data_fold_drn)) if args.data_fold_drn else None,
        "sat_fold_path" : args.data_fold_sat,
        "drn_fold_path" : args.data_fold_drn,
        "save_weights_fold" : args.save_weights_fold,
        "save_freq" : args.save_freq,
        "ckpt_path" : args.ckpt_path,
        "save_weights_only" : False if args.save_all_weights else True
    }
    
    #get_trainer(model_params, data_params)

    # ----------------------- Load Configuration for Models ---------------------- #

    with open("src/ssl_models/ssl_config.yml", "r") as f:
        config = yaml.safe_load(f)
            
    # -------------------------------- Train Model ------------------------------- #

    match args.ssl_model:
        case "mae":
            train_mae(model_params, data_params, args.backbone, pretrain_weights_file = None)
        case "dino":
            train_dino(model_params, data_params, args.backbone, pretrain_weight_file = None)


     
    