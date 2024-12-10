import os
import sys
from glob import glob
import yaml
import argparse
from termcolor import colored

from lightly.data import LightlyDataset
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightly.transforms import MAETransform


import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from timm.models.vision_transformer import vit_base_patch16_224, VisionTransformer

from mae import MaeBBViT
from ssl_utils import get_dataloaders

from typing import Union
# ---------------------------- Argeparse Arguments --------------------------- #
parser = argparse.ArgumentParser(description = "Pretrain SSL Models")
parser.add_argument("-ssl_model", type = str, help = "Enter SSL model for pretraining", choices = ["mae"])
parser.add_argument("-backbone", type = str, help = "Enter backbone model", choices = ["vit"])
parser.add_argument("-data_fold_drn", type = str, default = None, help = "Path to drone data")
parser.add_argument("-data_fold_sat", type = str, default = None, help = "Path to satelite data")
parser.add_argument("-save_weights_fold", type = str, default = "model_weights/pretrain_weights",help = "Path to where model weights are saved")
parser.add_argument("-epochs", type = int, default = 300, help = "Number of epochs")
parser.add_argument("-eff_batch_size", type = int,  default = 512, help = "Effective batch size (bs * nodes * devices)")
parser.add_argument("-lr", type = float,help = "Enter learning rate, this will remove any schedulers that are being used", default = None)
parser.add_argument("-input_size",type = int,help = "Enter input image size",default = 256)
parser.add_argument("-devices", type = int, default = 4, help = "No of GPU's")
parser.add_argument("-nodes", type = int, default = 1, help = "Number of nodes")
parser.add_argument("-precision", type = int, default = 32, help = "torch tensor precisions")
parser.add_argument("-dataloader_workers", type = int, default = 16, help = "number of workers for dataloader")
parser.add_argument("-save_freq", type = int, default = -1, help = "save_frequency")
args = parser.parse_args()

def train_mae(model_params:dict, data_params:dict, backbone_name:str, pretrain_weights_file:Union[str,None]):

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
    trainer.fit(mae, trainloader)

    

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
    logger = CSVLogger(save_dir = data_params["save_weight_fold"], name = save_name)
    checkpoint_callback = ModelCheckpoint(
        #dirpath=os.path.join(args.save_weights_fold, save_name), 
        filename="epoch:{epoch}",
        save_on_train_epoch_end=True,
        save_weights_only = True,
        save_top_k = -1,
        every_n_epochs = args.save_freq
    )

    # Model Traiining
    trainer = pl.Trainer(
        default_root_dir = os.path.join(args.save_weights_fold, save_name),
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

if __name__ == "__main__":


    # ------------------------------- Model params ------------------------------- #
    model_params = {
        "ssl_model" : args.ssl_model,
        "backbone" : args.backbone, #add these to save as hyperparams
        "batch_size" : int(args.eff_batch_size / (args.nodes * args.devices)),
        "eff_batch_size" : args.eff_batch_size,
        "epochs" : args.epochs,
        "epochs" : args.epochs,
        "input_size" : args.input_size,
        "devices" : args.devices,
        "nodes" : args.nodes,
        "precision" : args.precision,
        "dataloader_workers" : args.dataloader_workers,
    }
    if args.lr:
        model_params["lr"] = args.lr


    data_params = {
        "sat_data_name" : os.path.basename(os.path.dirname(args.data_fold_sat))if args.data_fold_sat else None,
        "drn_data_name" : os.path.basename(os.path.dirname(args.data_fold_drn)) if args.data_fold_drn else None,
        "sat_fold_path" : args.data_fold_sat,
        "drn_fold_path" : args.data_fold_drn,
        "save_weight_fold" : args.save_weights_fold,
        "save_freq" : args.save_freq,
    }
    
    get_trainer(model_params, data_params)

    # ----------------------- Load Configuration for Models ---------------------- #

    with open("src/ssl_models/ssl_config.yml", "r") as f:
        config = yaml.safe_load(f)
            
    # -------------------------------- Train Model ------------------------------- #

    match args.ssl_model:
        case "mae":
            train_mae(model_params, data_params, args.backbone, pretrain_weights_file = None)

     
    