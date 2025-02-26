# ==============================================================================
# This script retruns Geospatial Features from trained Backbone
# Unlike the malaria_train.py script this will only find the geovector
# which will be appended 
# ==============================================================================

# ---------------------------------- Imports --------------------------------- #
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import ToImage, Normalize, Compose, Resize
import rasterio
import warnings
from termcolor import colored
from typing import Union, List
from functools import partial
from ast import literal_eval
import argparse
import yaml
from tqdm import tqdm
import time
from glob import glob

from malaria_utils import setup_logger

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

sys.path.append("src/ssl_models")

from simsiam import SimSiamBBResnet, SimSiamBBSwinViT
from byol import ByolBBResnet, ByolBBSwinViT
from dino import DinoBBResnet, DinoBBSwinViT, DinoBBViT
from mae import MaeBBViT
from satmae import SatMaeGroupViTBB
from ssl_utils import print_model_weights

sys.path.append(os.path.join(os.getcwd(),"src","ssl_models","foundation_models","SatMAE"))

from models_mae_group_channels import MaskedAutoencoderGroupChannelViT
from models_vit_group_channels import GroupChannelsVisionTransformer

# ---------------------------------- Config ---------------------------------- #
with open("src/ssl_models/ssl_config.yml") as f:
    ssl_config = yaml.safe_load(f)

# --------------------------- Class to handle data --------------------------- #
class Get_Img_and_Csv_Data(Dataset):
    """
    This function returns the CSV Data & Images as tensors. We do this to speed
    up data loading - stop loading data for every epoch wait. This mean we dont
    pass the images through a trained encoder as this needs to be dont for every
    seperate weight
    """
    def __init__(
        self, df:pd.DataFrame, 
        img_p_colname:str,
        img_transforms:Union[None, dict], 
        num_channels:int, 
        drop_bands:List[int],
        data_root:str,
        norm:Union[float, None] = None, # Note this normalization is divide sentienal images by 10000
        clip:Union[float, None] = None
        ):
        super().__init__()
        self.df = df
        self.c = num_channels
        self.img_p_colname = img_p_colname
        self.img_transforms = img_transforms
        self.data_root = data_root
        self.drop_bands = drop_bands
        self.norm = norm
        self.clip = clip

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx:int):
        # Get a row row from pandas dataframe
        df_row = self.df.iloc[idx]
        # There are multiple images that correspond to lat/lon so select desired one
        img_fname = self.select_image_p(literal_eval(df_row[self.img_p_colname]))
        img_p = os.path.join(self.data_root, img_fname)
        # Load Image data
        with rasterio.open(img_p) as ds:
            img = ds.read()
            
        if self.norm:
            img /= self.norm

        if self.clip:
            img = np.clip(img, 0, self.clip)

        all_bands = np.arange(0,img.shape[0])
        selected_bands = [b for b in all_bands if b not in self.drop_bands]

        # We need to apply transformation before we drop bands
        img = torch.tensor(img)
        # These are either sentinal RGB, sentinel multi spectral or Drone RGB
        data_fold_name = os.path.basename(self.data_root)
        sen_or_drn = data_fold_name.split("_")[0]
        # Apply transformations
        if self.img_transforms:
            img = self.img_transforms[sen_or_drn](img) #i.e if sentinel (13,*,*) if drn (3,*,*)

        # We need to pick selected bands
        img = img[selected_bands] #(C,H,W)
        assert img.shape[0] == len(selected_bands)

        return df_row, img

    def select_image_p(self,image_paths:List[str]):
        """The following function is written to select an image.
        We will use a simple strategy of selecting the first image
        however this isnt the necessarily the best image as there
        maybe images which are more clearer
        """
        return image_paths[0]

def collate_pandas_numpy(batch):
    """Collate function to retrun a list of pandas series and numpy array of image"""
    df_row, img = zip(*batch)
    return list(df_row), np.array(img)


# ------------------------ Function to get Geo Vector ------------------------ #
def get_geo_vector(
    X_img:torch.Tensor,
    ssl_model:Union[SimSiamBBResnet,SimSiamBBSwinViT,ByolBBResnet, ByolBBSwinViT,DinoBBResnet, DinoBBSwinViT,MaeBBViT, SatMaeGroupViTBB],
    ssl_weight_p:str,
    backbone_name:str,
    ssl_strategy:str,
):
    """Function that applies SSL-pretrained model to an Image Tensor to X_geo (Environment features)"""
    
    # Loading Model Weights
    print(colored(f"Attempting to load pretrained weights : {ssl_weight_p}", "yellow"))
    if isinstance(ssl_model, (SimSiamBBResnet, SimSiamBBSwinViT,ByolBBResnet, ByolBBSwinViT,DinoBBResnet, DinoBBSwinViT, DinoBBViT,MaeBBViT)):
        try:
            ssl_model = load_ssl_weights(ssl_model, ssl_weight_p)
            print(colored(f"weights for simsiam/byol/dino/mae loaded successfully","green"))
        except:
            RuntimeError(colored("Failed to load weights", "red")) 
    else:
        try:
            model_checkpoint = torch.load(ssl_weight_p)
            ssl_model.load_state_dict(model_checkpoint["state_dict"], strict = False)
            print(colored(f"weights for satmae loaded successfully","green"))
        except:
            RuntimeError(colored("Failed to load weights", "red")) 

    # Extract Backbone
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
        case "satmae":
            ssl_model_bb = ssl_model.model
        case _:
            raise ValueError(f"Invalid ssl strategy : {ssl_stategy}")

    return ssl_model_bb

# ==============================================================================
# ----------------------------------- Main ----------------------------------- #
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser for Geo vect extraction")
    parser.add_argument("-data_csv", type = str, default = "data/processed/sshsph_mlr/lfmykmns_mlr_geopts_imgp_c13_256x_v1.csv")
    parser.add_argument("-data_img_root", type = str, default = "data/interim/gee_sat/sen2a_c13_256x_ext")
    parser.add_argument("-ssl_weights_root", type = str, help = "Path to root folder containing ssl trained weights, i.e byol-is256-effbs256-ep10-bbResnet-dsDrnSen2a-clClUcl-nmTTDrnSatNM")
    parser.add_argument("-version", type = int, help = "There might be multiple versions of models")
    parser.add_argument("-weight_every_n_epochs", type = int, default = 1, help = "Load ssl weights every n epochs to prevent long training times")
    parser.add_argument("-weight_at_epoch", type = int, default = None, help = "Load a weight saved at a specific epoch")
    args = parser.parse_args()

    # --------------------------------- Save Root -------------------------------- #
    # We will be saving the feature inside ssl_root folder
    save_root = os.path.join(args.ssl_weights_root, f"version_{args.version}", "downstream")
    # Create a folder to store all the downstream information
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    # We will also create a folder to save the csv files
    save_geo_csv_fold = os.path.join(save_root, "data_geo_csv")
    if not os.path.exists(save_geo_csv_fold):
        os.mkdir(os.path.join(save_geo_csv_fold))

    # ------------------------------- Read Dataset ------------------------------- #
    df = pd.read_csv(args.data_csv)
    
    # ------------------------------ Hyperparameters ----------------------------- #
    with open(os.path.join(args.ssl_weights_root, f"version_{args.version}", "hparams.yaml")) as f:
        hyper_config = yaml.safe_load(f)

    # -------------------------------- Getbackbone ------------------------------- #
    # we first need to identify ssl method since satmae handles things differently
    ssl_name =os.path.basename(args.ssl_weights_root).split("-")[0] #i.e satmae
    backbone_name = hyper_config["backbone"] #i.e vit
    print(colored(f"Identified, ssl_method : {ssl_name} ; backbone : {backbone_name}", "blue"))

    # Note we need get the backbone bit dont need to load the weights as that will be done for the entire SSL model ...
    # ... including the BB
    match backbone_name:
        case "resnet" :
            backbone_model = resnet50()
            backbone_model = torch.nn.Sequential(*list(backbone_model.children())[:-1])
        case "swin-vit":
            backbone_model = SwinTransformer(num_classes = 51)
        case "vit":
            # We need to handle these seperately as they are using Lightly SSL
            if ssl_name in ["simsiam","byol","dino","mae"]:
                backbone_model = vit_base_patch16_224(num_classes = 0)
            # We have to handle SatMAE seperately 
            else:
                # Force model params input size to be set to 96
                hyper_config["input_size"] = 96
                grouped_bands = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9))
                # backbone_model = MaskedAutoencoderGroupChannelViT(
                #     img_size = hyper_config["input_size"],
                #     patch_size = hyper_config["patch_size"],
                #     in_chans = 10,
                #     spatial_mask = hyper_config["spatial_mask"],
                #     channel_groups = grouped_bands,
                #     channel_embed = hyper_config["channel_embed"],
                #     embed_dim = hyper_config["embed_dim"], # default : 1024
                #     depth = hyper_config["depth"],
                #     num_heads = hyper_config["num_heads"],
                #     decoder_channel_embed = hyper_config["decoder_channel_embed"],
                #     decoder_embed_dim = hyper_config["decoder_embed_dim"],
                #     decoder_depth = hyper_config["decoder_depth"],
                #     decoder_num_heads = hyper_config["decoder_num_heads"],
                #     mlp_ratio = hyper_config["mlp_ratio"],
                #     norm_layer=partial(nn.LayerNorm, eps=1e-6),
                #     norm_pix_loss = hyper_config["norm_pix_loss"],
                # )

                backbone_model = GroupChannelsVisionTransformer(
                        channel_embed = 256, embed_dim = 768, depth = 12, 
                        num_heads = 12, mlp_ratio = 4, qkv_bias = True,
                        norm_layer  = partial(nn.LayerNorm, eps = 1e-6),
                        patch_size = 8, img_size = 96, in_chans = 10
                )
        case _:
            raise ValueError(colored("Incorrect backbone name found", "red"))

    # ------------------------------- Get SSL Model ------------------------------ #
    # Note we only identify and instantiate ssl model, but we dont load the weights
    # this will be done later as we need to load weights at each epoch

    # we need some model params, these are also avaialble in hyper_config but to
    # keep with the standard convention we will just add the hyperparams that are
    # necessary for some functions
    model_params = {
        "batch_size" : int(hyper_config["eff_batch_size"] / (hyper_config["devices"] * hyper_config["nodes"])),
        "lr" : hyper_config["lr"]
    }

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
        case "satmae":
            satmae_params = ssl_config["satmae_params"]
            if backbone_name == "vit":
                ssl_model = SatMaeGroupViTBB(model_params, backbone_model)

        case _:
            raise(KeyError(f"Model flag incorrect, got {ssl_name} but should get 'simsiam','byol','dino','mae"))

    # -------------------------------- DataLoader -------------------------------- #

    if backbone_name == "resnet":
        transforms = {
            "sen2a" : Compose([ToTensor(), Normalize(ssl_config["sat_img_mean"], ssl_config["sat_img_std"])]),
            "drn" : Compose([ToTensor(), Normalize(ssl_config["drn_img_mean"], ssl_config["drn_img_std"])])
        }
    else:
        # Resizing ... to work with ViT
        if ssl_name == "satmae":
            transforms = {
                "sen2a" : Compose([
                    Normalize(ssl_config["sat_hyp_img_mean"], ssl_config["sat_hyp_img_std"]), 
                    Resize(ssl_config["satmae_params"]["pretrain_patch_hw"])
                    ])
                }
        else:
            transforms = {
                "sen2a" : Compose([ToTensor(), Resize(224), Normalize(ssl_config["sat_img_mean"], ssl_config["sat_img_std"])]),
                "drn" : Compose([ToTensor(), Resize(224), Normalize(ssl_config["drn_img_mean"], ssl_config["drn_img_std"])])
            }

    # TO save time We will be loading images first
    img_csv_data = Get_Img_and_Csv_Data(
        df = df,
        img_p_colname = "sen2a_c13_ext_paths_v2",
        img_transforms = transforms,
        num_channels = 13,
        drop_bands = [1,9,10],
        data_root = args.data_img_root,
        norm = None,
        clip = None
    )

    dataloader = DataLoader(img_csv_data, batch_size = len(img_csv_data), shuffle = False, collate_fn = collate_pandas_numpy)

    # start = time.time()
    # df_r, img = next(iter(dataloader))
    # end = time.time()
    # print(colored(f"Time executed loading images : {(end-start):.3f}s", "blue"))

    # df_n = pd.DataFrame(df_r)
    # img_t = torch.from_numpy(img)
    # print(df_n.shape)
    # print(img_t.shape)
    torch.manual_seed(0)
    img_t = torch.randint(0,255, size = (1,10,96,96)).float()
    #print(img_t)


        
    # ------------------------------ Get Geo Vector ------------------------------ #
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
    
    for ssl_weight in ssl_weights:
        ssl_model_bb = get_geo_vector(
            X_img = img_t,
            ssl_model = ssl_model,
            ssl_weight_p = ssl_weight,
            backbone_name = backbone_name,
            ssl_strategy = ssl_name,
        )
        #x_feat, _, _ = ssl_model_bb.forward_encoder(img_t, 0)
        x_feat = ssl_model_bb.forward_features(img_t)
        print(x_feat.shape)
        break

    
        