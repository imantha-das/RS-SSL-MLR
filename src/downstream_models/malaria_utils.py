import os
import sys
import pandas as pd 
import rasterio

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.models import resnet50
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer

import pickle

from termcolor import colored

from typing import List, Union, Dict
from torchsummary import summary
import warnings
from tqdm import tqdm
import logging
from datetime import datetime

# sys.path.append("src/ssl_models")

# from byol_train import BYOL
# from simsiam_train import SimSiam

# ------------------------ Load Finetuned SSL weights ------------------------ #
def load_ssl_weights(model, ssl_weights:str = "model_weights/ssl_weights/simsiam-is256-effbs256-ep1-bbRes-dsDrnSen2a-clClUcl-nmTTDrnSatNM/epoch:epoch=0.ckpt"):
    """Loads weights finetuned weights after SSL training to model"""
    model_state = torch.load(ssl_weights)
    model.load_state_dict(model_state["state_dict"])
    return model

# -------------- To verify if model weights are loaded properly ------------- #
def print_model_weights(model):
    """Print to check if the weights are loaded properly"""
    for name, param in model.named_parameters():
        print("-"*20)
        print(f"name : {name}")
        print(f"values : \n{param}")

# ------------------------------- Setup logging ------------------------------ #
def setup_logger(save_loc):
    """Set up logging configuration"""
    # Clear existing log file or create new one
    with open(os.path.join(save_loc,'log.txt'), 'w') as f:
        f.write(f"=== New Run Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_loc,'log.txt'), mode='a'),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    return logging.getLogger(__name__)



