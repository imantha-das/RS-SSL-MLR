# ==============================================================================
# Desc : Python Script to train downstream task (Malaria dataset)
# ==============================================================================
import sys
import pandas as pd

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor,Normalize, Compose
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torchsummary import summary
from simsiam_train import SimSiam
from byol_train import BYOL

import cv2
from sklearn.model_selection import train_test_split

import argparse 
from typing import List
from termcolor import colored

from utils import SSHSPH_MALARIA_MY
import config
import malaria_config

# ----------------- Get model backbone from SSL traine model ----------------- #
def get_pretrained_model_backbone(cl_model:pl.LightningModule,model_weights_p:str):
    """
    Loads the model weights to SSL model and returns the backbone
    Inputs
        - model_weights_p : Path to state dict containing weights
    """
    
    model = cl_model.load_from_checkpoint(
        model_weights_p,
        resnet_hidden_dims = 2048,
        proj_hidden_dims = 2048,
        pred_hidden_dims = 512,
        out_dims = 2048
    )
    return model.backbone

# -------------- To verify if model weights are loaded properly ------------- #
def print_model_weight(model):
    """Print to check if the weights are loaded properly"""
    for name, param in model.named_parameters():
        print("-"*20)
        print(f"name : {name}")
        print(f"values : \n{param}")

# ------------------------- Malaria Prediction Model ------------------------- #
class MalariaClassifier(pl.LightningModule):
    def __init__(self, backbone:torch.nn.modules.Sequential, emb_size:int, feat_size:int):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(emb_size + feat_size, 1) #* One neuron at the end since we are using BCEWithLogitLoss
        self.emb_size = emb_size
        self.feat_size = feat_size

        # Freeze parameters of backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # note parameters of the linear layer will not be frozen

    def forward(self, img, feat): 
        z = self.backbone(img) #(*,2048,1,1)
        z = z.flatten(start_dim = 1) #(*,2048)
        #print(colored(f"z : {z.shape}", "green"))
        # Add the features 
        z_plus_feat = torch.cat([z, feat], dim = 1)  
        #print(colored(f"z_plus_feat : {z_plus_feat.shape}", "green"))

        assert z_plus_feat.shape[1] == self.emb_size + self.feat_size, "Embedding + Feature sizes dont mach output !"
        fc_out = self.fc(z_plus_feat) #(*,2) We dont need Softmax as we using CrossEntropyLoss that includes softmax
        
        return fc_out

    def training_step(self, batch, batch_idx):
        img, lab, feat = batch #(*,3,256,256), (*,) , (*, 55)
        logits = self.forward(img, feat).ravel() # usage of .ravel() to convert (*,1) -> (*,)
        # Compute loss : We need to convert the label from type long -> float
        loss = F.binary_cross_entropy_with_logits(logits, lab.float())
        self.log("train_loss", loss, on_epoch = True)
        # Compute accuracy
        correct_preds = self.get_correct_preds(logits, lab) 
        accuracy = correct_preds / img.shape[0]
        self.log("train_accuracy", accuracy.item(), on_epoch = True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, lab, feat = batch 
        logits = self.forward(img, feat).ravel()
        # Compute loss
        loss = F.binary_cross_entropy_with_logits(logits, lab.float())
        self.log("valid_loss", loss, on_epoch=True)
        #Compute accuracy
        correct_preds = self.get_correct_preds(logits, lab)
        accuracy = correct_preds / img.shape[0]
        self.log("valid_accuracy", accuracy.item(), on_epoch = True)    
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 0.001)
    
    def get_correct_preds(self,logits,y):
        probs = torch.sigmoid(logits) #(*,) <- we have already used ravel to convert logits from (*,1) -> (*,)
        preds = torch.round(probs) #(*,)
        correct_preds = torch.sum(preds == y)
        return correct_preds
    

if __name__ == "__main__":
    # ------------------------------ Argument Parser ----------------------------- #
    parser = argparse.ArgumentParser(description = "Malaria Classifier")
    parser.add_argument("-ssl_weight_p", type = str, help = "Path to SSL weights", default = "models/ssl_weights/simsiam-is256-bs128-ep99/version_2/checkpoints/epoch=98-step=48213.ckpt")
    parser.add_argument("-save_weight_p", type = str , help = "Path to save weights for malaria classifier", default = "models/mlr_weights")
    parser.add_argument("-mlr_csv_p", type = str, help = "Path to Malaria Dataset Processed", default = "data/processed/mlr_pts_no_missing.csv")
    parser.add_argument("-ssl_model", type = str, help = "Specify Name of SSL model, Options : simsiam | byol")
    args = parser.parse_args()

    # ------------------ Load SSL weights and get just backbone ------------------ #
    match args.ssl_model:
        case "simsiam":
            ssl_model = SimSiam
        case "byol":
            ssl_model = BYOL 
        case _:
            raise(ValueError("Incorrect Model Choose from options : simsiam, byol"))
        
    backbone = get_pretrained_model_backbone(ssl_model, args.ssl_weight_p)
    #print(summary(backbone))
    #print_model_weight(backbone)

    # --------------------------- Load Malaria Dataset --------------------------- #
    df = pd.read_csv(args.mlr_csv_p, index_col = "Unnamed: 0")
    train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 0)


    numeric_feat = malaria_config.numeric_feat
    cat_feat = malaria_config.cat_feat

    train_data = SSHSPH_MALARIA_MY(
        train_df, 
        target = 'hadMalaria',
        transform = Compose([ToTensor(), Normalize(mean = config.IMAGE_MEAN, std = config.IMAGE_STD)]),
        numeric_feat_names= numeric_feat,
        cat_feat_names= cat_feat,
        train=True,
        feat_transformer=None
    )
    #todo : We might need to Normalize these, only Images were normalized
    feat_transformer = train_data.column_trans #get the trained One-Hot-Encorder trained on categorical domain features only

    valid_data = SSHSPH_MALARIA_MY(
        valid_df, 
        target = "hadMalaria",
        transform = Compose([ToTensor(), Normalize(mean = config.IMAGE_MEAN, std = config.IMAGE_STD)]),
        numeric_feat_names= numeric_feat,
        cat_feat_names= cat_feat,
        train = False,
        feat_transformer=feat_transformer
    )

    train_loader = DataLoader(train_data, batch_size = malaria_config.BATCH_SIZE)
    valid_loader = DataLoader(valid_data, batch_size = malaria_config.BATCH_SIZE)

# -------------------------------- Test Model -------------------------------- #
    # print("Train Loader")
    # for img,lab,feat in train_loader:
    #     print(img.shape, lab.shape, feat.shape) #(*,3,256,256), (*,) , (*, 55)
    # print("\nValid Loader")
    # for img,lab,feat in valid_loader:
    #     print(img.shape, lab.shape, feat.shape) #(*,3,256,256), (*,) , (*, 55)
        

    # device = "cuda:0"
    # model = MalariaClassifier(backbone = backbone, emb_size = 2048, feat_size = 55)
    # model = model.to(torch.device(device))
    
    # # #print(df.columns.values)

    # for imgs, labs, feats in train_loader:
    #     imgs = imgs.to(torch.device(device))
    #     feats = feats.to(torch.device(device))
    #     labs = labs.to(torch.device(device))
    #     logits = model.forward(imgs, feats).ravel()
    #     loss = F.binary_cross_entropy_with_logits(logits, labs.float())
    #     probs = F.sigmoid(logits)
    #     preds = torch.round(probs)
    #     correct_preds = torch.sum(preds == labs)
    #     accuracy = correct_preds / imgs.shape[0]
    #     print(correct_preds, accuracy)
    #     break


# ------------------------ Pytorch Lightning Training ------------------------ #
    model = MalariaClassifier(backbone = backbone, emb_size = 2048, feat_size = 55)
    logger = logger = CSVLogger("models/mlr_weights", name = f"mlr-{args.ssl_model}-is{256}-bs{malaria_config.BATCH_SIZE}-ep{malaria_config.EPOCHS}")
    trainer = pl.Trainer(max_epochs = malaria_config.EPOCHS, default_root_dir = "tmp", logger = logger)
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders= valid_loader)
