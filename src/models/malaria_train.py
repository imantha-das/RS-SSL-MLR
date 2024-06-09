# ==============================================================================
# Desc : Python Script to train downstream task (Malaria dataset)
# ==============================================================================

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


import cv2
from sklearn.model_selection import train_test_split
from typing import List
from termcolor import colored

from utils import SSHSPH_MALARIA_MY
import config


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

class MalariaClassifier(pl.LightningModule):
    def __init__(self, backbone:torch.nn.modules.Sequential, emb_size:int, feat_size:int):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(emb_size + feat_size,2)
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
        img, lab, feat = batch
        logits = self.forward(img, feat) #(*,2) Since we are not explicitly applying Softmax
        # Compute loss
        loss = F.cross_entropy(logits, lab)
        self.log("train_loss", loss, on_epoch = True)
        # Compute accuracy
        correct_preds = self.compute_accuracy(logits, lab)
        self.log("accuracy", correct_preds, on_epoch = True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, lab, feat = batch
        logits = self.forward(img, feat)
        # Compute loss
        loss = F.crossentropy(yhat, lab)
        self.log("valid_loss", loss)
        #Compute accuracy
        correct_preds = self.compute_accuracy(logits, lab)
        self.log("valid_accuracy", correct_preds, on_epoch = True)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 0.001)

    def compute_accuracy(self,logits,y):
        yhat = torch.argmax(logits, dim = 1)
        correct_preds = torch.sum(yhat == y)
        return correct_preds
    

  

if __name__ == "__main__":
    path_to_weights = "ssl_weights/simsiam-is256-bs128-ep100/version_0/checkpoints/epoch=99-step=48700.ckpt"

    backbone = get_pretrained_model_backbone(SimSiam, path_to_weights)

    df = pd.read_csv("data/SSHSPH-malaria-prevelance-v1.0/malaria_pts_with_images.csv")
    train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 0)

    #! NOTE : Data is already in the range of 0-1 so no need to further normalize
    train_data = SSHSPH_MALARIA_MY(
        train_df, 
        target = 'pk',
        transform = Compose([ToTensor()], Normalize(mean = config.IMAGE_MEAN, std = config.IMAGE_STD)),
        features = ["PkSSP2_x","PkSera3.Ag2"]
        #features = []
    )

    valid_data = SSHSPH_MALARIA_MY(
        valid_df, 
        target = "pk",
        transform = Compose([ToTensor()]),
        features = []
    )
    train_loader = DataLoader(train_data, batch_size = 16)
    valid_loader = DataLoader(valid_data, batch_size = 16)

    # for img,lab,feat in train_loader:
    #     print(img.min(), img.max())

    # t1 = torch.rand([1,3,256,256], device = torch.device("cuda:0"))
    device = "cuda:0"
    model = MalariaClassifier(backbone, 2048, 2)
    #model = model.to(torch.device(device))
    
    #print(df.columns.values)

    # for imgs, labs, feats in train_loader:
    #     imgs = imgs.to(torch.device(device))
    #     feats = feats.to(torch.device(device))
    #     labs = labs.to(torch.device(device))
    #     logits = model.forward(imgs, feats)
    #     print(logits)
    #     print("\n")
    #     yhat = torch.argmax(logits, dim = 1)
    #     print(torch.sum(yhat == labs))
    #     break

    logger = logger = CSVLogger("tmp/loss_met", name = f"malclass-is{256}-bs{16}-ep{10}")
    trainer = pl.Trainer(max_epochs = 10, default_root_dir = "tmp", logger = logger)
    trainer.fit(model, train_dataloaders = train_loader)

    # img_t = torch.rand((1,3,256,256), device = device)
    # feat_t = torch.rand((1,8), device = device)
    # yhat = model.forward(img_t, feat_t)
    # print(yhat, torch.argmax(yhat, dim = 1))