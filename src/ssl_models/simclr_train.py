# ==============================================================================
# Desc : Training SimCLR to identify image embeddings for Remote sensing imagery
#   - We will be using a pretrained model Resnet50 model from the RSP repository
#   - SimCLR 
#       - backbone(x) -> hi -> mlp(hi) -> zi -> peform contrastive loss
#       - Once trained the simCLR paper uses hi instead of zi due to downstream 
#       task results performing better on hi
#       - Due to this, this script will not be freezing the weights of the
#       backbone as we want to extract hi instead of zi              
# ==============================================================================
import sys
import os
from glob import glob
import multiprocessing

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl 

from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss
import lightly.transforms.utils as lightly_utils
import argparse

import config

from utils import load_model_weights

sys.path.append("RSP/Scene Recognition")
from models.resnet import resnet50

NUM_WORKERS = multiprocessing.cpu_count()
SEED = 1
pl.seed_everything(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("-dfold", "--data_folder", type = str)
args = parser.parse_args()

class SimCLR(pl.LightningModule):
    """Desc : Implementation of simCLR algorithm using LightlySSL"""
    def __init__(self):
        super().__init__()
        # Load resnet50 model from RSP prepository
        resnet = load_rsp_weights(
            resnet50,
            path_to_weights = "pretrain_weights/rsp-aid-resnet-50-e300-ckpt.pth", 
            num_classes = 51
        )
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])

        # Define projection head
        hidden_dims = resnet.fc.in_features #2048
        self.projection_head = SimCLRProjectionHead(
            input_dim = hidden_dims, 
            hidden_dim = hidden_dims, #default 2048
            output_dim = config.Z_DIMS #default is 128
        )

        # Define loss function : Normalized Temperatre Scaled Cross Entropy Loss 
        self.criterion = NTXentLoss()

    def forward(self, X):
        h = self.backbone(X) #(b,3,512,512) -> (b,2048,1,1)
        z = self.projection_head(h.flatten(start_dim = 1)) #(b,2048,1,1) -> (b,2048) -> (b,128)

        return z

    def training_step(self, batch, batch_idx):
        X,y,f = batch #(images, labels,filepaths)
        Xi,Xj = X # Seperate the two augmented images
        zi = self.forward(Xi) 
        zj = self.forward(Xj)
        # Loss in embedding space 
        loss = self.criterion(zi,zj)
        self.log("train_loss_ntx" , loss)
        return loss
    
    def configure_optimizers(self):
        #todo, We need to get bigger batches and use LARS optimizer
        optim = torch.optim.SGD(
            self.parameters(),
            lr = config.LR,
            momentum = 0.9,
            weight_decay = 5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, 
            config.MAX_EPOCHS
        )

        return [optim], [scheduler]

if __name__ == "__main__":
    # Path to folder containg images (.tif)
    data_root = "data/SSHSPH-RSMosaics-MY-v2.1/images"
    path_to_data = os.path.join(data_root, args.data_folder)
    # Transforms that will be applied to training set
    simclr_transform = SimCLRTransform(
        input_size = config.INPUT_SIZE,
        normalize = {"mean" : config.IMAGE_MEAN, "std" : config.IMAGE_STD}
    ) #input image is 512, lets keep all other parameters the same
    # We can use LightlyDataset function straight away to create a dataset
    trainset = LightlyDataset(path_to_data, transform = simclr_transform)
    # Training Dataloader
    trainloader = DataLoader(
        trainset, 
        batch_size = config.BATCH_SIZE, #todo we need bigger batches
        shuffle = True, 
        drop_last = True,
        num_workers = NUM_WORKERS #Removing parallel workloads
    )

    simclr = SimCLR()
    # Train Model
    trainer = pl.Trainer(
        default_root_dir = f"ssl_weights/simclr-is{config.INPUT_SIZE}-bs{config.BATCH_SIZE}-ep{config.MAX_EPOCHS}",
        devices = 1,
        accelerator = "gpu",
        max_epochs = config.MAX_EPOCHS
    )

    trainer.fit(simclr, trainloader)
