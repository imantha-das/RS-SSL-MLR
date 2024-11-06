# ==============================================================================
# SimSiam Paper Implementation Details : Changes reflected on right after "|"

# Optimizer 
#   optimizer : SGD, momentum = 0.9 ; weight_decay = 0.0001
#   learning rate :  base lr = 0.05 ; lr = base_lr * BS/256
#   cosine decay | We have used CosineAnnealingLR from pytorch which maybe slightly different
# Batch Size : 512 | Due to computational restraints (Even with 8 GPUS's) we have set to 256
# Projection & Prediction Heads
#   Projection Head : input_size = 2048, output_size = 2048, hidden_size = 2048
#   Prediction Head : Input_size = 2048, Output_size = 2048, hidden_size = 512
# Loss Function : negative cosine similarity
# Default Backbone : RESNET-50
#   100 Epoch pre-training ablation study | Instead we use Pre-Trained model on RSP
# ==============================================================================

import os
import math
import argparse

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.transforms import Compose

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from lightly.transforms import SimSiamTransform
from lightly.data import LightlyDataset

from ssl_utils import load_model_weights
import sys

import ssl_config as config

from torchvision.models import resnet50

class SimSiam(pl.LightningModule):
    def __init__(self, model_params:dict, backbone:torch.nn.Sequential):
        super().__init__()
        self.projection_head = SimSiamProjectionHead(
            input_dim = model_params["proj_input_dim"],
            hidden_dim = model_params["proj_hidden_dim"],
            output_dim = model_params["proj_output_dim"]
        ) # 2048 > 2048 > 2048
        self.prediction_head = SimSiamPredictionHead(
            input_dim = model_params["pred_input_dim"],
            hidden_dim = model_params["pred_hidden_dim"],
            output_dim = model_params["pred_output_dim"]
        ) #2048 > 512 > 2048
        self.out_dims = model_params["pred_output_dim"] # we need access to out_dims for computing collapse levels
        
        #model parameters
        self.model_params = model_params

        # backbone
        self.backbone = backbone

        # Loss
        self.criterion = NegativeCosineSimilarity()

        # To check if model is collapsing
        self.avg_loss = 0.0
        self.avg_output_std = 0.0
        self.loss = 0.0

        # Batch size not required by this class but needed by "auto_scale_batch_size" in pl.Trainer when locating the "max" batchsize
        self.batch_size = config.BATCH_SIZE

    def forward(self, X):
        f = self.backbone(X) #(b,3,256,256) -> ... -> (b,2048,1,1) 
        f = f.flatten(start_dim = 1) #(b,2048,1,1) -> (b,2048)
        z = self.projection_head(f) #(b,2048) -> (b,2048) -> (b,2048) -> (b, 2048)
        p = self.prediction_head(z) #(b,2048) -> (b,512) -> (b,2048)
        z = z.detach() #SimSiams stop the gradient to prevent collapse
        return z, p
    
    def training_step(self, batch, batch_idx):
        # lightlyDataset passes X,y,path = batch, augmented versions X0,X1 = X
        (X0,X1) = batch[0]
        z0,p0 = self.forward(X0) #(b,2048),(b,2048)
        z1,p1 = self.forward(X1) #(b,2048),(b,2048)
        # Compute loss
        self.loss = 0.5 * (self.criterion(z0,p1) + self.criterion(z1, p0))
        # Compute Collapse
        if self.model_params["compute_collapse?"]:
            self.compute_std_per_dim(p0)

        return self.loss 
    
    def on_train_epoch_end(self):
        self.log("training_loss", self.loss)
        if self.model_params["apply_lr_scheduler?"]:
            self.log("current_lr", self.learning_rate_scheduler.get_lr()[0])
        if self.model_params["compute_collapse?"]:
            self.log("avg_output_std", self.avg_output_std)
            collapse_level = max(0.0, 1 - math.sqrt(self.out_dims) * self.avg_output_std)
            self.log("collapse_level", collapse_level)

    
    def compute_std_per_dim(self, p0):
        """Calculate per dimension std of outputs which we will use 
        to check whether embeddings collapse"""
        #output = p0.copy() # This is the prediction from left network
        output = p0.clone().detach() #(b, 2048)
        output = F.normalize(output, dim = 1) #(b, 2048)
        output_std = torch.std(output,dim = 0) #(2048,)
        output_std = output_std.mean() #(,) 
        # Check if embeddings are collapsing : Use a moving average to track the loss & std
        w = 0.9
        self.avg_loss = w * self.avg_loss + (1-w) * self.loss.item()
        self.avg_output_std = w * self.avg_output_std + (1-w) * output_std.item()

    def configure_optimizers(self):
        if self.model_params["apply_lr_scheduler?"]:
            lr = self.model_params["base_lr"] * config.BATCH_SIZE / 256
            optimizer = torch.optim.SGD(params = self.parameters(),lr = lr, weight_decay = 0.0001)
            self.learning_rate_scheduler = CosineAnnealingLR(optimizer, T_max = config.MAX_EPOCHS)
            return [optimizer], [{"scheduler":self.learning_rate_scheduler, "interval" : "epoch"}]
        else:
            return torch.optim.SGD(params = self.parameters(), lr = config.LR)
            

if __name__ == "__main__":
    # ------------------------------ Argument Parser ----------------------------- #
    
    parser = argparse.ArgumentParser(description = "Train BYOL algorithm") 
    parser.add_argument(
        "-data_fold_drn", 
        type = str, 
        help = "Path to drone data folder", 
        default = "data/processed/sshsph_drn/drn_c3_256x_pch"
    )
    parser.add_argument(
        "-data_fold_sat", 
        type = str, 
        help = "Path to sentinel data folder",
        default = "data/interim/gee_sat/sen2a_c3_256x_clp0.3_uint8_ucln_pch"
    )
    parser.add_argument(
        "-pretrain_weights_file", 
        type = str, 
        help = "Path to pretrained weights file", 
        default="model_weights/pretrain_weights/rsp-aid-resnet-50-e300-ckpt.pth"
    )
    parser.add_argument(
        "-save_weights_fold", 
        type = str, 
        help = "Path to where model weights + stats are saved", 
        default = "model_weights/ssl_weights"
    )
    args = parser.parse_args()

    # ----------------------------- Model Parameters ----------------------------- #

    model_params = config.simsiam_model_params # This is dict containing model params

    # ------------------------------- Get Backbone ------------------------------- #

    resnet = load_model_weights(
        resnet50, 
        path_to_weights=args.pretrain_weights_file, 
        num_classes = 51
    )
    resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])

    # ---------------------------- SimSiam Transforms ---------------------------- #
    drn_transforms = SimSiamTransform(
        input_size = 256, 
        normalize = {"mean" : config.drn_img_mean, "std" : config.drn_img_std}
    )
    sat_transforms = SimSiamTransform(
        input_size = 256, 
        normalize = {"mean" : config.sat_img_mean, "std" : config.sat_img_std}
    )

    # --------------------------- Dataset + DataLoader --------------------------- #

    # DRONE Dataset
    drn_trainset = LightlyDataset(input_dir = args.data_fold_drn, transform=Compose([drn_transforms])) # .__getitem__() returns -> view1,view2,fname
    #drn_trainloader = DataLoader(drn_trainset, batch_size=config.BATCH_SIZE)

    # SENTINEL Dataset
    sen2a_trainset = LightlyDataset(input_dir= args.data_fold_sat, transform=Compose([sat_transforms]))
    #sen2a_trainloader = DataLoader(sen2a_trainset, batch_size=config.BATCH_SIZE)

    # COMBINE datasets
    drnsen2a_trainset = ConcatDataset([drn_trainset, sen2a_trainset])
    drnsen2a_trainloader = DataLoader(
        drnsen2a_trainset, 
        batch_size=config.BATCH_SIZE,
        num_workers = config.DATALOADER_NUM_WORKERS
    )

    # ------------------------------- SimSiam Model ------------------------------ #
    simsiam = SimSiam(model_params,resnet_backbone)
    
    # --------------------------------- Training --------------------------------- #
    save_name = config.SAVE_NAME
    logger = CSVLogger(save_dir = args.save_weights_fold, name = save_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_weights_fold, save_name), 
        filename="epoch:{epoch}",
        save_on_train_epoch_end=True,
        save_weights_only = True,
        save_top_k = -1
    )
    trainer = pl.Trainer(
        default_root_dir = os.path.join(args.save_weights_fold, save_name),
        devices = -1,
        num_nodes= config.NODES,
        accelerator = "gpu",
        strategy = "ddp",
        max_epochs = config.MAX_EPOCHS,
        precision = config.PRECISION,
        logger = logger,
        callbacks = [checkpoint_callback],
        #auto_scale_batch_size = config.AUTO_SCALE_BATCH_SIZE # to find "max" batch_size that can be procesedwith resources (gpu)
    )

    trainer.fit(simsiam, drnsen2a_trainloader)
