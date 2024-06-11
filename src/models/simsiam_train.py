# ==============================================================================
# Training code for SimSiam
# Base line setting
#   optimizer : SGD, momentum = 0.9
#   learning rate :  base lr = 0.5 , lr = lr * BS/256
#   Cosine Decay : 0.0001
#   BS : 512
#   Projection Head : input_size = 2048, output_size = 2048, hidden_size = 2048
#   Prediction Head : Input_size = 2048, Output_size = 2048, hidden_size = 512

#todo : Need to incorporate cosine decay
#todo : lr = lr * BS/256 , where base_lr = 0.5
#! Cannot set batch size to 512 too big for memory
# ==============================================================================

# ---------------------------------- imports --------------------------------- #

import os
import sys
import math
import argparse

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from lightly.transforms import SimSiamTransform
from lightly.data import LightlyDataset

from utils import load_rsp_weights
import config

sys.path.append("RSP/Scene Recognition")
from models.resnet import resnet50


# ------------------------------- SimSIam Model ------------------------------ #
class SimSiam(pl.LightningModule):
    def __init__(self, resnet_hidden_dims, proj_hidden_dims, pred_hidden_dims, out_dims):
        super().__init__()
        resnet = load_rsp_weights(
            resnet50, 
            path_to_weights = "models/rsp_weights/rsp-aid-resnet-50-e300-ckpt.pth", 
            num_classes = 51
        )
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimSiamProjectionHead(resnet_hidden_dims,proj_hidden_dims,out_dims) #These are the default values
        self.prediction_head = SimSiamPredictionHead(out_dims,pred_hidden_dims,out_dims) #These are deafult values, bottleneck created in original paper
        self.out_dims = out_dims # we need access to out_dims for computing collapse levels
        # Loss
        self.criterion = NegativeCosineSimilarity()

        # To check if model is collapsing
        self.avg_loss = 0.0
        self.avg_output_std = 0.0
        self.loss = 0.0


    def forward(self, X):
        f = self.backbone(X).flatten(start_dim = 1) #(b,3,256,256) -> (b,512,1,1) -> (b,512)
        z = self.projection_head(f) #(b,2048)
        p = self.prediction_head(z) #(b,2048)
        z = z.detach() #We stop the gradient to prevent collapse
        return z, p

    def training_step(self, batch, batch_idx):
        (X0,X1) = batch[0]
        z0,p0 = self.forward(X0) #(b,2048),(b,2048)
        z1,p1 = self.forward(X1) #(b,2048),(b,2048)
        # Compute loss
        self.loss = 0.5 * (self.criterion(z0,p1) + self.criterion(z1, p0))
        
        # ======================================================================
        #! comment out if you DONT need to compute collapse
        # calculate the per-dimensional standard deviation of the outputs 
        # we use this later to check whether the embeddings are collapsing
        output = p0.detach() #(b, 512)
        output = F.normalize(output, dim = 1)
        output_std = torch.std(output, 0) #(2048,)
        output_std = output_std.mean() # () <- float value

        # Check if embeddings are collapsing : Use a moving average to track the loss & std
        w = 0.9
        self.avg_loss = w * self.avg_loss + (1-w) * self.loss.item()
        self.avg_output_std = w * self.avg_output_std + (1-w) * output_std.item()
        # ======================================================================
        
        return self.loss

    def compute_std(self,p0,loss):
        pass

    def on_train_epoch_end(self):
        # the level if collapse is karge uf the standard deviation if the 12 normalised
        # outputs are is much smaller than 1 / sqrt(dim)
        collapse_level = max(0.0, 1 - math.sqrt(self.out_dims) * self.avg_output_std)
        self.log("collapse_level", collapse_level)
        self.log("avg_output_std", self.avg_output_std)
        self.log("training loss", self.loss)


    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr = 0.001)
        return optim

if __name__ == "__main__":
    # ------------------------------ Argument Parse ------------------------------ #
    parser = argparse.ArgumentParser()
    parser.add_argument("-dfold", "--data_folder", type = str)
    args = parser.parse_args()
    path_to_data = args.data_folder

    # ---------------------- Simiam Transforms + Data loader --------------------- #
    transform = SimSiamTransform(input_size = 256, normalize = {"mean" : config.IMAGE_MEAN, "std" : config.IMAGE_STD})
    trainset = LightlyDataset(path_to_data, transform = transform)
    trainloader = DataLoader(trainset, batch_size = config.BATCH_SIZE, shuffle = True, drop_last = True)

    # ------------------------------- SimSiam Model ------------------------------ #
    simsiam = SimSiam(
        resnet_hidden_dims = 2048,
        proj_hidden_dims = 2048,
        pred_hidden_dims = 512,
        out_dims = 2048
    )

    # --------------------------------- Training --------------------------------- #
    logger = CSVLogger("models/ssl_weights", name = f"simsiam-is{config.INPUT_SIZE}-bs{config.BATCH_SIZE}-ep{config.MAX_EPOCHS}")
    trainer = pl.Trainer(
        default_root_dir = f"models/ssl_weights/simsiam-is{config.INPUT_SIZE}-bs{config.BATCH_SIZE}-ep{config.MAX_EPOCHS}",
        devices = 1,
        accelerator = "gpu",
        max_epochs = config.MAX_EPOCHS,
        logger = logger
    )

    trainer.fit(simsiam, trainloader)
