# ==============================================================================
# Implementation of BYOL (Bootstrap Your Latent) algorithm 
#! CHANGES from original paper implemntation
#! - we aren not using NegativeCosineSimilarity Loss function from lightly.loss
#! module. This is NOT the exact same loss function used in the paper
#! - We are using a BatchSize of 128 but the paper uses a much larger batchsize,
#!  however the Paper shows that BYOL works well with lower batchsizes.
#! - We will be using SGD instead of LARS optimizer. However LARS is not necessary 
#!  as our batch size is quite resonable (LARS used for very large batchsizes)
#! - Used a standard learning rate of 0.06 where the paper accomadates cosine decay 
#!  learning rate was used with a base value 0.2 to be scaled linearly with batch size

# ==============================================================================
import os 
import sys
import argparse
from copy import deepcopy

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl 
from pytorch_lightning.loggers import CSVLogger

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.transforms.byol_transform import BYOLTransform, BYOLView1Transform, BYOLView2Transform
from lightly.utils.scheduler import cosine_schedule
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.data import LightlyDataset

from torchsummary import summary

from utils import load_rsp_weights
import config
sys.path.append("RSP/Scene Recognition")
from models.resnet import resnet50

class BYOL(pl.LightningModule):
    def __init__(self):
        super(BYOL, self).__init__()
        resnet = load_rsp_weights(
            resnet50, 
            path_to_weights="models/rsp_weights/rsp-aid-resnet-50-e300-ckpt.pth", 
            num_classes = 51
        )
        # Online Network
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = BYOLProjectionHead(input_dim=2048,hidden_dim=4096, output_dim = 256)
        self.prediction_head = BYOLPredictionHead(input_dim = 256, hidden_dim = 4096, output_dim = 256)
        # Target Network
        self.backbone_momentum = deepcopy(self.backbone)
        self.projection_head_momentum = deepcopy(self.projection_head)
        # Freeze weights of target network as the weights are updated using exponential moving average 
        # of online network weights
        deactivate_requires_grad(self.backbone_momentum) 
        deactivate_requires_grad(self.projection_head_momentum)
        # Define Loss function : We will be using NegativeCosineSimilarity which is SimSiams Loss function
        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        # representation from resnet
        y = self.backbone(x).flatten(start_dim = 1) # (*,2048)
        # projection
        z = self.projection_head(y) #(*,256)
        # prediction
        p = self.prediction_head(z) #(*256)
        return p
    
    def forward_momentum(self, x):
        # representation from resent
        y = self.backbone_momentum(x).flatten(start_dim = 1) #(*,2048)
        # projection
        z = self.projection_head_momentum(y) #(*,256)
        # stop gradient
        z = z.detach()
        return z 
    
    def training_step(self, batch, batch_idx):
        # Updating momentum from online -> target
        momentum = cosine_schedule(step = self.current_epoch, max_steps = 10, start_value = 0.996, end_value = 1,)
        update_momentum(model = self.backbone, model_ema = self.backbone_momentum, m = momentum)
        update_momentum(model = self.projection_head, model_ema = self.projection_head_momentum, m = momentum)
        # Get two views from current batch
        (x0, x1) = batch[0]
        # Forward step for image view 1
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        # Forward step for image view 2
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        # Compute loss
        self.loss = 0.5 * (self.criterion(p0, p1) + self.criterion(p1, z0))
        return self.loss 
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr = config.LR)
    
    def on_train_epoch_end(self):
        self.log("training loss", self.loss)


if __name__ == "__main__":
    
    # ------------------------------ Argument Parser ----------------------------- #
    
    parser = argparse.ArgumentParser(description = "Train BYOL algorithm") 
    parser.add_argument("-dfold", type = str, help = "Path to data folder")
    parser.add_argument("-sveroot", type = str, help = "Path to where model weights + stats are saved")
    args = parser.parse_args()

    # ----------------------- DataLoader + BYOL Transforms ----------------------- #

    transforms = BYOLTransform(
        view_1_transform=BYOLView1Transform(input_size = config.INPUT_SIZE),
        view_2_transform=BYOLView2Transform(input_size = config.INPUT_SIZE)
    )
    trainset = LightlyDataset(input_dir = args.dfold, transform=transforms) # .__getitem__() returns -> view1,view2,fname
    trainloader = DataLoader(trainset, batch_size=config.BATCH_SIZE)

    # -------------------------- Instantiate BYOL model -------------------------- #

    byol = BYOL()

    # --------------------------------- Training --------------------------------- #

    svefold = f"byol-is{config.INPUT_SIZE}-bs{config.BATCH_SIZE}-ep{config.MAX_EPOCHS}-lr{config.LR}"
    logger = CSVLogger(save_dir = args.sveroot, name = svefold)
    trainer = pl.Trainer(
        default_root_dir= os.path.join(args.sveroot, svefold),
        devices = config.DEVICES,
        accelerator="gpu",
        max_epochs=config.MAX_EPOCHS,
        logger = logger
    )

    trainer.fit(byol, trainloader)