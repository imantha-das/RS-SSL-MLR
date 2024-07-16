# ==============================================================================
# Implementation of BYOL (Bootstrap Your Latent) algorithm 
# CHANGES from original paper implemntation
# - we are using NegativeCosineSimilarity Loss function from lightly.loss
# module. This is NOT the exact same loss function used in the paper
# - We are using a BatchSize of 128 but the paper uses a much larger batchsize,
#  however the Paper shows that BYOL works well with lower batchsizes.
# - We will be using SGD instead of LARS optimizer. However LARS is not necessary 
#  as our batch size is quite resonable (LARS used for very large batchsizes)

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

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from torchsummary import summary

from utils import load_model_weights
import config
sys.path.append("RSP/Scene Recognition")
from models.resnet import resnet50

class BYOL(pl.LightningModule):
    def __init__(self, pretrain_weights_file:str, model_params:dict):
        super(BYOL, self).__init__()
        resnet = load_model_weights(
            resnet50, 
            path_to_weights=pretrain_weights_file, 
            num_classes = 51
        )
        # Model parameters
        self.model_params = model_params
        # Online Network
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = BYOLProjectionHead(input_dim=2048,hidden_dim=4096, output_dim = 256) # 2048 > 4096 > 256
        self.prediction_head = BYOLPredictionHead(input_dim = 256, hidden_dim = 4096, output_dim = 256) # 256 > 4096 > 256
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
        z = self.projection_head(y) #(*,2048) -> (*,4096) -> (*,256)
        # prediction
        p = self.prediction_head(z) #(*256) -> (*,4096) -> (*,256)
        return p
    
    def forward_momentum(self, x):
        # representation from resent
        y = self.backbone_momentum(x).flatten(start_dim = 1) #(*,2048)
        # projection
        z = self.projection_head_momentum(y) #(*,2048) -> (*,4096) -> (*,256)
        # stop gradient
        z = z.detach()
        return z 
    
    def training_step(self, batch, batch_idx):
        # Updating momentum from online -> target
        #? In the dino paper it states that cosine_schedule for the momentum encoder runs from 0.996 to 1, but doesnt specify the total number of steps/epochs
        #? In LightlySSL docs they have set the max_steps to 10. We will set this to maximum number of epochs
        momentum = cosine_schedule(step = self.current_epoch, max_steps = config.MAX_EPOCHS, start_value = 0.996, end_value = 1) #? it hasnt clearnly been mention in paper that the max_steps is 10
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
        base_lr = model_params["lr"]
        if self.model_params["lr_schedule"]:
            #* Original paper uses LARS optimizer but as our batch sizes are small we will use SGD instead
            optimizer = torch.optim.SGD(params= self.parameters(), lr = base_lr)
            self.scheduler = LinearWarmupCosineAnnealingLR(
                optimizer = optimizer, 
                warmup_epochs=10, # Linearly rampup lr as then decay using cosine as indicated in paper
                max_epochs=self.model_params["max_epochs"], # Should be 1000 if we training for that long but we arnt
                warmup_start_lr=0, #* we linearly ramp up from 0 to base_lr, start value not indicated
                eta_min=0 #* We keep eta_min at 0 as byol Paper hasnt indicated a value
            )

            return [optimizer],[{"scheduler" : self.scheduler, "interval" : "epoch"}] 
        else:
            return torch.optim.SGD(self.parameters(), lr = config.LR)
    
    def on_train_epoch_end(self):
        self.log("training loss", self.loss)
        if self.model_params["lr_schedule"]:
            self.log("current lr", self.scheduler.get_lr()[0])


if __name__ == "__main__":
    
    # ------------------------------ Argument Parser ----------------------------- #
    
    parser = argparse.ArgumentParser(description = "Train BYOL algorithm") 
    parser.add_argument("-data_fold", type = str, help = "Path to data folder", default = "data/processed/channel3_256x256p")
    parser.add_argument("-pretrain_weights_file", type = str, help = "Path to pretrained weights file", default="models/rsp_weights/rsp-aid-resnet-50-e300-ckpt.pth")
    parser.add_argument("-save_weights_fold", type = str, help = "Path to where model weights + stats are saved", default = "models/ssl_weights")
    args = parser.parse_args()

    # ----------------------- DataLoader + BYOL Transforms ----------------------- #

    transforms = BYOLTransform(
        view_1_transform=BYOLView1Transform(
            input_size = config.INPUT_SIZE, 
            normalize={"mean" : config.IMAGE_MEAN, "std" : config.IMAGE_STD}
        ),
        view_2_transform=BYOLView2Transform(
            input_size = config.INPUT_SIZE,
            normalize={"mean" : config.IMAGE_MEAN, "std" : config.IMAGE_STD}
        )
    )
    trainset = LightlyDataset(input_dir = args.data_fold, transform=transforms) # .__getitem__() returns -> view1,view2,fname
    trainloader = DataLoader(trainset, batch_size=config.BATCH_SIZE)

    # -------------------------- Define model parameters ------------------------- #
    model_params = {
        "lr_schedule" : config.LR_SCHEDULE,
        "lr" : config.LR,
        "max_epochs" : config.MAX_EPOCHS
    }

    # -------------------------- Instantiate BYOL model -------------------------- #

    byol = BYOL(pretrain_weights_file=args.pretrain_weights_file, model_params = model_params)

    # --------------------------------- Training --------------------------------- #

    save_name = f"byol-is{config.INPUT_SIZE}-bs{config.BATCH_SIZE}-ep{config.MAX_EPOCHS}-lr{config.LR}-bb{"res"}"
    logger = CSVLogger(save_dir = args.save_weights_fold, name = save_name)
    trainer = pl.Trainer(
        default_root_dir= os.path.join(args.save_weights_fold, save_name),
        devices = config.DEVICES,
        accelerator="gpu",
        max_epochs=config.MAX_EPOCHS,
        logger = logger
    )

    trainer.fit(byol, trainloader)