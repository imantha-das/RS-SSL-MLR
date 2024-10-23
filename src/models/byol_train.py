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
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.transforms import Compose, Lambda
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
from typing import List

from utils import load_model_weights, SentinelAndDroneDataset, get_maxmin_stats
import config

sys.path.append("external/RSP/Scene Recognition")
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
        base_lr = self.model_params["lr"]
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

class RSDataset(Dataset):
    """Custom Dataset as we cannot use LightlyDataset for sentinel images for some reason"""
    def __init__(self, img_paths:List[str], transforms:Compose):
        super.__ini__(RSDataset, self)

        self.img_paths = img_paths
        self.transforms = transforms

    def __len__(self):
        """Dataset size"""
        return len(self.imag_paths)

    def __getitem__(self, idx):
        """Return two distinct views of the Image"""


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
        default="model_weights/rsp_weights/rsp-aid-resnet-50-e300-ckpt.pth"
    )
    parser.add_argument(
        "-save_weights_fold", 
        type = str, 
        help = "Path to where model weights + stats are saved", 
        default = "model_weights/ssl_weights"
    )
    args = parser.parse_args()

    # ---------------------------- BYOL Augmentations ---------------------------- #

    # DRONE Dataset
    transforms = BYOLTransform(
        view_1_transform=BYOLView1Transform(
            input_size = config.INPUT_SIZE, 
            #normalize={"mean" : config.drn_raw_img_mean, "std" : config.drn_raw_img_std}
            normalize = None
        ),
        view_2_transform=BYOLView2Transform(
            input_size = config.INPUT_SIZE,
            #normalize={"mean" : config.drn_raw_img_mean, "std" : config.drn_raw_img_std}
            normalize = None
        )
    )

    # --------------------------- Dataset + DataLoader --------------------------- #

    # DRONE Dataset
    drn_trainset = LightlyDataset(input_dir = args.data_fold_drn, transform=Compose([transforms])) # .__getitem__() returns -> view1,view2,fname
    drn_trainloader = DataLoader(drn_trainset, batch_size=config.BATCH_SIZE)

    # SENTINEL Dataset
    sen2a_trainset = LightlyDataset(input_dir= args.data_fold_sat, transform=Compose([transforms]))
    sen2a_trainloader = DataLoader(sen2a_trainset, batch_size=config.BATCH_SIZE)

    # COMBINE datasets
    drnsen2a_trainset = ConcatDataset([drn_trainset, sen2a_trainset])
    drnsen2a_trainloader = DataLoader(drnsen2a_trainset, batch_size=config.BATCH_SIZE)

    # -------------------------- Define model parameters ------------------------- #

    model_params = {
        "lr_schedule" : config.LR_SCHEDULE,
        "lr" : config.LR,
        "max_epochs" : config.MAX_EPOCHS
    }

    # -------------------------- Instantiate BYOL model -------------------------- #

    byol = BYOL(pretrain_weights_file=args.pretrain_weights_file, model_params = model_params)

    # --------------------------------- Training --------------------------------- #

    save_name = f"byol-is{config.INPUT_SIZE}-bs{config.BATCH_SIZE}-ep{config.MAX_EPOCHS}-lr{config.LR}-bb{"res"}-ds{"drnsen2aucln"}"
    logger = CSVLogger(save_dir = args.save_weights_fold, name = save_name)
    trainer = pl.Trainer(
        default_root_dir= os.path.join(args.save_weights_fold, save_name),
        devices = config.DEVICES,
        accelerator="gpu",
        max_epochs=config.MAX_EPOCHS,
        logger = logger
    )

    trainer.fit(byol, drnsen2a_trainloader)