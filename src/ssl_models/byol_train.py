# ==============================================================================
# Byol paper implementation details : Any changes done are reflected after "|"

# Optimizer
#   LARS + weight_decay = 1.5*10-6| SGD 
#   CosineDecay with Warmup of 10 epochs, lr = base_lr x BS/256 where base_lr = 0.2
#   Exponential moving average tau | hadled by lightly
# Batch Size
#   4096 | Max we can use is 128 * 8 = 1024 or 256 * 8 = 2048 if precision = 16
# Backbone
#   Resnet
# Image Augmentations
#   Same as SimCLR
# Loss Function
#   BYOL uses its own loss func | For ease of use we use "NegativeCosineSimilarity"
# ==============================================================================

import os 
import sys
import argparse
from copy import deepcopy

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.transforms import Compose
from torchvision.models import resnet50

import pytorch_lightning as pl 
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from flash.core.optimizers import LARS

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.transforms.byol_transform import BYOLTransform, BYOLView1Transform, BYOLView2Transform
from lightly.utils.scheduler import cosine_schedule
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.data import LightlyDataset

from torchsummary import summary
from typing import List
from termcolor import colored

from ssl_utils import load_model_weights
import ssl_config as config



class BYOL(pl.LightningModule):
    def __init__(self, model_params:dict, backbone:torch.nn.Sequential):
        super(BYOL, self).__init__()

        # Model parameters : dict containing parameters values
        self.model_params = model_params

        # Backbone
        self.backbone = backbone
        
        # Online Network
        self.projection_head = BYOLProjectionHead(
            input_dim = model_params["proj_input_dim"],
            hidden_dim = model_params["proj_hidden_dim"],
            output_dim = model_params["proj_output_dim"]           
        ) # 2048 > 4096 > 256
        self.prediction_head = BYOLPredictionHead(
            input_dim = model_params["pred_input_dim"],
            hidden_dim = model_params["pred_hidden_dim"],
            output_dim = model_params["pred_output_dim"]
        ) # 256 > 4096 > 256
        
        # Target Network
        self.backbone_momentum = deepcopy(self.backbone)
        self.projection_head_momentum = deepcopy(self.projection_head)
        
        # Freeze weights of target network as the weights are updated using exponential moving average 
        # of online network weights
        deactivate_requires_grad(self.backbone_momentum) 
        deactivate_requires_grad(self.projection_head_momentum)
        
        # Define Loss function 
        #* Incase you want to use Negative Cosine Similarity but BYOL uses a regression loss, which is defined below as part class method
        #self.criterion = NegativeCosineSimilarity()

        # batchsize not necessary but defined for "auto_scale_batch_size" func if needed 
        self.batch_size = config.BATCH_SIZE

    def forward(self, x):
        # representation from resnet
        y = self.backbone(x).flatten(start_dim = 1) # (*,2048)
        # projection
        z = self.projection_head(y) #(*,2048) -> (*,4096) -> (*,256)
        # prediction : This is represented as q(z) in paper
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
        #? In the byol paper it states that cosine_schedule for the momentum encoder runs from 0.996 to 1, but doesnt specify the total number of steps/epochs
        #? In LightlySSL docs they have set the max_steps to 10. We will set this to maximum number of epochs
        # Note dont get confused with this cosine_schedule and the one used by the optimizer, they are seperate.
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
        #? Note z0 & p1 used only if we use NegativeCosineSimilarity as shown in Lightly SSL
        # Compute loss
        self.loss = self.regression_loss(p0,z1)
        #* comment out below if you want to use NeagativeCosineSimilarity instead of BYOL's regression loss implemented above.
        #* you will also have to comment out line 75
        #self.loss = 0.5 * (self.criterion(p0, p1) + self.criterion(p1, z0))
        return self.loss 
    
    @staticmethod
    def regression_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, dim=1)  #L2-normalize
        y_norm = F.normalize(y, dim=1)  #L2-normalize
        loss = 2 - 2 * (x_norm * y_norm).sum(dim=-1)  #dot product
        return loss.mean()
    
    def configure_optimizers(self):
        base_lr = self.model_params["lr"]
        if self.model_params["apply_lr_scheduler?"]:
            #* Original paper uses LARS optimizer but as our batch sizes are small we will use SGD instead
            if (config.NODES * config.DEVICES * config.BATCH_SIZE) < 1000:
                optimizer = torch.optim.SGD(params= self.parameters(), lr = base_lr, weight_decay=1.5e-6)  
            else:
                print(colored("Using LARS optimizer", "green"))
                optimizer = LARS(params = self.parameters(), lr = base_lr, weight_decay = 1.5e-6)
            
            self.scheduler = LinearWarmupCosineAnnealingLR(
                optimizer = optimizer, 
                warmup_epochs=10, # Linearly rampup lr as then decay using cosine as indicated in paper
                max_epochs=config.MAX_EPOCHS, # Should be 1000 if we training for that long but we arnt
                warmup_start_lr=0, #* we linearly ramp up from 0 to base_lr, start value not indicated
                eta_min=0 #* We keep eta_min at 0 as byol Paper hasnt indicated a value
            )

            return [optimizer],[{"scheduler" : self.scheduler, "interval" : "epoch"}] 
        else:
            if (config.NODES * config.DEVICES * config.BATCH_SIZE) < 1000:
                optimizer = torch.optim.SGD(self.parameters(), lr = config.LR)
            else:
                print(colored("Using LARS optimizer", "green"))
                optimizer = LARS(params = self.parameters(), lr = config.LR)
            return optimizer
    
    def on_train_epoch_end(self):
        self.log("training loss", self.loss)
        if self.model_params["apply_lr_scheduler?"]:
            self.log("current lr", self.scheduler.get_lr()[0])

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

    # ----------------------------- Model Parameters ----------------------------- #

    model_params = config.byol_model_params # THis is a dict containing model params

    # ------------------------------- Get backbone ------------------------------- #
    
    resnet = load_model_weights(
        resnet50, 
        path_to_weights=args.pretrain_weights_file, 
        num_classes = 51
    )
    resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])

    # ---------------------------- BYOL Augmentations ---------------------------- #

    # Transforms for drones
    drn_transforms = BYOLTransform(
        view_1_transform=BYOLView1Transform(
            input_size = config.INPUT_SIZE, 
            normalize={"mean" : config.drn_img_mean, "std" : config.drn_img_std}
        ),
        view_2_transform=BYOLView2Transform(
            input_size = config.INPUT_SIZE,
            normalize={"mean" : config.drn_img_mean, "std" : config.drn_img_std}
        )
    )

    # Transforms for satellite
    sat_transforms = BYOLTransform(
        view_1_transform=BYOLView1Transform(
            input_size = config.INPUT_SIZE, 
            normalize={"mean" : config.sat_img_mean, "std" : config.sat_img_std}
        ),
        view_2_transform=BYOLView2Transform(
            input_size = config.INPUT_SIZE,
            normalize={"mean" : config.sat_img_mean, "std" : config.sat_img_std}
        )
    )

    

    # --------------------------- Dataset + DataLoader --------------------------- #

    # DRONE Dataset
    drn_trainset = LightlyDataset(input_dir = args.data_fold_drn, transform=Compose([drn_transforms])) # .__getitem__() returns -> view1,view2,fname
    drn_trainloader = DataLoader(drn_trainset, batch_size=config.BATCH_SIZE)

    # SENTINEL Dataset
    sen2a_trainset = LightlyDataset(input_dir= args.data_fold_sat, transform=Compose([sat_transforms]))
    sen2a_trainloader = DataLoader(sen2a_trainset, batch_size=config.BATCH_SIZE)

    # COMBINE datasets
    drnsen2a_trainset = ConcatDataset([drn_trainset, sen2a_trainset])
    drnsen2a_trainloader = DataLoader(drnsen2a_trainset, batch_size=config.BATCH_SIZE)


    # -------------------------- Instantiate BYOL model -------------------------- #

    byol = BYOL(model_params, resnet_backbone)

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

    trainer.fit(byol, drnsen2a_trainloader)