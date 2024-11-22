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

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
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
import yaml

with open("src/ssl_models/ssl_config.yml") as f:
    config = yaml.safe_load(f) 

byol_params = config["byol_params"]

# ==============================================================================
# BYOL with Resnet Backbone
# ==============================================================================

class ByolBBResnet(pl.LightningModule):
    """Implementation of Byol model that can only handle Resnet Backbone"""
    def __init__(self, model_params:dict, backbone:torch.nn.Sequential):
        """ 
        Inputs
            model_params : SimSiam model parameters, i.e projection_hidden_dims
            backbone : Pretrained Resnet Backbone
        """
        super().__init__()

        # Saving hyperparameters
        hyper_dict = {}
        hyper_dict.update(byol_params)
        hyper_dict.update(model_params)
        self.save_hyperparameters(hyper_dict)

        # Model parameters : dict containing parameters values
        self.model_params = model_params

        # Backbone
        self.backbone = backbone
        
        # Online Network
        self.projection_head = BYOLProjectionHead(
            input_dim = 2048,
            hidden_dim = byol_params["proj_hidden_dim"],
            output_dim = byol_params["proj_output_dim"]           
        ) # 2048 > 4096 > 256
        self.prediction_head = BYOLPredictionHead(
            input_dim = byol_params["proj_output_dim"],
            hidden_dim = byol_params["pred_hidden_dim"],
            output_dim = byol_params["pred_output_dim"]
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
        self.batch_size = model_params["batch_size"]

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
        momentum = cosine_schedule(step = self.current_epoch, max_steps = self.model_params["epochs"], start_value = 0.996, end_value = 1) #? it hasnt clearnly been mention in paper that the max_steps is 10
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
        if byol_params["apply_lr_scheduler?"]:
            #* Original paper uses LARS optimizer but as our batch sizes are small we will use SGD instead
            if self.model_params["eff_batch_size"] < 1000:
                optimizer = torch.optim.SGD(
                    params= self.parameters(), 
                    lr = byol_params["base_lr"] * self.model_params["eff_batch_size"] / 256, 
                    weight_decay= byol_params["weight_decay"]
                )  
            else:
                print(colored("Using LARS optimizer", "green"))
                optimizer = LARS(
                    params = self.parameters(), 
                    lr = byol_params["base_lr"] * self.model_params["eff_batch_size"] / 256, 
                    weight_decay = byol_params["weight_decay"]
                )
            
            #* Note Linear scaling rule in byol uses base_lr of 0.2 and scaled to 0.2 * BS / 256. This is a JAX related scheduler
            #* With PyTorchSchedule we need to set warmup epochs
            self.scheduler = LinearWarmupCosineAnnealingLR(
                optimizer = optimizer, 
                warmup_epochs= byol_params["scheduler_warmup_epochs"], # Linearly rampup lr as then decay using cosine as indicated in paper
                max_epochs= self.model_params["epochs"], # Should be 1000 if we training for that long but we arnt
                warmup_start_lr= byol_params["base_lr"], #* we linearly ramp up from 0 to base_lr, start value not indicated
                eta_min=byol_params["scheduler_eta_min"] #* We keep eta_min at 0 as byol Paper hasnt indicated a value
            )

            return [optimizer],[{"scheduler" : self.scheduler, "interval" : "epoch"}] 
        else:
            if self.model_params["eff_batch_size"] < 1000:
                optimizer = torch.optim.SGD(self.parameters(), lr = self.model_params["lr"])
            else:
                print(colored("Using LARS optimizer", "green"))
                optimizer = LARS(params = self.parameters(), lr = self.model_params["lr"])
            return optimizer
    
    def on_train_epoch_end(self):
        self.log("training loss", self.loss)
        if byol_params["apply_lr_scheduler?"]:
            self.log("current lr", self.scheduler.get_lr()[0])

# ==============================================================================
# Byol with Vit Backbone
# ==============================================================================

class ByolBBSwinVit(pl.LightningModule):
    """Implementation of Byol model that can only handle Resnet Swin-Vit Backbone"""
    def __init__(self, model_params:dict, backbone_model:torch.nn.Sequential):
        """ 
        Inputs
            model_params : Byol model parameters, i.e projection_hidden_dims
            backbone_model : Pretrained Swin-vit model (Note this includes the entire
                             model including feature extractor & classification head)
                             Unlike the resnet backbone we will use model.forward_features()
                             function to get the 768 feature vector that will be passed to 
                             SimSiam.
        """
        super().__init__()

        # Saving hyperparameters
        hyper_dict = {}
        hyper_dict.update(byol_params)
        hyper_dict.update(model_params)
        self.save_hyperparameters(hyper_dict)

        # Model parameters : dict containing parameters values
        self.model_params = model_params

        # Backbone
        self.backbone_model = backbone_model

        # Online Network
        self.projection_head = BYOLProjectionHead(
            input_dim = 768,
            hidden_dim = byol_params["proj_hidden_dim"],
            output_dim = byol_params["proj_output_dim"]           
        ) # 2048 > 4096 > 256
        self.prediction_head = BYOLPredictionHead(
            input_dim = byol_params["proj_output_dim"],
            hidden_dim = byol_params["pred_hidden_dim"],
            output_dim = byol_params["pred_output_dim"]
        ) # 256 > 4096 > 256

        # Target Network
        self.backbone_model_momentum = deepcopy(self.backbone_model)
        self.projection_head_momentum = deepcopy(self.projection_head)
        
        # Freeze weights of target network as the weights are updated using exponential moving average 
        # of online network weights
        deactivate_requires_grad(self.backbone_model_momentum) 
        deactivate_requires_grad(self.projection_head_momentum)
        
        # Define Loss function 
        #* Incase you want to use Negative Cosine Similarity but BYOL uses a regression loss, which is defined below as part class method
        #self.criterion = NegativeCosineSimilarity()

        # batchsize not necessary but defined for "auto_scale_batch_size" func if needed 
        self.batch_size = model_params["batch_size"]

    def forward(self, x):
        # representation from resnet
        y = self.backbone_model.forward_features(x) # (*,768)
        # projection
        z = self.projection_head(y) #(*,768) -> (*,4096) -> (*,256)
        # prediction : This is represented as q(z) in paper
        p = self.prediction_head(z) #(*256) -> (*,4096) -> (*,256)
        return p
    
    def forward_momentum(self, x):
        # representation from resent
        y = self.backbone_model_momentum.forward_features(x) #(*,768)
        # projection
        z = self.projection_head_momentum(y) #(*,768) -> (*,4096) -> (*,256)
        # stop gradient
        z = z.detach()
        return z 
    
    def training_step(self, batch, batch_idx):
        # Updating momentum from online -> target
        #? In the byol paper it states that cosine_schedule for the momentum encoder runs from 0.996 to 1, but doesnt specify the total number of steps/epochs
        #? In LightlySSL docs they have set the max_steps to 10. We will set this to maximum number of epochs
        # Note dont get confused with this cosine_schedule and the one used by the optimizer, they are seperate.
        momentum = cosine_schedule(step = self.current_epoch, max_steps = self.model_params["epochs"], start_value = 0.996, end_value = 1) #? it hasnt clearnly been mention in paper that the max_steps is 10
        update_momentum(model = self.backbone_model, model_ema = self.backbone_model_momentum, m = momentum)
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
        if byol_params["apply_lr_scheduler?"]:
            #* Original paper uses LARS optimizer but as our batch sizes are small we will use SGD instead
            if self.model_params["eff_batch_size"] < 1000:
                optimizer = torch.optim.SGD(
                    params= self.parameters(), 
                    lr = byol_params["base_lr"] * self.model_params["eff_batch_size"] / 256, 
                    weight_decay= byol_params["weight_decay"]
                )  
            else:
                print(colored("Using LARS optimizer", "green"))
                optimizer = LARS(
                    params = self.parameters(), 
                    lr = byol_params["base_lr"] * self.model_params["eff_batch_size"] / 256, 
                    weight_decay = byol_params["weight_decay"]
                )
            
            #* Note Linear scaling rule in byol uses base_lr of 0.2 and scaled to 0.2 * BS / 256. This is a JAX related scheduler
            #* With PyToorchSchedule we need to set warmup epochs
            self.scheduler = LinearWarmupCosineAnnealingLR(
                optimizer = optimizer, 
                warmup_epochs= byol_params["scheduler_warmup_epochs"], # Linearly rampup lr as then decay using cosine as indicated in paper
                max_epochs= self.model_params["epochs"], # Should be 1000 if we training for that long but we arnt
                warmup_start_lr= byol_params["base_lr"], #* we linearly ramp up from 0 to base_lr, start value not indicated
                eta_min=byol_params["scheduler_eta_min"] #* We keep eta_min at 0 as byol Paper hasnt indicated a value
            )

            return [optimizer],[{"scheduler" : self.scheduler, "interval" : "epoch"}] 
        else:
            if self.model_params["eff_batch_size"] < 1000:
                optimizer = torch.optim.SGD(self.parameters(), lr = self.model_params["lr"])
            else:
                print(colored("Using LARS optimizer", "green"))
                optimizer = LARS(params = self.parameters(), lr = self.model_params["lr"])
            return optimizer
    
    def on_train_epoch_end(self):
        self.log("training loss", self.loss)
        if byol_params["apply_lr_scheduler?"]:
            self.log("current lr", self.scheduler.get_lr()[0])
