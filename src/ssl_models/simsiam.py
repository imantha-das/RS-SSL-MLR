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

import math
import yaml

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import lightning.pytorch as pl

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead

import sys

from typing import List, Union

from torchvision.models import resnet50
#! Note for some reason torchvision.models swin_t does load the weights properly
sys.path.append("src/ssl_models/foundation_models/RSP/Scene Recognition/models")
from swin_transformer import SwinTransformer

# ------------------------- Dino Specific Parameters ------------------------- #
with open("src/ssl_models/ssl_config.yml") as f:
    config = yaml.safe_load(f) 

simsiam_params = config["simsiam_params"]
# ==============================================================================
# SimSiam Model with Resnet Backbone
# ==============================================================================          

class SimSiamBBResnet(pl.LightningModule):
    """Implementation of SimSiam model that can handle only handle Resnet Backbone"""
    def __init__(self, model_params:dict, backbone:torch.nn.Sequential):
        """
        Inputs
            model_params : SimSiam model parameters, i.e projection_hidden_dims
            backbone : Pretrained Resnet Backbone
        """
        super().__init__()

        # Saving hyperparameters
        hyper_dict = {}
        hyper_dict.update(simsiam_params)
        hyper_dict.update(model_params)
        self.save_hyperparameters(hyper_dict)

        #model parameters
        self.model_params = model_params

        # Resnet Backbone
        self.backbone = backbone

        self.projection_head = SimSiamProjectionHead(
            input_dim = 2048,
            hidden_dim = simsiam_params["proj_hidden_dim"],
            output_dim = simsiam_params["proj_output_dim"]
        ) # 2048 > 2048 > 2048
        self.prediction_head = SimSiamPredictionHead(
            input_dim = simsiam_params["proj_output_dim"], # input dims for prediction head is the same as the ouput dims
            hidden_dim = simsiam_params["pred_hidden_dim"],
            output_dim = simsiam_params["pred_output_dim"]
        ) #2048 > 512 > 2048
        self.out_dims = simsiam_params["pred_output_dim"] # we need access to out_dims for computing collapse levels
        
        # Loss
        self.criterion = NegativeCosineSimilarity()

        # To check if model is collapsing
        self.avg_loss = 0.0
        self.avg_output_std = 0.0
        self.loss = 0.0

        # Batch size not required by this class but needed by "auto_scale_batch_size" in pl.Trainer when locating the "max" batchsize
        self.batch_size = model_params["batch_size"]

    def forward(self, X):
        """Forward function for Resnet backbone"""
        f = self.backbone(X) #(b,3,256,256) -> ... -> (b,2048,1,1) 
        f = f.flatten(start_dim = 1) # (b,2048,1,1) -> (b,2048)
        z = self.projection_head(f) # (b,2048) -> (b,2048) -> (b,2048) -> (b, 2048) 
        p = self.prediction_head(z) # (b,2048) -> (b,512) -> (b,2048) 
        z = z.detach() #SimSiams stop the gradient to prevent collapse
        return z,p
    
    def training_step(self, batch, batch_idx):
        # lightlyDataset passes X,y,path = batch, augmented versions X0,X1 = X
        (X0,X1) = batch[0]
        z0,p0 = self.forward(X0) # (b,2048),(b,2048)
        z1,p1 = self.forward(X1) # (b,2048),(b,2048)
        # Compute loss
        self.loss = 0.5 * (self.criterion(z0,p1) + self.criterion(z1, p0))
        # Compute Collapse
        if simsiam_params["compute_collapse?"]:
            self.compute_std_per_dim(p0)

        return self.loss 
    
    def on_train_epoch_end(self):
        self.log("training_loss", self.loss)
        if simsiam_params["apply_lr_scheduler?"]:
            self.log("current_lr", self.learning_rate_scheduler.get_lr()[0])
        if simsiam_params["compute_collapse?"]:
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
        if simsiam_params["apply_lr_scheduler?"]:
            lr = simsiam_params["base_lr"] * self.model_params["eff_batch_size"] / 256
            optimizer = torch.optim.SGD(params = self.parameters(),lr = lr, weight_decay = 0.0001)
            self.learning_rate_scheduler = CosineAnnealingLR(optimizer, T_max = self.model_params["epochs"])
            return [optimizer], [{"scheduler":self.learning_rate_scheduler, "interval" : "epoch"}]
        else:
            return torch.optim.SGD(params = self.parameters(), lr = self.model_params["lr"]) 

# ==============================================================================
# SimSiam Model with Swin-Vit Backbone
# ==============================================================================

class SimSiamBBSwinViT(pl.LightningModule):
    """Implementation of SimSiam model that can handle only handle Resnet Backbone"""
    def __init__(self, model_params:dict, backbone_model:SwinTransformer):
        """
        Inputs
            model_params : SimSiam model parameters, i.e projection_hidden_dims
            backbone_model : Pretrained Swin-vit model (Note this includes the entire
                             model including feature extractor & classification head)
                             Unlike the resnet backbone we will use model.forward_features()
                             function to get the 768 feature vector that will be passed to 
                             SimSiam.
        """
        super().__init__()

        # Saving hyperparameters
        hyper_dict = {}
        hyper_dict.update(simsiam_params)
        hyper_dict.update(model_params)
        self.save_hyperparameters(hyper_dict)

        #model parameters
        self.model_params = model_params

        # Resnet Backbone
        self.backbone_model = backbone_model

        self.projection_head = SimSiamProjectionHead(
            input_dim = 768,
            hidden_dim = simsiam_params["proj_hidden_dim"],
            output_dim = simsiam_params["proj_output_dim"]
        ) # 2048 > 2048 > 2048
        self.prediction_head = SimSiamPredictionHead(
            input_dim = simsiam_params["proj_output_dim"], # input dims for prediction head is the same as the ouput dims
            hidden_dim = simsiam_params["pred_hidden_dim"],
            output_dim = simsiam_params["pred_output_dim"]
        ) #2048 > 512 > 2048
        self.out_dims = simsiam_params["pred_output_dim"] # we need access to out_dims for computing collapse levels
        
        # Loss
        self.criterion = NegativeCosineSimilarity()

        # To check if model is collapsing
        self.avg_loss = 0.0
        self.avg_output_std = 0.0
        self.loss = 0.0

        # Batch size not required by this class but needed by "auto_scale_batch_size" in pl.Trainer when locating the "max" batchsize
        self.batch_size = model_params["batch_size"]

    def forward(self, X):
        """Forward function for Resnet backbone"""
        f = self.backbone_model.forward_features(X) #(b,3,256,256) -> ... -> #(b, 768)
        z = self.projection_head(f) # (b,768) -> ... -> (b,2048)
        p = self.prediction_head(z) # (b,2048) -> (b,512) -> (b,2048) 
        z = z.detach() #SimSiams stop the gradient to prevent collapse
        return z,p
    
    def training_step(self, batch, batch_idx):
        # lightlyDataset passes X,y,path = batch, augmented versions X0,X1 = X
        (X0,X1) = batch[0]
        z0,p0 = self.forward(X0) #Resnet/Swin-Vit : (b,2048),(b,2048)
        z1,p1 = self.forward(X1) #Resnet/Swin-Vit : (b,2048),(b,2048)
        # Compute loss
        self.loss = 0.5 * (self.criterion(z0,p1) + self.criterion(z1, p0))
        # Compute Collapse
        if simsiam_params["compute_collapse?"]:
            self.compute_std_per_dim(p0)

        return self.loss 
    
    def on_train_epoch_end(self):
        self.log("training_loss", self.loss)
        if simsiam_params["apply_lr_scheduler?"]:
            self.log("current_lr", self.learning_rate_scheduler.get_lr()[0])
        if simsiam_params["compute_collapse?"]:
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
        if simsiam_params["apply_lr_scheduler?"]:
            lr = simsiam_params["base_lr"] * self.model_params["batch_size"] / 256
            optimizer = torch.optim.SGD(params = self.parameters(),lr = lr, weight_decay = 0.0001)
            self.learning_rate_scheduler = CosineAnnealingLR(optimizer, T_max = self.model_params["epochs"])
            return [optimizer], [{"scheduler":self.learning_rate_scheduler, "interval" : "epoch"}]
        else:
            return torch.optim.SGD(params = self.parameters(), lr = self.model_params["lr"]) 