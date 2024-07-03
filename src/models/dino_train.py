# ==============================================================================
# Dino Implementation
# 
# Paper Implementatin | What we use
# Backbone : Resnet or ViT | Resent50 or Swin-ViT from RSP Repo
# Projection Head : Gelu(Linear(*, 2048)) -> Gelu(Linear(2048,2048)) -> Linear(2048,4096) | we set the output dim
# loss : Cross Entropy with a temperaure
# ==============================================================================

import sys
import copy 
import torch 
import torch.nn as nn 
import pytorch_lightning as pl 
from torchvision.transforms import ToTensor, Resize
from torchsummary import summary

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.utils.scheduler import cosine_schedule

from utils import load_model_weights

sys.path.append("RSP/Scene Recognition/models")
from resnet import resnet50
from swin_transformer import SwinTransformer

class Dino(pl.LightningModule):
    def __init__(self, backbone_model:str = "swin-vit", proj_out:int = 4096):
        super().__init__()
        if backbone_model == "swin-vit":
            #swin_vit = SwinTransformer(num_classes = 51) 
            swin_vit = load_model_weights(SwinTransformer, path_to_weights="models/rsp_weights/rsp-aid-swin-vit-e300-ckpt.pth")
            #! Instead of the .foward() method you need to use .forward_features() method
            backbone = swin_vit # returns (*, 768) tensor
            student_proj_head = DINOProjectionHead(input_dim = 768, hidden_dim= 2048, output_dim= proj_out, freeze_last_layer=1) #note freeze_last_layer refers to Number of epochs during which we keep the output layer fixed
            teacher_proj_head = DINOProjectionHead(input_dim = 768, hidden_dim= 2048, output_dim= proj_out)
        else:
            resnet = load_model_weights(resnet50, path_to_weights="models/rsp_weights/rsp-aid-resnet-50-e300-ckpt.pth")
            backbone = nn.Sequential(*list(resnet.children())[:-1]) # returns a (*. 2048,1,1) tensor
            student_proj_head = DINOProjectionHead(input_dim = 2048, hidden_dim = 2048, output_dim= proj_out, freeze_last_layer= 1)
            teacher_proj_head = DINOProjectionHead(input_dim = 2048, hidden_dim = 2048, output_dim= proj_out)

        self.backbone_model = backbone_model
        self.student_backbone = backbone
        self.student_head = student_proj_head
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = teacher_proj_head 
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim = 4096, warmup_teacher_temp= 5)

    def forward(self, x):
        if self.backbone_model == "swin-vit":
            y = self.student_backbone.forward_features(x) #(*, 768)
            z = self.student_head(y) #(*,4096)
        else:
            y = self.student_backbone.forward(x).flatten(start_dim = 1) #(*,2048)
            z = self.student_head(y) #(*,4096)

        return z 
    
    def forward_teacher(self, x):
        if self.backbone_model == "swin-vit":
            y = self.teacher_backbone.forward_features(x) #(*, 768)
            z = self.teacher_head(y) #(*,4096)
        else:
            y = self.teacher_backbone.forward(x).flatten(start_dim = 1) #(*,2048)
            z = self.teacher_head(y) #(*,4096)

        return z 
    
    def training_step(self, batch, batch_idx):
        pass


def print_model_weights(model):
    for name, param in model.named_parameters():
        print("-"*20)
        print(f"name : {name}")
        print(f"values : \n{param}")

if __name__ == "__main__":
    dino = Dino(backbone_model= "swin-vit")
    torch.manual_seed(0)
    t1 = torch.randint(0,255, size = (1,3,224,224)).float()
    z = dino.forward(t1)
    print(z.shape)
    
