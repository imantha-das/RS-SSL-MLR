# ==============================================================================
# Dino Implementation
# 
# Paper Implementatin | What we use
# - Backbone : Resnet or ViT | Resent50 or Swin-ViT from RSP Repo
# - Projection Head : Gelu(Linear(*, 2048)) -> Gelu(Linear(2048,2048)) -> Linear(2048,4096) | we set the output dim
# - loss : Cross Entropy with a temperaure
# - Exponential moving average is implemented to update the teacher network from the student networks weights.
#   This used a Cosine Decay, 0.996 -> 1
# ==============================================================================

import os
import sys
import copy 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
import pytorch_lightning as pl 
from pytorch_lightning.loggers import CSVLogger
from torchvision.transforms import ToTensor, Resize
from torchsummary import summary

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.transforms.dino_transform import DINOTransform
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from lightly.data import LightlyDataset

import argparse
from termcolor import colored
import  config
from utils import load_model_weights
import plotly.express as px

sys.path.append("RSP/Scene Recognition/models")
from resnet import resnet50
from swin_transformer import SwinTransformer

# --------------------------- Model Class for Dino --------------------------- #

class Dino(pl.LightningModule):
    def __init__(self, backbone_model:str = "swin-vit", proj_out:int = 4096, batch_size = 128, local_view_size = 96, global_view_size = 224, lr = None):
        super().__init__()
        if backbone_model == "swin-vit":
            #swin_vit = SwinTransformer(num_classes = 51) 
            swin_vit = load_model_weights(SwinTransformer, path_to_weights="models/rsp_weights/rsp-aid-swin-vit-e300-ckpt.pth")
            #* Instead of the .foward() method you need to use .forward_features() method
            backbone = swin_vit # returns (*, 768) tensor
            student_proj_head = DINOProjectionHead(input_dim = 768, hidden_dim= 2048, output_dim= proj_out, freeze_last_layer=1) #note freeze_last_layer refers to Number of epochs during which we keep the output layer fixed
            teacher_proj_head = DINOProjectionHead(input_dim = 768, hidden_dim= 2048, output_dim= proj_out)
        else:
            resnet = load_model_weights(resnet50, path_to_weights="models/rsp_weights/rsp-aid-resnet-50-e300-ckpt.pth")
            backbone = nn.Sequential(*list(resnet.children())[:-1]) # returns a (*. 2048,1,1) tensor
            #todo : This probably needs to be the teacher network, In the Dino paper they have found that freezing the teacher network over one epoch works well, freeze_last_layer = 1 refers to this !. But in Lightly this is Done on the student head
            student_proj_head = DINOProjectionHead(input_dim = 2048, hidden_dim = 2048, output_dim= proj_out, freeze_last_layer= 1)
            teacher_proj_head = DINOProjectionHead(input_dim = 2048, hidden_dim = 2048, output_dim= proj_out)

        pad_size  = int((global_view_size - local_view_size) / 2)
        self.zero_pad = nn.ZeroPad2d(pad_size) #pads from LHS, RHS, To & bottom the specified amount
        self.backbone_model:str = backbone_model #This is just a string saying "swin-vit" or "resnet" etc.
        self.batch_size:int = batch_size
        self.lr = lr

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
        # update teacher network weights using an exponential moving average of student netowrks weights
        momentum = cosine_schedule(step = self.current_epoch, max_steps=10, start_value=0.996, end_value=1)
        update_momentum(model = self.student_backbone, model_ema = self.teacher_backbone, m = momentum)
        update_momentum(model = self.student_head, model_ema = self.teacher_head, m = momentum)

        # Get only image tensors, note lightly dataset outputs other attributes like filepath
        # dataloader returns : data, labs, filepath
        views = batch[0] #* Note len(data[0]) = 8 by default. This is because there is multiple views in each batch
        views = [view.to(self.device) for view in views] # [(*,3,224,224), (*,3,224,224), (*,3,96,96) ... , (*,3,96,96)]
        # Only the first two images from this set are global views rest are local ...
        global_views = views[:2] # bt default will have a shape of (*,3,224,224) , (*,3,224,224)
        # Only global views are passed through the teacher ...
        teacher_out = [self.forward_teacher(view) for view in global_views]
        # While all views including global are passed through the student 
        #* we have added pad_view function to convert local views image sizes, 96x96 -> 224x224, We ONLY need to do this if its swin-vit
        if self.backbone_model == "swin-vit":
            student_out = [self.forward(self.pad_view(view)) for view in views] 
        else:
            student_out = [self.forward(view) for view in views] 
        # Compute crossenropy loss
        self.loss = self.criterion(teacher_out, student_out, epoch = self.current_epoch)
        return self.loss 
    
    def on_after_backward(self):
        #todo : This probably needs to be the teacher network
        self.student_head.cancel_last_layer_gradients(current_epoch = self.current_epoch)

    def configure_optimizers(self):
        if self.lr is not None:
            return torch.optim.AdamW(params= self.parameters(), lr = self.lr)
        else:
            return torch.optim.AdamW(params = self.parameters(), lr = 0.0005 * self.batch_size / 256)

    def on_train_epoch_end(self) -> None:
        self.log("training loss" , self.loss)

    def pad_view(self, view_b):
        """
        The student network takes both Global and Local crops, these differ in sizes
        i.e A Global crop by default can be of size 224x 224 while a local crop
        will be 96x96. The Swin-Vit model is unable to produce a representation
        with varying view as such we will pad them to a size used by the global crop
        i.e (96x96) -> (224x224)
        Inputs
            - view_b : A batch of image views of shape : (*,3,96,96) or (*,3,224,224)
        """
        if view_b.shape[2] != 224:
            return self.zero_pad(view_b)
        else:
            return view_b

# ---------------------- Function to print model weights --------------------- #

def print_model_weights(model):
    for name, param in model.named_parameters():
        print("-"*20)
        print(f"name : {name}")
        print(f"values : \n{param}")

if __name__ == "__main__":

    # ------------------------------ Argument Parser ----------------------------- #

    parser = argparse.ArgumentParser(description = "Train DINO algorithm")
    parser.add_argument("-dfold", type = str, help = "Path to data folder")
    parser.add_argument("-bbmodel", type = str, help = "Select backbone mode (swin-vit | resent)", default="swin-vit")
    parser.add_argument("-savefold", type = str, help = "Path to where models weights + stats will be saved.")
    args = parser.parse_args()

    # ----------------------- DataLoader + DINO Transforms ----------------------- #
    transforms = DINOTransform(normalize = {"mean" : config.IMAGE_MEAN, "std" : config.IMAGE_STD})
    trainset = LightlyDataset(
        input_dir = args.dfold,
        transform = transforms
    )
    trainloader = DataLoader(dataset = trainset, batch_size=config.BATCH_SIZE, shuffle= False)

    # -------------------------- Instantiate Dino model -------------------------- #
    dino = Dino(backbone_model= args.bbmodel, batch_size=config.BATCH_SIZE)

    # -------------------------------- Train model ------------------------------- #
    foldname = f"dino-is{config.INPUT_SIZE}-bs{config.BATCH_SIZE}-ep{config.MAX_EPOCHS}-lr{0.0005 * config.BATCH_SIZE / 256}-bb{'svit' if args.bbmodel == 'swin-vit' else 'res'}"
    logger = CSVLogger(save_dir = args.savefold, name = foldname)
    trainer = pl.Trainer(
        default_root_dir= os.path.join(args.savefold, foldname),
        devices = config.DEVICES,
        accelerator="gpu",
        max_epochs=config.MAX_EPOCHS,
        logger = logger
    )

    trainer.fit(dino, trainloader)

    # for batch in trainloader:
    #     views = batch[0]
    #     #[print(dino.pad_view(view).shape) for view in views] 
    #     for i,view in enumerate(views):
    #         if i == 2:
    #             pad = dino.pad_view(view)
    #             print(pad[0,:,:,:])
    #             img = px.imshow(pad[57,2,:,:])
    #             img.show()

    #     break

    # torch.manual_seed(0)
    # t1 = torch.randint(0,255, size = (1,3,224,224)).float()
    # z = dino.forward(t1)
    # print(dino.student_backbone.patch_embed(t1).shape)
    
