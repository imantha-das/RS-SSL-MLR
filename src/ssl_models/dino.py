# ==============================================================================
# Dino Implementation
# 
# Paper Implementatin | What we use
# - Backbone : Resnet or ViT | Resent50 or Swin-ViT from RSP Repo
# - Projection Head : Gelu(Linear(*, 2048)) -> Gelu(Linear(2048,2048)) -> Linear(2048,4096) | we set the output dim
# - There is NO prediction head for Dino
# - loss : Cross Entropy with a temperaure
# - Batch Size : 1024
# - Optimizer : AdamW
#   - Scheduler : lr linearly ramped up for first 10 epochs, lr = 0.0005 * batch_size/256, once linear ramp is complete
#                 decay according to cosine schedule
#   - weight decay : Also follows cosine schedule 0.04 to 0.4. temperature (tau-s) 0.1 during linear warmup 
#                   (tau-t) 0.04 to 0.07 during first 30 epochs
# - Data Augmentaion : Same as BYOL
# - Exponential moving average is implemented to update the teacher network from the student networks weights.
#   This used a Cosine Decay, 0.996 -> 1, num_epochs not mentioned in paper

# - VIT's dont use BatchNorm, so these are NOT incorporate for prediction head as well.
# - Centering C updated with an EMA

# ==============================================================================

import os
import sys
import copy 
import torch 
import torch.nn as nn 
import lightning.pytorch as pl
from torchvision.transforms.functional import resize
from torchsummary import summary

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from timm.models.vision_transformer import vit_base_patch16_224, VisionTransformer

import argparse
import yaml
from termcolor import colored

import plotly.express as px

sys.path.append("src/ssl_models/foundation_models/RSP/Scene Recognition/models")
from swin_transformer import SwinTransformer

# ------------------------- Dino Specific Parameters ------------------------- #
with open("src/ssl_models/ssl_config.yml") as f:
    config = yaml.safe_load(f) 

dino_params = config["dino_params"]

# ==============================================================================
# Dinov1 model with resnet backbone
# ==============================================================================

class DinoBBResnet(pl.LightningModule):
    def __init__(self,model_params:dict, backbone:torch.nn.Sequential):
        super().__init__()

        # Saving hyperparameters
        hyper_dict = {}
        hyper_dict.update(dino_params)
        hyper_dict.update(model_params)
        self.save_hyperparameters(hyper_dict)

        # model parameters such batch_size, lr to be used later
        self.model_params = model_params

        # backbones for student and teacher networks
        self.student_backbone = backbone
        self.teacher_backbone = copy.deepcopy(backbone)

        #* freeze the gradient of the student network over x epochs for stability : according to lightly, this is done in the original implementation of dino
        student_proj_head = DINOProjectionHead(
            input_dim = 2048, hidden_dim = dino_params["proj_hidden_dim"], output_dim= dino_params["proj_out_dim"], 
            freeze_last_layer=dino_params["freeze_proj_out_over_x_epochs"] # default = 1
        ) 
        teacher_proj_head = DINOProjectionHead(
            input_dim = 2048, hidden_dim = dino_params["proj_hidden_dim"], output_dim= dino_params["proj_out_dim"]
        )
        self.student_head = student_proj_head
        self.teacher_head = teacher_proj_head 

        # Stop gradient applied to teacher to propgate gradients through student only. Teacher networks gradients updated through EMA
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        # Dino Loss is cross entropy loss. But dino incorporates centering & sharpnening to prevent collapse.
        # * linear warmup schedule for teacher and centering incorporated in DinoLoss func 
        self.criterion = DINOLoss(
            output_dim = dino_params["proj_out_dim"], 
            warmup_teacher_temp= 0.04, # as mentioned in paper
            teacher_temp = 0.04, # as mentioned in paper
            warmup_teacher_temp_epochs=30, # as mentioned in paper
            student_temp = 0.1, # as mentioned in paper
            center_momentum = 0.9, # in experimentation study this was the highest value
            center_mode = "mean" #centered with mean computed over batch
        )

        # Apply learning rate or not
        self.apply_lr_scheduler = False if model_params["lr"] else True

    def forward(self, x):
        # Backbone
        y = self.student_backbone.forward(x).flatten(start_dim = 1) #(*,2048)
        # Projection head
        z = self.student_head(y) #(*,4096)
        return z 
    
    def forward_teacher(self, x):
        # Backbone
        y = self.teacher_backbone.forward(x).flatten(start_dim = 1) #(*,2048)
        # Projection head 
        z = self.teacher_head(y) #(*,4096)
        return z 
    
    def training_step(self, batch, batch_idx):
        # update teacher network weights using an exponential moving average of student netowrks weights
        #? In the dino paper it states that cosine_schedule for the momentum encoder runs from 0.996 to 1, but doesnt specify the total number of steps/epochs
        #? In LightlySSL docs they have set the max_steps to 10. We will set this to maximum number of epochs
        momentum = cosine_schedule(
            step = self.current_epoch, 
            max_steps=self.model_params["epochs"], 
            start_value=0.996, 
            end_value=1
        )
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
        student_out = [self.forward(view) for view in views] 
        # Compute crossenropy loss
        self.loss = self.criterion(teacher_out, student_out, epoch = self.current_epoch)
        return self.loss 
    
    def on_after_backward(self):
        #* We do a stop gradient operation on the last layer of the teacher network - This is part of Dino Architecture 
        self.teacher_head.cancel_last_layer_gradients(current_epoch = self.current_epoch)

    def configure_optimizers(self):
        if self.apply_lr_scheduler:
            #todo Dino incorporates weight decay schedular that follows cosine decay from 0.04 - 0.4
            #todo This hasnt been implemented as ready made functions are not available for weight decay
            optimizer = torch.optim.AdamW(
                params= self.parameters(), 
                lr = dino_params["base_lr"] * self.model_params["eff_batch_size"] / 256, 
                weight_decay=dino_params["weight_decay"]
            )
            self.scheduler = LinearWarmupCosineAnnealingLR(
                optimizer = optimizer, 
                warmup_epochs=dino_params["scheduler_warmup_epochs"], # Linearly rampup lr as then decay using cosine as indicated in paper
                max_epochs=self.model_params["epochs"], 
                warmup_start_lr=dino_params["base_lr"], # we linearly ramp up from 0 to base_lr which is indicated in the optimizer
                eta_min=dino_params["scheduler_eta_min"] #* We keep eta_min at 0 as Dino Paper hasnt indicated a value
            )
            return [optimizer],[{"scheduler" : self.scheduler, "interval" : "epoch"}] 
        else:
            optimizer = torch.optim.AdamW(
                params = self.parameters(), lr = self.model_params["lr"], weight_decay=dino_params["weight_decay"]
            )
            return optimizer

    def on_train_epoch_end(self) -> None:
        self.log("training loss" , self.loss)
        if self.apply_lr_scheduler:
            self.log("current lr", self.scheduler.get_lr()[0])


# ==============================================================================
# Dinov1Model with Swin-vit Backbone
# ==============================================================================

class DinoBBSwinViT(pl.LightningModule):
    def __init__(self,model_params:dict, backbone_model:SwinTransformer):
        super().__init__()

        # Saving hyperparameters
        hyper_dict = {}
        hyper_dict.update(dino_params)
        hyper_dict.update(model_params)
        self.save_hyperparameters(hyper_dict)

        # model parameters such batch_size, lr to be used later
        self.model_params = model_params

        # backbones for student and teacher networks
        self.student_backbone = backbone_model
        self.teacher_backbone = copy.deepcopy(backbone_model)

        #* freeze the gradient of the student network over x epochs for stability : according to lightly, this is done in the original implementation of dino
        student_proj_head = DINOProjectionHead(
            input_dim = 768, hidden_dim = dino_params["proj_hidden_dim"], output_dim= dino_params["proj_out_dim"], 
            freeze_last_layer=dino_params["freeze_proj_out_over_x_epochs"] # default = 1
        ) 
        teacher_proj_head = DINOProjectionHead(
            input_dim = 768, hidden_dim = dino_params["proj_hidden_dim"], output_dim= dino_params["proj_out_dim"]
        )
        self.student_head = student_proj_head
        self.teacher_head = teacher_proj_head 

        # Stop gradient applied to teacher to propgate gradients through student only. Teacher networks gradients updated through EMA
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        # Unlike Resnet Swin-vit cannot handle different image views, Dino takes in Local & Global crops of 
        # images, 96x 224x image sizes. These need to be either resized or padded. Github thread proposes to
        # to pad (mentioned by Author)
        pad_size  = int((dino_params["global_view_size"] - dino_params["local_view_size"]) / 2)
        self.zero_pad = nn.ZeroPad2d(pad_size) #pads from LHS, RHS, To & bottom the specified amount         
         
        # Dino Loss is cross entropy loss. But dino incorporates centering & sharpnening to prevent collapse.
        # * linear warmup schedule for teacher and centering incorporated in DinoLoss func 
        self.criterion = DINOLoss(
            output_dim = dino_params["proj_out_dim"], 
            warmup_teacher_temp= 0.04, # as mentioned in paper
            teacher_temp = 0.04, # as mentioned in paper
            warmup_teacher_temp_epochs=30, # as mentioned in paper
            student_temp = 0.1, # as mentioned in paper
            center_momentum = 0.9, # in experimentation study this was the highest value
            center_mode = "mean" #centered with mean computed over batch
        )

        # Apply learning rate or not
        self.apply_lr_scheduler = False if model_params["lr"] else True

    def forward(self, x):
        # Backbone
        y = self.student_backbone.forward_features(x) #(*,768)
        # Projection head
        z = self.student_head(y) #(*,4096)
        return z 
    
    def forward_teacher(self, x):
        # Backbone
        y = self.teacher_backbone.forward_features(x) #(*,768)
        # Projection head 
        z = self.teacher_head(y) #(*,4096)
        return z 
    
    def training_step(self, batch, batch_idx):
        # update teacher network weights using an exponential moving average of student netowrks weights
        #? In the dino paper it states that cosine_schedule for the momentum encoder runs from 0.996 to 1, but doesnt specify the total number of steps/epochs
        #? In LightlySSL docs they have set the max_steps to 10. We will set this to maximum number of epochs
        momentum = cosine_schedule(
            step = self.current_epoch, 
            max_steps=self.model_params["epochs"], 
            start_value=0.996, 
            end_value=1
        )
        update_momentum(model = self.student_backbone, model_ema = self.teacher_backbone, m = momentum)
        update_momentum(model = self.student_head, model_ema = self.teacher_head, m = momentum)

        # Get only image tensors, note lightly dataset outputs other attributes like filepath
        # dataloader returns : data, labs, filepath
        views = batch[0] #* Note len(data[0]) = 8 by default. This is because there is multiple views in each batch
        views = [view.to(self.device) for view in views] # [(*,3,224,224), (*,3,224,224), (*,3,96,96) ... , (*,3,96,96)]
        # Only the first two images from this set are global views rest are local ...
        global_views = views[:2] # bt default will have a shape of (*,3,224,224) , (*,3,224,224)
        # Only global views are passed through the teacher ...
        # Note Global views dont needed to be padded as they are 224x224
        teacher_out = [self.forward_teacher(view) for view in global_views]
        # While all views including global are passed through the student 
        # Here we pass all views from which the local ones need to be padded
        student_out = [self.forward(self.pad_view(view)) for view in views] 
        
        # Compute crossenropy loss
        self.loss = self.criterion(teacher_out, student_out, epoch = self.current_epoch)
        return self.loss 
    
    def on_after_backward(self):
        #* We do a stop gradient operation on the last layer of the teacher network - This is part of Dino Architecture 
        self.teacher_head.cancel_last_layer_gradients(current_epoch = self.current_epoch)

    def configure_optimizers(self):
        if self.apply_lr_scheduler:
            #todo Dino incorporates weight decay schedular that follows cosine decay from 0.04 - 0.4
            #todo This hasnt been implemented as ready made functions are not available for weight decay
            optimizer = torch.optim.AdamW(
                params= self.parameters(), 
                lr = dino_params["base_lr"] * self.model_params["eff_batch_size"] / 256, 
                weight_decay=dino_params["weight_decay"]
            )
            self.scheduler = LinearWarmupCosineAnnealingLR(
                optimizer = optimizer, 
                warmup_epochs=dino_params["scheduler_warmup_epochs"], # Linearly rampup lr as then decay using cosine as indicated in paper
                max_epochs=self.model_params["epochs"], 
                warmup_start_lr=dino_params["base_lr"], # we linearly ramp up from 0 to base_lr which is indicated in the optimizer
                eta_min=dino_params["scheduler_eta_min"] #* We keep eta_min at 0 as Dino Paper hasnt indicated a value
            )
            return [optimizer],[{"scheduler" : self.scheduler, "interval" : "epoch"}] 
        else:
            optimizer = torch.optim.AdamW(
                params = self.parameters(), lr = self.model_params["lr"], weight_decay=dino_params["weight_decay"]
            )
            return optimizer

    def on_train_epoch_end(self) -> None:
        self.log("training loss" , self.loss)
        if self.apply_lr_scheduler:
            self.log("current lr", self.scheduler.get_lr()[0])

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

# ==============================================================================
# DinoV1ViT with ViT Backbone
# ==============================================================================

class DinoBBViT(pl.LightningModule):
    def __init__(self, model_params:dict, backbone:VisionTransformer):
        super(DinoBBViT,self).__init__()

        hyper_dict = {}
        hyper_dict.update(dino_params)
        hyper_dict.update(model_params)
        self.save_hyperparameters(hyper_dict)
        
        # model parameters such as batch_size, lr to be used later
        self.model_params = model_params

        # backbone for student and teacher network
        self.student_backbone = backbone
        self.teacher_backbone = copy.deepcopy(backbone)

        # Projection heads for student and teacher networks
        self.student_head = DINOProjectionHead(
            input_dim = 768,
            hidden_dim = dino_params["proj_hidden_dim"],
            bottleneck_dim = dino_params["bottleneck_dim"],
            output_dim = dino_params["proj_out_dim"],
            freeze_last_layer = dino_params["freeze_proj_out_over_x_epochs"] # For numerical stability, done in original dino git repo (according to a comment by lightly staff over github)
        )
        self.teacher_head = DINOProjectionHead(
            input_dim = 768,
            hidden_dim = dino_params["proj_hidden_dim"],
            bottleneck_dim = dino_params["bottleneck_dim"],
            output_dim = dino_params["proj_out_dim"],
        )
        # stop gradients in teacher, Teachers gradients updated through EMA
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        # Unlike Resnet, ViT model cannot handle varying size of images. As Dino takes in global
        # and local view of different size. We need to pad them. Or alternatively resize them
        pad_size = int((dino_params["global_view_size"] - dino_params["local_view_size"]) / 2)
        self.zero_pad = nn.ZeroPad2d(pad_size)

        # Dino Loss is cross entropy loss. But dino incorporates centering & sharpnening to prevent collapse.
        # * linear warmup schedule for teacher and centering incorporated in DinoLoss func 
        self.criterion = DINOLoss(
            output_dim = dino_params["proj_out_dim"], 
            warmup_teacher_temp= 0.04, # as mentioned in paper
            teacher_temp = 0.04, # as mentioned in paper
            warmup_teacher_temp_epochs=30, # as mentioned in paper
            student_temp = 0.1, # as mentioned in paper
            center_momentum = 0.9, # in experimentation study this was the highest value
            center_mode = "mean" #centered with mean computed over batch
        )

        # Apply learning rate or not
        self.apply_lr_scheduler = False if model_params["lr"] else True

    def forward(self, x):
        """
        Student Network
        Both local and global views are passed to this network. Local views are already padded
        before forward step is carried out. Refer to training_step method
        """
        # Backbone
        y = self.student_backbone(x) #(*,768) | Note local views are padded before passed, look at training step
        # Projection head
        z = self.student_head(y) #(*,4096)
        return z

        
    def forward_teacher(self,x):
        """
        Teacher network, gradient deactivated, only updates using EMA
        Only Global views are passed to this model, hence padding not required
        """
        # Backbone 
        y = self.teacher_backbone(x) #(*,768)
        # Projection head
        z = self.teacher_head(y) #(*,4096)
        return z

    def training_step(self,batch, batch_idx):
        # Define momentum to update teacher weights
        #? In the dino paper it states that cosine_schedule for the momentum encoder runs from 0.996 to 1, but doesnt specify the total number of steps/epochs
        #? In LightlySSL docs they have set the max_steps to 10. We will set this to maximum number of epochs
        momentum = cosine_schedule(
            step = self.current_epoch,
            max_steps = self.model_params["epochs"],
            start_value = 0.996,
            end_value = 1
        )
        # update weights of teacher backbone using EMA
        update_momentum(model = self.student_backbone, model_ema = self.teacher_backbone, m = momentum)
        # update weights of teacher head using EMA
        update_momentum(model = self.student_head, model_ema = self.teacher_head, m = momentum)
        # We need to show dino, local global views

        # All augment views, Dino has 8 pf which 2 are global and the remaining 6 are local
        views, _, _ = batch # len = 8 ; List[torch.tensor] <- content : [(*,3,224,224),(*,3,224,224),(*,3,96,96) ... (*,3,96,96)]
        # put in to gpu
        views = [view.to(self.device) for view in views]
        # get just the global views
        global_views = views[:2]
        # pass the global views through teacher
        teacher_out = [self.forward_teacher(view) for view in global_views] #[(*,4096),(*,4096),...,(*,4096)]
        # pass all views (global + local) through student
        # self.pad_view function handles both global and local views
        student_out = [self.forward(self.pad_view(view)) for view in views] #[(*,4096),(*,4096),...,(*,4096)]

        # loss
        self.loss = self.criterion(teacher_out, student_out, epoch = self.current_epoch)
        return self.loss

    def configure_optimizers(self):
        if self.apply_lr_scheduler:
            optimizer = torch.optim.AdamW(
                params= self.parameters(), 
                lr = dino_params["base_lr"] * self.model_params["eff_batch_size"] / 256, 
                weight_decay=dino_params["weight_decay"]
            )
            self.scheduler = LinearWarmupCosineAnnealingLR(
                optimizer = optimizer, 
                warmup_epochs=dino_params["scheduler_warmup_epochs"], # Linearly rampup lr as then decay using cosine as indicated in paper
                max_epochs=self.model_params["epochs"], 
                warmup_start_lr=dino_params["base_lr"], # we linearly ramp up from 0 to base_lr which is indicated in the optimizer
                eta_min=dino_params["scheduler_eta_min"] #* We keep eta_min at 0 as Dino Paper hasnt indicated a value
            )
            return [optimizer],[{"scheduler" : self.scheduler, "interval" : "epoch"}] 
        else:
            optimizer = torch.optim.AdamW(
                params = self.parameters(), lr = self.model_params["lr"], weight_decay=dino_params["weight_decay"]
            )
            return optimizer

    def on_train_epoch_end(self) -> None:
        self.log("training loss" , self.loss)
        if self.apply_lr_scheduler:
            self.log("current lr", self.scheduler.get_lr()[0])
        else:
            self.log("current lr", model_params["lr"])

    def pad_view(self, view_b):
        """
        The student network takes both global and local view which differ in size.
        Global View : 224x224 , Local View : 96x96. Vit/Swin-Vit models are unable to produce
        representations with varying views unlike resnet. So we pad them.
        Input
            - view_b : A batch of image views of shape : (*,3,96,96) or (*,3,224,224)-
        """
        if view_b.shape[2] != dino_params["global_view_size"]:
            return self.zero_pad(view_b)
        else:
            return view_b

    def resize(self, view_b):
        """
        Alternative approach to pad_view where instead we just resize rather than pad
        """
        if view_b.shape[2] != dino_params["global_view_size"]:
            return resize(view_b, dino_params["global_view_size"])
        else:
            return view_b

    

if __name__ == "__main__":
    # Create fake tensor
    from glob import glob
    from lightly.transforms.dino_transform import DINOTransform
    from lightly.data import LightlyDataset
    from torch.utils.data import DataLoader
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.strategies import DDPStrategy

    batch_size = 32
    devices = 2
    save_weights_fold = "tmp"
    save_name = f"dino_vit_test_bs{batch_size}"

    img_root = "data/processed/gee_sat/sen2a_c3_256x_clp0.3uint8_full_pch"
    img_paths = glob(os.path.join(img_root, "*"))

    dino_transform = DINOTransform()
    dataset = LightlyDataset(input_dir = img_root, transform = dino_transform)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

    model_params = {
        "devices" : devices,
        "lr" : None,
        "epochs" : 1,
        "eff_batch_size" : batch_size * devices,
        "precision" : 32
    }

    backbone_model = vit_base_patch16_224(num_classes = 0)
    dino_bb_vit = DinoBBViT(model_params, backbone_model)


    logger = CSVLogger(save_dir = save_weights_fold, name = save_name)
    checkpoint_callback = ModelCheckpoint(
        #dirpath=os.path.join(args.save_weights_fold, save_name), 
        filename="epoch:{epoch}",
        save_on_train_epoch_end=True,
        save_weights_only = True,
        save_top_k = -1,
        every_n_epochs = 1
    )

    trainer = pl.Trainer(
        default_root_dir = "tmp/dino_vit",
        devices= model_params["devices"],
        accelerator = "gpu",
        strategy = "ddp",
        max_epochs = model_params["epochs"],
        precision = model_params["precision"]
    )

    trainer.fit(dino_bb_vit, dataloader)

    

    #y = dino_bb_vit.student_backbone(fake_imgs)
    #print(y.shape)