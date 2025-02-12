# ==============================================================================
# The following code has been ammended from the SatMAE repository
# ==============================================================================
import torch
import lightning.pytorch as pl
import os
import sys
import yaml
import math
from termcolor import colored
import time

import timm.optim.optim_factory as optim_factory

sys.path.append(os.path.join(os.getcwd(),"src","ssl_models","foundation_models","SatMAE"))

import models_mae_group_channels
from models_mae_group_channels import MaskedAutoencoderGroupChannelViT
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# -------------------------- Load SatMAE hyperparams ------------------------- #
with open("src/ssl_models/ssl_config.yml") as f:
    config = yaml.safe_load(f) 

satmae_params = config["satmae_params"]

class SatMaeGroupViTBB(pl.LightningModule):
    """The following class is only an implementation of grouped channels MAE"""
    def __init__(self,model_params:dict, pretrained_model:MaskedAutoencoderGroupChannelViT):
        super(SatMaeGroupViTBB,self).__init__()
        # This loads "mae_vit_base_patch16" model
        self.model = pretrained_model
        # Apply lr scheduler
        self.apply_lr_scheduler = False if model_params["lr"] else True
        self.model_params = model_params
        

    def training_step(self, batch, batch_idx):
        start_time = time.time()
        # We dont need to define a forward step as the model already has one
        #? data loader only returns X
        X = batch
        loss, X_pred, mask  = self.model(X, mask_ratio = satmae_params["mask_ratio"])     
        
        # if infinite stop training
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training ...")
            raise ValueError(f"loss {loss.item()}, stop training")

        self.log("loss", loss.item())
        return loss
        
    def on_train_epoch_end(self):
        if self.apply_lr_scheduler:
            self.log("current lr", self.scheduler.get_lr()[0])
        else:
            self.log("current lr", self.model_params["lr"])

    def configure_optimizers(self):
        if self.apply_lr_scheduler:
            # satmae code incorporate the following line to comute effectibe batch size which we ignore,
            #! We have changed this to the following, have a look at line 204-207 in "main_pretrian.py" on how its been implemented
            optimizer = torch.optim.AdamW(
                params= self.parameters(), 
                lr = satmae_params["base_lr"] * self.model_params["eff_batch_size"] / 256, 
                weight_decay= satmae_params["weight_decay"]
            )
            self.scheduler = LinearWarmupCosineAnnealingLR(
                optimizer = optimizer, 
                warmup_epochs= satmae_params["warmup_epochs"], # Linearly rampup lr as then decay using cosine as indicated in paper
                max_epochs=self.model_params["epochs"], 
                warmup_start_lr= satmae_params["base_lr"], # we linearly ramp up from 0 to base_lr which is indicated in the optimizer
                eta_min= satmae_params["eta_min"] #* We keep eta_min at 0 as Dino Paper hasnt indicated a value
            )
            return [optimizer],[{"scheduler" : self.scheduler, "interval" : "epoch"}] 
        else:
            lr = self.model_params["lr"]


if __name__ == "__main__":
    from ssl_utils import Sen2aMultiDataset
    from torch.utils.data import DataLoader
    from glob import glob
    from torchvision.transforms.v2 import Compose, ToImage, Resize
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.callbacks import ModelCheckpoint


    model_params = {
        "input_size" : 224,
        "patch_size" : 8,
        "in_chans" : 10,
        "spatial_mask" : False,
        "norm_pixel_loss" : False,
        "model" : "mae_vit_base_patch16",
        "lr" : None,
        "eff_batch_size" : 512,
        "epochs" : 5,
        "precision" : 32
    }
    checkpoint_path = "model_weights/pretrain_weights/satmae-fmowsen2a-vit-b-e199.pth"
    # Instantiat Satmae model
    mae_vit_b_p16 = models_mae_group_channels.__dict__[model_params["model"]](
        img_size = model_params["input_size"],
        patch_size = model_params["patch_size"],
        in_chans = model_params["in_chans"],
        channel_groups = satmae_params["grouped_bands"],
        spatial_mask = model_params["spatial_mask"],
        norm_pix_loss  = model_params["norm_pixel_loss"]
    )

    #todo : We need a way to figure out how to load model weights here

    sat_mae_vit = SatMaeGroupViTBB(model_params,mae_vit_b_p16)
    
    # DataLoader
    root = "data/interim/gee_sat/sen2a_c13_512x_pch"
    img_paths = glob(os.path.join(root, "*")) 
    sen_mul_ds = Sen2aMultiDataset(img_paths, transforms = Compose([ToImage(), Resize(224)]))
    trainloader = DataLoader(sen_mul_ds, batch_size = 32, shuffle = False)

    save_name = "test_mae"
    # Pytorch lightning trainer
    logger  = CSVLogger(save_dir = "tmp", name = save_name)
    checkpoint_callback = ModelCheckpoint(
        #dirpath=os.path.join(args.save_weights_fold, save_name), 
        filename="epoch:{epoch}",
        save_on_train_epoch_end=True,
        save_weights_only = True,
        save_top_k = -1,
        every_n_epochs = 1
    )

    trainer = pl.Trainer(
        default_root_dir = os.path.join("tmp", save_name),
        devices = -1,
        num_nodes= 1,
        accelerator = "gpu",
        #strategy = "ddp" if model_params["backbone"] == "resnet" else DDPStrategy(find_unused_parameters = True),
        max_epochs = model_params["epochs"],
        precision = model_params["precision"],
        logger = logger,
        callbacks = [checkpoint_callback],
    )

    trainer.fit(sat_mae_vit, trainloader)


