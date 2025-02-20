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
from functools import partial
import torch.nn as nn
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
        self.loss, X_pred, mask  = self.model(X, mask_ratio = satmae_params["mask_ratio"])     
        
        
        # if infinite stop training
        if not math.isfinite(self.loss.item()):
            print(f"Loss is {self.loss.item()}, stopping training ...")
            raise ValueError(f"loss {self.loss.item()}, stop training")

        return self.loss
        
    def on_train_epoch_end(self):
        self.log("loss", self.loss.item())
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
        "input_size" : 96,
        "model" : "mae_vit_base_patch16",
        "lr" : None,
        "eff_batch_size" : 512,
        "epochs" : 5,
        "precision" : 32
    }

    # Instantiat Satmae model
    grouped_bands = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9))

    # Alternative method to run model
    # mae_vit_b_p8 = models_mae_group_channels.__dict__[model_params["model"]](
    #     img_size = model_params["input_size"],
    #     patch_size = satmae_params["patch_size"],
    #     in_chans = len(grouped_bands), # usually 3, as we have 3 groups
    #     spatial_mask = satmae_params["spatial_mask"],
    #     channel_groups = grouped_bands,
    #     norm_pix_loss = satmae_params["norm_pix_loss"],
    # )

    try:
        mae_vit_b_p8 = MaskedAutoencoderGroupChannelViT(
            img_size = model_params["input_size"],
            patch_size = satmae_params["patch_size"],
            #in_chans = len(grouped_bands), # usually 3, as we have 3 groups
            in_chans = 10,
            spatial_mask = satmae_params["spatial_mask"],
            channel_groups = grouped_bands,
            channel_embed = satmae_params["channel_embed"],
            embed_dim = satmae_params["embed_dim"], # default : 1024
            depth = satmae_params["depth"],
            num_heads = satmae_params["num_heads"],
            decoder_channel_embed = satmae_params["decoder_channel_embed"],
            decoder_embed_dim = satmae_params["decoder_embed_dim"],
            decoder_depth = satmae_params["decoder_depth"],
            decoder_num_heads = satmae_params["decoder_num_heads"],
            mlp_ratio = satmae_params["mlp_ratio"],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            norm_pix_loss = satmae_params["norm_pix_loss"],
        )
    except RuntimeError:
        print(colored("Model weight mismatch !", "red"))

    # Load Model Weights
    checkpoint_path = "model_weights/temp_pretrain_weights/satmae-fmowsen2a-vit-b-e199.pth"
    model_checkpoint = torch.load(checkpoint_path)
    mae_vit_b_p8.load_state_dict(model_checkpoint["model"])

    sat_mae_vit = SatMaeGroupViTBB(model_params,mae_vit_b_p8)
    
    # # DataLoader
    #todo : We need to normalize properly, perhaps use normalized values from paper
    root = "data/interim/gee_sat/sen2a_c13_512x_pch"
    img_paths = glob(os.path.join(root, "*")) 
    sen_mul_ds = Sen2aMultiDataset(
        img_paths, 
        drop_bands = [1,9,10],
        clip = None,
        norm = 1000,
        transforms = Compose([ToImage(), Resize(96)])
    )
    trainloader = DataLoader(sen_mul_ds, batch_size = 32, shuffle = False)


    save_name = "test_satmae"
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

    # 
    # for X in trainloader:
    #     print(X.shape) #(32,10,96,96)
        
    #     x, msk, ids_restore = mae_vit_b_p8.forward_encoder(X, 0.75) # ([32, 109, 768]) ([32, 3, 144]) ([32, 432])
        
    #     xr  = mae_vit_b_p8.forward_decoder(x, ids_restore) # ([32, 10, 144, 64])
    #     # Loss is where the error comes from
    #     loss = mae_vit_b_p8.forward_loss(X, xr, msk)
    #     # print(X.reshape(shape = (X.shape[0], 10, 12,8,12,8)).shape)
    #     # print(mae_vit_b_p8.in_c)
    #     break
