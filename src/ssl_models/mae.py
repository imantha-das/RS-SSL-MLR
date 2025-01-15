# ==============================================================================
# MAE Imp0lementation : THE PAPER HAS TWO DIFFERENT SETTINGS FOR PRETRAINING & FINETUNING
# What the paper uses | Any changes that deviate from paper
# Backbone : VIT-L/16 | VIT-B/16
#   Decoder : depth - 8 blocks, Width - 512d
#   Encoder : w/o mask tokens
# Optimizer
#   linear lr scaling (cosine decay) : lr = base_lr x bs / 256, base_lr = 1.5e-3/1e-3, warmup = 40/5 epochs
#   AdamW : Beta1,2 = 0.9,0.95 / 0.9,0.999
# Batch Size : 4096/1024 | 1024/1024
# Loss func : MSE
# Masking : 75%
# Augmenntation : RandomResized Crop     
#   Crop Size : 224x224
# Fine tuning : 50 epochs vs Pretraining : 200 epochs
#   Note that these are for Vit-L
# ==============================================================================

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchsummary import summary
from timm.models.vision_transformer import vit_base_patch16_224, VisionTransformer

from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightly.transforms import MAETransform
from lightly.models.utils import repeat_token, set_at_index, get_at_index, random_token_mask, patchify

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

import yaml
import timm

# libraries that you can delete later
from lightly.transforms import MAETransform
from lightly.data import LightlyDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from glob import glob
import os

with open("src/ssl_models/ssl_config.yml") as f:
    config = yaml.safe_load(f)

mae_params = config["mae_params"]

class MaeBBViT(pl.LightningModule):
    def __init__(self, model_params:dict, backbone:VisionTransformer):
        """
        Inputs 
            - model_params: dictionary containing model parameters such as lr, batch_size
            - backbone : vit model with last layer removed 
        """
        super().__init__()

        # Saving hyperparameters
        hyper_dict = {}
        hyper_dict.update(mae_params)
        hyper_dict.update(model_params)
        self.save_hyperparameters(hyper_dict)

        self.model_params = model_params

        self.backbone = backbone # we will need this later

        self.mask_ratio = mae_params["mask_ratio"] # 0.75
        self.patch_size = backbone.patch_embed.patch_size[0] #(16,16) so index [0] returns 16
        self.masked_encoder = MaskedVisionTransformerTIMM(vit = backbone)
        self.sequence_length = self.masked_encoder.sequence_length #197
        self.decoder = MAEDecoderTIMM(
            num_patches = backbone.patch_embed.num_patches, #196
            patch_size = self.patch_size,#196
            embed_dim = backbone.embed_dim, #768
            decoder_embed_dim = mae_params["decoder_dim"], #512
            decoder_depth = mae_params["decoder_depth"], #8
            mlp_ratio = 4.0,
            proj_drop_rate = 0.0, # drop out rate in projection head
            attn_drop_rate = 0.0 # drop out rate in 
        )
        self.criterion = nn.MSELoss()

        self.apply_lr_scheduler = False if model_params["lr"] else True

    def forward_encoder(self, images, idx_keep = None):
        # shape returned from .encode = (bs, num_unmasked_patches, embed size)
        # What you pass here as images are actually just unamsked portion of the image only
        # So if 50/197 patches were left not masked you get the shape below (total 197 including cls) of which 50 is not masked
        return self.masked_encoder.encode(images = images, idx_keep = idx_keep) #(*,3,224,224) -> (*,50,768)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # Build decoder input
        batch_size = x_encoded.shape[0]

        x_decode = self.decoder.embed(x_encoded) #(*,50,512) where 50 is number NOT makes patches, 512 is num dimension decalred above

        x_masked = repeat_token(
            self.decoder.mask_token,
            (batch_size, self.sequence_length)
        ) #(*,197,512) where 197 nuber of masked patches & 512 is the nub dimensions declared above

        x_masked = set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked)) #(*,197,512)

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked) #(*,197,512) 

        # Predict pixel values for masked tokens
        x_pred = get_at_index(x_decoded, idx_mask) #(*,147,512)
        x_pred = self.decoder.predict(x_pred) #(*,147,768)

        return x_pred

    def training_step(self, batch, batch_idx):
        X,y,f = batch 
        images = X[0] # (*,3, 224,224) There is only a single view but its within a list
        batch_size = images.shape[0]
        # These are unmasked and masked patches. Rememnber that there is 197 patches including ...
        # CLS token (14*14 + 1), 14 patches was obtained by 224/16 where each patch is 16 pixels in height and width 
        idx_keep, idx_mask = random_token_mask(
            size = (batch_size, self.sequence_length),
            mask_ratio = self.mask_ratio,
            device = images.device
        ) #(*,50),(*,147)
        x_encoded = self.forward_encoder(
            images = images, 
            idx_keep = idx_keep
        ) #(*,50,768) where 50 is the number of NOT maked patches
        x_pred = self.forward_decoder(
            x_encoded = x_encoded, 
            idx_keep = idx_keep, 
            idx_mask = idx_mask
        ) #(*,147,768) where 147 are the number of masked patches

        # get image patches for masked tokens
        patches = patchify(images, self.patch_size) #(*,196,768)
        # must adjust idx mask for missing class token
        target = get_at_index(patches, idx_mask - 1) #(*, 147, 768)

        self.loss = self.criterion(x_pred, target) #(,)
        return self.loss

    def on_train_epoch_end(self) -> None:
        self.log("training loss" , self.loss)

        if self.apply_lr_scheduler:
            self.log("current lr", self.scheduler.get_lr()[0])
        else:
            self.log("current lr", self.model_params["lr"])

    def configure_optimizers(self):
        if self.apply_lr_scheduler:
            optimizer = torch.optim.AdamW(
                params= self.parameters(), 
                lr = mae_params["base_lr"] * self.model_params["eff_batch_size"] / 256, 
                weight_decay=mae_params["weight_decay"]
            )
            self.scheduler = LinearWarmupCosineAnnealingLR(
                optimizer = optimizer, 
                warmup_epochs=mae_params["warmup_epochs"], # Linearly rampup lr as then decay using cosine as indicated in paper
                max_epochs=self.model_params["epochs"], 
                warmup_start_lr=mae_params["base_lr"], # we linearly ramp up from 0 to base_lr which is indicated in the optimizer
                eta_min=mae_params["eta_min"] #* We keep eta_min at 0 as Dino Paper hasnt indicated a value
            )
            return [optimizer],[{"scheduler" : self.scheduler, "interval" : "epoch"}] 
        else:
            optimizer = torch.optim.AdamW(
                params = self.parameters(), lr = self.model_params["lr"], weight_decay=mae_params["weight_decay"]
            )
            return optimizer

def save_bb_pretrain_weights(model:MaeBBViT, weights_p:str, save_p:str = "model_weights/pretrain_weights/sshsph-aid-maeptr-vit-e299.ckpt"):
    """Function to load model weights after training mae"""
    model_state = torch.load(weights_p)
    model.load_state_dict(model_state["state_dict"])
    backbone = model.backbone
    torch.save(backbone.state_dict(),save_p)

#todo Need to write pretraining script but for now

if __name__ == "__main__":
    # from lightning.pytorch.loggers import CSVLogger
    # from lightning.pytorch.callbacks import ModelCheckpoint
    # from lightning.pytorch.strategies import DDPStrategy

    model_params = {"epochs" : 10, "eff_batch_size" : 256, "lr" : None}

    backbone = vit_base_patch16_224(num_classes = 0)
    #mae_bb_vit = MaeBBViT({}, backbone)

    # data_p = "data/processed/sshsph_drn/drn_c3_256x_pch"
    # transforms = MAETransform()
    # train_data = LightlyDataset(input_dir = data_p, transform = Compose([transforms]))
    # trainloader = DataLoader(train_data, batch_size = model_params["eff_batch_size"], num_workers = 16)

    mae = MaeBBViT(model_params, backbone)
    

    # # Checkpoint + Logging
    # save_name = "-".join([
    #     f"{'mae'}",
    #     f"is{256}",
    #     f"effbs{model_params['eff_batch_size']}",
    #     f"ep{model_params['epochs']}",
    #     f"bb{'vit'}",
    #     "dsDrn",
    #     "clCl",
    #     "nmNone",
    #     "pretrnYes"
    # ])
    # logger = CSVLogger(save_dir = "model_weights/ssl_weights", name = save_name)
    # checkpoint_callback = ModelCheckpoint(
    #     #dirpath=os.path.join(args.save_weights_fold, save_name), 
    #     filename="epoch:{epoch}",
    #     save_on_train_epoch_end=True,
    #     save_weights_only = True,
    #     save_top_k = -1
    # )
    # trainer = pl.Trainer(
    #     default_root_dir = os.path.join("model_weights/ssl_weights", save_name),
    #     devices = 2,
    #     num_nodes= 1,
    #     accelerator = "gpu",
    #     strategy = DDPStrategy(find_unused_parameters = True),
    #     max_epochs = model_params["epochs"],
    #     precision = 32,
    #     logger = logger,
    #     callbacks = [checkpoint_callback],
    #     #auto_scale_batch_size = config.AUTO_SCALE_BATCH_SIZE # to find "max" batch_size that can be procesedwith resources (gpu)
    # )

    # trainer.fit(mae, trainloader)

    # --------------------------- Loading Model Weights -------------------------- #
    model_weights_p = "/hpc/home/idg/workspace/RS-SSL-MLR/model_weights/pretrain_weights_fold/mae-effbs1024-ep300-bbVit-dsmilaid/version_1/checkpoints/epoch:epoch=299.ckpt"
    save_bb_pretrain_weights(mae, model_weights_p)