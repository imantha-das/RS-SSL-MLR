# Note you may install the "better comments" extenstion from VSCode store to see colored comments, i.e #! refers to a red comment

# Training Hyperparameters
BATCH_SIZE = 128 #In an A40 GPU, its hard to run BS of 256 unless model parameters are reduced, futhur using DATALOADER_NUM_WORKERS adds to the cost.
INPUT_SIZE = 256
DATALOADER_NUM_WORKERS = 16
#! MAKE sure to change number of epochs
MAX_EPOCHS = 20

# Normalizing values for Satelites
sat_img_mean = [0.2132, 0.2890, 0.3737] ; sat_img_std = [0.2092, 0.1928, 0.1706]
# Normalizing values for Drones
drn_img_mean = [0.4768, 0.5559, 0.4325]; drn_img_std = [0.1557, 0.1466, 0.1245]

# Scaling Factors Implemented durung preprocessing steps - These vales are NOT used dueing SSL training
SENTINEL_SR_SF = 10000 # SR vqlues are scaled by 10,000 and hence need to divide by this value.
DRONE_SF = 255 # Divide by this nuber to scale to 0-1
CLIP_VAL = 0.3 # Clip values at 0.3 as most earth observation objects (i.e Vegetation) have reflectance value between 0-0.3

# Device and Nodes 
#! MAKE sure to change devices when working with "HPC" or "datta", To change effective batchsize
DEVICES = 8 #No of GPU devices
NODES = 1 # No of compute devices

# torch float size, reduce to 16 if cannot fit batch size
PRECISION = 32

# multi-gpu / multi-node data distribution method, set to "auto" or "ddp"
STRATEGY = "auto"

# This just to find what "max" batch size that can be applied for given set of gpu's. Set to False apart from when you want to find max number of batches that can be run
AUTO_SCALE_BATCH_SIZE = False

#! MAKE sure to change the first entry of the SAVE_NAME variable MANUALLY, i.e "simsiam", "byol"
SAVE_NAME = f"""\
byol\
-is{INPUT_SIZE}\
-effbs{BATCH_SIZE*DEVICES*NODES}\
-ep{MAX_EPOCHS}\
-bb{'Res'}\
-ds{'DrnSen2a'}\
-cl{'ClUcl'}\
-nm{'TTDrnSatNM'}\
"""
#Save name flag meanings - is : input size | effbs : effective batch size | ep : number of epochs | bb : backbone | ds : datasets used
#                          cl : "Cl" for clean, "Ucl" for unclean and refer to the ds used earlier for dataset names respectively, i.e "ClUcl" means first dataset (drn) is cleaned & Second dataset is unclean| 
#                          nm : normalization, TT refers ToTensor, Drn & Sat Nm refers normalization factors used above been applied to transformations

# ---------------------------- SimSiam Hyperparams --------------------------- #

simsiam_model_params = {
    "apply_lr_scheduler?" : True, 
    "compute_collapse?" : True,
    "lr" : 0.0001, #? ineffective when apply_lr_schedule? = True, uses base_lr to calculate lr decay using CosineAnnealingLR 
    "base_lr" : 0.05,
    "proj_input_dim" : 2048,
    "proj_hidden_dim" : 2048,
    "proj_output_dim" : 2048,
    "pred_input_dim" : 2048,
    "pred_hidden_dim" : 512,
    "pred_output_dim" : 2048
}

# ----------------------------- BYOL Hyperparams ----------------------------- #

byol_model_params = {
   "apply_lr_scheduler?" : True,  
   "lr" : 0.0001, #? ineffective when apply_lr_schedule? = True, uses base_lr to calculate lr decay using LinearWarmupCosineAnnelaingLR 
    "base_lr" : 0.02,
    "proj_input_dim" : 2048,
    "proj_hidden_dim" : 4096,
    "proj_output_dim" : 256,
    "pred_input_dim" : 256,
    "pred_hidden_dim" : 4096,
    "pred_output_dim" : 256
}

# ==============================================================================
# SimCLR Defaults
# ==============================================================================

# ==============================================================================
# SimSiam defaults
# - optimizer 
#       - SGD with momentum 0.9
#       - base lr = 0.05 with weight decay 0.0001
#       - lr = lr x BS/256
# - Batch
#       - 512
#       - Batch Normalization
# - Projection Head
#       - MLP with 3 Layers
#       - batch normalization applied to all layers including output
#       - output fc has no relu
#       - hidden fc 2048 dim
#  - Prediction head
#       - MLP with 2 layers
#       - Batch norm applied to hidden fc, output fc doesnt have BN
#       - 2048 input and output whicle 512 dims for hidden giving a bottleneck structure
# ==============================================================================

