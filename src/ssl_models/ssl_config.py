# Training Hyperparameters
BATCH_SIZE = 128
INPUT_SIZE = 256
MAX_EPOCHS = 3

# Normalizing values for Satelites
sat_img_mean = [0.2132, 0.2890, 0.3737] ; sat_img_std = [0.2092, 0.1928, 0.1706]
# Normalizing values for Drones
drn_img_mean = [0.4768, 0.5559, 0.4325]; drn_img_std = [0.1557, 0.1466, 0.1245]

# Scaling Factors Impplemented : More of a Preprocessing Step
SENTINEL_SR_SF = 10000 # SR vqlues are scaled by 10,000 and hence need to divide by this value.
DRONE_SF = 255 # Divide by this nuber to scale to 0-1

# Device and Nodes 
#! MAKE sure to change devices when working with "HPC" or "datta"
DEVICES = 2 #No of GPU devices
NODES = 1 # No of compute devices

# torch float size, reduce to 16 if cannot fit batch size
PRECISION = 32

# multi-gpu / multi-node data distribution method, set to "auto" or "ddp"
STRATEGY = "auto"

# This just to find what "max" batch size that can be applied for given set of gpu's. Set to False apart from when you want to find
AUTO_SCALE_BATCH_SIZE = False

#! MAKE sure to change the first entry of the SAVE_NAME variable, i.e "simsiam"
SAVE_NAME = f"""\
byol\
-is{INPUT_SIZE}\
-effbs{BATCH_SIZE*DEVICES}\
-ep{MAX_EPOCHS}\
-bb{'Res'}\
-ds{'DrnSen2a'}\
-cl{'ClUcl'}\
-nm{'TTDrnSatNM'}\
"""
#SAVE_NAME = " ".join(line.strip() for line in SAVE_NAME.splitlines())

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

