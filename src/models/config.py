# Training Hyperparameters
BATCH_SIZE = 128
INPUT_SIZE = 256
MAX_EPOCHS = 20
# Normalizing Values after Scaling
sen2a_scaled_img_mean = [0.0738, 0.0956, 0.1218] ; sen2a_scaled_img_std = [0.1069, 0.0973, 0.0943]
drn_scaled_img_mean = [0.4768, 0.5559, 0.4325] ; drn_scaled_img_std = [0.1557, 0.1466, 0.1245] 
# Normalizing values before scaling - normalizing values for Raw input
sen2a_raw_img_mean = [ 737.9805,  956.4913, 1218.3508] ; sen2a_raw_img_std = [1069.0999,  973.3830,  942.9846]
drn_raw_img_mean = [121.5762, 141.7526, 110.2758]; drn_raw_img_std = [39.7072, 37.3879, 31.7408]

SENTINEL_SR_SF = 10000 # SR vqlues are scaled by 10,000 and hence need to divide by this value.
DRONE_SF = 255 # Divide by this nuber to scale to 0-1
Z_DIMS = 128 # Embedding size for simCLR hidden state (zi)
LR = 0.0005 * BATCH_SIZE / 256 # For Dino use formulae : 0.0005 * batch_size / 256
LR_SCHEDULE = True # True if you want to use a lr_scheduler
DEVICES = 1 #No of GPU devices

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

