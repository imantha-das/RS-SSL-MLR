# Training Hyperparameters
BATCH_SIZE = 128
INPUT_SIZE = 256
MAX_EPOCHS = 2
IMAGE_MEAN = [0.4768, 0.5559, 0.4324] #mean for channel3_256x256p dataset with batchsize set to 512
IMAGE_STD = [0.1557, 0.1466, 0.1245] #std for channel3_256x256p dataset with batchsize set to 512
Z_DIMS = 128 # Embedding size for simCLR hidden state (zi)
LR = 6e-2 # Learning rate for optimizer

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

