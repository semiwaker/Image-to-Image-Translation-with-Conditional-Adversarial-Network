# Settings

# Dataset
PICTURE_SIZE = 256  # width and height of the input picture
PICTURE_CHANNELS = 3  # color channel of the input picture
TRAIN_PATH = "./data1/luozizhang/dataset"  # path of the training dataset
#TRAIN_PATH = "."
TEST_PATH = "./data"  # path of the testing dataset
DEV_PATH = "./data"  # path of the develep dataset

# Train
BATCH_SIZE = 10
EPOCHS = 10
ADAM_LR = 0.0002
ADAM_BETA1 = 0.5
ADAM_BETA2 = 0.999
OUTPUT_FREQ = 100
LOSS_PLOT_PATH = "./model/loss.jpg"

# G model
NGF = 64  # number of generator filters
LEAKY_RELU_ALPHA = 0.2  # Alpha value for leaky_relu
OUTPUT_CHANNELs = 3
G_SAVE_PATH="./model/G_model"
LAMBDA = 1.0

# D model
NDF = 64  # number of discriminator filters
D_SAVE_PATH="./model/D_model"

# test
TEST_FIG_PATH = "./test.jpg"
