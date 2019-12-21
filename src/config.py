# Settings

import tensorflow as tf

# Dataset
PICTURE_SIZE = 256  # width and height of the input picture
PICTURE_CHANNELS = 3  # color channel of the input picture
TRAIN_PATH = "/data1/luozizhang/datasets"  # path of the training dataset
#TRAIN_PATH = "."
TEST_PATH = "./data"  # path of the testing dataset
DEV_PATH = "./data"  # path of the develep dataset

DATASET_SIZE = {
    "facades": {"train": 400, "test": 206},
    "cityscapes": {"train": 2975, "test": 500},
    "night2day": {"train": 17823, "test": 2297},
    "edges2shoes": {"train": 49825, "test": 200},
    "edges2handbags": {"train": 138567, "test": 200}
}

# Train
BATCH_SIZE = 10
EPOCHS = 20
ADAM_LR = 0.0002
ADAM_BETA1 = 0.5
ADAM_BETA2 = 0.999
OUTPUT_FREQ = 50
LOSS_PLOT_PATH = "./model/loss.jpg"

# G model
NGF = 64  # number of generator filters
LEAKY_RELU_ALPHA = 0.2  # Alpha value for leaky_relu
OUTPUT_CHANNELs = 3
G_SAVE_PATH = "./model/G_model"
LAMBDA = 0.0

# D model
NDF = 64  # number of discriminator filters
D_SAVE_PATH = "./model/D_model"

# test
TEST_FIG_PATH = "./test.jpg"


# tf
def configure_gpu(gpu):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(gpus)
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)