import config
import data

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.activation import *
from tensorlayer.models import Model

import numpy as np
import pandas as pd


def G_model():
    NGF = config.NGF
    SIZE = config.PICTURE_SIZE
    CHANNELS = config.PICTURE_CHANNELS

    def relu_act(x):
        return leaky_relu(x, config.LEAKY_RELU_ALPHA)

    x_input = Input((-1, SIZE, SIZE, CHANNELS))
    x = x_input

    # encode

    NGFs = (NGF, NGF*2, NGF*4, NGF*8,
            NGF*8, NGF*8, NGF*8, NGF*8, NGF*8)

    encode_layer = []
    for ngf in NGFs:
        x = Conv2d(n_filter=ngf, filter_size=(4, 4), strides=(2, 2))(x)
        x = BatchNorm2d(act=relu_act)(x)
        encode_layer.append(x)

    # The last 1*1*NGF*8 layer does not need to be concatenated
    encode_layer.pop(-1)

    # decode

    Decode_spec = (
        (NGF*8, 0.5),
        (NGF*8, 0.5),
        (NGF*8, 0.5),
        (NGF*8, 0.0),
        (NGF*4, 0.0),
        (NGF*2, 0.0),
        (NGF, 0.0),
    )

    for i, (ngf, dropout) in enumerate(Decode_spec):
        x = DeConv2d(n_filter=ngf, filter_size=(4, 4), strides=(2, 2))(x)
        x = BatchNorm2d(act=tf.nn.relu)(x)
        x = Dropout(dropout)(x)
        x = tf.concat(x, encode_layer[-i])

    # Special treatment for the last layer

    x = DeConv2d(n_filter=config.OUTPUT_CHANNELs,
                 filter_size=(4, 4), strides=(2, 2))(x)
    x_output = hard_tanh(x)

    return Model(inputs=x_input, outputs=x_output, name="generator")


def D_model():
    pass
