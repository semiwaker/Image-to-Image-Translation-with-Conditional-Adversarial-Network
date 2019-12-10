import config
import data

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.activation import *
from tensorlayer.models import Model
from tensorlayer.array_ops import alphas_like
from tensorlayer.cost import binary_cross_entropy

import numpy as np
import pandas as pd


def G_model():
    NGF = config.NGF
    SIZE = config.PICTURE_SIZE
    CHANNELS = config.PICTURE_CHANNELS

    def relu_act(x):
        return leaky_relu(x, config.LEAKY_RELU_ALPHA)

    # x_input : image
    # image is (256 x 256 x channels)
    x_input = Input((-1, SIZE, SIZE, CHANNELS))
    x = x_input

    # encoder:
    # C64-①-C128-②-C256-③-C512-④-C512-⑤-C512-⑥-C512-⑦-C512-⑧-

    # e1 : C64
    # e1 output is (128 x 128 x 64)
    e1 = Conv2d(n_filter=NGF, filter_size=(4, 4), strides=(2, 2))(x_input)
    e1 = BatchNorm2d(act=relu_act, name="g_e1_conv")(e1)
    # e2 : C128
    # e2 output is (64 x 64 x 128)
    e2 = Conv2d(n_filter=NGF * 2, filter_size=(4, 4), strides=(2, 2))(e1)
    e2 = BatchNorm2d(act=relu_act, name="g_e2_conv")(e2)
    # e3 : C256
    # e3 output is (32 x 32 x 256)
    e3 = Conv2d(n_filter=NGF * 4, filter_size=(4, 4), strides=(2, 2))(e2)
    e3 = BatchNorm2d(act=relu_act, name="g_e3_conv")(e3)
    # e4 : C512
    # e4 output is (16 x 16 x 512)
    e4 = Conv2d(n_filter=NGF * 8, filter_size=(4, 4), strides=(2, 2))(e3)
    e4 = BatchNorm2d(act=relu_act, name="g_e4_conv")(e4)
    # e5 : C512
    # e5 output is (8 x 8 x 512)
    e5 = Conv2d(n_filter=NGF * 8, filter_size=(4, 4), strides=(2, 2))(e4)
    e5 = BatchNorm2d(act=relu_act, name="g_e5_conv")(e5)
    # e6 : C512
    # e6 output is (4 x 4 x 512)
    e6 = Conv2d(n_filter=NGF * 8, filter_size=(4, 4), strides=(2, 2))(e5)
    e6 = BatchNorm2d(act=relu_act, name="g_e6_conv")(e6)
    # e7 : C512
    # e7 output is (2 x 2 x 512)
    e7 = Conv2d(n_filter=NGF * 8, filter_size=(4, 4), strides=(2, 2))(e6)
    e7 = BatchNorm2d(act=relu_act, name="g_e7_conv")(e7)
    # e8 : C512
    # e8 output is (1 x 1 x 512)
    e8 = Conv2d(n_filter=NGF * 8, filter_size=(4, 4), strides=(2, 2))(e7)
    e8 = BatchNorm2d(act=relu_act, name="g_e8_conv")(e8)
    
    # decoder:
    # -⑧-CD512-⑨-CD512-⑩-CD512-⑪-C512-⑫-C256-⑬-C128-⑭-C64-⑮-~C3

    # d1 : CD512
    # d1 output is (2 x 2 x (512 + 512(concat)) )
    d1 = DeConv2d(n_filter=NGF * 8, filter_size=(4, 4), strides=(2, 2))(e8)
    d1 = BatchNorm2d(act=tf.nn.relu)(d1)
    d1 = Dropout(0.5)(d1)
    d1 = Concat(concat_dim=4, name="g_d1")(d1, e7)
    # d2 : CD512
    # d2 output is (4 x 4 x (512 + 512(concat)) )
    d2 = DeConv2d(n_filter=NGF * 8, filter_size=(4, 4), strides=(2, 2))(d1)
    d2 = BatchNorm2d(act=tf.nn.relu)(d2)
    d2 = Dropout(0.5)(d2)
    d2 = Concat(concat_dim=4, name="g_d2")(d2, e6)
    # d3 : CD512
    # d3 output is (8 x 8 x (512 + 512(concat)) )
    d3 = DeConv2d(n_filter=NGF * 8, filter_size=(4, 4), strides=(2, 2))(d2)
    d3 = BatchNorm2d(act=tf.nn.relu)(d3)
    d3 = Dropout(0.5)(d3)
    d3 = Concat(concat_dim=4, name="g_d3")(d3, e5)
    # d4 : C512
    # d4 output is (16 x 16 x (512 + 512(concat)) )
    d4 = DeConv2d(n_filter=NGF * 8, filter_size=(4, 4), strides=(2, 2))(d3)
    d4 = BatchNorm2d(act=tf.nn.relu)(d4)
    d4 = Concat(concat_dim=4, name="g_d4")(d4, e4)
    # d5 : C256
    # d5 output is (32 x 32 x (256 + 256(concat)) )
    d5 = DeConv2d(n_filter=NGF * 4, filter_size=(4, 4), strides=(2, 2))(d4)
    d5 = BatchNorm2d(act=tf.nn.relu)(d5)
    d5 = Concat(concat_dim=4, name="g_d5")(d5, e3)
    # d6 : C128
    # d6 output is (64 x 64 x (128 + 128(concat)) )
    d6 = DeConv2d(n_filter=NGF * 2, filter_size=(4, 4), strides=(2, 2))(d5)
    d6 = BatchNorm2d(act=tf.nn.relu)(d6)
    d6 = Concat(concat_dim=4, name="g_d6")(d6, e2)
    # d7 : C64
    # d7 output is (128 x 128 x (64 + 64(concat)) )
    d7 = DeConv2d(n_filter=NGF, filter_size=(4, 4), strides=(2, 2))(d6)
    d7 = BatchNorm2d(act=tf.nn.relu)(d7)
    d7 = Concat(concat_dim=4, name="g_d7")(d7, e1)
    # d8 : x_output
    # d8 output is (256 x 256 x 3)
    d8 = DeConv2d(n_filter=config.OUTPUT_CHANNELs, filter_size=(4, 4),
                  strides=(2, 2), act=tf.nn.tanh, name="g_d8")(d7)

    return Model(inputs=x_input, outputs=d8, name="generator")


def D_model():
    SIZE = config.PICTURE_SIZE
    CHANNELS = config.PICTURE_CHANNELS

    def relu_act(x):
        return leaky_relu(x, config.LEAKY_RELU_ALPHA)

    x_input = Input((-1, SIZE, SIZE, CHANNELS), name="x_input")
    y_input = Input((-1, SIZE, SIZE, CHANNELS), name="y_input")
    net = Concat(-1)(x_input, y_input)

    sz = SIZE / 2
    ndf = config.NDF
    while sz >= 32:
        net = PadLayer([[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        net = Conv2d(n_filter=ndf, filter_size=(4, 4),
                     strides=(2, 2), padding="valid")(net)
        net = BatchNorm2d(act=relu_act)(net)
        ndf *= 2
        sz /= 2

    net = PadLayer([[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    net = Conv2d(n_filter=1, filter_size=(4, 4),
                 strides=(2, 2), padding="valid")(net)
    net = tf.nn.sigmoid(net)
    net = Reshape((-1, net.shape[1] * net.shape[2]), net)

    return Model(inputs=(x_input, y_input), outputs=net, name="discriminator")

def loss(x, target):
    return binary_cross_entropy(x, alphas_like(x, target))
