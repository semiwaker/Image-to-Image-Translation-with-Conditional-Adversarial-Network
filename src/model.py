import config
import data

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.activation import *
from tensorlayer.models import Model
from tensorlayer.array_ops import alphas_like
from tensorlayer.cost import binary_cross_entropy, absolute_difference_error

import numpy as np
import pandas as pd


def generator():
    NGF = config.NGF
    SIZE = config.PICTURE_SIZE
    CHANNELS = config.PICTURE_CHANNELS

    def relu_act(x):
        return leaky_relu(x, config.LEAKY_RELU_ALPHA)

    # x_input : image
    # image is (256 x 256 x channels)
    x_input = Input((None, SIZE, SIZE, CHANNELS))
    x = x_input

    # encoder:
    # C64-①-C128-②-C256-③-C512-④-C512-⑤-C512-⑥-C512-⑦-C512-⑧-

    # e1 : C64
    # e1 output is (128 x 128 x 64)
    e1 = Conv2d(n_filter=NGF, filter_size=(4, 4), strides=(2, 2),
                act=relu_act, name="g_e1_conv")(x_input)
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
    d1 = Concat(concat_dim=-1, name="g_d1")([d1, e7])
    # d2 : CD512
    # d2 output is (4 x 4 x (512 + 512(concat)) )
    d2 = DeConv2d(n_filter=NGF * 8, filter_size=(4, 4), strides=(2, 2))(d1)
    d2 = BatchNorm2d(act=tf.nn.relu)(d2)
    d2 = Dropout(0.5)(d2)
    d2 = Concat(concat_dim=-1, name="g_d2")([d2, e6])
    # d3 : CD512
    # d3 output is (8 x 8 x (512 + 512(concat)) )
    d3 = DeConv2d(n_filter=NGF * 8, filter_size=(4, 4), strides=(2, 2))(d2)
    d3 = BatchNorm2d(act=tf.nn.relu)(d3)
    d3 = Dropout(0.5)(d3)
    d3 = Concat(concat_dim=-1, name="g_d3")([d3, e5])
    # d4 : C512
    # d4 output is (16 x 16 x (512 + 512(concat)) )
    d4 = DeConv2d(n_filter=NGF * 8, filter_size=(4, 4), strides=(2, 2))(d3)
    d4 = BatchNorm2d(act=tf.nn.relu)(d4)
    d4 = Concat(concat_dim=-1, name="g_d4")([d4, e4])
    # d5 : C256
    # d5 output is (32 x 32 x (256 + 256(concat)) )
    d5 = DeConv2d(n_filter=NGF * 4, filter_size=(4, 4), strides=(2, 2))(d4)
    d5 = BatchNorm2d(act=tf.nn.relu)(d5)
    d5 = Concat(concat_dim=-1, name="g_d5")([d5, e3])
    # d6 : C128
    # d6 output is (64 x 64 x (128 + 128(concat)) )
    d6 = DeConv2d(n_filter=NGF * 2, filter_size=(4, 4), strides=(2, 2))(d5)
    d6 = BatchNorm2d(act=tf.nn.relu)(d6)
    d6 = Concat(concat_dim=-1, name="g_d6")([d6, e2])
    # d7 : C64
    # d7 output is (128 x 128 x (64 + 64(concat)) )
    d7 = DeConv2d(n_filter=NGF, filter_size=(4, 4), strides=(2, 2))(d6)
    d7 = BatchNorm2d(act=tf.nn.relu)(d7)
    d7 = Concat(concat_dim=-1, name="g_d7")([d7, e1])
    # d8 : x_output
    # d8 output is (256 x 256 x 3)
    d8 = DeConv2d(n_filter=config.OUTPUT_CHANNELs, filter_size=(4, 4),
                  strides=(2, 2), act=tf.nn.tanh, name="g_d8")(d7)

    return Model(inputs=x_input, outputs=d8, name="generator")


def discriminator():
    # input size is 256 x 256 x ~(3 + 3), output size is 841=29*29 sigmoid
    FILTERS = config.NDF
    SIZE = config.PICTURE_SIZE
    CHANNELS = config.PICTURE_CHANNELS
    OUTPUT_CHANNELS = config.OUTPUT_CHANNELs

    def lrelu_act(x):
        return leaky_relu(x, config.LEAKY_RELU_ALPHA)

    # 70 x 70 PatchGAN :
    # C64-C128-C256-C512-~C1-Linear

    # Image : The image that is needed to be discriminate
    Image = Input((None, SIZE, SIZE, CHANNELS), name="Image")
    # Tag : Tag image
    Tag = Input((None, SIZE, SIZE, OUTPUT_CHANNELS), name="Tag")
    # Merged output is 256 x 256 x (input_channels + output_channels)
    # Receptive Field is 1 x 1
    Merged = Concat(-1)([Image, Tag])
    # h0 : C64 without BatchNorm
    # h0 output is 128 x 128 x 64
    # Receptive Field is 4 x 4
    h0 = Conv2d(n_filter=FILTERS, filter_size=(4, 4), strides=(2, 2),
                act=lrelu_act, name="d_h0_conv")(Merged)
    # h1 : C128
    # h1 output is 64 x 64 x 128
    # Receptive Field is 10 x 10
    h1 = Conv2d(n_filter=FILTERS * 2, filter_size=(4, 4), strides=(2, 2))(h0)
    h1 = BatchNorm2d(act=lrelu_act, name="d_h1_conv")(h1)
    # h2 : C256
    # h2 output is 32 x 32 x 256
    # Receptive Field is 22 x 22
    h2 = Conv2d(n_filter=FILTERS * 4, filter_size=(4, 4), strides=(2, 2))(h1)
    h2 = BatchNorm2d(act=lrelu_act, name="d_h2_conv")(h2)
    # h3 : C512
    # h3 output is 32 x 32 x 512
    # Receptive Field is 46 x 46
    h3 = Conv2d(n_filter=FILTERS * 8, filter_size=(4, 4), strides=(1, 1))(h2)
    h3 = BatchNorm2d(act=lrelu_act, name="d_h3_conv")(h3)
    # h4 : ~C1
    # h4 output is 29 x 29 x 1
    # Receptive Field is 70 x 70
    h4 = Conv2d(n_filter=1, filter_size=(4, 4),
                strides=(1, 1), act=tf.nn.sigmoid, padding="VALID")(h3)
    # lin : linear
    # lin output is 841
    lin = Reshape([h4.shape[0], -1])(h4)

    return Model(inputs=[Image, Tag], outputs=lin, name="Discriminator 70x70")


def entropy_loss(x, target):
    return binary_cross_entropy(x, alphas_like(x, target))


def L1_loss(y, z):
    return absolute_difference_error(y, z, is_mean=True)
