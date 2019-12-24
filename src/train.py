import config
import data
from model import *

import argparse
import time
import sys

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.models import Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# A global switch
_verbose = False


def VPrint(*args, **kwargs):
    'Print msg only when _verbose is on'
    if _verbose:
        print(*args, **kwargs)


class Timer:
    def __init__(self):
        self.reset()

    def __call__(self):
        delta = int(time.time() - self.start_time)
        h = delta//3600
        m = (delta % 3600) // 60
        s = delta % 60
        result = ""
        if h > 0:
            result += f"{h} hours "
        if h > 0 or m > 0:
            result += f"{m} minutes "
        result += f"{s} seconds"
        return result

    def reset(self):
        self.start_time = time.time()


class AccumulatedLoss:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.G_loss = 0.0
        self.D_loss = 0.0
        self.cnt = 0

    def accumulate(self, G_loss, D_loss):
        self.cnt += 1
        self.G_loss += G_loss
        self.D_loss += D_loss

    def __str__(self):
        return f"{self.name} loss: G {self.G_loss/self.cnt} D {self.D_loss/self.cnt}"


def calc_loss(x, y, G, D):
    z = G(x)
    d_logits = D(inputs=[y, x])
    d2_logits = D(inputs=[z, x])

    d_loss = entropy_loss(d_logits, 1.0) + entropy_loss(d2_logits, 0.0)
    g_loss = entropy_loss(d2_logits, 1.0) + config.LAMBDA * L1_loss(y, z)

    return g_loss, d_loss


def test(dataset, G, D):
    g_sum = 0.0
    d_sum = 0.0
    for i, (x, y) in enumerate(dataset):
        g_loss, d_loss = calc_loss(x, y, G, D)
        g_sum += g_loss
        d_sum += d_loss
    return np.average(g_sum), np.average(d_sum)


def train(dataset_name, verbose, make_graph):
    global _verbose
    _verbose = verbose
    EPOCHS = config.EPOCHS
    LR = config.ADAM_LR
    BETA1 = config.ADAM_BETA1
    BETA2 = config.ADAM_BETA2

    # Preparing
    train_dataset = data.make_dataset(dataset_name, "train")
    test_dataset = data.make_dataset(dataset_name, "test")

    G = generator()
    D = discriminator()

    G_optimizer = tf.optimizers.Adam(LR, BETA1, BETA2)
    D_optimizer = tf.optimizers.Adam(LR, BETA1, BETA2)

    # set up pyplot
    if make_graph:
        # plt.ion()
        g_loss_train = []
        g_loss_test = []
        d_loss_train = []
        d_loss_test = []
        batch_cnt = []
        total_batch = 0

    # Start training
    VPrint("Start training")
    global_timer = Timer()
    global_train_loss = AccumulatedLoss("Global train")

    G.train()
    D.train()

    for epoch in range(1, EPOCHS+1):
        epoch_timer = Timer()
        VPrint(f"Start epoch {epoch}")
        epoch_train_loss = AccumulatedLoss("Epoch train")
        running_loss = AccumulatedLoss("Running")

        for i, (x, y) in enumerate(train_dataset):
            total_batch += 1

            with tf.GradientTape(persistent=True) as tape:
                g_loss, d_loss = calc_loss(x, y, G, D)

            # train generator
            grad = tape.gradient(g_loss, G.trainable_weights)
            G_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            # save model
            G.save_weights(config.G_SAVE_PATH+'_'+dataset_name+".hdf5")

            # train discriminator
            grad = tape.gradient(d_loss, D.trainable_weights)
            D_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            # save model
            D.save_weights(config.D_SAVE_PATH+'_'+dataset_name+".hdf5")

            # Caculate batch loss
            batch_D_loss = np.average(d_loss)
            batch_G_loss = np.average(g_loss)
            global_train_loss.accumulate(batch_G_loss, batch_D_loss)
            epoch_train_loss.accumulate(batch_G_loss, batch_D_loss)
            running_loss.accumulate(batch_G_loss, batch_D_loss)

            if i % config.OUTPUT_FREQ == 0:
                VPrint(running_loss, f"time used:{epoch_timer()}")
                running_loss.reset()

                test_g_loss, test_d_loss = test(test_dataset, G, D)
                VPrint(f"Test loss: G {test_g_loss} D {test_d_loss}")

                # make graph

                if make_graph:
                    g_loss_train.append(batch_G_loss)
                    d_loss_train.append(batch_D_loss)
                    g_loss_test.append(test_g_loss)
                    d_loss_test.append(test_d_loss)
                    batch_cnt.append(total_batch)

                    # plt.plot(batch_cnt, g_loss_train,
                    #  label="generator train loss")
                    # plt.plot(batch_cnt, d_loss_train,
                    #  label="discriminator train loss")
                    # plt.plot(batch_cnt, g_loss_test,
                    #  label="generator test loss")
                    # plt.plot(batch_cnt, d_loss_test,
                    #  label="discriminator test loss")
                    # plt.legend(best)
                    # plt.draw()

        VPrint(f"Epoch {epoch} time used:{epoch_timer()}")
        VPrint(epoch_train_loss)

        test_g_loss, test_d_loss = test(test_dataset, G, D)
        VPrint(f"Test loss: G {test_g_loss} D {test_d_loss}")

        # make graph
        if make_graph:
            # plt.ioff()
            plt.figure()
            plt.plot(batch_cnt, g_loss_train,
                     label="generator train loss")
            plt.plot(batch_cnt, d_loss_train,
                     label="discriminator train loss")
            plt.plot(batch_cnt, g_loss_test,
                     label="generator test loss")
            plt.plot(batch_cnt, d_loss_test,
                     label="discriminator test loss")
            plt.legend(loc="best")
            plt.savefig(config.LOSS_PLOT_PATH, dpi=120, quality=100)
            plt.show()

    VPrint(f"End training. Time used {global_timer()}")
    VPrint(global_train_loss)
    test_g_loss, test_d_loss = test(test_dataset, G, D)
    VPrint(f"Test loss: G {test_g_loss} D {test_d_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pix2Pix")
    parser.add_argument(
        "dataset",
        metavar="D",
        type=str,
        help="the name of the dataset"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true"
    )
    parser.add_argument(
        "-g", "--makegraph",
        action="store_true"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="which gpu to use"
    )
    args = parser.parse_args()

    config.configure_gpu(args.gpu)

    train(args.dataset, args.verbose, args.makegraph)
