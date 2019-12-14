import config
import data
from model import *

import argparse
import datetime

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
        return (datetime.time() - self.start_time).strftime("%H:%M:%S")

    def reset(self):
        self.start_time = datetime.time()


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
        return f"{name} loss: G {self.G_loss/cnt} D {self.D_loss/cnt}"


def train(dataset_name, verbose):
    _verbose = verbose
    EPOCHS = config.EPOCHS
    LR = config.ADAM_LR
    BETA1 = config.ADAM_BETA1
    BETA2 = config.ADAM_BETA2

    # Preparing
    dataset = data.Dataset([dataset_name])

    G = generator()
    D = discriminator()

    G_optimizer = tf.optimizers.Adam(LR, BETA1, BETA2)
    D_optimizer = tf.optimizers.Adam(LR, BETA1, BETA2)

    # set up pyplot
    plt.ion()
    g_loss = []
    d_loss = []
    batch_cnt = []
    total_batch = 0
    plt.legend("best")

    # Start training
    VPrint("Start training")
    global_timer = Timer()
    global_loss = AccumulatedLoss("Global")

    G.train()
    D.train()

    for epoch in range(1, EPOCHS+1):
        epoch_timer = Timer()
        VPrint(f"Start epoch {epoch}")
        epoch_loss = AccumulatedLoss("Epoch")
        running_loss = AccumulatedLoss("Running")

        for (x, y), i in enumerate(dataset):
            total_batch += 1

            with tf.GradientTape(persistent=True) as tape:
                z = G(x)
                d_logits = D(inputs=(x, y))
                d2_logits = D(inputs=(z, y))

                d_loss = entropy_loss(d_logits, 1) + entropy_loss(d2_logits, 0)
                g_loss = entropy_loss(d2_logits, 1) + L1_loss(y, z)

            # train generator
            grad = tape.gradient(g_loss, G.trainable_weights)
            G_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            # train discriminator
            grad = tape.gradient(d_loss, D.trainable_weights)
            D_optimizer.apply_gradients(zip(grad, D.trainable_weights))

            # Caculate batch loss
            batch_D_loss = np.average(d_loss.to_numpy())
            batch_G_loss = np.average(g_loss.to_numpy())
            global_loss.accumulate(batch_G_loss, batch_D_loss)
            epoch_loss.accumulate(batch_G_loss, batch_D_loss)
            running_loss.accumulate(batch_G_loss, batch_D_loss)

            if i % config.OUTPUT_FREQ == 0:
                VPrint(running_loss, f"time used:{epoch_timer()}")
                running_loss.reset()

                # make graph
                g_loss.append(batch_G_loss)
                d_loss.append(batch_D_loss)
                batch_cnt.append(total_batch)
                plt.plot(batch_cnt, g_loss, label="generator loss")
                plt.plot(batch_cnt, d_loss, label="discriminator loss")
                plt.draw()

        VPrint(f"Epoch {epoch} time used:{epoch_timer()}")
        VPrint(epoch_loss)

        # make graph
        plt.ioff()
        plt.plot(batch_cnt, g_loss, label="generator loss")
        plt.plot(batch_cnt, d_loss, label="discriminator loss")
        plt.savefig(config.LOSS_PLOT_PATH, dpi=120, quality=100)
        plt.show()

    VPrint(f"End training. Time used {global_timer()}")
    VPrint(global_loss)

    # save model
    G.save_weights(config.G_SAVE_PATH)
    D.save_weights(config.D_SAVE_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pix2Pix")
    parser.add_argument(
        "dataset",
        metavar="D",
        type=str,
        nargs=1,
        help="the name of the dataset"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true"
    )
    args = parser.parse_args()
    train(args.dataset, args.verbose)
