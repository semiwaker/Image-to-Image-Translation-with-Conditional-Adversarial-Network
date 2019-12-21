import config
from model import generator
from data import read_picture

import argparse

import tensorflow as tf

import matplotlib.pyplot as plt


def show_generated(G, dataset, input_file):
    x, ground = read_picture(dataset, input_file)

    y = G(x)

    x = tf.reshape(x, (x.shape[1], x.shape[2], x.shape[3]))
    y = tf.reshape(y, (y.shape[1], y.shape[2], y.shape[3]))
    ground = tf.reshape(ground,
                        (ground.shape[1], ground.shape[2], ground.shape[3]))

    # Show aligned pictures
    plt.figure()
    display_list = [x, ground, y]
    title = ["Input image", "Ground truth", "Generated Image"]
    n = 3

    for i in range(0, n):
        plt.subplot(1, n, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis("off")
    plt.savefig(dataset+"/"+str(input_file)+".jpg",
                dpi=120, quality=100)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate input image")
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="which gpu to use"
    )
    args = parser.parse_args()

    config.configure_gpu(args.gpu)

    G = generator()
    G.load_weights(config.G_SAVE_PATH+"_"+args.dataset+".hdf5")
    G.eval()

    for i in range(1, config.DATASET_SIZE[args.dataset]["test"]+1):
        show_generated(G, args.dataset, i)
