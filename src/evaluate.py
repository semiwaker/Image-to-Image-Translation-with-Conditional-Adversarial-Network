import config
from model import generator
from data import read_picture

import argparse

import tensorflow as tf

import matplotlib.pyplot as plt


def show_generated(dataset, input_file, output_file):
    x, ground = read_picture(dataset, input_file)

    G = generator()
    G.load_weights(config.G_SAVE_PATH+"_"+dataset+".hdf5")

    G.eval()
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
    plt.savefig(config.TEST_FIG_PATH, dpi=120, quality=100)
    plt.show()

    # Save output file
    # FIXME: plt is not really suitable for this job
    if not len(output_file):
        output_file = dataset+"_out_"+str(input_file)+".jpg"

    plt.figure()
    plt.imshow(y)
    plt.axis("off")
    plt.savefig(output_file, quality=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate input image")
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset"
    )
    parser.add_argument(
        "input",
        type=int,
        help="Input file number"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help='Output file, default="input_file_name".out."jpg"',
        default=""
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="which gpu to use"
    )
    args = parser.parse_args()

    config.configure_gpu(args.gpu)

    show_generated(args.dataset, args.input, args.output)


def show_generated(dataset, input_file, output_file):
    x, ground = read_picture(dataset, input_file)

    G = generator()
    G.load_weights(config.G_SAVE_PATH+"_"+dataset+".hdf5")

    G.eval()
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
    plt.savefig(config.TEST_FIG_PATH, dpi=120, quality=100)
    plt.show()

    # Save output file
    # FIXME: plt is not really suitable for this job
    if not len(output_file):
        output_file = dataset+"_out_"+str(input_file)+".jpg"

    plt.figure()
    plt.imshow(y)
    plt.axis("off")
    plt.savefig(output_file, quality=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate input image")
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset"
    )
    parser.add_argument(
        "input",
        type=int,
        help="Input file number"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help='Output file, default="input_file_name".out."jpg"',
        default=""
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="which gpu to use"
    )
    args = parser.parse_args()

    config.configure_gpu(args.gpu)

    show_generated(args.dataset, args.input, args.output)
