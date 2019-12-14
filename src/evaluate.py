import config
from model import generator
from data import read_picture

import argparse

import matplotlib.pyplot as plt


def show_generated(dataset, input_file, output_file, ground_truth_file):
    x = read_picture(dataset, input_file)

    G = generator()
    G.load_weights(config.G_SAVE_PATH)

    y = G(x)

    # Show aligned pictures
    plt.figure()
    if len(ground_truth_file):
        ground = read_picture(ground_truth_file)
        display_list = [x, ground, y]
        title = ["Input image", "Ground truth", "Generated Image"]
        n = 3
    else:
        display_list = [x, y]
        title = ["Input image", "Generated Image"]
        n = 2
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

    plt.imshow(y)
    plt.axis("off")
    plt.savefig(output_file, quality=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate input image")
    parser.add_argument(
        "dataset",
        nargs=1,
        type=str,
        help="Dataset"
    )
    parser.add_argument(
        "input",
        nargs=1,
        type=int,
        help="Input file number"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help='Output file, default="input_file_name".out."input_file_suffix"',
        default=""
    )
    parser.add_argument(
        "-g", "--ground",
        type=str,
        help='Ground truth file"',
        default=""
    )
    args = parser.parse_args()

    show_generated(args.dataset, args.input, args.output, args.ground_truth)
