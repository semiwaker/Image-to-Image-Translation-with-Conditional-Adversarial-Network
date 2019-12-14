import config
import os
import pickle as pkl
import numpy as np
import random
import tensorflow as tf


def dump_pkl(data, pkl_path):
    with open(pkl_path, 'wb+') as f:
        pkl.dump(data, f)
        f.close()


def load_pkl(pkl_path):
    try:
        with open(pkl_path, 'rb+') as f:
            data = pkl.load(f)
            f.close()
            return data
    except Exception:
        raise Exception("Load pickle Error")


def normalize_and_split(img):
    img = (img / 127.5) - 1
    x_img = img[:, :256, :]
    y_img = img[:, 256:, :]
    return x_img, y_img


'''
string: the name of dataset
key: train or test
'''
def Dataset(string, key):
    pkl_name = os.path.join(config.TRAIN_PATH,  string + '_' + key + ".pkl")
    print("Loading ", pkl_name, "...")
    data = load_pkl(pkl_name)

    # Now, data is a list of images, there sizes are (256, 512, 3)

    data = np.array(data)
    if key is 'train':
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.map(normalize_and_split)
        dataset = dataset.shuffle(config.SHUFFLE_BUFFERSIZE)
        dataset = dataset.batch(config.BATCH_SIZE)
    elif key is 'test':
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.map(normalize_and_split)
        dataset = dataset.batch(config.BATCH_SIZE)
    else:
        raise Exception("Key must be 'train' or 'test' ")

    return dataset


def read_picture(input_file):
    'Read a picture, transform into model input format'
    # TODO: imple this function
    pass


if __name__ == '__main__':
    from PIL import Image
    dataset = Dataset('facade', 'train')
    for i in range(config.EPOCHS):
        x_list, y_list = dataset.generate()
        im = Image.fromarray(np.uint8(x_list[0]))
        im.show()
        from IPython import embed
        embed()
