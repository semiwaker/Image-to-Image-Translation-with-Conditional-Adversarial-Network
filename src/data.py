import os
import random

import pickle as pkl
import numpy as np
import tensorflow as tf

import config


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


# class Dataset:
#     def __init__(self, used_list):
#         self.data = dict()
#         self.data['train'] = dict()
#         self.data['val'] = dict()
#         self.random_list = used_list
#         if type(used_list) != list:
#             raise Exception("Type Error")
#         for string in used_list:
#             pkl_name = os.path.join(config.TRAIN_PATH,  string + ".pkl")
#             print("Loading ", pkl_name, "...")
#             data = load_pkl(pkl_name)
#             self.data['train'][string] = data['train']
#             self.data['val'][string] = data['val']
#             if 'test' in data:
#                 self.data['val'][string].append(data['test'])
#
#     def generate(self):
#         x_list = []
#         y_list = []
#         key_list = np.random.choice(self.random_list, config.BATCH_SIZE)
#         for key in key_list:
#             a = (random.choice(self.data['train'][key]))
#             x_img = a[:, 256:, :]
#             y_img = a[:, :256, :]
#             # from IPython import embed
#             # embed()
#             if key in ['edges2shoes', 'edges2handbags']:
#                 x_list.append(y_img)
#                 y_list.append(x_img)
#             else:
#                 x_list.append(x_img)
#                 y_list.append(y_img)
#
#         return x_list, y_list
#
#     def __iter__(self):
#         self.cnt = 0
#         return self
#
#     def __next__(self):
#         if self.cnt > len(self.data['train']):
#             raise StopIteration
#         self.cnt+=1
#         return self.generate()

def read_picture(input_file):
    'Read a picture, transform into model input format'
    # TODO: imple this function
    pass


def load_data(dataset, datatype, idx):
    pass


def make_dataset(dataset_name, dataset_type):
    def data_generator():
        for i in range(config.DATASET_SIZE[dataset_name][dataset_type]):
            yield load_data(dataset_name, dataset_type, i)
    d = tf.data.Dataset.from_generator(
        generator=data_generator,
        output_types=(tf.float32, tf.float32)
    )
    d.map(lambda x: (x-127.5)/127.5)
    d.shuffle(1000)
    d.batch(config.BATCH_SIZE)
    return d


if __name__ == '__main__':
    from PIL import Image
    dataset = Dataset(['facades', 'cityscapes'])
    for i in range(config.EPOCHS):
        x_list, y_list = dataset.generate()
        im = Image.fromarray(np.uint8(x_list[0]))
        im.show()
        from IPython import embed
        embed()
