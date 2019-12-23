import os
import random

import pickle as pkl
import numpy as np
import tensorflow as tf
import tensorlayer as tl

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

def read_picture(dataset, datatype, input_file):
    'Read a picture, transform into model input format'
    x, y = load_data(dataset, datatype, input_file)
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    x = (x-127.5)/127.5
    y = (y-127.5)/127.5
    return (tf.reshape(x, (1, x.shape[0], x.shape[1], x.shape[2])),
            tf.reshape(y, (1, y.shape[0], y.shape[1], y.shape[2])))


def load_data(dataset, datatype, idx):
    if datatype is 'train':
        pass
    elif datatype in ['val', 'test', 'val_test']:
        datatype = 'val_test'
    pkl_path = os.path.join(config.TRAIN_PATH, 'pkl',
                            dataset, datatype+'_'+str(idx)+'.pkl')
    x_img, y_img = load_pkl(pkl_path)
    x_img = tf.convert_to_tensor(x_img)
    y_img = tf.convert_to_tensor(y_img)
    return x_img, y_img


def create_matrix():
    M_rotate = tl.prepro.affine_rotation_matrix()
    M_flip = tl.prepro.affine_horizontal_flip_matrix()
    M_shift = tl.prepro.affine_shift_matrix(
        h=config.PICTURE_SIZE, w=config.PICTURE_SIZE)
    M_shear = tl.prepro.affine_shear_matrix()
    M_zoom = tl.prepro.affine_zoom_matrix()

    M_combined = M_shift.dot(M_zoom).dot(M_shear).dot(M_flip).dot(M_rotate)
    return M_combined


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, config.PICTURE_SIZE, config.PICTURE_SIZE, 3])

    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def make_dataset(dataset_name, dataset_type):
    def func(x, y):
        def transform(x, y):
            M = create_matrix()
            x = tl.prepro.affine_transform_cv2(x, M)
            y = tl.prepro.affine_transform_cv2(y, M)
            return x, y

        # x, y = tf.numpy_function( transform, [x,y], (tf.float32, tf.float32))
        x, y = tf.numpy_function(
            random_jitter, [x, y], (tf.float32, tf.float32))
        x = (x / 127.5)-1.0
        y = (y / 127.5)-1.0
        return x, y

    def generator():
        for i in range(1, config.DATASET_SIZE[dataset_name][dataset_type]+1):
            yield load_data(dataset_name, dataset_type, i)

    d = tf.data.Dataset.from_generator(
        generator, output_types=(tf.float32, tf.float32))
    d = d.map(func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    d = d.shuffle(4096)
    d = d.batch(config.BATCH_SIZE)
    d = d.prefetch(2)
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
