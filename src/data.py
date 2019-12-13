import config
import os
import pickle as pkl
import numpy as np
import random


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


class Dataset:
    def __init__(self, used_list):
        self.data = dict()
        self.data['train'] = dict()
        self.data['val'] = dict()
        self.random_list = used_list
        if type(used_list) != list:
            raise Exception("Type Error")
        for string in used_list:
            pkl_name = os.path.join(config.TRAIN_PATH,  string + ".pkl")
            print("Loading ", pkl_name, "...")
            data = load_pkl(pkl_name)
            self.data['train'][string] = data['train']
            self.data['val'][string] = data['val']
            if 'test' in data:
                self.data['val'][string].append(data['test'])

    def generate(self):
        x_list = []
        y_list = []
        key_list = np.random.choice(self.random_list, config.BATCH_SIZE)
        for key in key_list:
            a = (random.choice(self.data['train'][key]))
            x_img = a[:, 256:, :]
            y_img = a[:, :256, :]
            # from IPython import embed
            # embed()
            if key in ['edges2shoes', 'edges2handbags']:
                x_list.append(y_img)
                y_list.append(x_img)
            else:
                x_list.append(x_img)
                y_list.append(y_img)

        return x_list, y_list

    def __iter__(self):
        # TODO: Reset inner state
        self.cnt = 0
        return self

    def __next__(self):
        # TODO: check end of epoch and call generator
        if self.cnt > len(self.data['train']):
            raise StopIteration
        self.cnt+=1
        return self.generate()


if __name__ == '__main__':
    from PIL import Image
    dataset = Dataset(['facades', 'cityscapes'])
    for i in range(config.EPOCHS):
        x_list, y_list = dataset.generate()
        im = Image.fromarray(np.uint8(x_list[0]))
        im.show()
        from IPython import embed
        embed()
