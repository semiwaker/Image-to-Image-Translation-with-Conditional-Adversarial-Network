import config
import os
import pickle as pkl


def dump_pkl(data, pkl_path):
    with open(pkl_path, 'w') as f:
        pkl.dump(data, f)
        f.close()


def load_pkl(pkl_path):
    try:
        with open(pkl_path, 'r') as f:
            data = pkl.load(f)
            f.close()
            return data
    except Exception:
        raise Exception("Load pickle Error")


class dataset:
    def __init__(self, used_list):
        self.data = dict()
        self.data['train'] = dict()
        self.data['val'] = dict()
        if type(used_list) != list:
            raise Exception("Type Error")
        for string in used_list:
            pkl_name = os.path.join(os.path.abspath('.'), "dataset", used_list + ".pkl")
            print(pkl_name)
            data = load_pkl(pkl_name)
            self.data['train'][string] = data['train']
            self.data['val'][string] = data['val']

