import config
import pickle as pkl
import numpy as np
from PIL import Image
from IPython import embed
import os


def dump_pkl(data, pkl_path):
    with open(pkl_path, 'wb+') as f:
        #embed()
        pkl.dump(data, f)
        f.close()

dataset_name = "facades"
data = dict()
data['train'] = []
data['val'] = []
data['test'] = []
for root, dirs, files in os.walk("datasets/" + dataset_name):
    for filename in files:
        if(filename.endswith('.jpg')):
            imgpath = os.path.join(root, filename)
            im = Image.open(imgpath)
            #im.show()
            img = np.array(im)      # image类 转 numpy
            #img = img[:,:,0]        #第1通道
            #im=Image.fromarray(img)
            #embed()
            if root.endswith('train'):
                data['train'].append(img)
            elif root.endswith('val'):
                data['val'].append(img)
            elif root.endswith('test'):
                data['test'].append(img)
            else:
                raise Exception("Key error")

if len(data['test']) == 0:
    data.pop('test')
dump_pkl(data, dataset_name+'.pkl')