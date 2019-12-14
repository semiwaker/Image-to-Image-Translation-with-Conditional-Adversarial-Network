import config
import pickle as pkl
import numpy as np
from PIL import Image
from IPython import embed
import os


# C = 127.5

def dump_pkl(data, pkl_path):
    with open(pkl_path, 'wb+') as f:
        #embed()
        pkl.dump(data, f)
        f.close()

dataset_name = "maps"
train_data = []
val_test_data = []
for root, dirs, files in os.walk("/data1/luozizhang/datasets/" + dataset_name):
    for filename in files:
        if(filename.endswith('.jpg')):
            imgpath = os.path.join(root, filename)
            im = Image.open(imgpath)
            img = np.array(im)
            # img = (img - C) / C # C = 127.5
            x_img = img[:, 256:, :]
            y_img = img[:, :256, :]
            # from IPython import embed
            # embed()
            if dataset_name in ['edges2shoes', 'edges2handbags']:
                tmp = x_img
                x_img = y_img
                y_img = tmp

            img = np.concatenate((x_img, y_img), axis=1)
            if img.shape != (256, 512, 3):
                raise Exception(img.shape)

            if root.endswith('train'):
                train_data.append(img)
            elif root.endswith('val'):
                val_test_data.append(img)
            elif root.endswith('test'):
                val_test_data.append(img)
            else:
                raise Exception("Key error")

# x_img, y_img = train_data[0]
img = train_data[0] # C = 127.5
x_img = img[:, :256, :]
y_img = img[:, 256:, :]
im = Image.fromarray(np.uint8(x_img))
im.save(dataset_name+'_x.jpg')
im = Image.fromarray(np.uint8(y_img))
im.save(dataset_name+'_y.jpg')
print(len(train_data))
print(len(val_test_data))

dump_pkl(train_data, '/data1/luozizhang/datasets/' + dataset_name + '_train.pkl')
dump_pkl(val_test_data, '/data1/luozizhang/datasets/' + dataset_name + '_val_test.pkl')