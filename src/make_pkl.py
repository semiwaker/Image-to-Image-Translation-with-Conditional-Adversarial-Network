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

            # img = np.concatenate((x_img, y_img), axis=1)
            if x_img.shape != (256, 256, 3):
                raise Exception(img.shape)

            if root.endswith('train'):
                train_data.append((x_img, y_img))
            elif root.endswith('val'):
                val_test_data.append((x_img, y_img))
            elif root.endswith('test'):
                val_test_data.append((x_img, y_img))
            else:
                raise Exception("Key error")

# x_img, y_img = train_data[0]
x_img, y_img = train_data[0] # C = 127.5
# x_img = img[:, :256, :]
# y_img = img[:, 256:, :]
im = Image.fromarray(np.uint8(x_img))
im.save(dataset_name+'_x.jpg')
im = Image.fromarray(np.uint8(y_img))
im.save(dataset_name+'_y.jpg')
print(len(train_data))
print(len(val_test_data))

tot = 0
for t in train_data:
    tot += 1
    dump_pkl(t, '/data1/luozizhang/datasets/pkl/' + dataset_name + '/' + 'train_' + str(tot) + '.pkl')
tot = 0
for t in val_test_data:
    tot += 1
    dump_pkl(t, '/data1/luozizhang/datasets/pkl/' + dataset_name + '/' + 'val_test_' + str(tot) + '.pkl')
