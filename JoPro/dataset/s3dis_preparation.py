import os
import collections
import pickle
from math import ceil

import numpy as np
from tqdm import tqdm
from utils.data_util import crop_pc, voxelize


def generate_scene_mask(labels, ratio):

    # class-wise even sampling, according to PSD(ICCV 2021) and CPCM(ICCV 2023)

    mask_ = np.zeros((labels.shape[0], 1), dtype=np.int8)

    counts = collections.Counter(np.squeeze(labels.astype(np.int64())))

    for key, value in dict(counts).items():
        idx_label = [idx for idx, label_ in enumerate(labels) if label_ == key]
        num = round(ratio/100.0 * value)
        if num == 0:
            num = 1
        idx_ratio = np.random.permutation(np.array(idx_label))[0:num]

        mask_[idx_ratio] = 1

    return mask_


voxel_size = 0.04
label_proportion = 0.01     # init ratio = label_proportion/5
data_root = './Dataset/S3DIS'
raw_root = './Dataset/S3DIS/raw'
data_list = sorted(os.listdir(raw_root))

data_list = [item[:-4] for item in data_list if 'Area_' in item]
train_data_list = [item for item in data_list if 'Area_5' not in item]
val_data_list = [item for item in data_list if 'Area_5' in item]

np.random.seed(0)
train_data = []
mask_data = []
for item in tqdm(train_data_list):
    voxel_size = 0.04
    per_data = []
    per_mask = []
    data_path = os.path.join(raw_root, item + '.npy')
    cdata = np.load(data_path).astype(np.float32)
    cdata[:, :3] -= np.min(cdata[:, :3], 0)

    coord, feat, label = cdata[:, 0:3], cdata[:, 3:6], cdata[:, 6:7]

    uniq_idx = voxelize(coord, voxel_size)

    coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    per_data.append(coord)
    per_data.append(feat.astype(np.int16))
    per_data.append(label.astype(np.int8))

    train_data.append(per_data)

    mask = generate_scene_mask(label, label_proportion / 5)
    mask_data.append(np.squeeze(mask))


val_data = []
for item in tqdm(val_data_list):
    per_data = []
    data_path = os.path.join(raw_root, item + '.npy')
    cdata = np.load(data_path).astype(np.float32)
    cdata[:, :3] -= np.min(cdata[:, :3], 0)

    coord, feat, label = cdata[:, 0:3], cdata[:, 3:6], cdata[:, 6:7]
    uniq_idx = voxelize(coord, voxel_size)
    coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    per_data.append(coord)
    per_data.append(feat.astype(np.int16))
    per_data.append(label.astype(np.int8))
    val_data.append(per_data)

filename = os.path.join(data_root, 's3dis_train.pkl')
with open(filename, 'wb') as f:
    pickle.dump(train_data, f)
    print(f"{filename} saved successfully")

filename = os.path.join(data_root, 's3dis_val.pkl')
with open(filename, 'wb') as f:
    pickle.dump(val_data, f)
    print(f"{filename} saved successfully")

filename = os.path.join(data_root, 's3dis_mask_' + str(label_proportion) + '.pkl')
with open(filename, 'wb') as f:
    pickle.dump(mask_data, f)
    print(f"{filename} saved successfully")
