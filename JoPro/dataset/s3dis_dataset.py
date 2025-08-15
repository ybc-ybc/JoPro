import os
import pickle
import logging
import numpy as np

import torch
from torch.utils.data import Dataset
from utils.data_util import crop_pc


class s3dis_dataset(Dataset):

    def __init__(self, cfg, dataset_cfg, transform, split):

        super().__init__()

        self.label_proportion = cfg.label_proportion

        self.name = dataset_cfg.name
        self.data_root = dataset_cfg.data_root
        self.voxel_max = dataset_cfg.voxel_max
        self.sever_data_root = dataset_cfg.sever_data_root

        self.transform = transform
        self.split = split

        # Local or server devices
        if os.path.exists(self.data_root):
            self.raw_root = self.data_root + self.name
        else:
            self.raw_root = self.sever_data_root + self.name

        if self.split == 'train' or self.split == 'al':

            with open(self.raw_root + '/s3dis_train.pkl', 'rb') as f:
                self.data = pickle.load(f)
            with open(self.raw_root + '/s3dis_mask_' + str(self.label_proportion) + '.pkl', 'rb') as f:
                self.point_mask = pickle.load(f)

        if self.split == 'val':
            with open(self.raw_root + '/s3dis_val.pkl', 'rb') as f:
                self.data = pickle.load(f)

        assert len(self.data) > 0
        logging.info(f"\nTotally {len(self.data)} samples in {self.split} set")

        if self.split == 'train':
            self.loops = dataset_cfg.train_step
        else:
            self.loops = 1

    def __getitem__(self, idx):
        data_idx = idx % len(self.data)

        if self.split == 'val':
            point_coord, point_feat, point_label = self.data[data_idx][0], self.data[data_idx][1], self.data[data_idx][2]
            point_feat = point_feat.astype(np.float32)
            point_label = point_label.squeeze(-1).astype(np.int64)

            transform_data = self.transform({'pos': point_coord, 'x': point_feat, 'y': point_label})
            point_coord = transform_data['pos']
            point_feat = transform_data['x']
            point_label = transform_data['y']

            point_feat = torch.cat((point_feat, point_coord[:, 2:3]), dim=1).transpose(0, 1).contiguous()

            sample = {
                'pos': point_coord,
                'x': point_feat,
                'label': point_label,
            }
            return sample

        if self.split == 'train':
            point_coord, point_feat, point_label = self.data[data_idx][0], self.data[data_idx][1], self.data[data_idx][2]
            point_feat = point_feat.astype(np.float32)
            point_label = point_label.squeeze(-1).astype(np.int64)
            point_mask = self.point_mask[data_idx]

            point_coord, point_feat, point_label, point_mask = crop_pc(point_coord, point_feat, point_label, point_mask, self.split, self.voxel_max)

            transform_data = self.transform({'pos': point_coord, 'x': point_feat, 'y': point_label})
            point_coord = transform_data['pos']
            point_feat = transform_data['x']
            point_label = transform_data['y']

            point_feat = torch.cat((point_feat, point_coord[:, 2:3]), dim=1).transpose(0, 1).contiguous()

            sample = {
                'pos': point_coord,
                'x': point_feat,
                'label': point_label,
                'mask': torch.from_numpy(point_mask),
            }
            return sample

        if self.split == 'al':
            point_coord, point_feat, point_label = self.data[data_idx][0], self.data[data_idx][1], self.data[data_idx][2]
            point_feat = point_feat.astype(np.float32)
            point_label = point_label.squeeze(-1).astype(np.int64)
            point_mask = self.point_mask[data_idx]

            transform_data = self.transform({'pos': point_coord, 'x': point_feat, 'y': point_label})
            point_coord = transform_data['pos']
            point_feat = transform_data['x']
            point_label = transform_data['y']

            point_feat = torch.cat((point_feat, point_coord[:, 2:3]), dim=1).transpose(0, 1).contiguous()

            sample = {
                'pos': point_coord,
                'x': point_feat,
                'label': point_label,
                'mask': torch.from_numpy(point_mask),
            }
            return sample

    def __len__(self):
        return len(self.data) * self.loops
