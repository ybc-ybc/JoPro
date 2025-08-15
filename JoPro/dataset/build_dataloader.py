import torch
from openpoints.transforms import build_transforms_from_cfg
from dataset.s3dis_dataset import s3dis_dataset


def collate_fn_offset_train(batches):
    """collate fn and offset
    """
    pts, feats, labels, mask = [], [], [], []
    for i in range(0, len(batches)):
        pts.append(batches[i]['pos'])
        feats.append(batches[i]['x'])
        labels.append(batches[i]['label'])
        mask.append(batches[i]['mask'])

    data = {'pos': torch.stack(pts, dim=0),
            'x': torch.stack(feats, dim=0),
            'label': torch.stack(labels, dim=0),
            'mask': torch.stack(mask, dim=0),
            }
    return data


def collate_fn_offset_train_weak(batches):
    """collate fn and offset
    """
    pts_1, feats_1, pts_2, feats_2, labels, mask = [], [], [], [], [], []
    for i in range(0, len(batches)):
        pts_1.append(batches[i]['pos_1'])
        feats_1.append(batches[i]['x_1'])
        pts_2.append(batches[i]['pos_2'])
        feats_2.append(batches[i]['x_2'])

        labels.append(batches[i]['label'])
        mask.append(batches[i]['mask'])

    data_1 = {'pos': torch.stack(pts_1, dim=0),
              'x': torch.stack(feats_1, dim=0),
              'label': torch.stack(labels, dim=0),
              'mask': torch.stack(mask, dim=0),
              }
    data_2 = {'pos': torch.stack(pts_2, dim=0),
              'x': torch.stack(feats_2, dim=0),
              'label': torch.stack(labels, dim=0),
              'mask': torch.stack(mask, dim=0),
              }
    return data_1, data_2


def collate_fn_offset_val(batches):
    """collate fn and offset
    """
    pts, feats, labels = [], [], []
    for i in range(0, len(batches)):
        pts.append(batches[i]['pos'])
        feats.append(batches[i]['x'])
        labels.append(batches[i]['label'])

    data = {'pos': torch.unsqueeze(torch.cat(pts, dim=0), dim=0),
            'x': torch.unsqueeze(torch.cat(feats, dim=0), dim=0),
            'label': torch.unsqueeze(torch.cat(labels, dim=0), dim=0),
            }
    return data


def s3dis_dataloader(cfg, split):
    if split == 'train':
        train_transform = build_transforms_from_cfg('train', cfg.data_transforms)
        train_dataset = s3dis_dataset(cfg, cfg.dataset, train_transform, 'train')
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.train_batch_size,
            num_workers=cfg.num_workers,
            shuffle=True,
            collate_fn=collate_fn_offset_train,
            pin_memory=True,
            drop_last=True
        )
        return train_loader

    if split == 'val':
        val_transform = build_transforms_from_cfg('val', cfg.data_transforms)
        val_dataset = s3dis_dataset(cfg, cfg.dataset, val_transform, 'val')
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.val_batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
            collate_fn=collate_fn_offset_val,
            pin_memory=True,
            drop_last=False
        )
        return val_loader

    if split == 'al':
        al_transform = build_transforms_from_cfg('al', cfg.data_transforms)
        al_dataset = s3dis_dataset(cfg, cfg.dataset, al_transform, 'al')
        al_loader = torch.utils.data.DataLoader(
            al_dataset,
            batch_size=cfg.val_batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
            collate_fn=collate_fn_offset_train,
            pin_memory=True,
            drop_last=False
        )
        return al_loader
