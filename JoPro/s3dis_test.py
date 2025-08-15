import os
import sys
import glob
import time
import copy
import random
import logging

import torch
import numpy as np
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch_scatter import scatter

from openpoints.utils import EasyConfig
from openpoints.dataset.data_util import voxelize
from openpoints.loss import build_criterion_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.transforms import build_transforms_from_cfg
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious

from model.base_seg import BaseSeg
from model.joint_model import joint_model
from active_learning import active_learning
from utils.tools import save_model, load_model, load_data_to_gpu
from dataset.build_dataloader import s3dis_dataloader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def load_data(data_path, cfg):
    label, feat = None, None
    if 's3dis' in cfg.dataset.name.lower():
        data = np.load(data_path)  # xyzrgbl, N*7
        coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
        feat = np.clip(feat / 255., 0, 1).astype(np.float32)
    elif 'scannet' in cfg.dataset.name.lower():
        data = torch.load(data_path)  # xyzrgbl, N*7
        coord, feat = data[0], data[1]
        if cfg.dataset.test.split != 'test':
            label = data[2]
        else:
            label = None
        feat = np.clip((feat + 1) / 2., 0, 1).astype(np.float32)

    coord -= coord.min(0)

    idx_points = []
    voxel_idx, reverse_idx_part, reverse_idx_sort = None, None, None
    voxel_size = cfg.dataset.get('voxel_size', None)

    if voxel_size is not None:
        # idx_sort: original point indicies sorted by voxel NO.
        # voxel_idx: Voxel NO. for the sorted points
        idx_sort, voxel_idx, count = voxelize(coord, voxel_size, mode=1)
        if cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor':
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_part = idx_sort[idx_select]
            npoints_subcloud = voxel_idx.max() + 1
            idx_shuffle = np.random.permutation(npoints_subcloud)
            idx_part = idx_part[idx_shuffle]  # idx_part: randomly sampled points of a voxel
            reverse_idx_part = np.argsort(idx_shuffle, axis=0)  # revevers idx_part to sorted
            idx_points.append(idx_part)
            reverse_idx_sort = np.argsort(idx_sort, axis=0)
        else:
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                np.random.shuffle(idx_part)
                idx_points.append(idx_part)
    else:
        idx_points.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx_sort


@torch.no_grad()
def test(model, test_data_list, cfg):
    """using a part of original point cloud as input to save memory.
    Args:
        model (_type_): _description_
        test_loader (_type_): _description_
        cfg (_type_): _description_
        num_votes (int, optional): _description_. Defaults to 1.
    Returns:
        _type_: _description_
    """
    model.eval()  # set model to eval mode
    all_cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    set_seed(0)
    cfg.visualize = cfg.get('visualize', False)
    if cfg.visualize:
        from openpoints.dataset.vis3d import write_obj
        cfg.vis_dir = os.path.join(cfg.run_dir, 'visualization')
        os.makedirs(cfg.vis_dir, exist_ok=True)
        cfg.cmap = cfg.cmap.astype(np.float32) / 255.

    # data
    trans_split = 'val' if cfg.data_transforms.get('test', None) is None else 'test'
    pipe_transform = build_transforms_from_cfg(trans_split, cfg.data_transforms)

    len_data = len(test_data_list)

    cfg.save_path = cfg.log_path + '/result'
    os.makedirs(cfg.save_path, exist_ok=True)

    for cloud_idx, data_path in enumerate(test_data_list):

        cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
        all_logits = []
        coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx = load_data(data_path, cfg)
        if label is not None:
            label = torch.from_numpy(label.astype(np.int64).squeeze()).cuda(non_blocking=True)

        len_part = len(idx_points)
        pbar = tqdm(range(len(idx_points)))

        for idx_subcloud in pbar:
            pbar.set_description(f"Test on {cloud_idx + 1}-th cloud [{idx_subcloud}]/[{len_part}]]")

            idx_part = idx_points[idx_subcloud]
            coord_part = coord[idx_part]
            coord_part -= coord_part.min(0)

            feat_part = feat[idx_part] if feat is not None else None
            data = {'pos': coord_part}
            if feat_part is not None:
                data['x'] = feat_part
            if pipe_transform is not None:
                data = pipe_transform(data)

            data['x'] = torch.cat((data['x'], data['pos'][:, 2:3]), dim=1).transpose(0, 1).contiguous()

            data['x'] = data['x'].unsqueeze(0)
            data['pos'] = data['pos'].unsqueeze(0)

            load_data_to_gpu(data)

            logits, __ = model(data)
            all_logits.append(logits)

        all_logits = torch.cat(all_logits, dim=0)
        # if not cfg.dataset.common.get('variable', False):
        #     all_logits = all_logits.transpose(1, 2).reshape(-1, cfg.num_classes)

        # average merge overlapped multi voxels logits to original point set
        idx_points = torch.from_numpy(np.hstack(idx_points)).cuda(non_blocking=True)
        all_logits = scatter(all_logits, idx_points, dim=0, reduce='mean')

        pred = all_logits.argmax(dim=1)
        if label is not None:
            cm.update(pred, label)

        if 'scannet' in cfg.dataset.name.lower():
            pred = pred.cpu().numpy().squeeze()
            label_int_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 14,
                                 13: 16, 14: 24, 15: 28, 16: 33, 17: 34, 18: 36, 19: 39}
            pred = np.vectorize(label_int_mapping.get)(pred)
            save_file_name = data_path.split('/')[-1].split('_')
            save_file_name = save_file_name[0] + '_' + save_file_name[1] + '.txt'
            save_file_name = os.path.join(cfg.log_path + '/result/' + save_file_name)
            np.savetxt(save_file_name, pred, fmt="%d")

        if label is not None:
            tp, union, count = cm.tp, cm.union, cm.count
            miou, macc, oa, ious, accs = get_mious(tp, union, count)
            logging.info(
                f'[{cloud_idx + 1}/{len_data}] cloud,  test_oa , test_macc, test_miou: {oa:.2f} {macc:.2f} {miou:.2f}'
            )
            all_cm.value += cm.value

    if label is not None:
        miou, macc, oa, ious, accs = get_mious(all_cm.tp, all_cm.union, all_cm.count)
        seg_result = {
            'miou': miou,
            'macc': macc,
            'oa': oa,
            'ious': ious,
        }
        return seg_result
    else:
        return None


def main(cfg):
    model_path = '/home/ybc/桌面/AL_point/1/log/S3DIS_9474_2024.12.29-08.50.24'
    logging.info(f'{model_path}')

    logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Start Test >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    for iters in range(cfg.Iterations):

        logging.info(f'Loading the best model......')
        model = BaseSeg(cfg.model).cuda()

        _ = load_model(model, model_path + '/iter_' + str(iters) + '.pth')

        test_result = test(model, cfg.test_data_list, cfg)

        logging.info(">>>>>>>>>>>>>>>>>>>> Test Result >>>>>>>>>>>>>>>>>>>>")

        test_miou = test_result['miou']
        test_ious = test_result['ious']
        with np.printoptions(precision=2, suppress=True):
            logging.info(f'Test miou {test_miou:.2f}')
            logging.info(f'iou per class: {test_ious}')


if __name__ == "__main__":
    # load config
    config = EasyConfig()
    config.load("cfg_s3dis.yaml", recursive=True)

    config.seed = np.random.randint(1000, 10000)
    set_seed(config.seed)

    # create log dir
    config.log_path = './log/' + config.dataset.name + '_' + str(config.seed) + '_' + time.strftime('%Y.%m.%d-%H.%M.%S')

    os.makedirs(config.log_path)

    # create logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s",
        handlers=[logging.FileHandler('%s/%s.log' % (
            config.log_path,
            config.dataset.name + '_' + str(config.seed) + '_' + time.strftime('%Y.%m.%d-%H.%M'))),
                  logging.StreamHandler(sys.stdout)
                  ]
    )

    # test for s3dis
    raw_root = config.dataset.data_root + config.dataset.name + '/raw'
    data_list = sorted(os.listdir(raw_root))
    config.test_data_list = [os.path.join(raw_root, item) for item in data_list if 'Area_5' in item]

    main(config)
