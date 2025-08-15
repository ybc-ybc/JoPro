import os
import sys
import time
import copy
import random
import logging

import torch
import numpy as np
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F

from openpoints.utils import EasyConfig
from openpoints.loss import build_criterion_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from dataset.build_dataloader import s3dis_dataloader
from utils.tools import save_model, load_model, load_data_to_gpu

from model.base_seg import BaseSeg
from model.joint_model import joint_model
from active_learning import active_learning


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def generate_model_output(model, al_loader):
    model.eval()

    config.output_path = config.log_path + '/output'
    os.makedirs(config.output_path, exist_ok=True)

    pbar = tqdm(enumerate(al_loader), total=al_loader.__len__())
    for idx, data in pbar:
        pbar.set_description(f"Produce feat and logit")
        load_data_to_gpu(data)

        logit, feat = model(data)

        logit = F.softmax(logit, dim=1)
        feat = F.normalize(feat, dim=1)

        torch.save(feat, os.path.join(config.output_path, 'output_feat_' + str(idx) + '.pt'))
        torch.save(logit, os.path.join(config.output_path, 'output_logit_' + str(idx) + '.pt'))


@torch.no_grad()
def validate(model, val_loader, cfg):
    model.eval()

    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Val')

    for idx, data in pbar:
        load_data_to_gpu(data)
        logit, __ = model(data)

        label = data['label']
        cm.update(logit.argmax(dim=1), label)

    miou, macc, oa, ious, accs = get_mious(cm.tp, cm.union, cm.count)

    val_result = {
        'val_miou': miou,
        'val_macc': macc,
        'val_oa': oa,
    }
    return val_result


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, cfg):
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)

    model.train()

    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    for i, data in pbar:
        load_data_to_gpu(data)

        logit, feat = model(data)

        label = data['label']
        mask = data['mask']

        label = label.flatten()
        mask = mask.flatten()
        idx = mask == 1

        if cfg.ignore_index is not None:
            idx_not_ignore = (label != cfg.ignore_index)
            idx = idx & idx_not_ignore

        loss = criterion(logit[idx], label[idx])

        loss.backward()

        # optimize
        if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)

        optimizer.step()
        optimizer.zero_grad()

        # update confusion matrix
        cm.update(logit.argmax(dim=1), label)
        loss_meter.update(loss.item())

        pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}]")

    miou, _, _, _, _ = cm.all_metrics()

    return loss_meter.avg, miou


def main(cfg):
    model = BaseSeg(cfg.model).cuda()

    cfg.epochs = cfg.epochs_list[0]

    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, dampening=0.1, weight_decay=1e-4)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)
    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()

    # build dataset
    train_loader = s3dis_dataloader(cfg, 'train')
    val_loader = s3dis_dataloader(cfg, 'val')
    al_loader = s3dis_dataloader(cfg, 'al')

    pb_model = joint_model(cfg).cuda()
    pb_optimizer = optim.AdamW(pb_model.parameters(), lr=cfg.flow_lr)
    pb_scheduler = optim.lr_scheduler.CosineAnnealingLR(pb_optimizer, T_max=al_loader.__len__()*cfg.al_epochs)

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                             Start Active Learning Iteration
    # ---------------------------------------------------------------------------------------------------------------- #
    cfg.iters = 0
    AL_Result = []
    while 1:
        logging.info(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Start Iteration: {str(cfg.iters)} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        train_miou, best_iou = 0., 0.
        if not (cfg.pre_load and cfg.iters == 0):
            for epoch in range(1, cfg.epochs + 1):
                train_loss, train_miou = train_one_epoch(model, train_loader, criterion, optimizer, epoch, cfg)
                scheduler.step(epoch)

                if cfg.minval_list[cfg.iters] <= epoch and epoch % cfg.val_freq == 0:
                    val_result = validate(model, val_loader, cfg)

                    if val_result['val_miou'] > best_iou:
                        best_iou = val_result['val_miou']
                        save_model(model, val_result, cfg.log_path + '/iter_' + str(cfg.iters) + '.pth')
                        logging.info(f'Epoch {epoch} Find best ckpt, val_miou {best_iou:.2f}')

                lr = optimizer.param_groups[0]['lr']
                logging.info(f'Epoch {epoch} LR {lr:.6f} 'f'train_loss {train_loss:.2f}, train_miou {train_miou:.2f}'
                             f', best val_miou {best_iou:.2f}')

            logging.info("<<<<<<<<<<<<<<<<<<<< Train/Val End! <<<<<<<<<<<<<<<<<<<<")

            logging.info(f'Loading the best model......')
            model = BaseSeg(cfg.model).cuda()
            val_result = load_model(model, cfg.log_path + '/iter_' + str(cfg.iters) + '.pth')

        else:
            logging.info(f'Loading the pretrain model......')

            # Local or server devices
            if os.path.exists(cfg.model_path):
                model_path = cfg.model_path
            else:
                model_path = cfg.sever_model_path

            val_result = load_model(model, model_path)
            save_model(model, val_result, cfg.log_path + '/iter_' + str(cfg.iters) + '.pth')

        val_miou = val_result['val_miou']
        with np.printoptions(precision=2, suppress=True):
            logging.info(f'>>>>>>>>> Iteration {str(cfg.iters)}, val_miou: {val_miou:.2f} >>>>>>>>>>')

        AL_Result.append(val_miou)
        if cfg.iters == cfg.Iterations-1:
            break

        logging.info("<<<<<<<<<<<<<<<<<<<< Active Select <<<<<<<<<<<<<<<<<<<<")

        generate_model_output(model, al_loader)

        al_loader.dataset.point_mask = copy.deepcopy(train_loader.dataset.point_mask)
        al_loader = active_learning(al_loader, pb_model, pb_optimizer, pb_scheduler, cfg)
        train_loader.dataset.point_mask = copy.deepcopy(al_loader.dataset.point_mask)

        # reset optimizer, scheduler
        pb_optimizer = optim.AdamW(pb_model.parameters(), lr=cfg.flow_lr)
        pb_scheduler = optim.lr_scheduler.CosineAnnealingLR(pb_optimizer, T_max=al_loader.__len__() * cfg.al_epochs)

        cfg.epochs = cfg.epochs_list[cfg.iters + 1]
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, dampening=0.1, weight_decay=1e-4)
        scheduler = build_scheduler_from_cfg(cfg, optimizer)

        logging.info(f'Iteration_{cfg.iters} End!')
        cfg.iters = cfg.iters + 1

    logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished All !!! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    logging.info(f'AL iteration result: {AL_Result[0]:.2f}, {AL_Result[1]:.2f}, {AL_Result[2]:.2f}, '
                 f'{AL_Result[3]:.2f}, {AL_Result[4]:.2f}')


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

    main(config)
