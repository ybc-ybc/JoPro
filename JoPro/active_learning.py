import os
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import pgeof

criterion = torch.nn.CrossEntropyLoss()
log_theta = torch.nn.LogSigmoid()


def generate_class_prototype(al_loader, output_feat, output_logit, cfg):

    # change to labeled + unlabeled (high-quality)

    labeled_feature = []
    labeled_label = []
    certainty_feature = []
    certainty_label = []

    for i, batch in enumerate(al_loader):
        point_mask = batch['mask'].flatten()
        point_label = batch['label'].flatten().cuda()

        # select high-quality features and labeled features
        sorted_pyx, __ = torch.sort(output_logit[i], dim=1, descending=True)
        score = sorted_pyx[:, 0] - sorted_pyx[:, 1]

        sorted_index = score.sort(descending=True)[1]

        pred = output_logit[i].argmax(1)

        per_certainty_feature = output_feat[i][sorted_index[:int(score.shape[0]*0.2) + 1]]
        per_certainty_label = pred[[sorted_index[:int(score.shape[0]*0.2) + 1]]]

        per_labeled_feature = output_feat[i][point_mask == 1]
        per_labeled_label = point_label[point_mask == 1]

        if i == 0:
            labeled_feature = per_labeled_feature
            labeled_label = per_labeled_label
            certainty_feature = per_certainty_feature
            certainty_label = per_certainty_label
        else:
            labeled_feature = torch.cat((labeled_feature, per_labeled_feature), dim=0)
            labeled_label = torch.cat((labeled_label, per_labeled_label), dim=0)
            certainty_feature = torch.cat((certainty_feature, per_certainty_feature), dim=0)
            certainty_label = torch.cat((certainty_label, per_certainty_label), dim=0)

    class_prototype = torch.zeros((cfg.num_classes, cfg.model.encoder_args.width)).cuda()
    for k in range(cfg.num_classes):
        if (labeled_label == k).sum() != 0 and (certainty_label == k).sum() != 0:
            class_prototype[k] = (0.5 * torch.mean(labeled_feature[labeled_label == k], dim=0) +
                                  0.5 * torch.mean(certainty_feature[certainty_label == k], dim=0))

    return class_prototype


def active_learning(al_loader, pb_model, pb_optimizer, pb_scheduler, cfg):

    pre_points = 0
    post_points = 0
    dispersion_score = []

    # load model output
    output_feat = []
    output_logit = []
    for i in range(al_loader.__len__()):
        feat = torch.load(os.path.join(cfg.output_path, 'output_feat_' + str(i) + '.pt'))
        logit = torch.load(os.path.join(cfg.output_path, 'output_logit_' + str(i) + '.pt'))
        output_feat.append(feat)
        output_logit.append(logit)

    # -----------------------------------------  generate_class_prototype  -----------------------------------------
    class_prototype = generate_class_prototype(al_loader, output_feat, output_logit, cfg)

    # -------------------------------------------  Train joint prob model ------------------------------------------
    pb_model.train()
    pbar = tqdm(range(cfg.al_epochs))
    for _ in pbar:

        pbar.set_description(f"Train joint prob model")
        for i in range(len(output_logit)):
            train_feat = output_feat[i].cuda()
            train_logit = output_logit[i].cuda()

            prob, pred = train_logit.max(dim=1)

            log_px, logit_pyx = pb_model(train_feat)

            loss_px = -(log_theta(log_px)).mean()

            loss_pyx = criterion(logit_pyx[prob > 0.8], pred[prob > 0.8])             # select high-quality

            loss = 0.2 * loss_px + loss_pyx

            pb_optimizer.zero_grad()
            loss.backward()
            pb_optimizer.step()
            pb_scheduler.step()

    pb_model.eval()
    torch.cuda.empty_cache()

    # ---------------------------------  Feature mixing ----------------------------------

    pbar = tqdm(range(len(output_logit)))
    reliability_score = []
    for idx in pbar:
        pbar.set_description(f"Reliability assessment")

        train_feat = output_feat[idx].cuda()

        sum_pyx = []
        sum_px = []
        alpha = torch.randint(1, 11, (cfg.num_classes*cfg.mix_epochs,)) / 100.0
        for k in range(cfg.num_classes):

            per_pyx = []
            per_px = []
            for kn in range(cfg.mix_epochs):
                mix_feat = (1 - alpha[k*cfg.mix_epochs+kn]) * train_feat + alpha[k*cfg.mix_epochs+kn] * class_prototype[k]

                with torch.no_grad():
                    log_px, logit_pyx = pb_model(mix_feat)

                log_px -= torch.max(log_px)
                px = torch.exp(log_px)

                pyx = F.softmax(logit_pyx, dim=1)

                if kn == 0:
                    per_pyx = pyx.unsqueeze(0)
                    per_px = px.unsqueeze(1)
                else:
                    per_pyx = torch.cat((per_pyx, pyx.unsqueeze(0)), dim=0)
                    per_px = torch.cat((per_px, px.unsqueeze(1)), dim=1)

            if k == 0:
                sum_pyx = per_pyx
                sum_px = per_px
            else:
                sum_pyx = torch.cat((sum_pyx, per_pyx), dim=0)
                sum_px = torch.cat((sum_px, per_px), dim=1)

        mean_px = torch.mean(sum_px, dim=1)

        mean_pyx = torch.mean(sum_pyx, dim=0)
        var_pyx = torch.mean(torch.mean((sum_pyx - mean_pyx) ** 2, dim=0), dim=1)

        mean_pyx = mean_px.unsqueeze(1) * mean_pyx

        sorted_pyx, __ = torch.sort(mean_pyx, dim=1, descending=True)
        pyx_score = sorted_pyx[:, 0] - sorted_pyx[:, 1] - var_pyx

        joint_score = pyx_score - pyx_score.min()

        reliability_score.append(joint_score)

    # ---------------------------------------------  De-redundant  -------------------------------------------------

    pbar = tqdm(enumerate(al_loader), total=al_loader.__len__())
    for idx, batch in pbar:
        pbar.set_description(f"Redundancy elimination")

        point_mask = batch['mask'].flatten().detach().cpu().numpy()
        point_corrd = torch.squeeze(batch['pos'], dim=0).detach().cpu().numpy()

        pre_points += point_mask.sum()

        # top-K% sampling
        sorted_index = reliability_score[idx].sort(descending=False)[1]

        candidate_top_idx = sorted_index[:int(reliability_score[idx].shape[0] * 0.02)].detach().cpu().numpy()
        candidate_top__score = reliability_score[idx][candidate_top_idx]

        # distance sampling
        knn_num = 128
        if candidate_top__score.shape[0] < knn_num:
            knn_num = candidate_top__score.shape[0]

        knn, dist = pgeof.knn_search(point_corrd[candidate_top_idx], point_corrd[candidate_top_idx], knn_num)

        indicator_dist = np.ones((candidate_top_idx.shape[0])).astype(np.int8)
        for i in range(candidate_top_idx.shape[0]):
            for j in range(knn_num-1):
                if dist[i][j+1] < 0.2:    # and indicator_dist[knn[i][j+1]] == 1:
                    if candidate_top__score[i] > candidate_top__score[knn[i][j+1]]:
                        indicator_dist[i] = 0
                    else:
                        indicator_dist[knn[i][j+1]] = 0
                else:
                    break

        candidate_dist_idx = candidate_top_idx[indicator_dist == 1]
        candidate_dist_score = reliability_score[idx][candidate_dist_idx]

        # similarity sampling
        x = output_feat[idx][candidate_dist_idx].unsqueeze(1)
        y = class_prototype.unsqueeze(0)
        sim, label = torch.max(F.cosine_similarity(x, y, dim=-1), dim=1)

        indicator_sim = np.ones((candidate_dist_idx.shape[0])).astype(np.int8)
        for i in range(candidate_dist_idx.shape[0]-1):
            if sim[i] > 0.8:
                for j in range(i+1, candidate_dist_idx.shape[0]):
                    if sim[j] > 0.8 and label[i] == label[j]:
                        if candidate_dist_score[i] <= candidate_dist_score[j]:
                            indicator_sim[j] = 0

        candidate_sim_idx = candidate_dist_idx[indicator_sim == 1]

        # finish sampling, adjust score
        temp = np.ones((reliability_score[idx].shape[0])) * 1000
        temp[candidate_sim_idx] = 0

        point_score = reliability_score[idx].detach().cpu().numpy() + temp

        # point_score = reliability_score[idx].cpu()

        # -------  active labeling  -------
        idx_s = np.argsort(point_score)

        n, count = 0, 0
        labeling_num = point_mask.sum()/(cfg.iters + 1)

        while count < labeling_num:
            if point_mask[idx_s[n]] == 1:
                n += 1
                continue

            point_mask[idx_s[n]] = 1

            n += 1
            count += 1

        al_loader.dataset.point_mask[idx] = point_mask
        post_points += point_mask.sum()
    logging.info(f'Active select points: {post_points}, pre_points: {pre_points}')

    return al_loader
