import torch
from torch import nn
# FrEIA (https://github.com/VLL-HD/FrEIA/)
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch.nn.functional as F

coupling_layers = 8
clamp_alpha = 1.9
_GCONST_ = -0.9189385332046727  # ln(sqrt(2*pi))


def get_logp(C, z, logdet_J):
    logp = C * _GCONST_ - 0.5 * torch.sum(z ** 2, 1) + logdet_J
    return logp


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2 * dims_in), nn.ReLU(), nn.Linear(2 * dims_in, dims_out))


def flow_model(in_channels):
    coder = Ff.SequenceINN(in_channels)
    print('Normalizing Flow => Feature Dimension: ', in_channels)
    for k in range(coupling_layers):
        coder.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, affine_clamping=clamp_alpha,
                     global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def conditional_flow_model(in_channels):
    coder = Ff.SequenceINN(in_channels)
    print('Conditional Normalizing Flow => Feature Dimension: ', in_channels)
    for k in range(coupling_layers):  # 8
        coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(13,), subnet_constructor=subnet_fc,
                     affine_clamping=clamp_alpha,
                     global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


class joint_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.dim = cfg.model.feature_out_dim

        self.coder = Ff.SequenceINN(self.dim)
        for k in range(coupling_layers):
            self.coder.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, affine_clamping=clamp_alpha,
                              global_affine_type='SOFTPLUS', permute_soft=True)

        self.last_layer = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.BatchNorm1d(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, cfg.num_classes),
        )

    def forward(self, feat):

        z, log_jac_det = self.coder(feat)
        log_px = get_logp(self.dim, z, log_jac_det) / self.dim

        logit_pyx = self.last_layer(z)

        return log_px, logit_pyx
