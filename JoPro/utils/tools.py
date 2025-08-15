import os
import torch
import numpy as np


def save_model(model, result, save_path):

    save_dict = {
        'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'result': result,
    }

    torch.save(save_dict, save_path)


def load_model(model, pretrained_path):

    if not os.path.exists(pretrained_path):
        raise NotImplementedError('no checkpoint file from path %s...' % pretrained_path)

    # load state dict
    state_dict = torch.load(pretrained_path, map_location='cpu')

    model.load_state_dict(state_dict['model'])

    seg_result = state_dict['result']

    return seg_result


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if isinstance(val, torch.Tensor):
            batch_dict[key] = val.cuda()
        elif isinstance(val, np.ndarray):
            batch_dict[key] = torch.from_numpy(val).cuda()
        elif isinstance(val, dict):
            for k, v in val.items():
                batch_dict[key][k] = v.cuda()
        elif isinstance(val, list):
            batch_dict[key] = val
        else:
            raise ValueError("Invalid type of batch_dict")
