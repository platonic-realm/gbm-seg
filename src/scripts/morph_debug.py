import sys
sys.path.append('../')

import torch
import numpy as np
from torch import Tensor
from infer.morph import Morph


def load_voxel_space(_path: str) -> Tensor:
    vs = np.load(_path).astype(np.float32)
    return torch.from_numpy(vs).to('cuda')

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":
    voxel_space = load_voxel_space('/data/afatehi/gbm/cube.npy')
    voxel_space[voxel_space == 255] = 1
    morph = Morph('cuda', _ave_kernel_size=5, _inside_voxel_weight=1.0)
    result = morph(voxel_space)
    for i in range(result.shape[0]):
        print(f"Mean of Z={i}: {result[i, :, :].mean()}")
    for i in range(result.shape[1]):
        print(f"Mean of X={i}: {result[:, i, :].mean()}")
    for i in range(result.shape[2]):
        print(f"Mean of Y={i}: {result[:, :, i].mean()}")

    with open("/data/afatehi/gbm/result.npy", 'wb') as f:
        np.save(f, result.detach().cpu().numpy())
