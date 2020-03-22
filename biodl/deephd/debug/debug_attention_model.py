#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import os
import numpy as np
import scipy.spatial.distance as sdst
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import Tensor as T
from torch.nn import functional as F
import segmentation_models_pytorch.fpn


from biodl.deephd.dhd_model import MultiBottleneckSEBlock, \
    BottleneckASPPSEBlock, ASPPMultiConvBlock, count_model_parameters
from biodl.deephd.dhd_data import _load_idx, load_one_sample



def get_dst_mat(sample: dict) -> dict:
    ret = {}
    for atom_type in ('ca', 'cb'):
        tmp_ = []
        x1, x2 = sample[f'coords_{atom_type}']
        dst_x1x1 = sdst.cdist(x1, x1).astype(np.float32)
        dst_x1x2 = sdst.cdist(x1, x2).astype(np.float32)
        tmp_.append(dst_x1x1)
        tmp_.append(dst_x1x2)
        ret[atom_type] = tmp_
    return ret



def main():
    path_idx = '/home/ar/data/bioinformatics/deep_hd/idxok_debug0.txt'
    data_idx = _load_idx(path_idx)
    sample = load_one_sample(data_idx.iloc[0])
    sample_dst = get_dst_mat(sample)
    #
    x = torch.zeros([1, 3, 512, 512])
    x[..., x.shape[-2]//2, x.shape[-1]//2] = 1
    # m = SelfAttentionLayer(inp_size=3, emb_size=8, )
    # m = ConcatMultiHeadSelfAttention(3, 8, 14)
    m = DeepAttentionModel(use_attention=False)
    m.eval()
    with torch.no_grad():
        y = m(x)
    #
    plt.subplot(1, 2, 1)
    plt.imshow(x[0, 0])
    plt.subplot(1, 2, 2)
    plt.imshow(y[0, 0])
    plt.show()

    print('-')


if __name__ == '__main__':
    main()