#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import torch
from torch import nn
from segmentation_models_pytorch.utils.losses import BCEDiceLoss, BCEJaccardLoss, DiceLoss, JaccardLoss


def build_loss_by_name(loss_name: str) -> nn.Module:
    if loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'bcedice':
        return BCEDiceLoss()
    elif loss_name == 'bcejaccard':
        return BCEJaccardLoss()
    elif loss_name == 'dice':
        return DiceLoss()
    elif loss_name == 'jaccard':
        return JaccardLoss()
    else:
        raise NotImplementedError


def main_debug():
    y_pr = torch.rand([5, 5], dtype=torch.float32)
    y_gt = torch.randint(0, 2, [5, 5]).type(torch.float32)
    loss_names = ['bce', 'bcedice', 'bcejaccard', 'dice', 'jaccard']
    loss_functions = [build_loss_by_name(x) for x in loss_names]
    loss_ = [f(y_pr, y_gt) for f in loss_functions]
    print(loss_)

    print('-')


if __name__ == '__main__':
    main_debug()