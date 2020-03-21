#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import torch
from torch import nn
from pytorch_toolbelt.losses import JointLoss
from segmentation_models_pytorch.utils.losses import DiceLoss, JaccardLoss#, BCEDiceLoss, BCEJaccardLoss


def build_loss_by_name(loss_name: str, class_weight = 50.0) -> nn.Module:
    if loss_name == 'bce':
        # return nn.BCEWithLogitsLoss()
        return BCEWithLogitsLossW(class_weight=class_weight)
    elif loss_name == 'l1':
        return nn.L1Loss()
    elif loss_name == 'l2':
        return nn.MSELoss()
    elif loss_name == 'bcedice':
        # return BCEDiceLoss()
        # return JointLoss(first=nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([100])), second=DiceLoss(activation='sigmoid'))
        return JointLoss(first=BCEWithLogitsLossW(class_weight=class_weight), second=DiceLoss(activation='sigmoid'))
    elif loss_name == 'bcejaccard':
        # return BCEJaccardLoss()
        return JointLoss(first=BCEWithLogitsLossW(class_weight=class_weight), second=JaccardLoss(activation='sigmoid'))
    elif loss_name == 'dice':
        return DiceLoss(activation='sigmoid')
    elif loss_name == 'jaccard':
        return JaccardLoss(activation='sigmoid')
    else:
        raise NotImplementedError


class BCEWithLogitsLossW(nn.Module):

    def __init__(self, class_weight=1.):
        super().__init__()
        self.class_weight = class_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, y_pr, y_gt):
        msk_ = (y_gt > 0) * (self.class_weight - 1) + 1
        loss_ = self.bce(y_pr, y_gt) * msk_
        return loss_.mean()


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