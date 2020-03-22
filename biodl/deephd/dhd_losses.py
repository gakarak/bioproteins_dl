#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import torch
from torch import nn
from pytorch_toolbelt.losses import JointLoss
from segmentation_models_pytorch.utils.losses import DiceLoss, JaccardLoss#, BCEDiceLoss, BCEJaccardLoss
from segmentation_models_pytorch.utils.losses import Activation, F as F2
from segmentation_models_pytorch.utils import base


def build_loss_by_name(loss_name: str, class_weight = 30.0) -> nn.Module:
    if loss_name == 'bce':
        # return nn.BCEWithLogitsLoss()
        return BCEWithLogitsLossW(class_weight=class_weight)
    elif loss_name == 'l1':
        return nn.L1Loss()
    elif loss_name == 'l2':
        return nn.MSELoss()
    
    elif loss_name == 'bcedice':
        return JointLoss(first=BCEWithLogitsLossW(class_weight=class_weight), second=DiceLoss(activation='sigmoid'))
    elif loss_name == 'bceiou':
        return JointLoss(first=BCEWithLogitsLossW(class_weight=class_weight), second=JaccardLoss(activation='sigmoid'))
    
    elif loss_name == 'bceldice':
        return JointLoss(first=BCEWithLogitsLossW(class_weight=class_weight), second=LogDiceLoss(activation='sigmoid'))
    elif loss_name == 'bceliou':
        return JointLoss(first=BCEWithLogitsLossW(class_weight=class_weight), second=LogJaccardLoss(activation='sigmoid'))
    
    elif loss_name == 'dice':
        return DiceLoss(activation='sigmoid')
    elif loss_name == 'iou':
        return JaccardLoss(activation='sigmoid')
    elif loss_name == 'ldice':
        return LogDiceLoss(activation='sigmoid')
    elif loss_name == 'liou':
        return LogJaccardLoss(activation='sigmoid')
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

class LogDiceLoss(base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        loss_ = F2.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )
        return -torch.log(loss_.clamp_min(1e-6))


class LogJaccardLoss(base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        loss_ = F2.jaccard(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )
        return -torch.log(loss_.clamp_min(1e-6))



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