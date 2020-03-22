#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import copy
import torch
from torch import nn
from typing import Union as U, Optional as O
from segmentation_models_pytorch import Unet, FPN, Linknet, PSPNet
from segmentation_models_pytorch.encoders import get_encoder, get_encoder_names, encoders
from typing import Optional as O, Union as U
import pytorch_toolbelt
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules import decoders as D

from .dhd_model_dilated import ASPPResNetSE
from .dhd_model_attention import DeepAttentionModel


def build_model_from_cfg(cfg: dict) -> nn.Module:
    cfgm = cfg['model']
    num_inp = cfg['inp_size']
    num_out = cfg['out_size']
    if cfgm['type'] == 'ASPPResNetSE':
        model = ASPPResNetSE(inp=num_inp, out=num_out, nlin=cfgm['nlin'], num_stages=cfgm['num_stages'])
    elif cfgm['type'] == 'DeepAttention':
        model = DeepAttentionModel(inp_size=num_inp, out_size=num_out)
    else:
        type_enc = cfgm['ext']['enc']
        type_dec = cfgm['ext']['dec']
        if type_dec == 'fpn':
            model = FPN(type_enc, in_channels=num_inp, encoder_depth=cfgm['ext']['depth'], classes=num_out)
        elif type_dec == 'unet':
            model = Unet(type_enc, in_channels=num_inp, encoder_depth=cfgm['ext']['depth'], classes=num_out)
        elif type_dec == 'psp':
            model = PSPNet(type_enc, in_channels=num_inp, encoder_depth=cfgm['ext']['depth'], classes=num_out)
        else:
            raise NotImplementedError
    return model


def get_encoder_custom(encoder_name: str, encoder_params: dict, depth: int, in_channels=3, weights=None):
    Encoder = encoders[encoder_name]["encoder"]
    encoder_params.update(depth=depth)
    encoder = Encoder(**encoder_params)
    encoder.set_in_channels(in_channels)
    return encoder



def main_debug_segmm():
    x_inp = torch.zeros([1, 3, 256, 256], dtype=torch.float32)
    # encoder: E.EncoderModule = E.SEResnet50Encoder()
    # encoder_name = 'resnet34'
    encoder_name = 'densenet121'
    encoder_params = copy.deepcopy(encoders[encoder_name])
    encoder_params['params']['out_channels'] = (3, 64, 64, 128, 256, 256, 256, 256)
    # encoder_params['params']['layers'] = [3, 4, 4, 4, 6, 3]
    encoder_params['params']['block_config'] = [6, 12, 24, 12, 12, 16]
    #
    model_enc = get_encoder_custom(encoder_name, encoder_params['params'], depth=6)

    # y = model_enc(x_inp)
    print('-')


if __name__ == '__main__':
    main_debug_segmm()