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


from .dhd_model_dilated import MultiBottleneckSEBlock, \
    count_model_parameters


class ConvBlock_VGG(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=3,
                 num_conv=1,
                 act: str = 'prelu',
                 use_bn=True,
                 padding='same',
                 skip_last_activation=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        layers_ = []
        for idx in range(num_conv):
            if idx == 0:
                chn_ = in_channels
            else:
                chn_ = out_channels
            if padding == 'same':
                pad_ = kernel_size // 2
            else:
                pad_ = 0
            conv_ = nn.Conv2d(in_channels=chn_, out_channels=out_channels,
                              kernel_size=kernel_size, padding=pad_,
                              bias=not use_bn)
            layers_.append(conv_)
            if use_bn:
                layers_.append(nn.BatchNorm2d(num_features=out_channels))
            if act.lower() == 'relu':
                layers_.append(nn.ReLU(inplace=True))
            elif act.lower() == 'prelu':
                layers_.append(nn.PReLU())
            elif act.lower() == 'elu':
                layers_.append(nn.ELU(inplace=True))
            else:
                raise NotImplementedError('not supported activation [{}]'.format(act))
            if skip_last_activation:
                del layers_[-1]
        self.body = nn.Sequential(*layers_)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ret_ = self.body(input)
        return ret_


class SubConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 pool_kernel=3, pool_stride=2, use_bn=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.conv = ConvBlock_VGG(in_channels=self.in_channels,
                                  out_channels=out_channels // 2,
                                  num_conv=2, use_bn=use_bn)
        pad = self.pool_kernel // 2
        self.sub_max = nn.MaxPool2d(self.pool_kernel, stride=self.pool_stride, padding=pad)
        self.sub_avg = nn.AvgPool2d(self.pool_kernel, stride=self.pool_stride, padding=pad)

    def forward(self, x: T) -> T:
        x = self.conv(x)
        x_max, x_max = self.sub_max(x), self.sub_avg(x)
        x = torch.cat([x_max, x_max], dim=1)
        return x


class VGG_Decoder(nn.Module):

    def __init__(self, inp_size: int, out_size: int, up_sizes=(2, 2, 4, 4, ),
                 stage1_chn=64, stage2_chn=32, use_bn=False):
        super().__init__()
        layers = []
        for xi, x in enumerate(up_sizes):
            l_up = nn.UpsamplingBilinear2d(scale_factor=x)
            if xi == 0:
                l_conv = ConvBlock_VGG(inp_size, stage1_chn, kernel_size=3, use_bn=use_bn)
            elif xi == 1:
                l_conv = ConvBlock_VGG(stage1_chn, stage2_chn, kernel_size=3, use_bn=use_bn)
            else:
                l_conv = ConvBlock_VGG(stage2_chn, stage2_chn, kernel_size=3, use_bn=use_bn)
            layers.append(l_up)
            layers.append(l_conv)
        out_conv = nn.Conv2d(stage2_chn, out_size, kernel_size=1)
        layers.append(out_conv)
        self.body = nn.Sequential(*layers)

    def forward(self, x: T) -> T:
        return self.body(x)


class VGG_Encoder(nn.Module):

    def __init__(self, in_channels: int, num_stages=6, stage1_chn=32, stage2_chn=64, use_bn=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_stages = num_stages
        self.stage1_chn = stage1_chn
        self.stage2_chn = stage2_chn
        layers = []
        for x in range(num_stages):
            if x == 0:
                l = SubConvBlock(self.in_channels, self.stage1_chn, use_bn=use_bn)
            elif x == 1:
                l = SubConvBlock(self.stage1_chn, self.stage2_chn, use_bn=use_bn)
            else:
                l = SubConvBlock(self.stage2_chn, self.stage2_chn, use_bn=use_bn)
            layers.append(l)
        l = MultiBottleneckSEBlock(inp_dim=stage2_chn, emb_dim=stage1_chn,
                                   dilation=(2, 4), nlin='elu', attention=None)
        layers.append(l)
        self.body = nn.Sequential(*layers)

    def forward(self, x: T) -> T:
        return self.body(x)


class SelfAttentionLayer(nn.Module):

    def __init__(self, inp_size: int, emb_size: int, out_size=None, use_bn=False):
        super().__init__()
        self.inp_size = inp_size
        self.emb_size = emb_size
        if out_size is None:
            self.out_size = inp_size
        else:
            self.out_size = inp_size
        self.conv_q = ConvBlock_VGG(in_channels=inp_size, out_channels=emb_size, kernel_size=1,
                                    num_conv=2, skip_last_activation=True, use_bn=use_bn)
        self.conv_k = ConvBlock_VGG(in_channels=inp_size, out_channels=emb_size, kernel_size=1,
                                    num_conv=2, skip_last_activation=True, use_bn=use_bn)
        self.conv_v = ConvBlock_VGG(in_channels=inp_size, out_channels=self.out_size, kernel_size=1,
                                    num_conv=2, skip_last_activation=True, use_bn=use_bn)

    def forward(self, x: T) -> T:
        bsize, nch, h, w = x.size()
        HW = h * w
        HW_SQRT = np.sqrt(HW)
        proj_q = self.conv_q(x).view(bsize, -1, HW).permute(0, 2, 1)
        proj_k = self.conv_k(x).view(bsize, -1, HW)
        proj_v = self.conv_v(x).view(bsize, -1, HW)
        attention = F.softmax(proj_q.bmm(proj_k) / HW_SQRT, dim=-1)
        ret = proj_v.bmm(attention.permute(0, 2, 1)).view(bsize, nch, h, w)
        return ret


class ConcatMultiHeadSelfAttention(nn.Module):

    def __init__(self, inp_size, emb_size: int, num_heads=5, out_size=None, out_size_attention=None):
        super().__init__()
        if out_size is None:
            out_size = inp_size
        if out_size_attention is None:
            out_size_attention = inp_size
        self.attention_layers = nn.ModuleList([
            SelfAttentionLayer(inp_size, emb_size, out_size_attention)
            for _ in range(num_heads)
        ])
        self.out_prj = nn.Conv2d(inp_size + out_size_attention * num_heads, out_size, kernel_size=1)

    def forward(self, x: T) -> T:
        out = [x]
        for l in self.attention_layers:
            out.append(l(x))
        out = torch.cat(out, dim=1)
        out = self.out_prj(out)
        return out


class DeepAttentionModel(nn.Module):

    def __init__(self, inp_size: int = 3, out_size: int = 1,
                 num_enc_stages=6, stage1_chn=32, stage2_chn=64,
                 num_attention_stages=2, num_attention_heads=3,
                 emb_size=32, att_out_size=72,
                 up_sizes=(2, 2, 4, 4),
                 use_attention=True):
        super().__init__()
        self.inp_size = inp_size
        self.out_size = out_size
        self.use_attention = use_attention
        self.encoder = VGG_Encoder(in_channels=inp_size, num_stages=num_enc_stages,
                                   stage1_chn=stage1_chn, stage2_chn=stage2_chn,
                                   use_bn=False)
        if self.use_attention:
            layers_attentions = []
            for x in range(num_attention_stages):
                if x == 0:
                    l = ConcatMultiHeadSelfAttention(stage2_chn, emb_size,
                                                     num_heads=num_attention_heads,
                                                     out_size=att_out_size)
                else:
                    l = ConcatMultiHeadSelfAttention(att_out_size, emb_size,
                                                     num_heads=num_attention_heads,
                                                     out_size=att_out_size)
                layers_attentions.append(l)
            self.self_attention = nn.Sequential(*layers_attentions)
            inp_size_dec = att_out_size
        else:
            inp_size_dec = stage2_chn
        self.decoder = VGG_Decoder(inp_size=inp_size_dec,
                                   stage1_chn=stage2_chn, stage2_chn=stage1_chn,
                                   up_sizes=up_sizes,
                                   out_size=out_size)

    def forward(self, x: T) -> T:
        x = self.encoder(x)
        if self.use_attention:
            x = self.self_attention(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    pass