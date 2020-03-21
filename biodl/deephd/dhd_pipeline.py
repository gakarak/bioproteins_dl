#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import time
import json
import numpy as np
import pandas as pd
import pickle as pkl
from typing import Union as U
import logging
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch import nn
from .dhd_losses import build_loss_by_name
from .dhd_model import ASPPResNetSE, build_model_from_cfg
from .dhd_data import load_config, DHDDataset
from pytorch_lightning import LightningModule
import pytorch_lightning as pl



def _get_random_seed():
    seed = int(time.time() * 100000) % 10000000 + os.getpid()
    return seed


def worker_init_fn_random(idx):
    seed_ = _get_random_seed() + idx
    torch.manual_seed(seed_)
    np.random.seed(seed_)


def _preprocess_batch(data: dict) -> dict:
    data['inp'] = data['inp'].type(torch.float32)
    data['out'] = data['out'].type(torch.float32)
    return data


class DeepHDPipeline(LightningModule):

    def __init__(self, cfg: U[str, dict], num_workers=8, path_cfg: str = None):
        super().__init__()
        if isinstance(cfg, str):
            self.path_cfg = cfg
            self.cfg = load_config(self.path_cfg)
        else:
            self.cfg = cfg
            self.path_cfg = path_cfg
        self.path_model = self.cfg['path_model']
        self.num_workers = num_workers
        self.model = build_model_from_cfg(self.cfg)
        # self.model = ASPPResNetSE(inp=43, out=1,
        #                           nlin=self.cfg['model']['nlin'],
        #                           num_stages=self.cfg['model']['num_stages'])
        self.trn_loss = build_loss_by_name(self.cfg['loss'])
        self.val_losses = {}
        self.val_losses['val_loss'] = build_loss_by_name(self.cfg['loss'])
        for x in self.cfg['val_losses']:
            self.val_losses[f'val_loss_{x}'] = build_loss_by_name(x)

    def build(self):
        cfd=self.cfg['data']
        self.dataset_trn = DHDDataset(path_idx=self.cfg['trn_abs'],
                                      crop_size=self.cfg['crop_size'],
                                      use_sasa=cfd['use_sasa'],
                                      sasa_radiuses=cfd['sasa_rad'],
                                      res_types=cfd['res_types'],
                                      use_indices=cfd['use_indices'],
                                      params_aug=self.cfg['aug'],
                                      dst_contact=cfd['dst_contact'],
                                      test_mode=False,
                                      num_fake_iters=self.cfg['iter_per_epoch']).build()
        self.dataset_val = DHDDataset(path_idx=self.cfg['val_abs'],
                                      crop_size=self.cfg['crop_size'],
                                      use_sasa=cfd['use_sasa'],
                                      sasa_radiuses=cfd['sasa_rad'],
                                      res_types=cfd['res_types'],
                                      use_indices=cfd['use_indices'],
                                      dst_contact=cfd['dst_contact'],
                                      params_aug=None,
                                      test_mode=True).build()
        logging.info(f'Pipeline:\n\nmodel = {self.model}\n\tloss-train = {self.trn_loss}\n\tloss-val = {self.val_losses}')
        return self

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        batch = _preprocess_batch(batch)
        x = batch['inp']
        y_gt = batch['out'].type(torch.float32)
        y_pr = self.model(x)
        loss = self.trn_loss(y_pr, y_gt)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
        # return {'loss': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        batch = _preprocess_batch(batch)
        x = batch['inp']
        y_gt = batch['out'].type(torch.float32)
        y_pr = self.model(x)
        ret = {loss_name: loss_func(y_pr, y_gt) for loss_name, loss_func in self.val_losses.items()}
        return ret

    def validation_end(self, outputs):
        keys_ = list(outputs[0].keys())
        ret_avg = {f'{k}': torch.stack([x[k] for x in outputs]).mean() for k in keys_}
        tensorboard_logs = ret_avg
        ret = {'log': tensorboard_logs}
        return ret

    def configure_optimizers(self):
        ret = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        return ret

    @pl.data_loader
    def train_dataloader(self):
        ret = DataLoader(self.dataset_trn, num_workers=self.num_workers,
                         batch_size=self.cfg['batch_size'],
                         worker_init_fn=worker_init_fn_random)  # , collate_fn=skip_none_collate)
        return ret

    @pl.data_loader
    def val_dataloader(self):
        ret = DataLoader(self.dataset_val, num_workers=self.num_workers,
                         batch_size=self.cfg['batch_size_val'],
                         worker_init_fn=worker_init_fn_random)  # , collate_fn=skip_none_collate)
        return ret



def main_debug():
    logging.basicConfig(level=logging.INFO)
    path_cfg = '/home/ar/data/bioinformatics/deepdocking_experiments/homodimers/raw/cfg.json'
    pipe = DeepHDPipeline(path_cfg).build()


    print('-')


if __name__ == '__main__':
    main_debug()