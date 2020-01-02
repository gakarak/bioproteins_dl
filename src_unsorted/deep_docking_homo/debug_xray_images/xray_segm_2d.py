#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import skimage.io as io
import logging
from deephd_model import ASPPResNetSE
from deephd_losses import build_loss_by_name
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger


class XRay2dDataset(Dataset):

    def __init__(self, path_idx: str):
        self.path_idx = path_idx
        self.wdir = os.path.dirname(self.path_idx)
        self.data_idx = None
        self.data = None

    def build(self):
        logging.info(f':: loading data into memory [{self.path_idx}]')
        t1 = time.time()
        self.data_idx = pd.read_csv(self.path_idx)
        self.data_idx['path_img_abs'] = [os.path.join(self.wdir, x) for x in self.data_idx['path_img']]
        self.data_idx['path_msk_abs'] = [os.path.join(self.wdir, x) for x in self.data_idx['path_msk']]
        self.data = []
        for irow, row in self.data_idx.iterrows():
            img = io.imread(row['path_img_abs'])
            msk = io.imread(row['path_msk_abs'])
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
            msk = cv2.resize(msk, (128, 128), interpolation=cv2.INTER_NEAREST)
            img = np.stack([img] * 3).astype(np.float32) / 255.
            msk = (msk > 0).astype(np.float32)
            # print('-')
            self.data.append({
                'img': img,
                'msk': msk
            })
        dt = time.time() - t1
        logging.info('\t\t... done, dt ~ {:0.2f} (s), #samples = {}'.format(dt, len(self.data)))
        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class XRay2DPipeline(LightningModule):

    def __init__(self, path_trn: str, path_val: str, loss_trn='ce', batch_size=4, num_workers=1):
        super().__init__()
        self.path_trn = path_trn
        self.path_val = path_val
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model = ASPPResNetSE(inp=3, out=1, nlin='relu', num_stages=3)
        self.trn_loss = build_loss_by_name(loss_trn)
        self.val_losses = {'val_loss': build_loss_by_name(loss_trn)}
        for x in ['bcejaccard', 'jaccard']:
            self.val_losses[x] = build_loss_by_name(x)

    def build(self):
        self.dataset_trn = XRay2dDataset(self.path_trn).build()
        self.dataset_val = XRay2dDataset(self.path_val).build()
        return self

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x = batch['img']
        y_gt = batch['msk']
        y_pr = self.model(x)
        loss = self.trn_loss(y_pr, y_gt)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x = batch['img']
        y_gt = batch['msk'].type(torch.float32)
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
        ret = torch.optim.Adam(self.model.parameters(), lr=2e-4)
        return ret

    @pl.data_loader
    def train_dataloader(self):
        ret = DataLoader(self.dataset_trn, num_workers=self.num_workers, batch_size=self.batch_size)
        return ret

    @pl.data_loader
    def val_dataloader(self):
        ret = DataLoader(self.dataset_val, num_workers=self.num_workers, batch_size=4)
        return ret


def main_train():
    logging.basicConfig(level=logging.INFO)
    path_idx_trn = '/home/ar/gitlab.com/rsna-pneumonia.git/data/resources/idx_trn0.txt'
    path_idx_val = '/home/ar/gitlab.com/rsna-pneumonia.git/data/resources/idx_val0.txt'
    loss_trn = 'bce'
    batch_size = 2
    path_model = path_idx_trn + '_model_aspp_l{}_b{}'.format(loss_trn, batch_size)
    pipeline = XRay2DPipeline(path_idx_trn, path_idx_val, num_workers=2,
                              loss_trn=loss_trn, batch_size=batch_size).build()
    logging.info(pipeline.model)
    #
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(path_model, 'results'), verbose=True, monitor='val_loss', mode='min')
    logger = TestTubeLogger(save_dir=path_model, version=1)
    trainer = Trainer(default_save_path=path_model,
                      logger=logger,
                      # fast_dev_run=True,
                      # log_gpu_memory=True,
                      max_nb_epochs=100,
                      checkpoint_callback=checkpoint_callback,
                      early_stop_callback=False,
                      gpus=[0])
    trainer.fit(pipeline)
    print('-')

if __name__ == '__main__':
    main_train()