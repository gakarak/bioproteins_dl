#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import os
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from .dhd_data import DHDDataset, load_config
from .dhd_model import ASPPResNetSE
from .dhd_losses import build_loss_by_name
from .dhd_pipeline import DeepHDPipeline
#
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger



def main_train():
    logging.basicConfig(level=logging.INFO)
    path_cfg = '/mnt/data4t3/data/deepdocking_experiments/homodimers/raw/cfg.json'
    cfg = load_config(path_cfg)
    pipeline = DeepHDPipeline(path_cfg, num_workers=1).build()
    # checkpoint_callback = ModelCheckpoint(filepath=os.path.join(pipeline.path_model, 'results'), verbose=True, monitor='val_loss', mode='min')
    logger = TestTubeLogger(save_dir=pipeline.path_model, version=1)
    weights_path = glob.glob(os.path.join(logger.experiment.get_logdir(), '../../../results/*.ckpt'))[0]
    # pipeline.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))['state_dict'])
    pipeline.load_state_dict(torch.load(weights_path)['state_dict'])
    model = pipeline.model.to('cuda:0')
    # dataloader_ = pipeline.val_dataloader()[0]
    dataloader_ = DataLoader(DHDDataset(cfg['trn_abs'], crop_size=cfg['crop_size'], test_mode=True).build(), num_workers=4, batch_size=1)
    len_dataloader = len(dataloader_)
    step_ = int(np.ceil(len_dataloader / 10))
    t1 = time.time()
    pipeline.eval()
    with torch.no_grad():
        print('-')
        for xi, x in enumerate(dataloader_.dataset):
            x_inp = torch.from_numpy(x['inp']).unsqueeze(dim=0).to('cuda:0')
            y_gt = x['out']
            y_pr = torch.sigmoid(model.forward(x_inp)).cpu().numpy()[0]
            plt.subplot(1, 3, 1)
            plt.imshow(y_pr)
            plt.subplot(1, 3, 2)
            plt.imshow(y_gt)
            plt.subplot(1, 3, 3)
            plt.imshow(x_inp.cpu().numpy()[0, 0])
            plt.show()
            print('-')
    dt = time.time() - t1
    logging.info(f'\t\t... done, dt ~ {dt:0.2f} (s)')


if __name__ == '__main__':
    main_train()