#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import fire

# from .dhd_data import DHDDataset
# from .dhd_model import ASPPResNetSE
# from .dhd_losses import build_loss_by_name
from .dhd_pipeline import DeepHDPipeline
#
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TensorBoardLogger



def main_train(path_cfg: str, num_workers=0):
    logging.basicConfig(level=logging.INFO)
    pipeline = DeepHDPipeline(cfg=path_cfg, num_workers=num_workers).build()
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(pipeline.path_model, 'results'),
                                          verbose=True, monitor='val_loss', mode='min')
    logger = TensorBoardLogger(save_dir=pipeline.path_model, version=1)
    t1 = time.time()
    trainer = Trainer(default_save_path=pipeline.path_model,
                      logger=logger,
                      # log_gpu_memory=True,
                      max_nb_epochs=pipeline.cfg['epochs'],
                      checkpoint_callback=checkpoint_callback,
                      early_stop_callback=False,
                      fast_dev_run=True,
                      gpus=[0])
    trainer.fit(pipeline)
    dt = time.time() - t1
    logging.info(f'\t\t... done, dt ~ {dt:0.2f} (s)')


if __name__ == '__main__':
    fire.Fire(main_train)
