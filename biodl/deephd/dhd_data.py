#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import time
import json
import numpy as np
import pandas as pd
import pickle as pkl
import logging
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from biodl.bio_utils import parallel_tasks_run_def
import scipy.spatial.distance as sdst
import skimage.io as io



all_res = ('UNK', 'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
           'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU',
           'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR',
           'TRP', 'TYR', 'VAL')

map_res2idx = {x: xi for xi, x in enumerate(all_res)}
map_idx2res = {y: x for x, y in map_res2idx.items()}


def load_config(path_cfg: str) -> dict:
    cfg = json.load(open(path_cfg, 'r'))
    wdir = os.path.dirname(path_cfg)
    cfg['wdir'] = wdir
    cfg['trn_abs'] = os.path.join(wdir, cfg['trn'])
    cfg['val_abs'] = os.path.join(wdir, cfg['val'])
    return cfg


def get_pairwise_res_1hot_matrix(res: np.ndarray, res2idx: dict = None) -> np.ndarray:
    if res2idx is None:
        res2idx = map_res2idx
    res_idx = np.array([res2idx[x] for x in res])
    num_res = len(res2idx)
    X, Y = np.meshgrid(res_idx, res_idx)
    shp_inp = X.shape
    X = np.eye(num_res)[X.reshape(-1)]
    Y = np.eye(num_res)[Y.reshape(-1)]
    X = X.reshape(shp_inp + (num_res,))
    Y = Y.reshape(shp_inp + (num_res,))
    XY = np.dstack([X, Y])
    return XY


def _load_idx(path_idx: str) -> pd.DataFrame:
    wdir_ = os.path.dirname(path_idx)
    data_ = pd.read_csv(path_idx)
    keys_ = data_.keys()
    for k in keys_:
        data_[f'{k}_abs'] = [os.path.join(wdir_, x) for x in data_[k]]
    return data_


def load_one_sample(row: pd.Series, load_ca: bool, load_cb: bool, load_sasa: bool, sasa_radiuses: tuple) -> dict:
    data_ca = pkl.load(open(row['path_ca_abs'], 'rb')) if load_ca else None
    data_cb = pkl.load(open(row['path_cb_abs'], 'rb')) if load_cb else None
    if load_sasa:
        data_sasa = pd.read_csv(row['path_sasa_abs'])
        print('-')
    else:
        data_sasa = None
    return {
        'ca': data_ca,
        'cb': data_cb,
        'sasa': data_sasa
    }


class DHDDataset(Dataset):

    def __init__(self, path_idx: str, crop_size: int, params_aug: dict = None,
                 num_fake_iters=100, test_mode=False, test_mode_crop=False,
                 use_ca=True, use_cb=True, use_sasa=True,
                 sasa_radiuses=(3, ), crop_coef=2**5):
        self.path_idx = path_idx
        self.params_aug = params_aug
        self.test_mode = test_mode
        self.test_mode_crop = test_mode_crop
        self.crop_size = crop_size
        self.data = None
        self.num_fake_iters = num_fake_iters
        #
        self.use_ca = use_ca
        self.use_cb = use_cb
        self.use_sasa = use_sasa
        self.sasa_radiuses = sasa_radiuses
        self.crop_coef = crop_coef

    def build(self):
        self.wdir = os.path.dirname(self.path_idx)
        self.data_idx = _load_idx(self.path_idx)
        t1 = time.time()
        logging.info('\t::load dataset into memory, #samples = {}'.format(len(self.data_idx)))
        # self.data = [pkl.load(open(x, 'rb')) for x in self.data_idx['path_abs']]
        self.data = [load_one_sample(row, load_ca=self.use_ca, load_cb=self.use_cb,
                                     load_sasa=self.use_cb, sasa_radiuses=self.sasa_radiuses)
                     for _, row in self.data_idx.iterrows()]
        self.data = [x for x in self.data if (x['res'].shape[1] > self.crop_size)
                     and (len(set(x['res'][0]) - set(all_res)) < 1)
                     and (len(set(x['res'][1]) - set(all_res)) < 1)]
        dt = time.time() - t1
        logging.info('\t\t\t... done, dt ~ {:0.2f} (s), #samples={} with size >= {}'
                     .format(dt, len(self.data), self.crop_size))
        return self

    def __len__(self):
        if self.test_mode:
            return len(self.data)
        else:
            return self.num_fake_iters

    def __get_aug_coords(self, coords: np.ndarray, aug_params: dict = None) -> np.ndarray:
        if aug_params is None:
            ret = coords
        else:
            dxyz = np.random.uniform(*aug_params['shift_xyz'], coords.shape)
            ret = coords + dxyz
        return ret

    def __get_distance_mat(self, sample: dict, aug_params: dict = None) -> dict:
        res_pw = get_pairwise_res_1hot_matrix(sample['res'][0]).astype(np.float32)
        x1, x2 = sample['coords']
        x1 = self.__get_aug_coords(x1, aug_params=aug_params)
        x2 = self.__get_aug_coords(x2, aug_params=aug_params)
        x1 /= 10.
        x2 /= 10.
        dst_x1x1 = sdst.cdist(x1, x1).astype(np.float32)
        dst_x1x2 = sdst.cdist(x1, x2).astype(np.float32)# < 1.4
        inp = np.dstack([dst_x1x1[..., None], res_pw])
        ret = {
            'inp': inp,
            'out': dst_x1x2,
            'pdb': sample['pdb']
        }
        return ret

    def _get_random_crop(self, dst_info: dict, crop_size: int) -> dict:
        nrc = dst_info['inp'].shape[0]
        if crop_size < nrc:
            rr, cc = np.random.randint(0, nrc - crop_size, 2)
            inp_crop = dst_info['inp'][rr: rr + crop_size, cc: cc + crop_size, ...]
            out_crop = dst_info['out'][rr: rr + crop_size, cc: cc + crop_size, ...]
        else:
            inp_crop = dst_info['inp']
            out_crop = dst_info['out']
        ret = {
            'inp': inp_crop.transpose((2, 0, 1)),
            'out': out_crop,
            'pdb': dst_info['pdb']
        }
        return ret

    def __getitem__(self, item):
        if self.test_mode:
            sample = self.data[item]
        else:
            rnd_idx = np.random.randint(0, len(self.data))
            sample = self.data[rnd_idx]
        dst_info = self.__get_distance_mat(sample, aug_params=self.params_aug)
        if not self.test_mode:
            dst_info = self._get_random_crop(dst_info, self.crop_size)
        else:
            dst_info = self._get_random_crop(dst_info, crop_size=len(sample['res'][0]))
        return dst_info


def main_run():
    logging.basicConfig(level=logging.INFO)
    # path_idx = '/home/ar/data/bioinformatics/deepdocking_experiments/homodimers/raw/idx-okl.txt'
    # path_cfg = '/home/ar/data/bioinformatics/deepdocking_experiments/homodimers/raw/cfg.json'
    # path_cfg = '/mnt/data4t3/data/deepdocking_experiments/homodimers/raw/cfg.json'
    path_cfg = '/home/ar/data/bioinformatics/deep_hd/cfg.json'
    cfg = load_config(path_cfg)
    dataset = DHDDataset(path_idx=cfg['trn_abs'],
                         crop_size=cfg['crop_size'],
                         params_aug=cfg['aug'],
                         test_mode=True).build()
    for xi, x in enumerate(dataset):
        print('inp-shape/out-shape = {}/{}'.format(x['inp'].shape, x['out'].shape))
        plt.subplot(1, 2, 1)
        plt.imshow(x['inp'][0])
        plt.subplot(1, 2, 2)
        plt.imshow(x['out'])
        plt.show()
    print('-')


if __name__ == '__main__':
    main_run()