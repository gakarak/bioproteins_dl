#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
#
from scipy.spatial.distance import cdist
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adam
import logging


class RotationMatrix(nn.Module):

    def __init__(self, params=None, to_device: str=None):
        super().__init__()
        if params is None:
            par_3d_mat_npy = [0, 0, 0, 0, 0, 0]
        par_3d_mat_npy = np.array(params, dtype=np.float32)
        self.par3d_mat = torch.tensor(par_3d_mat_npy, requires_grad=True, device=to_device)
        self.c0 = torch.tensor(float(0.), requires_grad=False, device=to_device)
        self.c1 = torch.tensor(float(1.), requires_grad=False, device=to_device)
        self.pc = np.pi / 180.

    def __get_c01(self) -> tuple:
        return self.c0, self.c1

    def _build_rot_x(self):
        c0, c1 = self.c0, self.c1
        ax = self.par3d_mat[0]
        cx, sx = torch.cos(self.pc * ax), torch.sin(self.pc * ax)
        rot_x = torch.stack((
            torch.stack([c1, c0, c0, c0]),
            torch.stack([c0, cx, -sx, c0]),
            torch.stack([c0, sx, cx, c0]),
            torch.stack([c0, c0, c0, c1]),
        ))
        return rot_x

    def _build_rot_y(self):
        c0, c1 = self.c0, self.c1
        ay = self.par3d_mat[1]
        cy, sy = torch.cos(self.pc * ay), torch.sin(self.pc * ay)
        rot_y = torch.stack((
            torch.stack([cy, c0, sy, c0]),
            torch.stack([c0, c1, c0, c0]),
            torch.stack([-sy, c0, cy, c0]),
            torch.stack([c0, c0, c0, c1]),
        ))
        return rot_y

    def _build_rot_z(self):
        c0, c1 = self.c0, self.c1
        az = self.par3d_mat[2]
        cz, sz = torch.cos(self.pc * az), torch.sin(self.pc * az)
        rot_z = torch.stack((
            torch.stack([cz, -sz, c0, c0]),
            torch.stack([sz, cz, c0, c0]),
            torch.stack([c0, c0, c1, c0]),
            torch.stack([c0, c0, c0, c1]),
        ))
        return rot_z

    def _build_shift_xyz(self):
        c0, c1 = self.c0, self.c1
        dx, dy, dz = self.par3d_mat[3], self.par3d_mat[4], self.par3d_mat[5]
        d_xyz = torch.stack((
            torch.stack([c1, c0, c0, dx]),
            torch.stack([c0, c1, c0, dy]),
            torch.stack([c0, c0, c1, dz]),
            torch.stack([c0, c0, c0, c1]),
        ))
        return d_xyz

    def build_3d_mat(self):
        rot_x = self._build_rot_x()
        rot_y = self._build_rot_y()
        rot_z = self._build_rot_z()
        dxyz = self._build_shift_xyz()
        #
        tmp_ = [dxyz, rot_z, rot_y, rot_x]
        mat_ = None
        for x in tmp_:
            if mat_ is None:
                mat_ = x
            else:
                mat_ = torch.mm(mat_, x)
        return mat_

    def transform_coords(self, x: torch.tensor):
        T = self.build_3d_mat()
        x_ = x.mm(T.T)
        return x_


def cdist_torch(v1: torch.tensor, v2: torch.tensor, norm=2, eps=1e-4) -> torch.tensor:
    n_1, n_2 = v1.size(0), v2.size(0)
    dim = v1.size(1)
    expanded_1 = v1.unsqueeze(1).expand(n_1, n_2, dim)
    expanded_2 = v2.unsqueeze(0).expand(n_1, n_2, dim)
    differences = torch.abs(expanded_1 - expanded_2) ** norm
    inner = torch.sum(differences, dim=2, keepdim=False)
    return (eps + inner) ** (1. / norm)


def calc_loss_dst(dst12: torch.tensor, x1: torch.tensor, x2: torch.tensor, loss_type='l1'):
    tmp_ = dst12 - cdist_torch(x1[:, :3], x2[:, :3])
    if loss_type == 'l1':
        loss_ = torch.mean(torch.abs(tmp_))
    elif loss_type == 'l2':
        loss_ = torch.mean(torch.sqrt(tmp_ ** 2))
    else:
        raise NotImplementedError
    return loss_


def calc_loss_dst_mat(dst12: torch.tensor, x1: torch.tensor, T: torch.tensor, loss_type='l1'):
    tmp_ = dst12 - cdist_torch(x1[:, :3], x1.mm(T.T))
    if loss_type == 'l1':
        loss_ = torch.mean(torch.abs(tmp_))
    elif loss_type == 'l2':
        loss_ = torch.mean(torch.sqrt(tmp_ ** 2))
    else:
        raise NotImplementedError
    return loss_


def calc_loss_t(dst_t: float, dst12: torch.tensor, x1: torch.tensor, T: torch.tensor):
    dst_t2 = dst_t * 0.5
    loss_ = torch.mean(torch.abs(
        dst_t2 - cdist_torch(x1[:, :3], x1.mm(T.T))[dst12 < dst_t]
    ))
    return loss_


def calc_RMSDT(x2: torch.tensor, x1: torch.tensor, T: torch.tensor) -> torch.tensor:
    with torch.no_grad():
        score_ = torch.sqrt( torch.mean((x2[..., :3] - x1.mm(T.T)[..., :3]) ** 2))
    return score_


def calc_RMSD(x2: torch.tensor, x1: torch.tensor) -> torch.tensor:
    with torch.no_grad():
        score_ = torch.sqrt( torch.mean((x2[..., :3] - x1[..., :3]) ** 2))
    return score_


def coords3d_to_homo(coords_3d: np.ndarray) -> np.ndarray:
    ret = np.insert(coords_3d, 3, values=1, axis=1)
    return ret


def optimize_rigidaffine_matrix(X1_gt: np.ndarray, dst_mat_gt: np.ndarray,
                                X2_gt: np.ndarray = None,
                                num_iter=20000, num_iter_eval=2000, opt_lr=0.1,
                                init_params=None,
                                dstmat_threshold=None, #TODO: add more explicit threshold-optimization, if threshold is not None, dst_mat_gt -> binary array
                                to_device: str = None) -> dict:
    if isinstance(X1_gt, np.ndarray):
        X1_gt = torch.tensor(X1_gt.astype(np.float32), requires_grad=False, device=to_device)
    if (X2_gt is not None) and isinstance(X2_gt, np.ndarray):
        X2_gt = torch.tensor(X2_gt.astype(np.float32), requires_grad=False, device=to_device)
    if isinstance(dst_mat_gt, np.ndarray):
        dst_mat_gt = torch.tensor(dst_mat_gt.astype(np.float32), requires_grad=False, device=to_device)
    if init_params is None:
        init_params = [0, 0, 0, 0, 0, 0]
    mat3d = RotationMatrix(params=init_params, to_device=to_device)
    optimizer = Adam([mat3d.par3d_mat], lr=opt_lr)
    t1 = time.time()
    for xi in range(num_iter):
        optimizer.zero_grad()
        X2_pr = mat3d.transform_coords(X1_gt)
        dst_mat_pr = cdist_torch(X1_gt, X2_pr)
        if dstmat_threshold is None:
            loss_ = torch.mean(torch.abs(dst_mat_gt - dst_mat_pr))
        else:
            loss_ = torch.mean(torch.abs(dst_mat_pr[dst_mat_gt > 0] - dstmat_threshold * 0.5))
        loss_.backward()
        optimizer.step()
        if (xi % num_iter_eval) == 0:
            with torch.no_grad():
                if X2_gt is not None:
                    T = mat3d.build_3d_mat()
                    X2_pr = X1_gt.mm(T.T)
                    RMSD = calc_RMSD(X2_pr, X2_gt)
                else:
                    RMSD = None
                logging.info('\t({}/{}) : loss ~ {:0.3f}, RMSD ~ {:0.3f}'.format(xi, num_iter, float(loss_), RMSD))
    dt = time.time() - t1
    logging.info(f'\t... done, dt ~ {dt:0.1} (s)')
    #
    with torch.no_grad():
        params = mat3d.par3d_mat.detach().cpu().numpy()
        T = mat3d.build_3d_mat().cpu().numpy()
        mtx = {
            'rotx': mat3d._build_rot_x().cpu().numpy(),
            'roty': mat3d._build_rot_y().cpu().numpy(),
            'rotz': mat3d._build_rot_z().cpu().numpy(),
            'dxyz': mat3d._build_shift_xyz().cpu().numpy()
        }
    ret = {
        'params': params,
        'T': T,
        'mtx': mtx
    }
    return ret


def main():
    logging.basicConfig(level=logging.INFO)
    to_device = 'cuda:0'
    path_sample = '/home/ar/data/bioinformatics/deep_docking_homo/10gs_raw_dumpl.pkl'
    data_sample = pkl.load(open(path_sample, 'rb'))
    X1_gt, X2_gt = data_sample['coords']
    X1_gt, X2_gt = coords3d_to_homo(X1_gt), coords3d_to_homo(X2_gt)
    coords_t = 10
    dst_mat12 = cdist(X2_gt[..., :3], X1_gt[..., :3]).astype(np.float32)
    if coords_t is not None:
        dst_mat12 = dst_mat12 < coords_t
    ret = optimize_rigidaffine_matrix(X1_gt, dst_mat12, X2_gt, to_device=to_device, dstmat_threshold=coords_t)

    print('-')
    #

    # mat3d = RotationMatrix(params=[0, 0, 0, 0, 0, 0], to_device=to_device)
    # #
    # num_ = 30
    # X1_gt_npy = 1 * np.ones((num_, 4), dtype=np.float32)
    # X2_gt_npy = 2 * np.ones((num_, 4), dtype=np.float32)
    # X2_gt_npy[:, -1] = 1
    # #
    # X1_gt = torch.tensor(X1_gt_npy, requires_grad=False, device=to_device)
    # X2_gt = torch.tensor(X2_gt_npy, requires_grad=False, device=to_device)
    # dst_mat12_npy = cdist(X1_gt_npy, X2_gt_npy).astype(np.float32)
    # dst_mat12 = torch.tensor(dst_mat12_npy, requires_grad=False, device=to_device)
    # #
    # optimizer = Adam([mat3d.par3d_mat], lr=0.1)
    # num_iter = 10000
    # for xi in range(num_iter):
    #     optimizer.zero_grad()
    #     X2_pr = mat3d.transform_coords(X1_gt)
    #     dst_ = cdist_torch(X1_gt, X2_pr)
    #     loss_ = torch.mean(torch.abs(dst_mat12 - dst_))
    #     # loss_ = calc_loss_dst(dst_mat12, X2_gt, X2_pr)
    #     loss_.backward()
    #     optimizer.step()
    #     if (xi % 100) == 0:
    #         with torch.no_grad():
    #             # print('{} : loss ~ {:0.3f}'.format(xi, float(loss_)))
    #             T = mat3d.build_3d_mat()
    #             X2_pr_ = X1_gt.mm(T.T)
    #             RMSD = calc_RMSD(X2_pr_, X2_gt)
    #             # RMSD = calc_RMSDT(X2_gt, X1_gt, T)
    #             print('{} : loss ~ {:0.3f}, RMSD ~ {:0.3f}'.format(xi, float(loss_), RMSD))
    #     # print('-')
    # print('-')


if __name__ == '__main__':
    main()