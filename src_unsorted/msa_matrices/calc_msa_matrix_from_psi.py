#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import time
import numpy as np
import matplotlib.pyplot as plt


amino_str = ('-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y')
amino_str2idx = {x: xi for xi, x in enumerate(amino_str)}
amino_idx2str = {y: x for x, y in amino_str2idx.items()}
num_amino = 22


def map_amino_str_to_idx(seq_str: np.ndarray) -> np.ndarray:
    uniq_, inv_ = np.unique(seq_str.reshape(-1), return_inverse=True)
    uniq_map_ = np.array([amino_str2idx[x] for x in uniq_])
    ret = uniq_map_[inv_].reshape(seq_str.shape)
    return ret


def read_alignment_from_psi(path_psi: str) -> dict:
    with open(path_psi, 'r') as f:
        strs = f.readlines()
    ids_ = [x[:20] for x in strs]
    seq_str_ = [x[21:-1] for x in strs]
    map_ = {x: y for x, y in zip(ids_, seq_str_)}
    id_pdb = seq_str_[0].split(':')[0]
    #
    seq_ = np.array([list(x) for x in seq_str_])
    seq_idx_ = map_amino_str_to_idx(seq_).T
    ret = {
        'id_pdb': id_pdb,
        'id': seq_[0],
        'map': map_,
        'ids': ids_,
        'seq_str': seq_str_,
        'seq': seq_,
        'arr': seq_idx_
    }
    return ret

def calc_msa(arr: np.ndarray, use_tensordot=True) -> np.ndarray:
    arr_1h = (np.eye(num_amino)[arr.reshape(-1)]).reshape(arr.shape + (num_amino,))
    arr_1h_freq = np.mean(arr_1h, axis=1)
    arr_1hm = arr_1h - arr_1h_freq[:, None, :]
    arr_1hm = arr_1hm.transpose([0, 2, 1])
    # arr_1hm = arr_1hm[:30]
    if use_tensordot:
        ret = np.tensordot(arr_1hm, arr_1hm, [2, 2]).transpose([0, 2, 1, 3])
    else:
        ret = np.einsum('ijk,lmk->iljm', arr_1hm, arr_1hm)
    ret = ret.reshape(ret.shape[:2] + (-1,))
    return ret


def main_run():
    # path_psi = '/home/ar/data/pdb_subset_hydrolyse2k/1A5I.fasta.txt.psi'
    path_psi = '/home/ar/data/bioinformatics/pdb_subset_hydrolyse2k/1A5I.fasta.txt.psi'
    align_raw = read_alignment_from_psi(path_psi)
    #
    t1 = time.time()
    msa = calc_msa(align_raw['arr'], use_tensordot=True)
    dt = time.time() - t1
    print('msa-shape={}, dt ~ {:0.2f} (s), speed={:0.2f} (samples/s)'.format(msa.shape, dt, 1/dt))

    print('-')


if __name__ == '__main__':
    main_run()
