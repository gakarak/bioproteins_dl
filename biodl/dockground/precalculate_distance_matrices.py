#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import logging
import numpy as np
import pandas as pd
import prody
import matplotlib.pyplot as plt
import scipy
import scipy.spatial.distance as sdst


aacids_name2id = {
    'UNK': 0, 'ALA': 1, 'ARG': 2, 'ASN': 3, 'ASP': 4, 'CYS': 5, 'GLN': 6, 'GLU': 7,
    'GLY': 8, 'HIS': 9, 'ILE': 10, 'LEU': 11, 'LYS': 12, 'MET': 13, 'PHE': 14,
    'PRO': 15, 'SER': 16, 'THR': 17, 'TRP': 18, 'TYR': 19, 'VAL': 20
}
aacids_id2name = {v:k for k,v in aacids_name2id.items()}


def build_distance_matrix(tmpl:prody.atomic.atomgroup.AtomGroup, acid_name2idx):
    select_ = tmpl.select('ca')
    coords_ca = select_.getCoords()
    resnames = select_.getResnames()
    tmpl_residx = np.array([acid_name2idx[x] for x in resnames])
    # print('-')
    dst_mat = sdst.cdist(coords_ca, coords_ca, 'euclidean')
    return dst_mat, coords_ca


if __name__ == '__main__':
    # logging.basicConfig(level=logging.CRITICAL)
    prody.DEPRECATION_WARNINGS = True
    # path_idx1 = '/home/ar/data2/bioinformatics/dockground/full_structures_v1.1/idx.txt'
    path_idx2 = '/home/ar/data2/bioinformatics/dockground/full_structures_v1.1/idx2.txt'
    wdir = os.path.dirname(path_idx2)
    data_idx = pd.read_csv(path_idx2)
    data_idx['path_tmpl1'] = [os.path.join(wdir, x) for x in data_idx['path_tmpl1']]
    data_idx['path_tmpl2'] = [os.path.join(wdir, x) for x in data_idx['path_tmpl2']]
    #
    num_data = len(data_idx)
    err_info = []
    list_sizes = []
    for irow, row in data_idx.iterrows():
        path_tmpl1 = row['path_tmpl1']
        path_tmpl2 = row['path_tmpl2']
        tmpl1 = prody.parsePDB(path_tmpl1, subsets='ca')
        tmpl2 = prody.parsePDB(path_tmpl2, subsets='ca')
        if len(tmpl1) == len(tmpl2):
            continue
        ## tmpl1_coords_ca = tmpl1.select('ca').getCoords()
        # tmpl2_coords_ca = tmpl2.select('ca').getCoords()
        tmpl1_dstmat, tmpl1_coords = build_distance_matrix(tmpl1, acid_name2idx=aacids_name2id)
        tmpl2_dstmat, tmpl2_coords = build_distance_matrix(tmpl2, acid_name2idx=aacids_name2id)
        #
        print('[{}/{}] #err = {}, T1/T2 = {}/{}'.format(irow, num_data, len(err_info), len(tmpl1), len(tmpl2)))
    #
    print('-')
