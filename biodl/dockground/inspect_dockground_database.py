#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import time
import logging
import numpy as np
import pandas as pd
import prody
import matplotlib.pyplot as plt
import scipy
import scipy.spatial.distance as sdst

if __name__ == '__main__':
    # logging.basicConfig(level=logging.CRITICAL)
    prody.DEPRECATION_WARNINGS = True
    path_idx1 = '/home/ar/data2/bioinformatics/dockground/full_structures_v1.1/idx.txt'
    path_idx2 = '/home/ar/data2/bioinformatics/dockground/full_structures_v1.1/idx2.txt'
    wdir = os.path.dirname(path_idx1)
    data_idx = pd.read_csv(path_idx1)
    data_idx['path_tmpl1'] = [os.path.join(wdir, x) for x in data_idx['path_tmpl1']]
    data_idx['path_tmpl2'] = [os.path.join(wdir, x) for x in data_idx['path_tmpl2']]
    #
    num_data = len(data_idx)
    err_info = []
    list_sizes = []
    t1 = time.time()
    lst_aminoacids = []
    unk_acids = []
    for irow, row in data_idx.iterrows():
    # for irow, row in data_idx.loc[:100].iterrows():
        path_tmpl1 = row['path_tmpl1']
        path_tmpl2 = row['path_tmpl2']
        try:
            tmpl1 = prody.parsePDB(path_tmpl1, subsets='ca')
            tmpl2 = prody.parsePDB(path_tmpl2, subsets='ca')
            tmp_acids = tmpl1.getResnames().tolist()
            lst_aminoacids += tmp_acids
            dct_acids = {x:y for x,y in zip(*np.unique(tmp_acids, return_counts=True))}
            if 'UNK' in dct_acids:
                unk_acids.append([dct_acids['UNK'], path_tmpl1])
                print('\t@@@ found UNK amino-acid: [{}]'.format(path_tmpl1))
            list_sizes.append([len(tmpl1), len(tmpl2)])
            print('[{}/{}] #err = {}, T1/T2 = {}/{}'.format(irow, num_data, len(err_info), len(tmpl1), len(tmpl2)))
        except Exception as err:
            err_info.append(path_tmpl1)
            print(f'**** ERROR: [{err}] -> [{path_tmpl1}]/[{path_tmpl2}]')
    dt = time.time() - t1
    print('\t... #pdb={}, dt ~ {:0.2f} (s), #pdb/s = {:0.3f} (s)'.format(num_data, dt, num_data / dt))
    tmp_ = np.unique(lst_aminoacids, return_counts=True)
    print(tmp_)
    print('\n----\nacids = {}\nfreqs = {}'.format(tmp_[0], tmp_[1] / np.sum(tmp_[1])))
    sizes = np.array(list_sizes)
    data_idx['size_tmpl1'] = sizes[:, 0]
    data_idx['size_tmpl2'] = sizes[:, 1]
    print(':: export into [{}]'.format(path_idx2))
    data_idx.to_csv(path_idx2, index=None)
    print('-')
