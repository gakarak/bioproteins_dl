#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import pandas as pd
import re


if __name__ == '__main__':
    path_db1 = '/home/ar/data2/bioinformatics/alphafold_experiments/ecod.latest.domains_flt1.txt'
    path_db2 = '/home/ar/data2/bioinformatics/alphafold_experiments/ecod.latest.domains_flt2.txt'
    path_db2_trn_tmpl = '/home/ar/data2/bioinformatics/alphafold_experiments/ecod.latest.domains_flt2_trn_s{}.txt'
    path_db2_val_tmpl = '/home/ar/data2/bioinformatics/alphafold_experiments/ecod.latest.domains_flt2_val_s{}.txt'
    data_db1 = pd.read_csv(path_db1, delimiter='\t')
    #
    re_str = r'^\d+\.\d+\.\d+\.\d+$'
    data_db2 = data_db1[data_db1['f_id'].str.contains(re_str)]
    data_db2.to_csv(path_db2, sep='\t', index=None)
    #
    p_val = 30
    num_splits = 3
    for idx_split in range(num_splits):
        path_db2_trn = path_db2_trn_tmpl.format(idx_split)
        path_db2_val = path_db2_val_tmpl.format(idx_split)
        uniq_idx = np.array(data_db2['f_id'].unique())
        num_uniq_idx = len(uniq_idx)
        num_uniq_idx_trn = int((100 - p_val) * num_uniq_idx / 100)
        #
        idx_all = np.random.permutation(uniq_idx)
        idx_all_trn = idx_all[:num_uniq_idx_trn]
        idx_all_val = idx_all[num_uniq_idx_trn:]
        #
        data_db2_trn = data_db2[data_db2['f_id'].isin(idx_all_trn)]
        data_db2_val = data_db2[data_db2['f_id'].isin(idx_all_val)]
        #
        data_db2_trn.to_csv(path_db2_trn, sep='\t', index=None)
        data_db2_val.to_csv(path_db2_val, sep='\t', index=None)
        print('[{}/{}] --> {}'.format(idx_split, num_splits, path_db2_trn))
        # print('-')
