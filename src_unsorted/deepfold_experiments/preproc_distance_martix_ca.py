#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import pandas as pd
import prody
import matplotlib.pyplot as plt
import scipy
import scipy.spatial.distance as sdst

if __name__ == '__main__':
    # pdb = prody.fetchPDB('6h8k')
    # path_data = '/home/ar/tmp/111/6h8k.cif.gz'
    # dat = prody.parseCIF(path_data)
    path_pdb = '/home/ar/data2/bioinformatics/alphafold_experiments/data/ecod/domain_data/01437/000143792/000143792.pdbnum.pdb'
    data_pdb = prody.parsePDB(path_pdb, subsets='ca')
    #
    coords = data_pdb.getCoords()
    dst_mat = sdst.cdist(coords, coords, 'euclidean')

    print('-')