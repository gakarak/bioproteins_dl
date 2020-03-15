#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import pandas as pd
import scipy.spatial.distance as sdst
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import prody
from prody import showProtein


map_3to1 = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
    'XXX': 'X', '---': '-'
}

map_1to3 = {y:x for x, y in map_3to1.items()}


def main_run():
    path_pdb = '/home/ar/github.com/bioproteins_dl.git/data/1a5i.pdb'
    pdb = prody.parsePDB(path_pdb, subset='ca')
    coords = pdb.getCoords()
    dst_mat = sdst.cdist(coords, coords, 'euclidean')
    aminos3 = pdb.getResnames()
    aminos1 = [map_3to1[x] for x in aminos3]

    print('-')
    #
    plt.figure()
    plt.subplot(1, 2, 1, projection='3d')
    plt.plot(coords[:, 0], coords[:, 1], coords[:, 2], '-o')
    plt.title('ca-atoms: {}'.format(os.path.basename(path_pdb)))
    plt.subplot(1, 2, 2)
    plt.imshow(dst_mat)
    plt.title('distance-matrix')
    plt.show()

    print('-')



if __name__ == '__main__':
    main_run()