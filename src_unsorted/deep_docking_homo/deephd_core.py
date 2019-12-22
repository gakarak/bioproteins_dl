#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import pandas as pd
import prody
import matplotlib.pyplot as plt
import scipy.spatial.distance as sdst
from itertools import combinations

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio import BiopythonWarning
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', BiopythonWarning)


def read_homo_pdb_coords(path_pdb: str, calc_dst=True, pdb_parser=None) -> dict:
    if pdb_parser is None:
        pdb_parser = PDBParser()
    ppb = PPBuilder()
    models = list(pdb_parser.get_structure(os.path.basename(path_pdb), path_pdb).get_models())
    # models = list(ppb.build_peptides(pdb_parser.get_structure(os.path.basename(path_pdb), path_pdb)))
    # if len(models) != 2:
    #     raise IndexError('Invalid number of chains, required 2 chains in PDB file, but present only {}'.format(len(models)))
    models_coords = []
    models_res = []
    for m in models:
        atoms_ = [x for x in m.get_atoms() if (x.name == 'CA') and (not x.get_full_id()[3][0].strip()) ]
        res_ = [x.get_parent().resname for x in atoms_]
        ca_coords_ = np.array([x.coord for x in atoms_])
        models_coords.append(ca_coords_)
        models_res.append(res_)
    models_coords = np.stack(models_coords)
    models_res = np.stack(models_res)
    if calc_dst:
        models_dstm = np.stack([sdst.cdist(x, x, 'euclidean') for x in models_coords])
        model_combs_pw = combinations(list(range(len(models))), 2)
        models_dstm_pw = {x: sdst.cdist(models_coords[x[0]], models_coords[x[1]], 'euclidean') for x in model_combs_pw}
    else:
        models_dstm = None
        models_dstm_pw = None
    ret = {
        'coords': models_coords,
        'dst': models_dstm,
        'dst_pw': models_dstm_pw,
        'res': models_res,
        'num': len(models_coords),
        'pdb': os.path.basename(path_pdb)
    }
    return ret


def main_debug():
    # path_pdb = '/home/ar/data/bioinformatics/deepdocking_experiments/homodimers/raw/homo/1a18AA_raw.pdb'
    # path_pdb = '/home/ar/data/bioinformatics/deepdocking_experiments/homodimers/raw/homo/1a3xAA_raw.pdb'
    # path_pdb = '/home/ar/data/bioinformatics/deepdocking_experiments/homodimers/raw/homo/1f02TT_raw.pdb'
    path_pdb = '/home/ar/data/bioinformatics/deepdocking_experiments/homodimers/raw/homo/3qtcAA_raw.pdb'
    q = read_homo_pdb_coords(path_pdb)



    print('-')


if __name__ == '__main__':
    main_debug()