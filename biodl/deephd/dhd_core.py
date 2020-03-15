#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import copy
import numpy as np
import pandas as pd
import prody
import matplotlib.pyplot as plt
import scipy.spatial.distance as sdst
from itertools import combinations

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio import BiopythonWarning
from Bio.PDB.PDBExceptions import PDBConstructionException, PDBConstructionWarning
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', PDBConstructionException)
    warnings.simplefilter('ignore', PDBConstructionWarning)
    warnings.simplefilter('ignore', BiopythonWarning)
    

def read_homo_pdb_coords_cacb(path_pdb: str, calc_dst=True,
                              return_models=False, pdb_parser=None) -> dict:
    if pdb_parser is None:
        pdb_parser = PDBParser()
    ppb = PPBuilder()
    models_ = list(pdb_parser.get_structure(os.path.basename(path_pdb), path_pdb).get_models())
    models = []
    for m in models_:
        for c in list(m.get_chains()):
            models.append(c)
    # models = list(ppb.build_peptides(pdb_parser.get_structure(os.path.basename(path_pdb), path_pdb)))
    # if len(models) != 2:
    #     raise IndexError('Invalid number of chains, required 2 chains in PDB file, but present only {}'.format(len(models)))
    models_coords_ca, models_coords_cb = [], []
    models_res = []
    for m in models:
        atoms_ca = [x for x in m.get_atoms() if (x.name == 'CA') and (not x.get_full_id()[3][0].strip()) ]
        atoms_cb = [x for x in m.get_atoms() if
            (
                ((x.parent.resname=='GLY') and (x.name == 'CA')) 
                or (x.name == 'CB')
            ) 
            and 
            (not x.get_full_id()[3][0].strip()) 
        ]
        res_ = [x.get_parent().resname for x in atoms_ca]
        coords_ca_ = np.array([x.coord for x in atoms_ca])
        coords_cb_ = np.array([x.coord for x in atoms_cb])
        models_coords_ca.append(coords_ca_)
        models_coords_cb.append(coords_cb_)
        models_res.append(res_)
    models_coords_ca = np.stack(models_coords_ca).astype(np.float32)
    models_coords_cb = np.stack(models_coords_cb).astype(np.float32)
    models_res = np.stack(models_res)
    if calc_dst:
        models_dstm_ca = np.stack([sdst.cdist(x, x, 'euclidean') for x in models_coords_ca]).astype(np.float16)
        models_dstm_cb = np.stack([sdst.cdist(x, x, 'euclidean') for x in models_coords_cb]).astype(np.float16)
        # model_combs_pw = combinations(list(range(len(models))), 2)
        # models_dstm_pw = {x: sdst.cdist(models_coords[x[0]], models_coords[x[1]], 'euclidean') for x in model_combs_pw}
        models_dstm_pw_ca, models_dstm_pw_cb = None, None
    else:
        models_dstm = None
        models_dstm_pw_ca = None
        models_dstm_pw_cb = None
    if not return_models:
        models = None
    ret = {
        'coords_ca': models_coords_ca,
        'coords_cb': models_coords_cb,
        'dst_ca': models_dstm_ca,
        'dst_cb': models_dstm_cb,
        'dst_pw_ca': models_dstm_pw_ca,
        'dst_pw_cb': models_dstm_pw_cb,
        'res': models_res,
        'len_ca': len(models_coords_ca[0]),
        'len_cb': len(models_coords_cb[0]),
        'num_ca': len(models_coords_ca),
        'num_cb': len(models_coords_cb),
        'pdb': os.path.basename(path_pdb),
        'models': models
    }
    return ret


def read_homo_pdb_coords(path_pdb: str, calc_dst=True, pdb_parser=None) -> dict:
    if pdb_parser is None:
        pdb_parser = PDBParser()
    ppb = PPBuilder()
    models_ = list(pdb_parser.get_structure(os.path.basename(path_pdb), path_pdb).get_models())
    models = []
    for m in models_:
        for c in list(m.get_chains()):
            models.append(c)
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
    # path_pdb = '/home/ar/data/bioinformatics/deepdocking_experiments/homodimers/raw/homo/3qtcAA_raw.pdb'
    path_pdb = '/home/ar/data/bioinformatics/deepdocking_experiments/homodimers/raw/homo/2cjxAA_raw.pdb'
    q = read_homo_pdb_coords(path_pdb)



    print('-')


if __name__ == '__main__':
    main_debug()