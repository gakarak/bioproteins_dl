#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import sys
import glob
import copy
import pickle as pkl
import shutil
import numpy as np
import pandas as pd
import prody
import matplotlib.pyplot as plt
import scipy.spatial.distance as sdst
from biodl.deephd import read_homo_pdb_coords_cacb
from biodl.bio_utils import parallel_tasks_run_def, get_temp_path
import logging
import fire

from Bio.PDB import PDBIO
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder


def prepare_coords(path_pdb: str, build_separate_models=True) -> bool:
    tmp_root_dir = os.path.dirname(os.path.dirname(path_pdb))
    path_out = os.path.splitext(path_pdb)[0] + '_dumpl_cacb.pkl'
    if os.path.isfile(path_out):
        logging.warning(f'!!! output file exst, skip... [{path_out}]')
        return True
    try:
        tmp_ = read_homo_pdb_coords_cacb(path_pdb, return_models=True)
        if len(tmp_['models']) > 2:
            raise NotImplementedError('#models > 2: [{}]'.format(len(tmp_['models'])))
        if tmp_['len_ca'] != tmp_['len_cb']:
            raise NotImplementedError('#ca <> #cb, ca/ab = {}{}'.format( tmp_['len_ca'], tmp_['len_cb'] ))
        if build_separate_models:
            ppb = PPBuilder()
            for mi, m in enumerate(tmp_['models']):
                path_model_out = os.path.splitext(path_pdb)[0] + f'_dumpl_model_{mi}.pdb'
                if os.path.isfile(path_model_out):
                    continue
                path_model_out_tmp = get_temp_path(path_model_out, root_dir=tmp_root_dir)
                pdb_io = PDBIO()
                pdb_io.set_structure(copy.deepcopy(m.copy()))
                # ms = ppb.build_peptides(m)
                pdb_io.save(path_model_out_tmp)
                shutil.move(path_model_out_tmp, path_model_out)
        del tmp_['models']
        #
        path_out_tmp = get_temp_path(path_out, root_dir=tmp_root_dir)
        with open(path_out_tmp, 'wb') as f:
            pkl.dump(tmp_, f)
        shutil.move(path_out_tmp, path_out)
        return True
    except Exception as err:
        logging.info('\t\terr=[{}] <- [{}]'.format(err, path_pdb))
        return False
        


def main_debug_pproc(path_idx: str, jobs=1):
    # path_idx = '/mnt/data1T/data/annaha/homod/idx-pdb-raw.txt'
    path_log = path_idx + '_log.txt'
    logging.basicConfig(level=logging.INFO, filename=path_log,
                        format='%(asctime)s %(name)s %(levelname)s:%(message)s',
                        handlers=[
                            logging.FileHandler(path_log),
                            logging.StreamHandler()
                        ])
    wdir = os.path.dirname(path_idx)
    paths_pdb = [os.path.join(wdir, x) for x in pd.read_csv(path_idx)['path']]
    task_data = [{'path_pdb': x} for x in paths_pdb]
    ret = parallel_tasks_run_def(prepare_coords, task_data, num_workers=jobs)
    print('-')


if __name__ == '__main__':
    fire.Fire(main_debug_pproc)
    
