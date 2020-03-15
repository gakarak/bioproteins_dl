#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import matplotlib.pyplot as plt

from Bio.PDB import PDBIO
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder


def main_run():
    path_pdb = '/home/ar/tmp/111/19hc_raw.pdb'
    pdb_parser = PDBParser()
    models_ = list(pdb_parser.get_structure(os.path.basename(path_pdb), path_pdb).get_models())
    models = []
    for m in models_:
        for c in list(m.get_chains()):
            models.append(c)
    #
    ppb = PPBuilder()
    for mi, m in enumerate(models):
        path_model_out = os.path.splitext(path_pdb)[0] + f'_dumpl_model_{mi}.pdb'
        # if os.path.isfile(path_model_out):
        #     continue
        pdb_io = PDBIO()
        # ms = ppb.build_peptides(m)
        pdb_io.set_structure(m)
        # pdb_io.set_structure(ms)
        pdb_io.save(path_model_out)
        print('-')
    print('-')


if __name__ == '__main__':
    main_run()