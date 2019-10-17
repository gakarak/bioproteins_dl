#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import matplotlib.pyplot as plt


def read_alignment_from_psi(path_psi: str) -> dict:
    with open(path_psi, 'r') as f:
        strs = f.readlines()
    ids_ = [x[:20] for x in strs]
    seq_ = [x[21:] for x in strs]
    map_ = {x: y for x, y in zip(ids_, seq_)}
    id_pdb = seq_[0].split(':')[0]
    ret = {
        'id_pdb': id_pdb,
        'id': seq_[0],
        'map': map_,
        'ids': ids_,
        'seq': seq_
    }
    return ret


def main_run():
    path_psi = '/home/ar/data/pdb_subset_hydrolyse2k/1A5I.fasta.txt.psi'
    align_raw = read_alignment_from_psi(path_psi)

    print('-')


if __name__ == '__main__':
    main_run()
