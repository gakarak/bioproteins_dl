#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import sys
import glob
from pympler.asizeof import asizeof
import numpy as np
import pandas as pd
import prody
import matplotlib.pyplot as plt
import scipy.spatial.distance as sdst
from itertools import combinations
from biodl.deephd import read_homo_pdb_coords_cacb
from biodl.bio_utils import parallel_tasks_run_def
import logging

from Bio.PDB import PDBIO
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio import BiopythonWarning
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', BiopythonWarning)


def main_debug_pproc():
    logging.basicConfig(level=logging.INFO)

    print('-')


if __name__ == '__main__':
    main_debug_pproc()
