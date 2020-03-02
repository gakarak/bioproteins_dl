#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
from Bio.PDB import PDBParser
import freesasa


def main():
    p = '/home/ar/github.com/bioproteins_dl.git/data/1a5i.pdb'
    parser = PDBParser()
    structure = parser.get_structure(os.path.basename(os.path.splitext(p)[0]), p)
    #
    result, sasa_classes = freesasa.calcBioPDB(structure)

    print('-')


if __name__ == '__main__':
    main()