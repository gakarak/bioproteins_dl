import copy
import heapq
import numpy as np
from collections import deque

from Bio.PDB import PDBParser, DSSP

HELIX_CODES = ['H', 'G', 'I']
BETA_CODES = ['B', 'E']
SHORT_CODES = ['T', 'S', 'C', '-', '', ' ']
if __name__ == '__main__':
    res = []
    path = 'D:/work/bioproteins_dl/deep/alpha-fold/train_dataset/'
    protname = '6ryj'
    pdb_filename = path+''+protname+'.pdb'
    p = PDBParser()
    structure = p.get_structure(protname, pdb_filename)
    model = structure[0]
    dssp = DSSP(model, pdb_filename)
    f = open(path+protname+'.ss_sa','w')
    f.write('>'+protname+'\n')
    for residue in dssp:
        print(residue[2])
        if residue[2] in HELIX_CODES:
            res.append('H')
        elif residue[2] in BETA_CODES:
            res.append('E')
        else:
            res.append('C')
    r = ''.join(res)
    f.write(r)
    f.flush()
    print(len(r))
    f.close()
