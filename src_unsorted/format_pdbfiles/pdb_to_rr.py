import os
import numpy as np
import pandas as pd
import prody
import matplotlib.pyplot as plt
import scipy
import scipy.spatial.distance as sdst
from Bio import SeqIO

map_3to1 = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
    'GLY': 'G'
}

map_1to3 = {y: x for x, y in map_3to1.items()}


if __name__ == '__main__':
    path = 'D:/work/bioproteins_dl/deep/alpha-fold/train_dataset/'
    protname = '1a5i'
    data_pdb = prody.parsePDB(path+'pdb/'+protname+'.pdb')
    aminos = ''.join([map_3to1[x] for x in data_pdb.select('ca').getResnames()])
    f = open(path+'rr/'+protname+'.rr','w')
    f.write(aminos + '\n')
    coords_sel = []
    for sample in data_pdb:
        resname = sample.getResname()
        atomname = sample.getName()
        coord_ = sample.getCoords()
        if resname not in map_3to1.keys():
            continue
        if ((resname == 'GLY') and (atomname == 'CA')) or (atomname == 'CB'):
            coords_sel.append(coord_)
    coords_sel = np.array(coords_sel)
    dst_mat = sdst.cdist(coords_sel, coords_sel, 'euclidean')
    for i in range(dst_mat.shape[0]):
        for j in range(i+1, dst_mat.shape[1]):
            f.write(str(i)+' '+str(j)+' '+str(round(dst_mat[i][j],2))+' '+str(round(dst_mat[i][j],2))+' 1.0'+'\n')
    print(aminos)
    print(f'#amino-residues = {len(aminos)}')
    print(f'dst-mat-shape = {dst_mat.shape}')
    f.flush()
    f.close()
