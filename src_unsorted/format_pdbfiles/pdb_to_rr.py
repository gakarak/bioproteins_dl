import os
import numpy as np
import pandas as pd
import prody
import matplotlib.pyplot as plt
import scipy
import scipy.spatial.distance as sdst
from Bio import SeqIO
import argparse
import logging

map_3to1 = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
    'GLY': 'G'
}

map_1to3 = {y: x for x, y in map_3to1.items()}


def get_aminos(path_pdb: str) -> str:
    pdb = prody.parsePDB(path_pdb)
    aminos = ''.join([map_3to1[x] for x in pdb.select('ca').getResnames()])
    return aminos


def get_dst_matrix_cb(path_pdb: str) -> np.ndarray:
    pdb = prody.parsePDB(path_pdb)
    coords_sel = []
    for sample in pdb:
        resname = sample.getResname()
        atomname = sample.getName()
        coord_ = sample.getCoords()
        if resname not in map_3to1.keys():
            continue
        if ((resname == 'GLY') and (atomname == 'CA')) or (atomname == 'CB'):
            coords_sel.append(coord_)
    coords_sel = np.array(coords_sel)
    dst_mat = sdst.cdist(coords_sel, coords_sel, 'euclidean')
    return dst_mat


def main_run(path_pdb: str, path_out: str = None):
    if path_out is None:
        path_out = os.path.splitext(path_pdb)[0] + '.dist.rr'
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    aminos_str = get_aminos(path_pdb)
    dst_mat = get_dst_matrix_cb(path_pdb)
    rr_mat_str = [aminos_str]
    for i in range(dst_mat.shape[0]):
        for j in range(i + 1, dst_mat.shape[1]):
            d = dst_mat[i, j]
            rr_mat_str.append(f'{i + 1} {j + 1} {d:0.2f} {d:0.2f} 1.0')
    rr_mat_str = '\n'.join(rr_mat_str)
    logging.info(aminos_str)
    logging.info(f'#amino-residues = {len(aminos_str)}')
    logging.info(f'dst-mat-shape = {dst_mat.shape}')
    logging.info(f'\t:: export rr-matrix to file [{path_out}]')
    with open(path_out, 'w') as f:
        f.write(rr_mat_str)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inp_pdb', type=str, required=True, default=None, help='Path to PDB')
    parser.add_argument('-o', '--out', type=str, required=False, default=None, help='Path to output file for rr-matrix')
    args = parser.parse_args()
    logging.info(f'args = {args}')
    #
    main_run(
        path_pdb=args.inp_pdb,
        path_out=args.out
    )
    # path = 'D:/work/bioproteins_dl/deep/alpha-fold/train_dataset/'
    # protname = '1a5i'
    # data_pdb = prody.parsePDB(path+'pdb/'+protname+'.pdb')
    # aminos = ''.join([map_3to1[x] for x in data_pdb.select('ca').getResnames()])
    # f = open(path+'rr/'+protname+'.rr','w')
    # f.write(aminos + '\n')
    # coords_sel = []
    # for sample in data_pdb:
    #     resname = sample.getResname()
    #     atomname = sample.getName()
    #     coord_ = sample.getCoords()
    #     if resname not in map_3to1.keys():
    #         continue
    #     if ((resname == 'GLY') and (atomname == 'CA')) or (atomname == 'CB'):
    #         coords_sel.append(coord_)
    # coords_sel = np.array(coords_sel)
    # dst_mat = sdst.cdist(coords_sel, coords_sel, 'euclidean')
    # for i in range(dst_mat.shape[0]):
    #     for j in range(i+1, dst_mat.shape[1]):
    #         f.write(str(i)+' '+str(j)+' '+str(round(dst_mat[i][j],2))+' '+str(round(dst_mat[i][j],2))+' 1.0'+'\n')
    # print(aminos)
    # print(f'#amino-residues = {len(aminos)}')
    # print(f'dst-mat-shape = {dst_mat.shape}')
    # f.flush()
    # f.close()
