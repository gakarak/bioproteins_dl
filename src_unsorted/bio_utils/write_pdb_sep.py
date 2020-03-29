from Bio.PDB import PDBParser, PDBIO
import pandas as pd
from Bio.PDB.Polypeptide import PPBuilder
from Bio import BiopythonWarning
import warnings
import os
import numpy as np
import pickle as pkl
with warnings.catch_warnings():
    warnings.simplefilter('ignore', BiopythonWarning)


def write_pdb_sep():
    parser = PDBParser()
    io = PDBIO()
    path_str = '/mnt/data2t2/data/annaha/'
    data = pd.read_csv(path_str+'pdb_raw/idx.txt', sep = ' ')
    print(len(data))
    wfile = open(path_str + 'idx_chains.txt','w')
    for i in range(len(data['path'])):

        try:
            structure = parser.get_structure(data['path'][i].lower()+'_raw.pdb', path_str+'pdb_raw/'+data['path'][i].lower()+'_raw.pdb')

            models_ = list(structure.get_models())
            models = []
            j = 0
            ar = []
            for m in models_:
                for c in list(m.get_chains()):
                    #models.append(c)
                    ar.append(data['path'][i].lower()+c.id+str(j)+'.pdb')
                    io.set_structure(c)
                    j+=1
                    io.save(path_str+'pdb_sep/'+data['path'][i].lower()+c.id+str(j)+'.pdb')
                    #print(path_str+'pdb_sep/'+data['path'][i].lower()+c.id+str(j)+'.pdb')
            wfile.write(data['path'][i].lower()+' '+ar[0]+'+'+ar[1]+'\n')
        except:
            print('fail')

def write_sasa_sep():
    parser = PDBParser()
    io = PDBIO()
    path_str = '/mnt/data2t2/data/annaha/'
    data = pd.read_csv(path_str+'pdb_raw/idx.txt', sep = ' ')
    print(len(data))
    wfile = open(path_str + 'failed_sasa_pkl.txt','w')
    for i in range(len(data['path'])):
        #if(True):
        try:
            structure = parser.get_structure(data['path'][i].lower()+'_raw.pdb', path_str+'pdb_raw/'+data['path'][i].lower()+'_raw.pdb')

            models_ = list(structure.get_models())
            models = []
            j = 1
            ar = []
            for m in models_:
                for c in list(m.get_chains()):
                    #models.append(c)
                    modelssub_ = list(parser.get_structure('bsa_res/atoms/'+data['path'][i].lower()+c.id+str(j)+'_sasa_.pdb',path_str+'bsa_res/atoms/'+data['path'][i].lower()+c.id+str(j)+'_sasa_.pdb').get_models())
                    modelssub = []
                    for msub in modelssub_:
                        for csub in list(msub.get_chains()):
                            modelssub.append(csub)


                    sasa = []
                    cur_chain = data['path'][i].lower() + c.id + str(j)
                    for msub in modelssub:
                        atoms_ = [x for x in msub.get_atoms() if (x.name == 'CA') and (not x.get_full_id()[3][0].strip())]
                        atoms_ = [x for x in msub.get_atoms() if
                                  ((x.get_parent().resname == 'GLY' and x.name == 'CA') or (x.name == 'CB')) and (
                                      not x.get_full_id()[3][0].strip())]

                        sasa_coords_ = np.array([x.bfactor for x in atoms_])
                        sasa.append(sasa_coords_)
                    j+=1

                    #io.save(path_str+'pdb_sep/'+data['path'][i].lower()+c.id+str(j)+'.pdb')
                    #print(path_str+'pdb_sep/'+data['path'][i].lower()+c.id+str(j)+'.pdb')
            #wfile.write(data['path'][i].lower()+' '+ar[0]+'+'+ar[1]+'\n')
                    pdb_info={'sasa': sasa}
                    #print(pdb_info)
                    with open(path_str + 'bsa_res/atoms_pkl_cb/' + cur_chain + '_sasa.pkl', 'wb') as f:
                        pkl.dump(pdb_info, f)
        except:
            #break
            wfile.write(data['path'][i].lower() + '\n')
            #print('fail')


if __name__ == '__main__':
    write_sasa_sep()