from Bio.PDB import PDBParser, PDBIO
import pandas as pd
from Bio.PDB.Polypeptide import PPBuilder
from Bio import BiopythonWarning
import warnings
import os
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



if __name__ == '__main__':
    write_pdb_sep()