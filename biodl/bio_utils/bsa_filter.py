from Bio.PDB import PDBParser, PDBIO
import pandas as pd
from Bio.PDB.Polypeptide import PPBuilder
from Bio import BiopythonWarning
import warnings
import os
import pickle as pkl

with warnings.catch_warnings():
    warnings.simplefilter('ignore', BiopythonWarning)


def write_surf_pkl():
    parser = PDBParser()

    path_str = '/mnt/data2t2/data/annaha/'
    #data = pd.read_csv(path_str+'pdb_raw/idx.txt', sep = ' ')
    data = pd.read_csv(path_str + 'fail_bsa.txt', sep=' ')
    print(len(data))
    wf = open(path_str+'fail_bsa2.txt','w')
    for i in range(len(data['path'])):
        #if(True):
        try:
            print(data['path'][i].lower())
            structure = parser.get_structure(data['path'][i].lower()+'_raw.pdb', path_str+'pdb_raw/'+data['path'][i].lower()+'_raw.pdb')
            data_sample = pkl.load(open(path_str+'pdb_raw/'+data['path'][i].lower()+'_raw_dumpl_cb.pkl', 'rb'))

            num = len(data_sample['coords'][0])
            models_ = list(structure.get_models())
            models = []
            j = 0
            ar = []
            for m in models_:
                for c in list(m.get_chains()):
                    j += 1
                    cur_chain = data['path'][i].lower()+c.id+str(j)
                    data_bsa = pd.read_csv(path_str+'pdb_sep/'+cur_chain+'_sasa_r3.txt', skiprows=1, header=None,sep=',')

                    if(len(data_bsa)==num):
                        list_bsa = []
                        for index, row in data_bsa.iterrows():
                            list_bsa.append(row[3])
                        ar.append(list_bsa)
                    pdb_info = {
                        'bsa_res': ar}
            with open(path_str+'bsa_res/'+cur_chain+'_sasa_r3.pkl', 'wb') as f:
                 pkl.dump(pdb_info, f)
            #return


        except:
            wf.write(data['path'][i].lower()+'\n')
            print('fail')
    wf.flush()
    wf.close()


if __name__ == '__main__':
    write_surf_pkl()