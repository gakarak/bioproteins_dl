#from Bio.PDB import PDBParser
import pandas as pd
from Bio import SeqIO
import Bio
from Bio.PDB.Polypeptide import PPBuilder
from Bio import BiopythonWarning
import warnings
import os
#import freesasa
import glob
import os
import glob
import Bio
from Bio import SeqIO
import pandas as pd
from Bio import pairwise2

with warnings.catch_warnings():
    warnings.simplefilter('ignore', BiopythonWarning)

def filter_dataset():

    path_str = '/mnt/data2t2/data/annaha/'
    wf = open(path_str + 'seq_sim_max.txt', 'w')
    data = pd.read_csv(path_str + 'idx-okl_cb_val0_sh_fix.txt', sep=' ')
    files = data['path']

    wf.write('pdb2 score' + '\n')
    for i in range(len(files)):
        pdb2 = files[i][-21:-17]
        ar = calc_seq_sim(pdb2)
        res = max(ar)
        print(max(ar))
        wf.write(pdb2 + ' ' + str(res) + '\n')

    wf.flush()
    wf.close()


def calc_seq_sim(pdb2):

    path_str = '/mnt/data2t2/data/annaha/'
    wf = open(path_str+'seq_sim'+pdb2+'temp.txt', 'w')
    data = pd.read_csv(path_str + 'idx-okl_cb_trn0_sh_fix.txt', sep=' ')
    files = data['path']
    f2 = glob.glob("".join(['/mnt/data2t2/data/annaha/fasta/' + pdb2, "*.fa"]))
    s2 = SeqIO.read(f2[0], "fasta")
    #wf.write('pdb1 pdb2 score'+'\n')
    ar = []
    for i in range(len(files)):

        pdb1 = files[i][-21:-17]

        f1 = glob.glob("".join(['/mnt/data2t2/data/annaha/fasta/' + pdb1, "*.fa"]))
        s1 = SeqIO.read(f1[0], "fasta")
        alignments = pairwise2.align.globalxx(s1, s2)
        res = float(alignments[0][2])/max(len(s1),len(s2))
        #print(res)

        ar.append(res)
        if(res>=0.7):
            return ar
        #wf.write(pdb1+' '+pdb2+' '+str(res)+'\n')

    #wf.flush()
    #wf.close()
    return ar

def tofasta():
    path_str = '/mnt/data2t2/data/annaha/'

    data = pd.read_csv(path_str+'idx-okl_cb_val0_sh_fix.txt', sep=' ')

    files = data['path']



    for i in range(len(files)):
        try:
            print(files[i])
            pdb = files[i][-21:-17]
            print(path_str + pdb + '.pdb')
            sequences = list(Bio.SeqIO.parse(path_str + 'pdb_raw/' + pdb + '_raw.pdb', 'pdb-atom'))

            print(sequences)

            for s in sequences:
                s.id = pdb + s.id[5:]
                SeqIO.write(s, path_str + 'fasta/' + s.id + '.fa', 'fasta');
        except:
            print('exception')



if __name__ == '__main__':
    filter_dataset()
    #calc_seq_sim('1ika')