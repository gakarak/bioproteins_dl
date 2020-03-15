from Bio.PDB import PDBParser
import pandas as pd
from Bio.PDB.Polypeptide import PPBuilder
from Bio import BiopythonWarning
import warnings
import os
import freesasa

with warnings.catch_warnings():
    warnings.simplefilter('ignore', BiopythonWarning)


def bsa_calc():
    data = pd.read_csv('/home/annaha/bioinf/data/auc_cfg_21_02_cb_cont8.txt', sep=' ')
    parser = PDBParser()

    wf = open('/home/annaha/bioinf/data/cb_train_bsatemp2.txt', 'w')
    for i in range(len(data)):
        try:
            structure = parser.get_structure(data['pdb'][i], '/home/annaha/bioinf/data/pdb_raw/' + data['pdb'][i])
            result, sasa_classes = freesasa.calcBioPDB(structure)

            models_ = list(structure.get_models())
            models = []

            for m in models_:
                for c in list(m.get_chains()):
                    models.append(c)
            result1, sasa_classes = freesasa.calcBioPDB(models[0])
            result2, sasa_classes = freesasa.calcBioPDB(models[1])
            bsa = (result2.totalArea() + result1.totalArea() - result.totalArea()) / 2.0

            wf.write(data['pdb'][i] + ' ' + str(bsa) + '\n')

        except:
            print('fail')

    wf.flush()
    wf.close()


if __name__ == '__main__':
    bsa_calc()