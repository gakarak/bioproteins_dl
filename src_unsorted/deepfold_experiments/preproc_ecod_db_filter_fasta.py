#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import pandas as pd
import re

from Bio import SeqIO

if __name__ == '__main__':
    path_db_flt = '/home/ar/data2/bioinformatics/alphafold_experiments/ecod.latest.domains_flt2.txt'
    path_fasta_inp = '/home/ar/data2/bioinformatics/alphafold_experiments/ecod.latest.fasta.txt'
    #
    data_db = pd.read_csv(path_db_flt, sep='\t')
    gen_fasta = SeqIO.parse(path_fasta_inp, 'fasta')
    for x in gen_fasta:

        print('-')
