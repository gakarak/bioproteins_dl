#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import math
import numpy as np
import pandas as pd
import prody
import argparse
import logging
from pathlib import Path


def __append_error(path_txt, err_msg = None):
    if err_msg is not None:
        with open(path_txt, 'a') as f:
            f.write('---\n\n')
            f.write(str(err_msg))
            f.write('\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    prody.confProDy(verbosity='none', typo_warnings=False, selection_warning=False)
    prody.LOGGER._logger.setLevel(logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=str, required=True, default=None, help='path to proteins index file')
    args = parser.parse_args()
    logging.info(f'args = {args}')
    num_log_iter = 10
    path_csv = args.idx
    path_csv_out = os.path.splitext(path_csv)[0] + '-chains.txt'
    path_err_txt = os.path.splitext(path_csv)[0] + '-chains-errlog.txt'
    Path(path_err_txt).touch()
    wdir = os.path.dirname(path_csv)
    data_csv = pd.read_csv(path_csv, converters={'uid': str})
    paths_pdb = [os.path.join(wdir, x) for x in data_csv['path']]
    #
    data_csv_out = {x: [] for x in ['path2', 'chains_num', 'chains_legnth', 'chains_legnth_total']}
    step_ = math.ceil(len(paths_pdb) / num_log_iter)
    arr_is_ok = []
    for xi, x in enumerate(paths_pdb):
        try:
            pdb = prody.parsePDB(x)
            chains_num = pdb.numChains()
            chains_lengths = [len(c) for c in pdb.iterChains()]
            chains_length_total = int(np.sum(chains_lengths))
            chains_lengths_str = '/'.join([str(x) for x in chains_lengths])
            #
            data_csv_out['path2'].append(os.path.basename(x))
            data_csv_out['chains_num'].append(chains_num)
            data_csv_out['chains_legnth'].append(chains_lengths_str)
            data_csv_out['chains_legnth_total'].append(chains_length_total)
            arr_is_ok.append(True)
        except Exception as err:
            logging.error(' [!!!!] cant process file, skip... [{}], err = {}'.format(x, err))
            __append_error(path_err_txt, err_msg='pdb = {}, error = {}'.format(x, err))
            arr_is_ok.append(False)
            continue
        if (xi % step_) == 0:
            logging.info('\t[{}/{}]'.format(xi, len(paths_pdb)))
    data_csv = data_csv.iloc[np.array(arr_is_ok)>0]
    data_csv = data_csv.reset_index(drop=True)
    data_csv_out = pd.DataFrame(data=data_csv_out)
    logging.info('#data_csv/#data_csv_out = {}/{}'.format(len(data_csv), len(data_csv_out)))
    data_csv_out = pd.concat([data_csv, data_csv_out], axis=1, ignore_index=False)
    logging.info('#data_csv/#data_csv_out = {}/{}'.format(len(data_csv), len(data_csv_out)))
    #
    logging.info(':: export CSV into: [{}]'.format(path_csv_out))
    data_csv_out.to_csv(path_csv_out, index=None)
