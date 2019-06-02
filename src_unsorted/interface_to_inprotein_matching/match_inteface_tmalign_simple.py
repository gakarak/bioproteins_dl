#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import re
import shutil
import copy
import json
import glob
import math
import typing
import numpy as np
import subprocess
import pandas as pd
import concurrent.futures as mp
import time
import logging
import argparse


##################################################
def _split_list_by_blocks(lst, psiz) -> list:
    tret = [lst[x:x + psiz] for x in range(0, len(lst), psiz)]
    return tret


##################################################
def get_args(params = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=str, required=True, default=None, help='path to proteins index file')
    parser.add_argument('-i', type=str, required=True, default=None, help='path to interfaces index file')
    parser.add_argument('--num_log_steps', type=int, required=False, default=5, help='#log steps')
    parser.add_argument('--odir', type=str, required=False, default=None, help='output directory')
    parser.add_argument('--batch', type=int, required=False, default=1, help='batch size for splitting proteins data per interface')
    parser.add_argument('--threads', type=int, required=False, default=1, help='#threads for processing')
    parser.add_argument('--use_process', action="store_true", help='flag, if present - use process insted threads')
    parser.add_argument('--debug', action="store_true", help='flag, if present - use debug messaging')
    #
    parser.add_argument('--run_tmalign', action="store_true", help='run task mm-align')
    parser.add_argument('--run_pproc_i', action="store_true", help='run task postrocessing mmalign data for every interface')
    parser.add_argument('--run_pproc_all', action="store_true", help='run aggregation of postrocessing mmalign data')
    parser.add_argument('--run_all', action="store_true", help='run all mmalign processing steps')
    if params is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(params)
    return args, parser


class Config(object):

    def __init__(self, args=None):
        if args is None:
            args, _ = get_args()
        self.idx_proteins = getattr(args, 'p')
        self.idx_interfaces = getattr(args, 'i')
        self.num_log_steps = getattr(args, 'num_log_steps')
        self.batch = getattr(args, 'batch')
        self.odir = getattr(args, 'odir')
        self.debug = getattr(args, 'debug')
        self.threads = getattr(args, 'threads')
        self.use_process = getattr(args, 'use_process')
        #
        self.run_tmalign = getattr(args, 'run_tmalign')
        self.run_pproc_i = getattr(args, 'run_pproc_i')
        self.run_pproc_all = getattr(args, 'run_pproc_all')
        self.run_all = getattr(args, 'run_all')
        if self.run_all:
            self.run_tmalign = True
            self.run_pproc_i = True
            self.run_pproc_all = True

    def to_json(self):
        return json.dumps(self.__dict__, indent=4)


##################################################
def helper_threaded_processing(ptr_task_function, lst_task_data, str_name_task, max_threads, use_process=True):
    num_tasks = len(lst_task_data)
    lst_task_data = [[xi, num_tasks, x] for xi, x in enumerate(lst_task_data)]
    logging.info(' [**] (start) ::{}: #tasks/#threads = {}/{}'.format(str_name_task, num_tasks, max_threads))
    t1 = time.time()
    if (num_tasks < 2) or (max_threads < 2):
        lst_ret = [ptr_task_function(xx) for xx in lst_task_data]
    else:
        if use_process:
            pool = mp.ProcessPoolExecutor(max_workers=max_threads)
        else:
            pool = mp.ThreadPoolExecutor(max_workers=max_threads)
        lst_ret = list(pool.map(ptr_task_function, lst_task_data))
        pool.shutdown(wait=True)
    dt = time.time() - t1
    logging.info('\t\tdone... dt ~ {:0.4f} (s) [{}]'.format(dt, str_name_task))
    return lst_ret


##################################################
def get_local_path_tmp(file_path, create_tmp_dir=True):
    wdir = os.path.dirname(file_path)
    tmp_dir = os.path.join(wdir, '.tmp')
    if create_tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
    tmp_file_path = os.path.join(tmp_dir, os.path.basename(file_path))
    return tmp_file_path


def __load_paths(path_csv:str) -> list:
    wdir = os.path.dirname(path_csv)
    data = pd.read_csv(path_csv)
    ret = [os.path.join(wdir, x) for x in data['path']]
    return ret


def __tmalign_i_to_p(path_i, path_p) -> typing.Tuple[dict, bytes]:
    dir_p = os.path.dirname(path_p)
    bn_p = os.path.basename(path_p)
    path_i_rel = os.path.relpath(path_i, dir_p)
    cmd = ['TMalign', bn_p, path_i_rel]
    ret = subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd=dir_p)
    out_str_raw, _ = ret.communicate()
    out_str_raw_split = out_str_raw.splitlines()
    len_aligned = int(re.findall(r'Aligned length=\s*(\d+)', out_str_raw_split[16].decode())[0])
    rmsd = float(re.findall(r'RMSD=\s+(\d+.\d*)', out_str_raw_split[16].decode())[0])
    Seq_ID = float(re.findall(r'Seq_ID=n_identical/n_aligned=\s+(\d+.\d*)', out_str_raw_split[16].decode())[0])
    #
    len_chain_1 = int(re.findall('Length of Chain_1:\s*(\d+)', out_str_raw_split[13].decode())[0])
    len_chain_2 = int(re.findall('Length of Chain_2:\s*(\d+)', out_str_raw_split[14].decode())[0])
    tm_chain_1 = float(re.findall(r'TM-score=\s+(\d+.\d*)', out_str_raw_split[17].decode())[0])
    tm_chain_2 = float(re.findall(r'TM-score=\s+(\d+.\d*)', out_str_raw_split[18].decode())[0])
    #
    seq_indices = '/'.join([str(m.start()) for m in re.finditer(':|\.', out_str_raw_split[23].decode())])
    bn_p = os.path.basename(path_p)
    bn_i = os.path.basename(path_i)
    df_data = {
        'interface':    bn_i,
        'protein':      bn_p,
        'len_aligned':  len_aligned,
        'len_chain1':   len_chain_1,
        'len_chain2':   len_chain_2,
        'tm_chain_1':   tm_chain_1,
        'tm_chain_2':   tm_chain_2,
        'rmsd':         rmsd,
        'seq_id':       Seq_ID,
        'seq_indices':  seq_indices
    }
    return df_data, out_str_raw


def __append_error(path_txt, err_msg = None):
    if err_msg is not None:
        with open(path_txt, 'a') as f:
            f.write('---\n\n')
            f.write(str(err_msg))
            f.write('\n')


def task_tmalign_interface_to_proteins_simple(pdata):
    idx, idx_num, [row_i, path_p, cfg] = pdata
    dir_i = os.path.dirname(cfg.idx_interfaces)
    dir_p = os.path.dirname(cfg.idx_proteins)
    if cfg.odir is None:
        odir = dir_i
    else:
        odir = os.path.join(cfg.odir, os.path.basename(dir_i))
    os.makedirs(odir, exist_ok=True)
    path_i = os.path.join(dir_i, row_i['path'])
    path_idx_score = os.path.join(odir, os.path.basename(path_i)) + '_score_tmalign.txt'
    path_idx_score_error = os.path.join(odir, os.path.basename(path_i)) + '_score_tmalign-errors.txt'
    if os.path.isfile(path_idx_score):
        logging.warning('*** output scoring file exist, skip... [{}]'.format(path_idx_score))
        return True
    data_p = pd.read_csv(path_p, converters={'uid': str})
    num_data_p = len(data_p)
    step_ = math.ceil(num_data_p / cfg.num_log_steps)
    t0 = time.time()
    list_df_series = []
    len_i = row_i['chains_legnth_total2']
    for xi, (_, row_p) in enumerate(data_p.iterrows()):
        len_p = row_p['chains_legnth_total']
        if cfg.debug and (len_p < len_i):
            logging.warning('\t*** protein length less than interface len_i/len_p={}/{}, skip... i=[{}], p=[{}]'
                            .format(len_i, len_p, row_i['path'], row_p['path']))
            continue
        t1 = time.time()
        path_p = os.path.join(dir_p, row_p['path'])
        try:
            df_data, _ = __tmalign_i_to_p(path_i, path_p)
            list_df_series.append(df_data)
        except Exception as err:
            str_err = ' [!!!] cant process TMAlign fpr p=[{}], i=[{}], skip... err=[{}]'.format(row_p['path'], row_i['path'], err)
            logging.error(str_err)
            __append_error(path_idx_score_error, str_err)
            continue
        if (xi % step_) == 0:
            dt = time.time() - t1
            logging.info('\t\t[{}/{}] [{}/{}] : [{}] -> [{}], dt ~ {:0.2f} (s)'.format(idx, idx_num, xi, num_data_p, row_i['path'], row_p['path'], dt))
    if len(list_df_series) > 0:
        df_ = pd.DataFrame(list_df_series, columns=list_df_series[0].keys())
        path_idx_score_tmp = get_local_path_tmp(path_idx_score)
        df_.to_csv(path_idx_score_tmp, index=False)
        shutil.move(path_idx_score_tmp, path_idx_score)
        dt = time.time() - t0
        dt_n = dt / num_data_p
        speed_ = num_data_p / dt
        logging.info('\t[{}/{}] ... done, dt ~{:0.2f} (s), #proteins={}, dt/item ~ {:0.2f} (s), speed = {:0.3f} (item/sec)'
                     .format(idx, idx_num, dt, num_data_p, dt_n, speed_))
    else:
        logging.error('Cant process any pairs for TMalign, skip... [{}]'.format(path_i))
        return False
    return True


def task_mmalign_pproc_i(pdata):
    idx, idx_num, [path_i, paths_p, num_log_steps, tmpl] = pdata
    t1 = time.time()
    dir_i = path_i + '_match'
    paths_scores = glob.glob(os.path.join(dir_i, tmpl))
    path_out_score = path_i + '-score2.txt'
    num_scores = len(paths_scores)
    num_p = len(paths_p)
    if num_scores < 1:
        logging.info(f'[{idx}/{idx_num}] *** cant find any precalculated score for interface [{path_i}]')
    else:
        step_ = math.ceil(num_scores / num_log_steps)
        df_i = None
        for ii, path_score in enumerate(paths_scores):
            #TODO: check performance: concat all list (more memory), or by iter (less memory usage)
            try:
                df_ = pd.read_csv(path_score)
            except Exception as err:
                logging.error(f'[!!!] cant read CSV file with score, skip... [{path_score}], err=[{err}]')
                continue
            if df_i is None:
                df_i = df_
            else:
                df_i = pd.concat([df_i, df_])
            if (ii % step_) == 0:
                logging.info(f'\t\t[{idx}/{idx_num}] [{ii}/{num_scores}/{num_p}] [pproc] -> [{path_score}]')
        df_i.to_csv(path_out_score, index=False)
    dt = time.time() - t1
    logging.info(f'\t[{idx}/{idx_num}] ... done, dt ~ {dt:0.2f} (s), #scores/#proteins = {num_scores}/{num_p}')
    return [num_scores, num_p]



##################################################
def run_tasks_tmaling_simple(path_i:list, path_p:list, cfg:Config):
    task_data = []
    data_i = pd.read_csv(path_i)
    for ii, row_i in data_i.iterrows():
        task_data.append([row_i, path_p, copy.copy(cfg)])
    #
    helper_threaded_processing(task_tmalign_interface_to_proteins_simple, task_data, 'tmalign i->p (simple)',
                               max_threads=cfg.threads, use_process=cfg.use_process)

def run_task_pproc_i(paths_i:list, paths_p:list, cfg:Config, tmpl = 'mmalign-*-scores.txt'):
    task_data = [[x, paths_p, cfg.num_log_steps, tmpl] for x in paths_i]
    ret_ = helper_threaded_processing(task_mmalign_pproc_i, task_data, 'pproc mmalign: interfaces',
                                      max_threads=cfg.threads, use_process=cfg.use_process)
    ret_ = np.array(ret_)
    logging.info(' ** #scores/#proteins = {}'.format(np.sum(ret_, axis=0).tolist()))


def run_task_pproc_all(paths_i:list, cfg:Config, path_scores_db):
    df_all = None
    num_i = len(paths_i)
    step_ = math.ceil(num_i / cfg.num_log_steps)
    t1 = time.time()
    for idx, path_i in enumerate(paths_i):
        path_score = path_i + '-score2.txt'
        if not os.path.isfile(path_score):
            logging.warning(f'*** cant find score file, skip... [{path_score}]')
            continue
        df_ = pd.read_csv(path_score)
        if df_all is None:
            df_all = df_
        else:
            df_all = pd.concat([df_all, df_])
        if (idx % step_) == 0:
            logging.info(f'\t[{idx}/{num_i}] ... loading [{path_i}]')
    dt = time.time() - t1
    logging.info(f' export results into [{path_scores_db}]')
    if df_all is None:
        logging.warning(' *** no valid data for interfaces, skip... (probably: no scores)')
    else:
        num_scores = len(df_all)
        df_all.to_csv(path_scores_db, index=None)
        logging.info(f'... done, dt ~ {dt:0.3f}, #interfaces/#scores = {num_i}/{num_scores}')


##################################################
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args, args_parser_ = get_args()
    cfg = Config(args)
    logging.info(f'config:\n{cfg.to_json()}')
    #
    # paths_i = __load_paths(cfg.idx_interfaces)
    # paths_p = __load_paths(cfg.idx_proteins)
    path_idx_i = cfg.idx_interfaces
    path_idx_p = cfg.idx_proteins
    if cfg.run_tmalign:
        num_i = len(pd.read_csv(path_idx_i))
        num_p = len(pd.read_csv(path_idx_p))
        logging.info('\n (1) :: TM/MM-Align processing: #interfaces/#proteins = {}/{}'.format(num_i, num_p))
        run_tasks_tmaling_simple(path_idx_i, path_idx_p, cfg)
    # if cfg.run_pproc_i:
    #     logging.info('\n (2) :: MM-Align postprocessing interfaces score: #interfaces/#proteins = {}/{}'.format(len(paths_i), len(paths_p)))
    #     run_task_pproc_i(paths_i, paths_p, cfg)
    # if cfg.run_pproc_all:
    #     logging.info('\n (3) :: MM-Align postprocessing score aggregation: #interfaces/#proteins = {}/{}'.format(len(paths_i), len(paths_p)))
    #     path_scores = cfg.idx_interfaces + '-db-scores.txt'
    #     run_task_pproc_all(paths_i, cfg, path_scores_db=path_scores)
    if not (cfg.run_tmalign or cfg.run_pproc_i or cfg.run_pproc_all):
        args_parser_.print_help()
