#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import re
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
    parser.add_argument('--batch', type=int, required=False, default=1, help='batch size for splitting proteins data per interface')
    parser.add_argument('--threads', type=int, required=False, default=1, help='#threads for processing')
    parser.add_argument('--use_process', action="store_true", help='flag, if present - use process insted threads')
    #
    parser.add_argument('--run_mmalign', action="store_true", help='run task mm-align')
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
        self.threads = getattr(args, 'threads')
        self.use_process = getattr(args, 'use_process')
        #
        self.run_mmalign = getattr(args, 'run_mmalign')
        self.run_pproc_i = getattr(args, 'run_pproc_i')
        self.run_pproc_all = getattr(args, 'run_pproc_all')
        self.run_all = getattr(args, 'run_all')
        if self.run_all:
            self.run_mmalign = True
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
def __load_paths(path_csv:str) -> list:
    wdir = os.path.dirname(path_csv)
    data = pd.read_csv(path_csv)
    ret = [os.path.join(wdir, x) for x in data['path']]
    return ret


def __mmalign_i_to_p(path_i, path_p) -> typing.Tuple[dict, bytes]:
    cmd = ['MMalign', path_p, path_i, '-b']
    ret = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out_str_raw, _ = ret.communicate()
    # LEN, RMSD, TMSCORE, ID = re.findall('\d+[.]?\d+', out_str_raw.splitlines()[10].decode())
    LEN, RMSD, TMSCORE, ID = [float(x.split(',')[0]) for x in out_str_raw.splitlines()[10].decode().split('=')[1:]]
    bn_p = os.path.basename(path_p)
    bn_i = os.path.basename(path_i)
    df_data = {
        'interface': [bn_i],
        'protein': [bn_p],
        'len': [int(LEN)],
        'rmsd': [float(RMSD)],
        'tmscore': [float(TMSCORE)],
        'ID': [float(ID)]
    }
    return df_data, out_str_raw


def task_mmalign_interface_to_proteins(pdata):
    idx, idx_num, [path_i, paths_p, num_log_steps] = pdata
    path_i_out_dir = path_i + '_match'
    os.makedirs(path_i_out_dir, exist_ok=True)
    #
    if not isinstance(paths_p, list):
        paths_p = [paths_p]
    batch_size = len(paths_p)
    step_ = math.ceil(batch_size / num_log_steps)
    bn_i = os.path.basename(path_i)
    t0 = time.time()
    for xi, path_p in enumerate(paths_p):
        bn_p = os.path.basename(path_p)
        path_out_mmalign = os.path.join(path_i_out_dir, f'mmalign-{bn_i}-{bn_p}.txt')
        path_out_mmalign_scores = os.path.join(path_i_out_dir, f'mmalign-{bn_i}-{bn_p}-scores.txt')
        if os.path.isfile(path_out_mmalign_scores):
            # logging.warning(f'[{idx}/{idx_num}] *** out exist, skip... [{path_out_mmalign_scores}]')
            continue
        else:
            t1 = time.time()
            try:
                df_data, str_raw_ = __mmalign_i_to_p(path_i, path_p)
            except Exception as err:
                logging.error(' [!!!] cant process MMAlign fpr p=[{}], i=[{}], skip... err=[{}]'.format(path_p, path_i, err))
                continue
            with open(path_out_mmalign, 'wb') as f:
                f.write(str_raw_)
            df_ = pd.DataFrame(data=df_data)
            df_.to_csv(path_out_mmalign_scores, index=False)
            dt = time.time() - t1
            if (xi % step_) == 0:
                logging.info('\t\t[{}/{}] [{}/{}] : [{}] -> [{}], dt ~ {:0.2f} (s)'.format(idx, idx_num, xi, batch_size, bn_i, bn_p, dt))
    dt = time.time() - t0
    dt_n = dt / batch_size
    speed_ = batch_size / dt
    logging.info('\t[{}/{}] ... done, dt ~{:0.2f} (s), #batch={}, dt/item ~ {:0.2f} (s), speed = {:0.3f} (item/sec)'
                 .format(idx, idx_num, dt, batch_size, dt_n, speed_))


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
def run_tasks_mmaling(paths_i:list, paths_p:list, cfg:Config):
    task_data = []
    paths_p_split = _split_list_by_blocks(paths_p, cfg.batch)
    for ii, path_i in enumerate(paths_i):
        task_data += [[path_i, paths_p, cfg.num_log_steps] for paths_p in paths_p_split]
    #
    helper_threaded_processing(task_mmalign_interface_to_proteins, task_data, 'mmalign i->p',
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
    paths_i = __load_paths(cfg.idx_interfaces)
    paths_p = __load_paths(cfg.idx_proteins)
    if cfg.run_mmalign:
        logging.info('\n (1) :: MM-Align processing: #interfaces/#proteins = {}/{}'.format(len(paths_i), len(paths_p)))
        run_tasks_mmaling(paths_i, paths_p, cfg)
    if cfg.run_pproc_i:
        logging.info('\n (2) :: MM-Align postprocessing interfaces score: #interfaces/#proteins = {}/{}'.format(len(paths_i), len(paths_p)))
        run_task_pproc_i(paths_i, paths_p, cfg)
    if cfg.run_pproc_all:
        logging.info('\n (3) :: MM-Align postprocessing score aggregation: #interfaces/#proteins = {}/{}'.format(len(paths_i), len(paths_p)))
        path_scores = cfg.idx_interfaces + '-db-scores.txt'
        run_task_pproc_all(paths_i, cfg, path_scores_db=path_scores)
    if not (cfg.run_mmalign or cfg.run_pproc_i or cfg.run_pproc_all):
        args_parser_.print_help()
