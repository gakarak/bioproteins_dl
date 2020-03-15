#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os


def get_temp_path(fn_out: str, root_dir: str, tmp_dir='.tmp', autocreate_dirs=True):
    if root_dir is None:
        root_dir = os.path.dirname(fn_out)
    rel_path = os.path.relpath(fn_out, root_dir)
    tmp_path = os.path.join(root_dir, tmp_dir, rel_path)
    if autocreate_dirs:
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    return tmp_path


if __name__ == '__main__':
    pass
