#!/usr/bin/env python3
"""把 train 数据预加载为连续 float32 memmap，供 LRHRDataset 内存索引。

目的：消除每个样本 ~19 次 scipy.io.loadmat 的磁盘解析开销（当前训练的主要瓶颈）。

输出（写到 /mnt，避免占满根分区）：
  {out}/{dose}/data.bin       float32 (n_subjects, 128, 256, 128)
                              dims = (subject, slice, row[0:128=lpet / 128:256=hpet], col)
  {out}/{dose}/subjects.json  subject_id 列表（与 data.bin 第 0 维对齐，按字母序）

用法：
  python3 preload_train_data.py --mat-dir ../data_root/train_mat_1 \
      --dose 1 --out ../preload

路径可用环境变量 ``PET_PRELOAD_DIR`` 覆盖（推理/训练侧读取同一变量）。
"""

import argparse
import json
import os
import sys

import numpy as np
import scipy.io as sio

SLICES = 128


def list_subjects(mat_dir):
    subs = set()
    for f in os.listdir(mat_dir):
        if f.endswith('.mat'):
            subs.add(f.rsplit('_', 1)[0])
    return sorted(subs)   # 字母序 == get_paths_from_images 的最终 sorted(images) 顺序


def build(mat_dir, dose, out_dir, verbose=True):
    subjects = list_subjects(mat_dir)
    n = len(subjects)
    os.makedirs(out_dir, exist_ok=True)

    data_path = os.path.join(out_dir, 'data.bin')
    subj_path = os.path.join(out_dir, 'subjects.json')

    mm = np.memmap(data_path, dtype=np.float32, mode='w+',
                   shape=(n, SLICES, 256, 128))
    for si, sid in enumerate(subjects):
        for idx in range(SLICES):
            p = os.path.join(mat_dir, f'{sid}_{idx}.mat')
            img = sio.loadmat(p)['img']          # (1, 256, 128)
            mm[si, idx] = img[0]                 # (256, 128) float32
        if verbose and (si % 20 == 0 or si == n - 1):
            print(f'  subject {si + 1}/{n} done', flush=True)

    mm.flush()
    del mm
    with open(subj_path, 'w') as f:
        json.dump(subjects, f, indent=1)
    print(f'[preload] dose {dose}: {n} subjects -> {data_path}')
    print(f'           size {os.path.getsize(data_path) / 1e9:.2f} GB, subjects -> {subj_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mat-dir', required=True)
    ap.add_argument('--dose', required=True, help='剂量标签，用于输出子目录命名')
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'preload'))
    args = ap.parse_args()
    build(args.mat_dir, args.dose, os.path.join(args.out, str(args.dose)))


if __name__ == '__main__':
    main()
