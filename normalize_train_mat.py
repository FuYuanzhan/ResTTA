#!/usr/bin/env python3
"""预处理：为每剂量 train_mat_{d} 的每个 subject 计算 per-subject 归一化因子 scale。

口径（见 data/normalize_common.py）：scale = LPET 全卷的 99.5 分位数，仅由 LPET
可算、不依赖 HPET。

输出：
  {out}/{d}_scale.json    ->  {"subject_id": scale, ...}
  {out}/_summary.json     ->  每剂量的 subject 数 / 缺失切片数 / 分位数 / scale 分布

用法：
  python3 normalize_train_mat.py --data-root ../data_root \
      --doses 1 2 5 10 25 50 --out normalize_scales

训练侧 LRHR_dataset 查本脚本产出的 scale 表；推理侧 load_subject_volume 在加载完整
LPET 体积后用同一 compute_scale 实时计算，二者共用同一函数，保证逐位一致。
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import scipy.io as sio

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from normalize_common import DEFAULT_PERCENTILE, compute_scale, extract_lpet_hpet

SLICES_PER_SUBJECT = 128


def list_subjects(mat_dir):
    """按 subject_id 分组，返回 {subject_id: [slice_idx, ...]}。

    subject_id = 文件名去掉末尾 ``_<idx>.mat``（文件名内可含下划线）。
    """
    subjects = defaultdict(list)
    for fname in os.listdir(mat_dir):
        if not fname.endswith('.mat'):
            continue
        stem = fname[:-4]                       # 去掉 .mat
        subject_id, _, idx_str = stem.rpartition('_')
        subjects[subject_id].append(int(idx_str))
    return subjects


def process_dose(mat_dir, percentile=DEFAULT_PERCENTILE, max_subjects=0):
    """计算一个剂量目录下所有 subject 的 scale，返回 (scales, missing)。"""
    subjects = list_subjects(mat_dir)
    ids = sorted(subjects)
    if max_subjects > 0:
        ids = ids[:max_subjects]

    scales = {}
    missing = []          # (subject_id, slice_idx, errmsg)
    wrong_count = []      # (subject_id, n_slices)

    for subject_id in ids:
        idxs = subjects[subject_id]
        if len(idxs) != SLICES_PER_SUBJECT:
            wrong_count.append((subject_id, len(idxs)))

        lpet_slices = []
        failed = False
        for idx in range(SLICES_PER_SUBJECT):
            p = os.path.join(mat_dir, f'{subject_id}_{idx}.mat')
            try:
                data = sio.loadmat(p)['img']
            except Exception as e:
                failed = True
                missing.append((subject_id, idx, str(e)[:90]))
                break
            lpet, _ = extract_lpet_hpet(data)
            lpet_slices.append(lpet)
        if failed:
            continue

        lpet_vol = np.stack(lpet_slices, axis=0).astype(np.float32)
        scales[subject_id] = compute_scale(lpet_vol, percentile=percentile)

    return scales, missing, wrong_count


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', default=os.path.join(ROOT, '..', 'data_root'),
                    help='含 train_mat_{d} 的根目录（默认 ../data_root）')
    ap.add_argument('--doses', nargs='+', default=['1', '2', '5', '10', '25', '50'],
                    help='剂量列表，对应 train_mat_{d} 目录')
    ap.add_argument('--out', default=os.path.join(ROOT, 'normalize_scales'),
                    help='scale 表输出目录')
    ap.add_argument('--percentile', type=float, default=DEFAULT_PERCENTILE,
                    help='稳健分位数（默认 99.5）')
    ap.add_argument('--max-subjects', type=int, default=0,
                    help='每剂量最多处理几个 subject（0=全部，用于冒烟测试）')
    args = ap.parse_args()

    data_root = os.path.abspath(args.data_root)
    os.makedirs(args.out, exist_ok=True)

    summary = {}
    for d in args.doses:
        mat_dir = os.path.join(data_root, f'train_mat_{d}')
        if not os.path.isdir(mat_dir):
            print(f'[skip] dose {d}: 目录不存在 {mat_dir}')
            continue

        scales, missing, wrong_count = process_dose(
            mat_dir, percentile=args.percentile, max_subjects=args.max_subjects)

        out_path = os.path.join(args.out, f'{d}_scale.json')
        with open(out_path, 'w') as f:
            json.dump(scales, f, indent=1, sort_keys=True)

        vals = list(scales.values())
        dist = ''
        if vals:
            dist = (f'min={min(vals):.4g} median={float(np.median(vals)):.4g} '
                    f'max={max(vals):.4g}')

        print(f'[done] dose {d}: {len(scales)} subjects -> {out_path}  ({dist})')

        if wrong_count:
            print(f'       !!! {len(wrong_count)} 个 subject 切片数 != {SLICES_PER_SUBJECT}: '
                  f'{wrong_count[:3]} ...')
        if missing:
            print(f'       !!! {len(missing)} 个切片加载失败（数据缺失 / 断链）:')
            for sid, idx, msg in missing[:5]:
                print(f'           {sid}_{idx}.mat  {msg}')
            if len(missing) > 5:
                print(f'           ... 共 {len(missing)} 个')

        summary[d] = {
            'n_subjects': len(scales),
            'n_missing_slices': len(missing),
            'n_wrong_slice_count': len(wrong_count),
            'percentile': args.percentile,
            'out': out_path,
        }

    summary_path = os.path.join(args.out, '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    print(f'\n汇总 -> {summary_path}')


if __name__ == '__main__':
    main()
