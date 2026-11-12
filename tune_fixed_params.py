#!/usr/bin/env python3
"""
验证集固定超参调参脚本 —— 路径 A 的第①步
==========================================
在【带 GT 的验证集】上为每个剂量搜索出全局固定的 TTA 超参 (lr_mid, start_ratio)。

为什么合法：验证集 ≠ 测试集。这里在验证集上用 GT（hpet）搜索是标准做法；
得到的固定值随后通过 inference.py 的 `--fixed_lr/--fixed_ratio`
用于测试阶段，测试时全程零 GT。

默认从 ``data_root/train_mat_{dose}`` 抽取前 N 个 subject 作为验证集
（train 数据自带 hpet）。复用 inference.py 里的全部 TTA 组件
（load_subject_volume / build_pixel_bank / tta_infer_slice），保证与主流程
逐位一致。

验证集必须与测试集 subject 隔离。可用 ``--val_dir`` 显式指定数据目录，例如：
  - ``--val_dir ../data_root/train_mat``  （有 hpet 的训练集）
  - 或先从 ``test_mat_{dose}`` 划一个 val 子集（与报告的 test subject 隔离）。

用法（单 checkpoint 跨所有剂量，类似官方模型）:
    python tune_fixed_params.py --checkpoint /path/I500000_E93 \
        --doses 1,2,5,10,25,50 --val_subjects 4 --device cuda:0

用法（每剂量一个 checkpoint，用 bash 循环，同 run_tta_all_doses.sh 的 CKPT 表）:
    for d in 1 5 10 25 50; do
        python tune_fixed_params.py --checkpoint "${CKPT[$d]}" --doses "$d" \
            --val_subjects 4 --device cuda:0
    done

输出：每个剂量的推荐固定 (lr, ratio) + 每个候选的平均 PSNR 表（可 --out_json 落盘）。
"""

import os
import sys
import argparse
import json
import math

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import inference as rc


def _psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))


def tune_dose(checkpoint, dose, val_dir, val_subjects, lr_values, start_ratios,
              search_slices, K_mid, K_late, lam, sigma, sigma_est,
              pixelbank_ws, pixelbank_nn, pixelbank_nr, ms_scales,
              device, device_id, seed):
    """在 dose 的验证集 val_dir 上跑 (lr, ratio) 网格，返回汇总 dict 或 None。"""
    config_dose = dose if dose != '2' else '1'
    config_path = os.path.join(rc.DEFAULT_CONFIG_DIR, f'sr_sr3_{config_dose}_128.json')
    if not os.path.exists(config_path):
        print(f'  [SKIP] config not found: {config_path}')
        return None

    data_dir = val_dir
    if not os.path.isdir(data_dir):
        print(f'  [SKIP] val data dir not found: {data_dir}')
        return None

    # 收集验证 subject（前 val_subjects 个）
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])
    subjects = {}
    for f in files:
        sid = f.rsplit('_', 1)[0]
        subjects.setdefault(sid, []).append(f)
    subject_ids = sorted(subjects.keys())[:val_subjects]
    print(f'  Val subjects ({len(subject_ids)}): {subject_ids}')

    print(f'  Loading model from {checkpoint} ...')
    diffusion, _ = rc.load_model_for_dose(config_path, checkpoint, device_id)

    # 聚合每个 (lr, ratio) 候选的 PSNR
    scores = {(lr, r): [] for lr in lr_values for r in start_ratios}
    baseline_scores = []

    for abs_idx, sid in enumerate(tqdm(subject_ids, desc=f'  dose {dose}')):
        # 逐 subject 绝对序号种子：与 inference.py 的 subj_seed = seed + subject_idx*128
        # 完全一致（这里 subject_ids 是 sorted(...)[:val_subjects]，abs_idx 即绝对下标 0..3）。
        subj_seed = seed + abs_idx * 128
        lpet_vol, hpet_vol = rc.load_subject_volume(data_dir, sid)

        if ms_scales:
            pixel_bank = rc.build_pixel_bank_multi_scale(
                lpet_vol, scales=ms_scales, device=device, sigma=sigma)
        else:
            pixel_bank = rc.build_pixel_bank_indices(
                lpet_vol, window_size=pixelbank_ws,
                num_neighbors=pixelbank_nn, num_rows=pixelbank_nr,
                device=device, sigma=sigma)

        n_search = min(search_slices, lpet_vol.shape[0])
        for s in range(n_search):
            pb_s = rc._extract_pb_slice(pixel_bank, s)
            pb_t = rc._pb_to_device(pb_s, device)

            # baseline（TTA 关闭）
            final_bl, _, _, hr_np = rc.tta_infer_slice(
                diffusion, lpet_vol[s], hpet_vol[s], lpet_vol, hpet_vol, s,
                pb_t, 0.0, 0.0, 0.0, 0, 0, 0.0,
                sigma_est=0.0, num_ensemble=1, base_seed=subj_seed)
            baseline_scores.append(_psnr(final_bl, hr_np))

            # 网格搜索 TTA 参数（num_ensemble=1）
            for lr in lr_values:
                for ratio in start_ratios:
                    final_np, _, _, _ = rc.tta_infer_slice(
                        diffusion, lpet_vol[s], hpet_vol[s], lpet_vol, hpet_vol, s,
                        pb_t, ratio, lr, lr * 0.1, K_mid, K_late, lam,
                        sigma_est=sigma_est,
                        num_neighbors=pixelbank_nn, window_size=pixelbank_ws,
                        num_ensemble=1, base_seed=subj_seed)
                    scores[(lr, ratio)].append(_psnr(final_np, hr_np))

        del pixel_bank
        torch.cuda.empty_cache()

    del diffusion
    torch.cuda.empty_cache()

    bl_avg = float(np.mean(baseline_scores))
    table = {f'lr={lr:.2e},r={r}': float(np.mean(scores[(lr, r)]))
             for lr in lr_values for r in start_ratios}
    best_params, best_avg = max(
        ((k, float(np.mean(v))) for k, v in scores.items()),
        key=lambda kv: kv[1])
    return {'dose': dose, 'baseline_psnr': bl_avg,
            'best_lr': best_params[0], 'best_ratio': best_params[1],
            'best_psnr': best_avg, 'table': table}


def main():
    p = argparse.ArgumentParser(description='验证集固定超参调参（路径 A 第①步）')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--doses', type=str, default='1,2,5,10,25,50')
    p.add_argument('--val_dir', type=str, default=None,
                   help='验证集数据目录（含 hpet 的 .mat）。默认按剂量取 '
                        'data_root/train_mat_{dose}；也可显式指定 '
                        '（如 --val_dir ../data_root/train_mat 或某个 test_mat 子集）')
    p.add_argument('--val_subjects', type=int, default=4,
                   help='每个剂量抽取的验证 subject 数（默认 4）')
    p.add_argument('--lr_values', type=str, default='1e-2,5e-2,1e-1')
    p.add_argument('--start_ratios', type=str, default='1.0')
    p.add_argument('--search_slices', type=int, default=5)
    p.add_argument('--lam', type=float, default=0.0)
    p.add_argument('--K_mid', type=int, default=10)
    p.add_argument('--K_late', type=int, default=20)
    p.add_argument('--pixelbank_ws', type=int, default=17)
    p.add_argument('--pixelbank_nn', type=int, default=16)
    p.add_argument('--pixelbank_nr', type=int, default=20)
    p.add_argument('--use_hybrid_pb', action='store_true', default=True)
    p.add_argument('--no-use_hybrid_pb', dest='use_hybrid_pb', action='store_false')
    p.add_argument('--no-bilateral', action='store_true', default=False)
    p.add_argument('--hybrid_sigma_ratio', type=float, default=0.3)
    p.add_argument('--ms_pb_scales', type=str, default='9,17,33')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_json', type=str, default=None,
                   help='把结果写成 JSON（默认仅打印）')
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device_id = int(args.device.split(':')[-1]) if ':' in args.device else 0
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    lr_values = [float(x) for x in args.lr_values.split(',')]
    start_ratios = [float(x) for x in args.start_ratios.split(',')]
    doses = [d.strip() for d in args.doses.split(',')]

    ms_scales = None
    if args.ms_pb_scales.strip():
        ms_scales = []
        for ws in [int(x) for x in args.ms_pb_scales.split(',')]:
            if ws <= 9:
                ms_scales.append((ws, 8, 8))
            elif ws <= 17:
                ms_scales.append((ws, 16, min(20, ws)))
            else:
                ms_scales.append((ws, 24, 20))

    results = {}
    for dose in doses:
        print(f'\n===== Tuning dose {dose} =====')
        dose_int = int(dose) if dose.isdigit() else 10
        sigma = rc.SIGMA_MAP.get(dose_int, rc.DEFAULT_SIGMA)
        if args.no_bilateral:
            sigma = 0.0
        sigma_est = sigma * args.hybrid_sigma_ratio if args.use_hybrid_pb else 0.0
        val_dir = args.val_dir or os.path.join(rc.DEFAULT_DATA_ROOT, f'train_mat_{dose}')
        r = tune_dose(
            args.checkpoint, dose, val_dir, args.val_subjects, lr_values, start_ratios,
            args.search_slices, args.K_mid, args.K_late, args.lam,
            sigma, sigma_est, args.pixelbank_ws, args.pixelbank_nn,
            args.pixelbank_nr, ms_scales, device, device_id, args.seed)
        if r is not None:
            results[dose] = r
            print(f'  dose {dose}: baseline={r["baseline_psnr"]:.4f}  '
                  f'=> best(lr={r["best_lr"]:.2e}, r={r["best_ratio"]}) = '
                  f'{r["best_psnr"]:.4f}')
            for k, v in sorted(r['table'].items()):
                print(f'      {k}: {v:.4f}')

    print('\n' + '=' * 72)
    print('  推荐固定超参（传给 --fixed_lr / --fixed_ratio）')
    print('=' * 72)
    for dose, r in sorted(results.items(), key=lambda kv: int(kv[0])):
        print(f'  dose {dose:>3s}%: baseline PSNR={r["baseline_psnr"]:.4f}  '
              f'=>  FIXED lr={r["best_lr"]:.2e}, ratio={r["best_ratio"]}  '
              f'(val PSNR={r["best_psnr"]:.4f})')
    print('=' * 72)

    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  Saved -> {args.out_json}')


if __name__ == '__main__':
    main()
