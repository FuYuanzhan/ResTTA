#!/usr/bin/env python3
"""ResTTA inference: baseline and residual TTA over multiple dose levels.

For each dose the script loads the low-dose/full-dose volumes, builds the
pixel banks, runs (optionally) the PreNet refinement, then samples the residual
with or without TTA.  Predictions are saved as ``.img`` volumes and evaluated
with ``evaluate.py``'s protocol.

Two hyper-parameter modes are supported:

* ``--fixed_lr`` (recommended, test-time ground-truth free): use hyper-parameters
  selected on a validation set with ``tune_fixed_params.py``.
* no ``--fixed_lr``: per-subject grid search against the ground truth.  This is
  an oracle upper bound for diagnostics only and must not be used to report
  test numbers.

Example:
    python inference.py --checkpoint checkpoints/I500000_E93 \
        --doses 1,5,10,25,50 --fixed_lr 0.05 --fixed_ratio 1.0 \
        --output_root outputs/restta
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from common import (DEFAULT_CONFIG_DIR, DEFAULT_DATA_ROOT, DEFAULT_SIGMA,
                    SIGMA_MAP, compute_3d_context, compute_spectrum,
                    load_subject_volume)
from evaluate import (evaluate_results_dir, metrics_mean_std,
                      write_per_subject_csv)
from model_loader import load_model_for_dose
from pixel_bank import (build_pixel_bank_indices, build_pixel_bank_multi_scale,
                        scales_from_sizes)
from refinement import refine_prenet_ip, tta_p_sample_loop


def _extract_pb_slice(pb_input, s):
    """Extract slice ``s`` from a single-scale or multi-scale pixel bank."""
    if isinstance(pb_input, dict):
        return {k: torch.from_numpy(v[:, s, :, :]).long()
                for k, v in pb_input.items()}
    return torch.from_numpy(pb_input[:, s, :, :]).long()


def _pb_to_device(pb_input, device):
    if isinstance(pb_input, dict):
        return {k: v.to(device) for k, v in pb_input.items()}
    return pb_input.to(device)


def compute_prenet_ip_volume(diffusion, lpet_vol, hpet_vol, device,
                             zero_hpet_guide=False):
    """Run PreNet slice by slice to obtain the full coarse image volume."""
    D, H, W = lpet_vol.shape
    ip_vol = np.zeros((D, H, W), dtype=np.float32)

    model = diffusion
    for n in ("netguide_3D_1", "netguide_3D_2", "netguide_3D_3",
              "netguide_spectrum_1", "netguide_spectrum_2",
              "netguide_spectrum_3", "netP"):
        getattr(model, n).eval()

    for s in range(D):
        sr_t = torch.from_numpy(lpet_vol[s]).float()[None, None].to(device)
        l3d = compute_3d_context(lpet_vol, s).unsqueeze(0).to(device)
        h3d = compute_3d_context(hpet_vol, s).unsqueeze(0).to(device)
        lp = compute_spectrum(lpet_vol[s]).unsqueeze(0).to(device)
        hp = compute_spectrum(hpet_vol[s]).unsqueeze(0).to(device)

        if zero_hpet_guide:
            h3d = torch.zeros_like(h3d)
            hp = torch.zeros_like(hp)

        with torch.no_grad():
            _ = model.netguide_3D_1(l3d, h3d, t=None)
            _ = model.netguide_3D_2(l3d, h3d, t=None)
            _ = model.netguide_3D_3(l3d, h3d, t=None)
            _ = model.netguide_spectrum_1(lp, hp, t=None)
            _ = model.netguide_spectrum_2(lp, hp, t=None)
            _ = model.netguide_spectrum_3(lp, hp, t=None)

            ax_feature = [model.netguide_3D_1.get_feature(),
                          model.netguide_3D_2.get_feature(),
                          model.netguide_3D_3.get_feature()]
            fr_feature = [model.netguide_spectrum_1.get_feature(),
                          model.netguide_spectrum_2.get_feature(),
                          model.netguide_spectrum_3.get_feature()]

            ip = model.netP(sr_t, time=None, ax_feature=ax_feature,
                            fr_feature=fr_feature)
            ip_vol[s] = ip.detach().squeeze().cpu().numpy()

    return ip_vol


def tta_infer_slice(model, lpet_slice, hpet_slice, lpet_vol, hpet_vol,
                    slice_idx, pb_input, constraint_start_ratio,
                    lr_mid, lr_late, K_mid, K_late, lam,
                    sigma_est=0.0, num_neighbors=16, window_size=17,
                    freq_low_radius=16, freq_low_scale=0.3,
                    num_ensemble=1, base_seed=42,
                    zero_hpet_guide=False,
                    tta_target="denoisenet",
                    prenet_K=0, prenet_lr=0.0, prenet_pb_scale="medium"):
    """Run one slice through guide networks, PreNet and residual sampling.

    Args:
        tta_target: ``'denoisenet'`` refines only the residual sampling,
            ``'prenet'`` refines only the coarse prediction ``IP``, ``'both'``
            refines ``IP`` first and then the residual (coupled).
        num_ensemble: number of random seeds averaged (1 disables ensembling).

    Returns:
        ``(final_result, residual, IP, ground_truth)`` as numpy arrays.
    """
    device = next(model.netP.parameters()).device

    # Bit-exact reproducibility for a given (subject, slice): the caller passes
    # base_seed = seed + subject_abs_idx * 128.
    _seed = int(base_seed) + int(slice_idx)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    np.random.seed(_seed)

    sr_t = torch.from_numpy(lpet_slice).float()[None, None].to(device)
    l3d = compute_3d_context(lpet_vol, slice_idx).unsqueeze(0).to(device)
    h3d = compute_3d_context(hpet_vol, slice_idx).unsqueeze(0).to(device)
    lp = compute_spectrum(lpet_slice).unsqueeze(0).to(device)
    hp = compute_spectrum(hpet_slice).unsqueeze(0).to(device)

    # Diagnostic: zeroing the HPET guide inputs should leave the result
    # unchanged if HPET does not leak into inference.
    if zero_hpet_guide:
        h3d = torch.zeros_like(h3d)
        hp = torch.zeros_like(hp)

    model.netguide_3D_1.eval()
    model.netguide_3D_2.eval()
    model.netguide_3D_3.eval()
    model.netguide_spectrum_1.eval()
    model.netguide_spectrum_2.eval()
    model.netguide_spectrum_3.eval()

    _, _ = model.netguide_3D_1(l3d, h3d, t=None)
    _, _ = model.netguide_3D_2(l3d, h3d, t=None)
    _, _ = model.netguide_3D_3(l3d, h3d, t=None)
    _, _ = model.netguide_spectrum_1(lp, hp, t=None)
    _, _ = model.netguide_spectrum_2(lp, hp, t=None)
    _, _ = model.netguide_spectrum_3(lp, hp, t=None)

    ax_feature = [model.netguide_3D_1.get_feature(),
                  model.netguide_3D_2.get_feature(),
                  model.netguide_3D_3.get_feature()]
    fr_feature = [model.netguide_spectrum_1.get_feature(),
                  model.netguide_spectrum_2.get_feature(),
                  model.netguide_spectrum_3.get_feature()]

    model.netP.eval()
    with torch.no_grad():
        IP = model.netP(sr_t, time=None, ax_feature=ax_feature,
                        fr_feature=fr_feature)

    if tta_target in ("prenet", "both") and prenet_K > 0 and prenet_lr > 0:
        IP = refine_prenet_ip(
            IP, pb_input, prenet_K, prenet_lr,
            freq_low_radius=freq_low_radius, freq_low_scale=freq_low_scale,
            pb_scale=prenet_pb_scale)

    use_denoise_tta = (tta_target != "prenet") and (
        constraint_start_ratio > 0 and lr_mid > 0)

    if num_ensemble > 1:
        rand_state = torch.get_rng_state()
        cuda_rand_state = (torch.cuda.get_rng_state()
                           if torch.cuda.is_available() else None)
        np_rand_state = np.random.get_state()

    model.netG.eval()

    if num_ensemble > 1:
        residuals = []
        for ens in range(num_ensemble):
            ens_seed = base_seed + ens * 1000
            torch.manual_seed(ens_seed)
            torch.cuda.manual_seed(ens_seed)
            np.random.seed(ens_seed)

            if use_denoise_tta:
                residual = tta_p_sample_loop(
                    model.netG, sr_t, ax_feature, fr_feature, pb_input,
                    constraint_start_ratio=constraint_start_ratio,
                    lr_mid=lr_mid, lr_late=lr_late,
                    K_mid=K_mid, K_late=K_late, lam=lam,
                    IP=IP, sigma_est=sigma_est,
                    num_neighbors=num_neighbors, window_size=window_size,
                    freq_low_radius=freq_low_radius,
                    freq_low_scale=freq_low_scale)
            else:
                with torch.no_grad():
                    residual = model.netG.super_resolution(
                        sr_t, ax_feature, fr_feature, continous=False)
            residuals.append(residual)

        residual = torch.stack(residuals).mean(dim=0)
        torch.set_rng_state(rand_state)
        if cuda_rand_state is not None:
            torch.cuda.set_rng_state(cuda_rand_state)
        np.random.set_state(np_rand_state)
    else:
        if use_denoise_tta:
            residual = tta_p_sample_loop(
                model.netG, sr_t, ax_feature, fr_feature, pb_input,
                constraint_start_ratio=constraint_start_ratio,
                lr_mid=lr_mid, lr_late=lr_late,
                K_mid=K_mid, K_late=K_late, lam=lam,
                IP=IP, sigma_est=sigma_est,
                num_neighbors=num_neighbors, window_size=window_size,
                freq_low_radius=freq_low_radius,
                freq_low_scale=freq_low_scale)
        else:
            with torch.no_grad():
                residual = model.netG.super_resolution(
                    sr_t, ax_feature, fr_feature, continous=False)

    final_result = (residual + IP).squeeze().cpu().numpy()
    return (final_result, residual.squeeze().cpu().numpy(),
            IP.squeeze().cpu().numpy(), hpet_slice)


def process_subject(diffusion, lpet_vol, hpet_vol, subject_idx, subject_id,
                    output_dir_bl, output_dir_tta,
                    pixel_bank_ws, pixel_bank_nn, pixel_bank_nr,
                    search_slices, lr_values, start_ratios,
                    K_mid, K_late, lam, device, sigma=1.5,
                    sigma_est=0.0, fallback_ratio=1.0,
                    ms_scales=None, freq_low_radius=16, freq_low_scale=0.3,
                    num_ensemble=1, seed=42,
                    fixed_lr=None, fixed_ratio=None,
                    zero_hpet_guide=False,
                    tta_target="denoisenet", prenet_K=0, prenet_lr=0.0,
                    prenet_pb_scale="medium", fixed_lr_late=None,
                    pb_source="lpet"):
    """Run baseline and TTA inference for one subject and save ``.img`` files.

    ``fixed_lr`` enables the ground-truth-free fixed-hyper-parameter mode and
    skips the oracle search.  Otherwise a per-subject grid search is run on the
    first ``search_slices`` slices and the best setting is applied to the rest.
    """
    D, H, W = lpet_vol.shape
    subj_seed = seed + subject_idx * 128

    t0 = time.time()
    if pb_source == "prenet":
        pb_src_vol = compute_prenet_ip_volume(
            diffusion, lpet_vol, hpet_vol, device,
            zero_hpet_guide=zero_hpet_guide)
    else:
        pb_src_vol = lpet_vol

    if ms_scales:
        pixel_bank = build_pixel_bank_multi_scale(
            pb_src_vol, scales=ms_scales, device=device, sigma=sigma)
    else:
        pixel_bank = build_pixel_bank_indices(
            pb_src_vol, window_size=pixel_bank_ws,
            num_neighbors=pixel_bank_nn, num_rows=pixel_bank_nr,
            device=device, sigma=sigma)
    pb_time = time.time() - t0

    n_search = min(search_slices, D)
    n_eval_start = n_search

    full_fallback = False
    if fixed_lr is not None:
        best_lr = fixed_lr
        best_ratio = fixed_ratio if fixed_ratio is not None else 1.0
        best_lr_late = (fixed_lr_late if fixed_lr_late is not None
                        else best_lr * 0.1)
        print(f"    Subject {subject_idx}: pb={pb_time:.1f}s, "
              f"FIXED(lr={best_lr:.2e}, lr_late={best_lr_late:.2e}, "
              f"r={best_ratio}) (no search, eval slices "
              f"{n_eval_start}..{D - 1})")
    else:
        # Oracle per-subject grid search (diagnostic upper bound; uses GT).
        baseline_psnrs = []
        param_scores = {}

        for s in range(n_search):
            pb_t = _pb_to_device(_extract_pb_slice(pixel_bank, s), device)

            final_np, _, _, hr_np = tta_infer_slice(
                diffusion, lpet_vol[s], hpet_vol[s], lpet_vol, hpet_vol, s,
                pb_t, 0.0, 0.0, 0.0, 0, 0, 0.0,
                sigma_est=0.0, num_ensemble=1, base_seed=subj_seed,
                zero_hpet_guide=zero_hpet_guide)
            mse = np.mean((final_np - hr_np) ** 2)
            baseline_psnrs.append(
                float("inf") if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse)))

            for lr in lr_values:
                for ratio in start_ratios:
                    final_np, _, _, _ = tta_infer_slice(
                        diffusion, lpet_vol[s], hpet_vol[s], lpet_vol,
                        hpet_vol, s, pb_t, ratio, lr, lr * 0.1,
                        K_mid, K_late, lam,
                        sigma_est=sigma_est,
                        num_neighbors=pixel_bank_nn,
                        window_size=pixel_bank_ws,
                        num_ensemble=1, base_seed=subj_seed,
                        zero_hpet_guide=zero_hpet_guide,
                        tta_target=tta_target,
                        prenet_K=prenet_K, prenet_lr=prenet_lr,
                        prenet_pb_scale=prenet_pb_scale)
                    mse = np.mean((final_np - hr_np) ** 2)
                    param_scores.setdefault((lr, ratio), []).append(
                        float("inf") if mse == 0
                        else 20 * math.log10(1.0 / math.sqrt(mse)))

        param_avg = {k: np.mean(v) for k, v in param_scores.items()}
        best_params, best_avg_psnr = max(param_avg.items(), key=lambda x: x[1])
        best_lr, best_ratio = best_params
        bl_avg = np.mean(baseline_psnrs)

        print(f"    Subject {subject_idx}: pb={pb_time:.1f}s, "
              f"BL_PSNR(search)={bl_avg:.4f}, "
              f"best(lr={best_lr:.2e}, r={best_ratio})={best_avg_psnr:.4f}")

        if best_avg_psnr <= bl_avg:
            best_lr, best_ratio = 0.0, 0.0
            full_fallback = True
            print("      => no improvement, falling back to baseline")
        elif fallback_ratio < 1.0:
            best_lr_orig = best_lr
            best_lr = best_lr * fallback_ratio
            print(f"      => partial fallback: lr {best_lr_orig:.2e} -> "
                  f"{best_lr:.2e} (ratio={fallback_ratio})")

    if fixed_lr is None:
        best_lr_late = best_lr * 0.1

    epet_bl = np.zeros([D, H, W], dtype=np.float32)
    spet_bl = np.zeros([D, H, W], dtype=np.float32)
    epet_tta = np.zeros([D, H, W], dtype=np.float32)
    spet_tta = np.zeros([D, H, W], dtype=np.float32)
    cnt = 0

    for s in range(n_eval_start, D):
        pb_t = _pb_to_device(_extract_pb_slice(pixel_bank, s), device)

        final_bl, _, _, hr_np = tta_infer_slice(
            diffusion, lpet_vol[s], hpet_vol[s], lpet_vol, hpet_vol, s,
            pb_t, 0.0, 0.0, 0.0, 0, 0, 0.0,
            sigma_est=0.0, num_ensemble=num_ensemble, base_seed=subj_seed,
            zero_hpet_guide=zero_hpet_guide)
        epet_bl[cnt] = final_bl
        spet_bl[cnt] = hr_np

        final_tta, _, _, _ = tta_infer_slice(
            diffusion, lpet_vol[s], hpet_vol[s], lpet_vol, hpet_vol, s,
            pb_t, best_ratio, best_lr, best_lr_late, K_mid, K_late, lam,
            sigma_est=sigma_est,
            num_neighbors=pixel_bank_nn, window_size=pixel_bank_ws,
            freq_low_radius=freq_low_radius, freq_low_scale=freq_low_scale,
            num_ensemble=num_ensemble, base_seed=subj_seed,
            zero_hpet_guide=zero_hpet_guide,
            tta_target=tta_target,
            prenet_K=prenet_K, prenet_lr=prenet_lr,
            prenet_pb_scale=prenet_pb_scale)
        epet_tta[cnt] = final_tta
        spet_tta[cnt] = hr_np
        cnt += 1

    if cnt > 0:
        import core.metrics as Metrics

        epet_bl, spet_bl = epet_bl[:cnt], spet_bl[:cnt]
        epet_tta, spet_tta = epet_tta[:cnt], spet_tta[:cnt]
        os.makedirs(output_dir_bl, exist_ok=True)
        os.makedirs(output_dir_tta, exist_ok=True)

        Metrics.save_img(epet_bl, os.path.join(
            output_dir_bl, f"{subject_id}_result.img"))
        Metrics.save_img(spet_bl, os.path.join(
            output_dir_bl, f"{subject_id}_hr.img"))
        Metrics.save_img(epet_tta, os.path.join(
            output_dir_tta, f"{subject_id}_result.img"))
        Metrics.save_img(spet_tta, os.path.join(
            output_dir_tta, f"{subject_id}_hr.img"))

    del pixel_bank
    torch.cuda.empty_cache()

    return {"subject_id": subject_id, "subject_idx": subject_idx,
            "effective_lr": float(best_lr),
            "effective_ratio": float(best_ratio),
            "full_fallback": bool(full_fallback)}


def _list_subjects(data_dir):
    subjects = {}
    for f in sorted(os.listdir(data_dir)):
        if f.endswith(".mat"):
            subjects.setdefault(f.rsplit("_", 1)[0], []).append(f)
    return sorted(subjects)


def main():
    parser = argparse.ArgumentParser(
        description="ResTTA inference over multiple dose levels.")
    parser.add_argument("--checkpoint", default="checkpoints/I500000_E93",
                        help="Checkpoint stem (absolute or relative to cwd)")
    parser.add_argument("--doses", default="1,2,5,10,25,50")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT,
                        help="Root containing test_mat_{dose}/ directories")
    parser.add_argument("--data_dir", default="",
                        help="Override the data directory for a single dose")
    parser.add_argument("--config_dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_root", default="outputs/restta")
    # TTA hyper-parameters
    parser.add_argument("--lr_values", default="1e-2,5e-2,1e-1")
    parser.add_argument("--start_ratios", default="1.0")
    parser.add_argument("--search_slices", type=int, default=5,
                        help="Slices used for the oracle search; in fixed mode "
                             "also the index of the first evaluated slice")
    parser.add_argument("--lam", type=float, default=0.0,
                        help="Proximal term weight (0 = deployment setting)")
    parser.add_argument("--K_mid", type=int, default=10)
    parser.add_argument("--K_late", type=int, default=20)
    parser.add_argument("--pixelbank_ws", type=int, default=17)
    parser.add_argument("--pixelbank_nn", type=int, default=16)
    parser.add_argument("--pixelbank_nr", type=int, default=20)
    parser.add_argument("--use_hybrid_pb", action="store_true", default=True)
    parser.add_argument("--no-use_hybrid_pb", dest="use_hybrid_pb",
                        action="store_false")
    parser.add_argument("--no-bilateral", action="store_true", default=False)
    parser.add_argument("--hybrid_sigma_ratio", type=float, default=0.3)
    parser.add_argument("--pb_source", default="lpet",
                        choices=["lpet", "prenet"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_subjects", type=int, default=0)
    parser.add_argument("--subject_start", type=int, default=0)
    parser.add_argument("--subject_end", type=int, default=None)
    parser.add_argument("--subject_indices", default="",
                        help="Comma-separated absolute subject indices")
    parser.add_argument("--fallback_ratio", type=float, default=0.3)
    # Fixed hyper-parameters (recommended, GT-free at test time)
    parser.add_argument("--fixed_lr", type=float, default=None)
    parser.add_argument("--fixed_ratio", type=float, default=None)
    parser.add_argument("--fixed_lr_late", type=float, default=None)
    # Ablation switches
    parser.add_argument("--tta_target", default="denoisenet",
                        choices=["denoisenet", "prenet", "both"])
    parser.add_argument("--prenet_K", type=int, default=20)
    parser.add_argument("--prenet_lr", type=float, default=0.3)
    parser.add_argument("--prenet_pb_scale", default="medium")
    parser.add_argument("--ms_pb_scales", default="9,17,33",
                        help="Comma-separated multi-scale window sizes; "
                             "empty string disables multi-scale banks")
    parser.add_argument("--freq_low_radius", type=int, default=16)
    parser.add_argument("--freq_low_scale", type=float, default=0.3)
    parser.add_argument("--num_ensemble", type=int, default=1)
    parser.add_argument("--zero_hpet_guide", action="store_true", default=False,
                        help="[diagnostic] zero the HPET guide inputs")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    doses = [d.strip() for d in args.doses.split(",") if d.strip()]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_root, exist_ok=True)

    lr_values = [float(x) for x in args.lr_values.split(",")]
    start_ratios = [float(x) for x in args.start_ratios.split(",")]
    device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0

    ms_scales = (scales_from_sizes([int(x) for x in
                                    args.ms_pb_scales.split(",") if x.strip()])
                 if args.ms_pb_scales.strip() else None)

    ckpt_display = (args.checkpoint if os.path.isabs(args.checkpoint)
                    else os.path.join(os.getcwd(), args.checkpoint))

    print("=" * 72)
    print("  ResTTA inference")
    print(f"  Checkpoint : {ckpt_display}")
    print(f"  Doses      : {doses}")
    print(f"  Device     : {device}")
    print(f"  Output     : {args.output_root}")
    print(f"  Hybrid PB  : {args.use_hybrid_pb}")
    print(f"  PB source  : {args.pb_source}")
    print(f"  Multi-scale: {ms_scales if ms_scales else f'single (ws={args.pixelbank_ws})'}")
    print(f"  Freq filter: radius={args.freq_low_radius}, "
          f"low_scale={args.freq_low_scale}")
    print(f"  Ensemble   : {args.num_ensemble}")
    if args.fixed_lr is not None:
        print(f"  TTA mode   : FIXED lr_mid={args.fixed_lr:.2e}, "
              f"ratio={args.fixed_ratio if args.fixed_ratio is not None else 1.0}")
    else:
        print("  TTA mode   : oracle per-subject GT search (diagnostic only)")
    print("=" * 72)

    all_results = {}

    for dose in doses:
        print(f"\n{'=' * 72}\n  Dose {dose}%\n{'=' * 72}")

        config_dose = dose if dose != "2" else "1"
        config_path = os.path.join(
            args.config_dir, f"sr_sr3_{config_dose}_128.json")
        if not os.path.exists(config_path):
            print(f"  [SKIP] config not found: {config_path}")
            continue

        data_dir = args.data_dir or os.path.join(
            args.data_root, f"test_mat_{dose}")
        if not os.path.isdir(data_dir):
            print(f"  [SKIP] data dir not found: {data_dir}")
            continue

        diffusion, _ = load_model_for_dose(
            config_path, args.checkpoint, device_id)

        dose_out = os.path.join(args.output_root, f"dose_{dose}")
        bl_dir = os.path.join(dose_out, "baseline")
        tta_dir = os.path.join(dose_out, "tta")

        all_subject_ids = _list_subjects(data_dir)
        if args.subject_indices.strip():
            wanted = [int(x) for x in args.subject_indices.split(",")]
            bad = [i for i in wanted if i < 0 or i >= len(all_subject_ids)]
            if bad:
                raise ValueError(
                    f"--subject_indices out of range: {bad} "
                    f"(dataset has {len(all_subject_ids)} subjects)")
            subject_ids = [all_subject_ids[i] for i in wanted]
            subject_abs = {sid: i for sid, i in zip(subject_ids, wanted)}
        else:
            subject_ids = all_subject_ids[args.subject_start:args.subject_end]
            if args.max_subjects > 0:
                subject_ids = subject_ids[:args.max_subjects]
            subject_abs = {sid: args.subject_start + j
                           for j, sid in enumerate(subject_ids)}

        print(f"  Subjects: {len(subject_ids)}")

        dose_selections = []
        for subject_id in tqdm(subject_ids, desc=f"  Dose {dose}"):
            abs_subj_idx = subject_abs[subject_id]
            bl_file = os.path.join(bl_dir, f"{subject_id}_result.img")
            tta_file = os.path.join(tta_dir, f"{subject_id}_result.img")
            if os.path.exists(bl_file) and os.path.exists(tta_file):
                print(f"    Skip {subject_id}: already processed")
                continue

            lpet_vol, hpet_vol = load_subject_volume(data_dir, subject_id)
            dose_int = int(dose) if dose.isdigit() else 10
            sigma = SIGMA_MAP.get(dose_int, DEFAULT_SIGMA)
            if args.no_bilateral:
                sigma = 0.0
            sigma_est = sigma * args.hybrid_sigma_ratio if args.use_hybrid_pb else 0.0

            dose_selections.append(process_subject(
                diffusion, lpet_vol, hpet_vol, abs_subj_idx, subject_id,
                bl_dir, tta_dir,
                args.pixelbank_ws, args.pixelbank_nn, args.pixelbank_nr,
                args.search_slices, lr_values, start_ratios,
                args.K_mid, args.K_late, args.lam, device, sigma=sigma,
                sigma_est=sigma_est, fallback_ratio=args.fallback_ratio,
                ms_scales=ms_scales,
                freq_low_radius=args.freq_low_radius,
                freq_low_scale=args.freq_low_scale,
                num_ensemble=args.num_ensemble, seed=args.seed,
                fixed_lr=args.fixed_lr, fixed_ratio=args.fixed_ratio,
                zero_hpet_guide=args.zero_hpet_guide,
                tta_target=args.tta_target,
                prenet_K=args.prenet_K, prenet_lr=args.prenet_lr,
                prenet_pb_scale=args.prenet_pb_scale,
                pb_source=args.pb_source,
                fixed_lr_late=args.fixed_lr_late))

        print(f"\n  --- Baseline evaluation (dose={dose}) ---")
        bl_psnr, bl_ssim, bl_nmse, bl_per_subj = evaluate_results_dir(bl_dir)
        print(f"\n  --- TTA evaluation (dose={dose}) ---")
        tta_psnr, tta_ssim, tta_nmse, tta_per_subj = evaluate_results_dir(tta_dir)

        bl_m = metrics_mean_std(bl_per_subj)
        tta_m = metrics_mean_std(tta_per_subj)
        metrics_dir = os.path.join(dose_out, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        write_per_subject_csv(
            os.path.join(metrics_dir, "baseline_per_subject.csv"), bl_per_subj)
        write_per_subject_csv(
            os.path.join(metrics_dir, "tta_per_subject.csv"), tta_per_subj)

        with open(os.path.join(metrics_dir, "lr_selection.csv"), "w") as f:
            f.write("subject_id,abs_idx,effective_lr,effective_ratio,full_fallback\n")
            for sel in dose_selections:
                f.write(f"{sel['subject_id']},{sel['subject_idx']},"
                        f"{sel['effective_lr']:.6f},{sel['effective_ratio']:.6f},"
                        f"{int(sel['full_fallback'])}\n")

        all_results[dose] = {
            "bl": {"psnr": bl_m[0][0], "psnr_std": bl_m[0][1],
                   "ssim": bl_m[1][0], "ssim_std": bl_m[1][1],
                   "nmse": bl_m[2][0], "nmse_std": bl_m[2][1]},
            "tta": {"psnr": tta_m[0][0], "psnr_std": tta_m[0][1],
                    "ssim": tta_m[1][0], "ssim_std": tta_m[1][1],
                    "nmse": tta_m[2][0], "nmse_std": tta_m[2][1]},
        }

        del diffusion
        torch.cuda.empty_cache()

    print(f"\n\n{'=' * 90}\n  FINAL COMPARISON\n{'=' * 90}")
    header = (f"{'Dose':>5s} | {'BL PSNR':>15s} | {'TTA PSNR':>15s} | "
              f"{'dPSNR':>8s} | {'BL SSIM':>15s} | {'TTA SSIM':>15s} | "
              f"{'dSSIM':>8s}")
    print(header)
    print("-" * len(header))
    for dose in sorted(all_results, key=int):
        r = all_results[dose]
        bl, tta = r["bl"], r["tta"]
        print(f"{dose:>5s} | {bl['psnr']:6.3f}+-{bl['psnr_std']:.3f} | "
              f"{tta['psnr']:6.3f}+-{tta['psnr_std']:.3f} | "
              f"{tta['psnr'] - bl['psnr']:+8.4f} | "
              f"{bl['ssim']:6.4f}+-{bl['ssim_std']:.3f} | "
              f"{tta['ssim']:6.4f}+-{tta['ssim_std']:.3f} | "
              f"{tta['ssim'] - bl['ssim']:+8.4f}")
    print("-" * len(header))

    with open(os.path.join(args.output_root, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Summary -> {os.path.join(args.output_root, 'summary.json')}")


if __name__ == "__main__":
    main()
