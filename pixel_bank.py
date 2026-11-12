"""Pixel-bank construction for residual test-time adaptation.

A pixel bank stores, for every voxel, the flattened indices of its most
similar neighbours inside a local search window.  The indices (not the noisy
intensity values) are stored, so the bank serves as a graph regulariser over
the *denoised residual estimate* rather than pulling it back to the noisy
observation.

The bank is built from the low-dose PET (LPET) volume after a dose-adaptive
bilateral pre-filter (edge-preserving denoising).  For every z-slice:

  1. Split the slice into blocks, search a ``window_size`` neighbourhood
     around each block and keep the ``num_neighbors`` closest intensities.
  2. Rank those candidates by distance to the centre voxel and keep the
     ``num_rows`` closest ones.
  3. Store their flattened indices.

Multi-scale banks (window sizes 9/17/33 by default) are supported; the TTA
loop mixes them with a schedule that favours small windows early and large
windows late in the reverse diffusion.

The same functions are used by ``inference.py`` (computed on the fly) and by
the standalone CLI below (optionally cached as ``.npy``).
"""

import argparse
import os
import re
import time

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

from common import (BILATERAL_COLOR_RATIO, DEFAULT_SIGMA, SIGMA_MAP,
                    load_lpet_volume)


def apply_bilateral_slice(image_2d, sigma_spatial):
    """Edge-preserving bilateral filter for a single 2-D slice."""
    if sigma_spatial is None or sigma_spatial <= 0:
        return image_2d.astype(np.float32)
    sigma_color = sigma_spatial * BILATERAL_COLOR_RATIO
    img_sitk = sitk.GetImageFromArray(image_2d.astype(np.float64))
    filtered_sitk = sitk.Bilateral(img_sitk,
                                   domainSigma=sigma_spatial,
                                   rangeSigma=sigma_color)
    return sitk.GetArrayFromImage(filtered_sitk).astype(np.float32)


def apply_bilateral_volume(volume, sigma_spatial):
    """Apply the bilateral filter slice by slice (no smoothing across z)."""
    if sigma_spatial is None or sigma_spatial <= 0:
        return volume.astype(np.float32)
    filtered = np.zeros_like(volume)
    for z in range(volume.shape[0]):
        filtered[z] = apply_bilateral_slice(
            volume[z].astype(np.float64), sigma_spatial)
    return filtered.astype(np.float32)


def build_pixel_bank_indices(volume_lpet, window_size=17, num_neighbors=16,
                             num_rows=20, device="cuda", sigma=1.5):
    """Build a 3-D pixel bank of neighbour indices.

    Args:
        volume_lpet: ``(D, H, W)`` normalised low-dose volume.
        window_size: side length of the local search window.
        num_neighbors: candidates kept per voxel inside the search window.
        num_rows: final number of entries stored per voxel (<= num_neighbors).
        device: torch device used for the search.
        sigma: bilateral pre-filter spatial sigma (0 disables filtering).

    Returns:
        ``(num_rows, D, H, W)`` int32 array of flattened in-slice indices.
    """
    if sigma is not None and sigma > 0:
        volume_lpet = apply_bilateral_volume(
            volume_lpet.astype(np.float64), sigma_spatial=sigma)

    D, H, W = volume_lpet.shape
    num_rows = min(num_rows, num_neighbors)
    blk_sz = min(32, H // 2)
    num_blk_h = H // blk_sz
    num_blk_w = W // blk_sz

    pixel_bank = np.zeros((num_rows, D, H, W), dtype=np.int32)

    for z in range(D):
        z_s = max(0, z - 1)
        z_e = min(D, z + 2)
        slice_stack = [torch.from_numpy(volume_lpet[zz]).float().to(device)
                       for zz in range(z_s, z_e)]
        multi_slice = torch.stack(slice_stack, dim=0)
        center_slice = multi_slice[multi_slice.shape[0] // 2]

        all_idx_blocks = []
        all_val_blocks = []

        for bi in range(num_blk_w):
            for bj in range(num_blk_h):
                r0 = max(0, bj * blk_sz - window_size // 2)
                r1 = min(H, (bj + 1) * blk_sz + window_size // 2)
                c0 = max(0, bi * blk_sz - window_size // 2)
                c1 = min(W, (bi + 1) * blk_sz + window_size // 2)

                tr0, tr1 = bj * blk_sz, (bj + 1) * blk_sz
                tc0, tc1 = bi * blk_sz, (bi + 1) * blk_sz
                search_w = c1 - c0

                target = center_slice[tr0:tr1, tc0:tc1].reshape(-1)
                search = center_slice[r0:r1, c0:c1].reshape(-1)

                dist = (target.unsqueeze(1) - search.unsqueeze(0)) ** 2
                k = min(num_neighbors, search.shape[0])
                _, topk_idx = torch.topk(dist, k=k, dim=1, largest=False)

                global_rows = r0 + (topk_idx // search_w)
                global_cols = c0 + (topk_idx % search_w)
                global_flat = (global_rows * W + global_cols).long()

                all_idx_blocks.append(global_flat)
                all_val_blocks.append(search[topk_idx])

        all_idx = torch.cat(all_idx_blocks, dim=0).reshape(H, W, num_neighbors)
        all_vals = torch.cat(all_val_blocks, dim=0).reshape(H, W, num_neighbors)

        center_vals = torch.from_numpy(volume_lpet[z]).float().to(device)
        idx_flat = all_idx.reshape(-1, num_neighbors)
        vals_flat = all_vals.reshape(-1, num_neighbors)
        center_flat = center_vals.reshape(-1, 1)

        dist_to_center = (vals_flat - center_flat) ** 2
        _, sorted_order = torch.sort(dist_to_center, dim=1)
        sorted_indices = torch.gather(idx_flat, 1, sorted_order)
        selected = sorted_indices[:, :num_rows].cpu().numpy()

        for r in range(num_rows):
            pixel_bank[r, z, :, :] = selected[:, r].reshape(H, W)

    return pixel_bank


def build_pixel_bank_slice_2d(image_2d, window_size=17, num_neighbors=16,
                              num_rows=8, device="cuda"):
    """Build a 2-D pixel bank (used by the hybrid bank inside the TTA loop).

    Returns:
        ``(num_rows, H, W)`` int32 array.
    """
    H, W = image_2d.shape
    num_rows = min(num_rows, num_neighbors)
    blk_sz = min(32, H // 2)
    num_blk_h = H // blk_sz
    num_blk_w = W // blk_sz

    slice_t = torch.from_numpy(image_2d).float().to(device)
    all_idx_blocks = []
    all_val_blocks = []

    for bi in range(num_blk_w):
        for bj in range(num_blk_h):
            r0 = max(0, bj * blk_sz - window_size // 2)
            r1 = min(H, (bj + 1) * blk_sz + window_size // 2)
            c0 = max(0, bi * blk_sz - window_size // 2)
            c1 = min(W, (bi + 1) * blk_sz + window_size // 2)

            tr0, tr1 = bj * blk_sz, (bj + 1) * blk_sz
            tc0, tc1 = bi * blk_sz, (bi + 1) * blk_sz
            search_w = c1 - c0

            target = slice_t[tr0:tr1, tc0:tc1].reshape(-1)
            search = slice_t[r0:r1, c0:c1].reshape(-1)

            dist = (target.unsqueeze(1) - search.unsqueeze(0)) ** 2
            k = min(num_neighbors, search.shape[0])
            _, topk_idx = torch.topk(dist, k=k, dim=1, largest=False)

            global_rows = r0 + (topk_idx // search_w)
            global_cols = c0 + (topk_idx % search_w)
            global_flat = (global_rows * W + global_cols).long()

            all_idx_blocks.append(global_flat)
            all_val_blocks.append(search[topk_idx])

    all_idx = torch.cat(all_idx_blocks, dim=0).reshape(H, W, num_neighbors)
    all_vals = torch.cat(all_val_blocks, dim=0).reshape(H, W, num_neighbors)

    idx_flat = all_idx.reshape(-1, num_neighbors)
    vals_flat = all_vals.reshape(-1, num_neighbors)
    center_flat = slice_t.reshape(-1, 1)

    dist_to_center = (vals_flat - center_flat) ** 2
    _, sorted_order = torch.sort(dist_to_center, dim=1)
    sorted_indices = torch.gather(idx_flat, 1, sorted_order)
    selected = sorted_indices[:, :num_rows].cpu().numpy()

    pixel_bank = np.zeros((num_rows, H, W), dtype=np.int32)
    for r in range(num_rows):
        pixel_bank[r, :, :] = selected[:, r].reshape(H, W)
    return pixel_bank


def scales_from_sizes(sizes):
    """Map window sizes to ``(window_size, num_neighbors, num_rows)`` tuples."""
    scales = []
    for ws in sizes:
        if ws <= 9:
            scales.append((ws, 8, 8))
        elif ws <= 17:
            scales.append((ws, 16, min(20, ws)))
        else:
            scales.append((ws, 24, 20))
    return scales


def build_pixel_bank_multi_scale(volume_lpet, scales=None, device="cuda",
                                 sigma=1.5):
    """Build one bank per scale, returning ``{'s{ws}': array}``."""
    if scales is None:
        scales = scales_from_sizes([9, 17, 33])
    banks = {}
    for ws, nn, nr in scales:
        t0 = time.time()
        banks[f"s{ws}"] = build_pixel_bank_indices(
            volume_lpet, window_size=ws, num_neighbors=nn, num_rows=nr,
            device=device, sigma=sigma)
        print(f"    pixel bank s{ws} (ws={ws}, nn={nn}, nr={nr}): "
              f"{time.time() - t0:.1f}s")
    return banks


# ---------------------------------------------------------------------------
# Standalone builder CLI
# ---------------------------------------------------------------------------

def _infer_dose(path):
    m = re.search(r"(?:test|train)_mat_(\d+)", path)
    return int(m.group(1)) if m else None


def _list_subjects(data_dir, num_slices=128):
    subjects = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith(".mat"):
            continue
        sid = f.rsplit("_", 1)[0]
        subjects.setdefault(sid, []).append(f)
    return {sid: len(files) for sid, files in sorted(subjects.items())
            if len(files) == num_slices}


def main():
    parser = argparse.ArgumentParser(
        description="Build and cache pixel banks for a PET .mat dataset.")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing {subject_id}_{slice}.mat files")
    parser.add_argument("--save_dir", default=None,
                        help="Output directory (default: outputs/pixel_bank/dose_<d>)")
    parser.add_argument("--dose", type=int, default=None,
                        help="Dose level (1/2/5/10/25/50) for the sigma map")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Explicit bilateral sigma (overrides --dose)")
    parser.add_argument("--scales", type=str, default="9,17,33",
                        help="Comma-separated window sizes; empty string = single scale")
    parser.add_argument("--window_size", type=int, default=17,
                        help="Window size used when --scales is empty")
    parser.add_argument("--num_neighbors", type=int, default=16)
    parser.add_argument("--num_rows", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_subjects", type=int, default=0)
    args = parser.parse_args()

    dose = args.dose if args.dose is not None else _infer_dose(args.data_dir)
    if args.sigma is not None:
        sigma = args.sigma
    elif dose is not None:
        sigma = SIGMA_MAP.get(dose, DEFAULT_SIGMA)
    else:
        sigma = DEFAULT_SIGMA

    save_dir = args.save_dir or os.path.join(
        "outputs", "pixel_bank", f"dose_{dose if dose is not None else 'custom'}")
    os.makedirs(save_dir, exist_ok=True)

    scales = scales_from_sizes([int(x) for x in args.scales.split(",") if x.strip()])
    subjects = list(_list_subjects(args.data_dir))
    if args.max_subjects > 0:
        subjects = subjects[:args.max_subjects]

    print(f"[pixel_bank] data={args.data_dir} subjects={len(subjects)} "
          f"sigma={sigma} scales={[s[0] for s in scales]}")

    for sid in tqdm(subjects, desc="Building pixel banks"):
        lpet_vol = load_lpet_volume(args.data_dir, sid)
        if scales:
            banks = build_pixel_bank_multi_scale(
                lpet_vol, scales=scales, device=args.device, sigma=sigma)
            items = [(f"{sid}_s{ws}.npy", bank) for ws, bank in
                     zip([s[0] for s in scales], banks.values())]
        else:
            bank = build_pixel_bank_indices(
                lpet_vol, window_size=args.window_size,
                num_neighbors=args.num_neighbors, num_rows=args.num_rows,
                device=args.device, sigma=sigma)
            items = [(f"{sid}.npy", bank)]

        for name, bank in items:
            np.save(os.path.join(save_dir, name), bank)
        del lpet_vol

    print(f"[pixel_bank] done -> {save_dir}")


if __name__ == "__main__":
    main()
