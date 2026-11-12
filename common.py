"""Shared constants, data loading and feature utilities.

ResTTA operates on 2-D slices of a 3-D PET volume.  Each subject is stored as
128 ``.mat`` files (``{subject_id}_{slice_idx}.mat``); every file contains a
``(1, 256, W)`` array whose first 128 rows are the low-dose PET (LPET) and the
last 128 rows are the full-dose PET (HPET, ground truth during evaluation).

All volumes are normalised per subject with ``compute_scale`` from
``normalize_common.py`` (99.5th percentile of LPET).  The training loader uses
the same definition, so training and inference are numerically consistent.
"""

import os

import numpy as np
import scipy.io as sio
import torch

from normalize_common import compute_scale, extract_lpet_hpet

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
DEFAULT_DATA_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "data_root")

# Dose-adaptive bilateral-filter strength used before building the pixel bank.
# Lower dose -> noisier input -> stronger smoothing.
SIGMA_MAP = {1: 1.2, 2: 0.8, 5: 0.5, 10: 0.3, 25: 0.0, 50: 0.0}
DEFAULT_SIGMA = 1.5
# range sigma / spatial sigma for the bilateral filter
BILATERAL_COLOR_RATIO = 0.06


def _read_mat_slice(mat_path):
    data = sio.loadmat(mat_path)["img"]
    return extract_lpet_hpet(data)


def load_lpet_volume(mat_dir, subject_id, num_slices=128):
    """Load and normalise a subject's LPET volume as ``(D, H, W)`` float32."""
    lpet_slices = []
    for idx in range(num_slices):
        lpet, _ = _read_mat_slice(os.path.join(mat_dir, f"{subject_id}_{idx}.mat"))
        lpet_slices.append(lpet)
    lpet_vol = np.stack(lpet_slices, axis=0).astype(np.float32)
    lpet_vol /= compute_scale(lpet_vol)
    return lpet_vol


def load_subject_volume(mat_dir, subject_id, num_slices=128):
    """Load and normalise one subject's LPET and HPET volumes.

    Returns:
        (lpet_vol, hpet_vol), both ``(D, H, W)`` float32, normalised by the
        per-subject LPET scale (HPET is divided by the same scale).
    """
    lpet_slices, hpet_slices = [], []
    for idx in range(num_slices):
        lpet, hpet = _read_mat_slice(os.path.join(mat_dir, f"{subject_id}_{idx}.mat"))
        lpet_slices.append(lpet)
        hpet_slices.append(hpet)

    lpet_vol = np.stack(lpet_slices, axis=0).astype(np.float32)
    hpet_vol = np.stack(hpet_slices, axis=0).astype(np.float32)
    scale = compute_scale(lpet_vol)
    lpet_vol /= scale
    hpet_vol /= scale
    return lpet_vol, hpet_vol


def compute_3d_context(volume, slice_idx):
    """Build the 5-slice neighbourhood ``(5, H, W)`` around ``slice_idx``."""
    D = volume.shape[0]
    slices = []
    for offset in (-2, -1, 0, 1, 2):
        s = max(0, min(D - 1, slice_idx + offset))
        slices.append(volume[s])
    return torch.from_numpy(np.stack(slices, axis=0)).float()


def compute_spectrum(slice_2d):
    """Log-magnitude, min-max normalised 2-D Fourier spectrum."""
    img = slice_2d.cpu().numpy() if isinstance(slice_2d, torch.Tensor) else slice_2d
    if img.ndim == 3:
        img = img.squeeze()
    fft = np.fft.fft2(img)
    fshift = np.fft.fftshift(fft)
    fshift += 1e-8
    mag = 20 * np.log(np.abs(fshift))
    mn, mx = mag.min(), mag.max()
    mag = (mag - mn) / (mx - mn) if mx > mn else np.zeros_like(mag)
    return torch.from_numpy(mag).float().unsqueeze(0)
