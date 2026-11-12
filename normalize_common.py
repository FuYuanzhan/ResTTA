"""归一化口径的唯一实现（训练 / 推理共用，防止两处口径再次漂移）。

口径：per-subject 缩放因子 ``scale = robust_max(LPET 全卷)``，默认取 LPET 的
99.5 分位数（对噪声尖峰稳健），异常时逐级回退到 LPET max、再回退到 1。

**仅由 LPET 可算，不依赖 HPET**——这是与旧口径
``max(lpet.max(), hpet.max())`` 的关键区别：真实部署去噪时拿不到 HPET，
而 LPET 一定在手，故 scale 必须只由 LPET 计算。

训练侧 ``data/LRHR_dataset.py``（逐切片加载，无法直接算整卷分位数）查
``normalize_train_mat.py`` 预计算的 ``{subject_id: scale}`` 表；
推理侧 ``load_subject_volume`` 手里有完整 LPET 体积，直接用本模块的
``compute_scale`` 实时计算——两者共用同一函数，保证结果逐位一致。

注意：本模块只依赖 numpy，故意不放在 ``data/`` 包内（``data/__init__.py``
会 import torch，而预处理脚本不需要 torch，避免拖入损坏的 torch 环境）。
"""

import numpy as np

DEFAULT_PERCENTILE = 99.5
_EPS = 1e-6


def compute_scale(lpet_volume, percentile=DEFAULT_PERCENTILE):
    """由 LPET 体积计算 per-subject 缩放因子（仅 LPET 可算）。

    Args:
        lpet_volume: (D, H, W) 或任意 shape 的 float array，LPET 原始值。
        percentile:  稳健分位数，默认 99.5。

    Returns:
        float scale，保证 > 0。
    """
    flat = np.asarray(lpet_volume, dtype=np.float64).ravel()
    if flat.size == 0:
        return 1.0

    if flat.size > 1:
        scale = float(np.percentile(flat, percentile))
    else:
        scale = float(flat[0])

    if not np.isfinite(scale) or scale <= _EPS:
        scale = float(flat.max())
    if not np.isfinite(scale) or scale <= _EPS:
        scale = 1.0
    return scale


def extract_lpet_hpet(mat_data):
    """从 .mat 的 ``img`` 数组提取 (lpet, hpet)。

    适配两种历史格式：
      - ``(1, 256, W)``：channel-first 3D，LPET=0:128 行，HPET=128:256 行；
      - ``(256, W)``：    2D，LPET=rows 0:128，HPET=rows 128:256。
    """
    data = np.asarray(mat_data)
    if data.ndim == 3 and data.shape[0] == 1:
        return data[0, 0:128, :], data[0, 128:256, :]
    return data[0:128, :], data[128:256, :]
