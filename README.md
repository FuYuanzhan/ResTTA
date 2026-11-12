# ResTTA: Residual Test-Time Adaptation for Low-Dose PET Reconstruction

ResTTA reconstructs a full-dose PET image from a single low-dose PET (LPET)
slice with a guided residual diffusion model, and improves it at **test time**
with a pixel-bank consistency prior — no ground truth and no re-training.

The code is organised around the four stages of the method:

1. **Training** (`train.py`) — jointly train a coarse predictor (PreNet), a
   residual denoiser (DenoiseNet), and six guide networks (3-D context and
   spectral guidance).
2. **Pixel-bank construction** (`pixel_bank.py`) — for every voxel, store the
   indices of its most similar neighbours in a local window.
3. **Residual refinement** (`refinement.py`) — during reverse diffusion, take a
   few gradient steps on a neighbourhood-consistency objective; a
   frequency-domain filter attenuates low-frequency gradient components so
   global structure is preserved while detail is corrected.
4. **Inference and evaluation** (`inference.py`, `evaluate.py`) — save
   baseline/TTA volumes and compute PSNR / SSIM / NMSE.

## Repository layout

```
ResTTA/
├── train.py                 # original model training
├── pixel_bank.py            # pixel-bank construction (library + CLI)
├── refinement.py            # residual refinement (TTA sampling / PreNet refinement)
├── model_loader.py          # config + checkpoint loading
├── inference.py             # multi-dose baseline + ResTTA inference
├── evaluate.py              # PSNR / SSIM / NMSE evaluation
├── tune_fixed_params.py     # validation-set tuning of (lr, start_ratio)
│
├── common.py                # data loading, normalisation, context/spectrum features
├── normalize_common.py      # shared per-subject normalisation definition
├── normalize_train_mat.py   # precompute per-subject scale tables
├── preload_train_data.py    # optional memmap cache for faster training
│
├── model/                   # PreNet, DenoiseNet, guide networks, diffusion
│   ├── sr3_modules/         # Gaussian diffusion + U-Net
│   └── ddpm_modules/
├── data/                    # training dataset / dataloader
├── core/                    # logging and metrics helpers
├── configs/                 # model configs per dose level
├── checkpoints/             # (place model weights here)
├── data_root/               # (place data here; not committed)
└── outputs/                 # inference results
```

## Installation

```bash
pip install -r requirements.txt
```

`medpy` is used only for reading/writing `.img` volumes; `SimpleITK` provides
the bilateral pre-filter used before building pixel banks.

## Data format

Each subject is stored as 128 `.mat` files named
`{subject_id}_{slice_idx}.mat` (`slice_idx` = 0…127). Every file contains an
`img` array of shape `(1, 256, W)`:

* rows `0:128` — low-dose PET (input),
* rows `128:256` — full-dose PET (ground truth).

Place the data as:

```
data_root/
├── train_mat_1/  train_mat_5/  train_mat_10/  train_mat_25/  train_mat_50/
└── test_mat_1/   test_mat_2/   test_mat_5/   test_mat_10/   test_mat_25/  test_mat_50/
```

Dose `2` has no dedicated config (the architecture is identical to dose `1`);
`inference.py` falls back to `sr_sr3_1_128.json` for it.

Training and inference share one normalisation definition: each subject is
divided by the 99.5th percentile of its LPET volume
(`normalize_common.compute_scale`). Training reads a precomputed per-subject
scale table; inference computes it on the fly.

## Checkpoints

Model weights are **not** included. Place the eight files of a checkpoint in
`checkpoints/`:

```
checkpoints/I500000_E93_PreNet_gen.pth
checkpoints/I500000_E93_DenoiseNet_gen.pth
checkpoints/I500000_E93_guide_3D_{1,2,3}_gen.pth
checkpoints/I500000_E93_guide_spectrum_{1,2,3}_gen.pth
```

Any stem works; pass it with `--checkpoint checkpoints/<stem>`.

## Quick start

### 1. Preprocess (once)

```bash
# per-subject normalisation table (written to normalize_scales/)
python normalize_train_mat.py --data-root ../data_root --doses 1 5 10 25 50

# optional: cache training data as a float32 memmap (faster epochs)
python preload_train_data.py --mat-dir ../data_root/train_mat_1 --dose 1 --out ../preload
```

### 2. Train the base model

```bash
python train.py -c configs/sr_sr3_1_128.json \
    --train_dir ../data_root/train_mat_1 \
    --val_dir   ../data_root/test_mat_1 \
    --n_iter 125000 --batch_size 16 -gpu 0
```

Checkpoints and TensorBoard logs are written under `experiments/`.

### 3. Choose TTA hyper-parameters on a validation set

`lr_mid` and `start_ratio` are tuned on a validation split that is disjoint
from the test subjects. The selected values are then fixed for test-time
inference, so **no ground truth is used at test time**.

With the released checkpoint the paper uses subjects 0–3 of each test set for
tuning and subjects 4–31 for reporting; for a model you train yourself, hold
out training subjects instead. Tune each dose separately:

```bash
for d in 1 5 10 25 50; do
  python tune_fixed_params.py --checkpoint checkpoints/I500000_E93 \
      --doses "$d" --val_dir ../data_root/test_mat_"$d" --val_subjects 4 \
      --lr_values 0.01,0.05,0.1,0.2,0.3 \
      --out_json outputs/tune_$d.json
done
```

The selected `best_lr` is reported per dose. Pass it to `inference.py` with
`--fixed_lr` (and optionally `--fixed_lr_late`, default `best_lr * 0.1`).

### 4. Inference

```bash
python inference.py --checkpoint checkpoints/I500000_E93 \
    --doses 1,5,10,25,50 \
    --data_root ../data_root \
    --fixed_lr 0.05 --fixed_ratio 1.0 \
    --subject_start 4 --subject_end 32 \
    --output_root outputs/restta
```

`--fixed_lr` enables the recommended ground-truth-free mode. Omitting it runs
a per-subject oracle grid search (diagnostic upper bound only; it uses the test
ground truth).

Outputs:

```
outputs/restta/
├── dose_1/
│   ├── baseline/  {subject_id}_result.img, {subject_id}_hr.img
│   ├── tta/       {subject_id}_result.img, {subject_id}_hr.img
│   └── metrics/   per-subject CSV + lr_selection.csv
├── dose_5/ ...
└── summary.json
```

### 5. Evaluation

```bash
# evaluate everything written by inference.py
python evaluate.py --root outputs/restta

# or one directory
python evaluate.py --result_dir outputs/restta/dose_1/tta
```

Metrics follow `our_paper_metric.py`: background (`gt <= 0.05`) zeroed, PSNR
and NMSE on the non-zero region, SSIM on the full volume, dynamic data range.

## Using cached pixel banks

Pixel banks are built on the fly during inference. To build/save them
separately (e.g. for inspection or reuse), use:

```bash
python pixel_bank.py --data_dir ../data_root/test_mat_1 --dose 1 \
    --scales 9,17,33 --save_dir outputs/pixel_bank/dose_1
```

## Method notes

* **Residual formulation.** PreNet predicts a coarse image `IP`; DenoiseNet
  samples a residual `R` with a 10-step reverse diffusion; the output is
  `IP + R`.
* **Neighbourhood consistency.** For a residual estimate `x`, the prior is
  `L_nc(x) = mean_p ||x[p] - mean_{q in N(p)} x[q]||^2`, where `N(p)` comes
  from the pixel bank. The bank stores indices (not noisy intensities) and is
  anchored on the LPET volume after a dose-adaptive bilateral filter.
* **Why the frequency filter.** Low-frequency components of the gradient are
  scaled by `freq_low_scale` (default 0.3); this prevents TTA from changing
  global structure while keeping high-frequency corrections.
* **Hybrid bank.** Half of the medium-scale bank comes from LPET and half from
  the current estimate `IP + x_t`, rebuilt once during refinement.
* **Multi-scale mixing.** Small windows are weighted early in the refinement
  window, large windows late.

### Ablation switches

`inference.py` exposes the ablations used in the paper:

| Flag | Effect |
| --- | --- |
| `--no-use_hybrid_pb` | use only the LPET-derived bank |
| `--no-bilateral` | disable the bilateral pre-filter |
| `--ms_pb_scales ""` | single-scale bank (`--pixelbank_ws`) |
| `--freq_low_scale 1.0` | disable the frequency filter |
| `--tta_target prenet\|both` | refine PreNet output (`prenet`) or both stages |
| `--pb_source prenet` | anchor the bank on PreNet's prediction instead of LPET |
| `--num_ensemble N` | average N random seeds |

## What is not included

* Raw PET datasets (`.mat`) — place them under `data_root/` as described above.
* Model weights (`.pth`) — place them under `checkpoints/`.
* Experiment outputs and precomputed caches.

## License

Apache License 2.0 (see `LICENSE`).
