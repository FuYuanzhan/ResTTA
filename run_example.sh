#!/bin/bash
# End-to-end example. Edit the paths/values for your environment.
set -euo pipefail

PY=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-../data_root}
CKPT=${CKPT:-checkpoints/I500000_E93}
DOSES=${DOSES:-1,5,10,25,50}
# Fixed TTA hyper-parameters selected on the validation set (see tune_fixed_params.py)
FIXED_LR=${FIXED_LR:-0.05}
FIXED_RATIO=${FIXED_RATIO:-1.0}

# 1) Preprocessing ---------------------------------------------------------
$PY normalize_train_mat.py --data-root "$DATA_ROOT" --doses 1 5 10 25 50

# Optional: cache training data for faster epochs
# for d in 1 5 10 25 50; do
#   $PY preload_train_data.py --mat-dir "$DATA_ROOT/train_mat_$d" --dose "$d" --out ../preload
# done

# 2) Training (one model per dose, or reuse one checkpoint across doses) ----
# $PY train.py -c configs/sr_sr3_1_128.json \
#     --train_dir "$DATA_ROOT/train_mat_1" --val_dir "$DATA_ROOT/test_mat_1" \
#     -gpu 0

# 3) Validation-set tuning (subjects 0-3; test subjects start at 4) ---------
for d in ${DOSES//,/ }; do
  $PY tune_fixed_params.py --checkpoint "$CKPT" --doses "$d" \
      --val_dir "$DATA_ROOT/test_mat_$d" --val_subjects 4 \
      --lr_values 0.01,0.05,0.1,0.2,0.3 --out_json "outputs/tune_$d.json"
done

# 4) Inference (test-time zero GT) -----------------------------------------
$PY inference.py --checkpoint "$CKPT" --doses "$DOSES" \
    --data_root "$DATA_ROOT" \
    --fixed_lr "$FIXED_LR" --fixed_ratio "$FIXED_RATIO" \
    --subject_start 4 --subject_end 32 \
    --output_root outputs/restta

# 5) Evaluation -------------------------------------------------------------
$PY evaluate.py --root outputs/restta
