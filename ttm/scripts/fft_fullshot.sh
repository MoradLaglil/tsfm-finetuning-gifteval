#!/usr/bin/env bash
# TTM-R3-PT full fine-tuning (FFT) on the whole training split, seed 42.
# All parameter groups are trained (decoder, bias, head, patch, quantile, backbone, norm).
set -euo pipefail
cd "$(dirname "$0")/.."     # run from ttm/ (the runner uses relative resource paths)
export PYTHONUNBUFFERED=1
python run_ttm_r3.py -tv TTM-R3-FullFinetuning-FullShot-Seed42 \
    -as 1 -ne 20 -dt 1 -bt 1 -aff 1 -ht 1 -prt 0 -pp 0 -qt 1 -bkt 1 -nt 1 -pat 1 -rn 1 \
    -lr 1e-4 -ff 1 \
    -seed 42 "$@"
