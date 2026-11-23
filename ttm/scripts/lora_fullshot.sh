#!/usr/bin/env bash
# TTM-R3-PT LoRA (r=8, alpha=16, dropout 0) on the whole training split, seed 42.
# The multi-quantile output block is trained fully (-qt 1).
set -euo pipefail
cd "$(dirname "$0")/.."     # run from ttm/ (the runner uses relative resource paths)
export PYTHONUNBUFFERED=1
python run_ttm_r3.py -tv TTM-R3-LORA-FullShot-Seed42 \
    -as 1 -aff 1 -ne 20 -qt 1 -prt 0 -pp 0 -rn 1 \
    -lr 1e-3 -ulora 1 -lrank 8 -lalpha 16 -ldrop 0.0 \
    -ff 1 \
    -seed 42 "$@"
