#!/usr/bin/env bash
# TTM-R3-PT LoRA few-shot, seed 42: default fewshot_fraction=0.2 with fewshot_location="rand"
# (20% of the training windows sampled at random, capped at 100,000 windows by -ubfs 1; raised to 90%
# when 20% gives fewer than 200 windows, -aff 1).
set -euo pipefail
cd "$(dirname "$0")/.."     # run from ttm/ (the runner uses relative resource paths)
export PYTHONUNBUFFERED=1
python run_ttm_r3.py -tv TTM-R3-LORA-FewShot-20-Seed42 \
    -as 1 -aff 1 -ne 20 -qt 1 -prt 0 -pp 0 -rn 1 \
    -lr 1e-3 -ulora 1 -lrank 8 -lalpha 16 -ldrop 0.0 \
    -ubfs 1 \
    -seed 42 "$@"
