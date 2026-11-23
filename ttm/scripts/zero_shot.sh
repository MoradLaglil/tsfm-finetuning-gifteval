#!/usr/bin/env bash
# TTM-R3-PT zero-shot (no fine-tuning: -ne 0). Extra arguments are passed to run_ttm_r3.py.
set -euo pipefail
cd "$(dirname "$0")/.."     # run from ttm/ (the runner uses relative resource paths)
export PYTHONUNBUFFERED=1
python run_ttm_r3.py -tv TTM-R3-PT-Zero-Shot \
    -ubfs 1 -as 1 -ne 0 -rn 1 \
    -seed 42 "$@"
