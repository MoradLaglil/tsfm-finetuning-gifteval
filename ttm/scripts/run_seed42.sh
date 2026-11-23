#!/usr/bin/env bash
# Run the four TTM-R3-PT strategies of the paper for seed 42 on all 97 GIFT-Eval configurations.
# Usage (with the tsfm-ttm environment active): bash ttm/scripts/run_seed42.sh [extra run_ttm_r3.py args]
set -euo pipefail
HERE="$(dirname "$0")"
for s in zero_shot fft_fullshot lora_fullshot lora_fewshot; do
    bash "$HERE/$s.sh" "$@"
done
