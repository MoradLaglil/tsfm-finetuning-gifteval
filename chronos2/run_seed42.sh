#!/usr/bin/env bash
# Run the four Chronos-2 strategies of the paper for seed 42 on all 94 GIFT-Eval configurations.
# Usage (from the repository root, with the tsfm-chronos2 environment active):
#     bash chronos2/run_seed42.sh                     # all strategies
#     bash chronos2/run_seed42.sh lora-fullshot fft   # a subset
# Extra options can be passed through EXTRA_ARGS, e.g. EXTRA_ARGS="--datasets ett1/D --terms short".
set -euo pipefail
cd "$(dirname "$0")/.."
STRATEGIES=("${@:-zero-shot fft lora-fullshot lora-fewshot}")
for s in ${STRATEGIES[@]}; do
    python chronos2/run_chronos2.py --strategy "$s" --seed 42 ${EXTRA_ARGS:-}
done
