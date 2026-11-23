#!/usr/bin/env bash
# Smoke test: run the four strategies of one model on a single small configuration and check that a result
# row is written for each. Results go to results/smoke/ (not used by analysis/aggregate.py).
#
#   conda activate tsfm-chronos2 && bash scripts/smoke_test.sh chronos2
#   conda activate tsfm-ttm      && bash scripts/smoke_test.sh ttm
#
# Optional: DATASET (default ett1/D, short horizon) and, for Chronos-2, STEPS (default 200 instead of 2000).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?usage: bash scripts/smoke_test.sh chronos2|ttm}"
DATASET="${DATASET:-ett1/D}"
OUT="results/smoke"

case "$MODEL" in
chronos2)
    for s in zero-shot fft lora-fullshot lora-fewshot; do
        python chronos2/run_chronos2.py --strategy "$s" --datasets "$DATASET" --terms short \
            --steps "${STEPS:-200}" --output-dir "$OUT/chronos2"
    done
    files=("$OUT"/chronos2/*/seed42/leaderboard.csv)
    ;;
ttm)
    for s in zero_shot fft_fullshot lora_fullshot lora_fewshot; do
        bash "ttm/scripts/$s.sh" --datasets "$DATASET" --terms short --results_dir "../$OUT/ttm"
    done
    files=("$OUT"/ttm/*/all_results.csv)
    ;;
*) echo "unknown model: $MODEL" >&2; exit 1 ;;
esac

echo; echo "Result files:"
for f in "${files[@]}"; do
    rows=$(($(wc -l < "$f") - 1))
    printf "  %-70s %d row(s)\n" "$f" "$rows"
    [[ "$rows" -ge 1 ]] || { echo "FAILED: no result in $f" >&2; exit 1; }
done
[[ "${#files[@]}" -eq 4 ]] || { echo "FAILED: expected 4 result files, found ${#files[@]}" >&2; exit 1; }
echo "Smoke test passed."
