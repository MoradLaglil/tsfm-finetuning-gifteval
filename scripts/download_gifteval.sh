#!/usr/bin/env bash
# Download the GIFT-Eval benchmark (Salesforce/GiftEval on the Hugging Face Hub, ~1.5 GB) and register its
# location in the .env file read by the runners.
# Usage: bash scripts/download_gifteval.sh [target_dir]      (default: data/gift-eval)
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET="${1:-data/gift-eval}"
python - "$TARGET" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Salesforce/GiftEval", repo_type="dataset", local_dir=sys.argv[1])
PY
ABS="$(cd "$TARGET" && pwd)"
touch .env
grep -v '^GIFT_EVAL=' .env > .env.tmp || true
echo "GIFT_EVAL=$ABS" >> .env.tmp && mv .env.tmp .env
echo "GIFT-Eval downloaded to $ABS (GIFT_EVAL written to .env)"
