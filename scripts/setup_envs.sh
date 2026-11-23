#!/usr/bin/env bash
# Create the two conda environments used in the paper.
# Usage: bash scripts/setup_envs.sh [chronos2|ttm|all]   (default: all)
# Set ENV_PREFIX=/some/dir to create the environments as prefixes instead of named envs.
set -euo pipefail
cd "$(dirname "$0")/.."

GIFT_EVAL_PKG="salesforce-gift-eval @ git+https://github.com/SalesforceAIResearch/gift-eval.git@795b246e307941b05316a8f00103002a869b7af0"
GRANITE_TSFM_PKG="granite-tsfm @ git+https://github.com/ibm-granite/granite-tsfm.git@e911cd16a0901d6a51159f636a50ee9db895ae78"
WHICH="${1:-all}"

eval "$(conda shell.bash hook)"

make_env() {  # $1 = env name; an existing environment is reused
    local target="$1" flag="-n"
    if [[ -n "${ENV_PREFIX:-}" ]]; then target="${ENV_PREFIX}/$1"; flag="-p"; fi
    if ! conda activate "$target" 2>/dev/null; then
        conda create -y -q "$flag" "$target" python=3.11
        conda activate "$target"
    fi
    python -m pip install -q --upgrade pip
}

if [[ "$WHICH" == "chronos2" || "$WHICH" == "all" ]]; then
    make_env tsfm-chronos2
    pip install -r chronos2/requirements.txt
    # GIFT-Eval data loader only (its pinned dependencies conflict with AutoGluon and are not needed here)
    pip install --no-deps "$GIFT_EVAL_PKG"
    conda deactivate
fi

if [[ "$WHICH" == "ttm" || "$WHICH" == "all" ]]; then
    make_env tsfm-ttm
    pip install -r ttm/requirements.txt
    pip install --no-deps "$GIFT_EVAL_PKG"
    # tsfm_public from the TTM-R3 release branch (ttm-r3-release-mq2), installed without its
    # dependency pins so that the torch/transformers versions used in the paper are kept.
    pip install --no-deps "$GRANITE_TSFM_PKG"
    conda deactivate
fi

echo "Done. Activate with: conda activate tsfm-chronos2   or   conda activate tsfm-ttm"
