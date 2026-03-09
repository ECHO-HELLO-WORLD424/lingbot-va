#!/usr/bin/env bash
# Run batch arm-distinction evaluation via i2va.
#
# Usage:
#   NGPU=1 bash scripts/run_arm_eval.sh [--input-dir eval_starting_images] [--save-root train_out/arm_eval] [--offload false] [extra args...]
#
# --offload true  (default) offloads VAE/text-encoder to CPU to conserve VRAM.
# --offload false keeps all model components on GPU; use this on VRAM-rich clusters.
#
# Must be run from the lingbot-va repo root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${REPO_ROOT}/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found. Did you create the venv? (see SETUP.md)"
    exit 1
fi

NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29501"}

export TOKENIZERS_PARALLELISM=false

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node="${NGPU}" \
    --master_port "${MASTER_PORT}" \
    "${SCRIPT_DIR}/run_arm_eval.py" "$@"
