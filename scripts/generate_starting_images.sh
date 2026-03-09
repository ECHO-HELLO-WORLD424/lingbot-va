#!/usr/bin/env bash
# Generate starting images for instruction-following evaluation.
# Usage: bash scripts/generate_starting_images.sh [--num-samples 50] [--output-dir eval_starting_images]
#
# Must be run from the lingbot-va repo root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found. Did you create the venv? (see SETUP.md)"
    exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/generate_starting_images.py" "$@"
