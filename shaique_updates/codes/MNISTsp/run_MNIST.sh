#!/usr/bin/env bash
set -euo pipefail

#python3 -m pip install --user --no-cache-dir pandas

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MNISTSP_SPLIT_ROOT="${MNISTSP_SPLIT_ROOT:-$PWD/data/MNIST/sparsified_pt_splits}"
echo "Using MNISTSP_SPLIT_ROOT=$MNISTSP_SPLIT_ROOT"
exec python3 -u bcos_MNIST.py