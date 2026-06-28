#!/usr/bin/env bash
set -euo pipefail

#python3 -m pip install --user --no-cache-dir pandas

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export BCOSGNN_DATA_ROOT="${BCOSGNN_DATA_ROOT:-$PWD/bcosgnn/data}"
echo "Using BCOSGNN_DATA_ROOT=$BCOSGNN_DATA_ROOT"
exec python3 -u GSAT_Node_classification.py