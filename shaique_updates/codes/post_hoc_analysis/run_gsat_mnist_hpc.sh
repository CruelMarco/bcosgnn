#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

exec python3 -u gsat_mnist_hpc.py "$@"
