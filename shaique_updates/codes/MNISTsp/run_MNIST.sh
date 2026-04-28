#!/usr/bin/env bash
set -euo pipefail

#python3 -m pip install --user --no-cache-dir pandas

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
exec python3 -u MNIST.py