#!/usr/bin/env bash
set -euo pipefail

#python3 -m pip install --user --no-cache-dir pandas

cd "$(dirname "$0")"

exec python3 -u /home/moso00002/bcosgnn/bcosgnn/shaique_updates/codes/MolHIV/molhiv_mean_agg_v2.py