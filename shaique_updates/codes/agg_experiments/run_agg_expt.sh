#!/usr/bin/env bash
set -euo pipefail

#python3 -m pip install --user --no-cache-dir pandas

cd "$(dirname "$0")"

exec python3 -u /home/moso00002/bcosgnn/bcosgnn/shaique_updates/codes/agg_experiments/GINE_Di_Halo_mean_agg.py