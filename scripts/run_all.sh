#!/usr/bin/env bash
# Whole pipeline on the GPU server: 3DGS, InstantNGP, then evaluation + figures.
# Run inside tmux:  bash scripts/run_all.sh 2>&1 | tee logs/run_all.log
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$DIR/run_gs.sh"
bash "$DIR/run_ngp.sh"
bash "$DIR/run_eval.sh"
