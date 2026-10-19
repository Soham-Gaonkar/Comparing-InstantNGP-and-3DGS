#!/usr/bin/env bash
# 3DGS: train (timed), render every checkpoint's test views, benchmark, render novel paths,
# and run the official render.py + metrics.py once as a cross-check of our metric code.
# Usage (from anywhere):  bash scripts/run_gs.sh        DRY_RUN=1 bash scripts/run_gs.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$ROOT"
mkdir -p "$LOGS" "$RES/timing"
activate "$GS_ENV"

GS_OUT="$OUT/gs"
FINAL="${GS_ITERS[-1]}"
check_gpu_idle

timed_train gs "$LOGS/gs_train.log" \
    python "$GS_ROOT/train.py" -s "$SCENE" -m "$GS_OUT" --eval -r 1 \
    --iterations "$FINAL" --save_iterations "${GS_ITERS[@]}" --test_iterations "$FINAL" --disable_viewer

for it in "${GS_ITERS[@]}"; do
    run python scripts/render_gs.py --gs_root "$GS_ROOT" -m "$GS_OUT" --iteration "$it" \
        --out "$OUT/eval/gs/iter_$(pad6 "$it")"
done

check_gpu_idle
run python scripts/render_gs.py --gs_root "$GS_ROOT" -m "$GS_OUT" --iteration "$FINAL" \
    --bench "$BENCH_REPEATS" --timing_out "$RES/timing/gs.json"

for path in "${NOVEL_PATHS[@]}"; do
    run python scripts/render_gs.py --gs_root "$GS_ROOT" -m "$GS_OUT" --iteration "$FINAL" \
        --novel "$SCENE/novel/$path.json" --out "$OUT/novel/gs/$path"
done

# official pipeline on the final model (writes outputs/gs/test/ours_<FINAL> and outputs/gs/results.json)
run python "$GS_ROOT/render.py" -m "$GS_OUT" --iteration "$FINAL" --skip_train \
    || echo "WARNING: official render.py failed (cross-check only)"
run python "$GS_ROOT/metrics.py" -m "$GS_OUT" \
    || echo "WARNING: official metrics.py failed (cross-check only)"

echo "3DGS done."
