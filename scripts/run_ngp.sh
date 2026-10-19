#!/usr/bin/env bash
# InstantNGP: train (timed, snapshots at every checkpoint), render every checkpoint's test
# views, benchmark, render novel paths.
# Usage (from anywhere):  bash scripts/run_ngp.sh       DRY_RUN=1 bash scripts/run_ngp.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$ROOT"
mkdir -p "$LOGS" "$RES/timing"
activate "$NGP_ENV"

NGP_OUT="$OUT/ngp"
FINAL="${NGP_STEPS[-1]}"
FINAL_SNAP="$NGP_OUT/step_$(pad6 "$FINAL").ingp"
check_gpu_idle

timed_train ngp "$LOGS/ngp_train.log" \
    python scripts/train_ngp.py --ngp_root "$NGP_ROOT" --scene "$SCENE/transforms_train.json" \
    --out_dir "$NGP_OUT" --checkpoints "${NGP_STEPS[@]}"

for s in "${NGP_STEPS[@]}"; do
    run python scripts/render_ngp.py --ngp_root "$NGP_ROOT" --snapshot "$NGP_OUT/step_$(pad6 "$s").ingp" \
        --transforms "$SCENE/transforms_test.json" --train_json "$SCENE/transforms_train.json" \
        --out "$OUT/eval/ngp/step_$(pad6 "$s")"
done

check_gpu_idle
run python scripts/render_ngp.py --ngp_root "$NGP_ROOT" --snapshot "$FINAL_SNAP" \
    --transforms "$SCENE/transforms_test.json" --train_json "$SCENE/transforms_train.json" \
    --bench "$BENCH_REPEATS" --timing_out "$RES/timing/ngp.json"

for path in "${NOVEL_PATHS[@]}"; do
    run python scripts/render_ngp.py --ngp_root "$NGP_ROOT" --snapshot "$FINAL_SNAP" \
        --transforms "$SCENE/novel/$path.json" --train_json "$SCENE/transforms_train.json" \
        --out "$OUT/novel/ngp/$path"
done

echo "InstantNGP done."
