#!/usr/bin/env bash
# Metrics for every checkpoint of both methods, then the summary and all figures.
# Needs torch + lpips + matplotlib + scipy (the 3DGS env plus `pip install lpips matplotlib scipy`).
# Usage (from anywhere):  bash scripts/run_eval.sh      DRY_RUN=1 bash scripts/run_eval.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$ROOT"
mkdir -p "$RES/metrics"
activate "$GS_ENV"

for it in "${GS_ITERS[@]}"; do
    tag=$(pad6 "$it")
    run python scripts/eval_metrics.py --gt_dir "$SCENE/images" --list "$RES/test_list.txt" \
        --pred_dir "$OUT/eval/gs/iter_$tag" --method 3dgs --tag "$it" --out "$RES/metrics/gs_iter_$tag.json"
done
for s in "${NGP_STEPS[@]}"; do
    tag=$(pad6 "$s")
    run python scripts/eval_metrics.py --gt_dir "$SCENE/images" --list "$RES/test_list.txt" \
        --pred_dir "$OUT/eval/ngp/step_$tag" --method instant-ngp --tag "$s" --out "$RES/metrics/ngp_step_$tag.json"
done

run python scripts/collect_results.py --outputs "$OUT" --results "$RES" --logs "$LOGS"
run python scripts/make_figures.py --outputs "$OUT" --results "$RES" --scene "$SCENE" \
    --test_list "$RES/test_list.txt" --train_list "$RES/train_list.txt" \
    --fig_dir "$RES/figures" --video_dir "$RES/videos"
echo "Evaluation done: $RES/summary.md, $RES/figures/, $RES/videos/"
