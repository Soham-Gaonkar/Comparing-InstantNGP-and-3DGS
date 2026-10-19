# Shared settings for run_gs.sh / run_ngp.sh / run_eval.sh / run_all.sh.
# Edit environment names or iteration lists here. DRY_RUN=1 prints commands without running them.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE="$ROOT/data/scene"
GS_ROOT="$ROOT/repos/gaussian-splatting"
NGP_ROOT="$ROOT/repos/instant-ngp"
OUT="$ROOT/outputs"
RES="$ROOT/results"
LOGS="$ROOT/logs"

GS_ENV="${GS_ENV:-gs}"       # conda env with 3DGS (+ lpips, matplotlib, scipy for evaluation)
NGP_ENV="${NGP_ENV:-ngp}"    # conda env with the instant-ngp build

GS_ITERS=(1000 3000 7000 15000 30000)    # last one = final 3DGS model
NGP_STEPS=(1000 3000 7000 15000 35000 100000)   # last one = final InstantNGP model (35k = paper schedule, 100k = converged)
NOVEL_PATHS=(trajectory shift_up shift_down spiral dolly orbit look_around)
BENCH_REPEATS=20

DRY_RUN="${DRY_RUN:-0}"

run() {
    echo "+ $*"
    if [ "$DRY_RUN" != 1 ]; then "$@"; fi
}

activate() {
    echo "+ conda activate $1"
    if [ "$DRY_RUN" = 1 ]; then return; fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$1"
}

pad6() { printf "%06d" "$1"; }

# Sample GPU memory every second while process $1 runs; lines: epoch,proc|total,MiB
gpu_monitor() {
    local pid=$1 out=$2 gpu="${CUDA_VISIBLE_DEVICES:-0}"
    gpu="${gpu%%,*}"
    : > "$out"
    while kill -0 "$pid" 2>/dev/null; do
        local now; now=$(date +%s)
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', *' -v p="$pid" -v t="$now" '$1==p {print t",proc,"$2}' >> "$out"
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null \
            | awk -v t="$now" 'NF {print t",total,"$1}' >> "$out"
        sleep 1
    done
}

# timed_train <name> <logfile> <command...>
# Runs the command in the background with a GPU-memory monitor and writes
# $LOGS/<name>_train_time.json with wall-clock start, end and duration.
timed_train() {
    local name=$1 log=$2
    shift 2
    echo "+ $*  > $log"
    if [ "$DRY_RUN" = 1 ]; then return; fi
    local t0 t1 pid mon status=0
    t0=$(date +%s.%N)
    "$@" > "$log" 2>&1 &
    pid=$!
    gpu_monitor "$pid" "$LOGS/${name}_gpumem.csv" &
    mon=$!
    wait "$pid" || status=$?
    t1=$(date +%s.%N)
    wait "$mon" 2>/dev/null || true
    printf '{"start": %s, "end": %s, "wall_s": %s, "exit_status": %d}\n' \
        "$t0" "$t1" "$(awk -v a="$t0" -v b="$t1" 'BEGIN {printf "%.3f", b - a}')" "$status" \
        > "$LOGS/${name}_train_time.json"
    if [ "$status" -ne 0 ]; then
        echo "ERROR: $name training failed (exit $status); last lines of $log:"
        tail -n 30 "$log"
        exit "$status"
    fi
    echo "  $name training finished in $(awk -v a="$t0" -v b="$t1" 'BEGIN {printf "%.1f", b - a}') s"
}

check_gpu_idle() {
    if [ "$DRY_RUN" = 1 ]; then return; fi
    echo "GPU status before timing-sensitive work:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
    local others
    others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$others" -gt 0 ]; then
        echo "WARNING: $others other process(es) are using the GPU - timings will not be clean."
    fi
}
